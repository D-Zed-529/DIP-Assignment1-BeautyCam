"""自动 HDR 连拍融合（Phase 1，拍照模式，不进逐帧效果链）。

经典线实现（PLAN §2 关键绕坑决策）：
  macOS 的 AVFoundation 不支持可靠的手动曝光控制，真实包围曝光拍不了，
  改用「连拍（自然手持抖动）+ gamma 曲线模拟不同曝光」——效果等价、
  完全可控、可复现（答辩也好讲）。

流程：
  连拍 N 张（间隔捕获自然抖动）
    → 每张施加 EV 偏移（LUT gamma 模拟：ev>0 提亮模拟过曝）
    → findTransformECC 帧间对齐（EUCLIDEAN：旋转+平移，手持抖动模型）
    → MergeMertens 曝光融合（输出 float [0,1]，无需真实辐射标定）
    →（可选）Drago / Reinhard 色调映射，高光滚降更柔（显示风格选项）

Mertens 融合不产生真 HDR（32F 辐射图），它是多尺度加权融合直接出
显示域图——对"逆光成片高光不过曝、暗部有细节"的验收线正合适，
tonemap 选项作为观感风格作用于融合结果（伪 HDR 域）。
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ------- 调参常量 -------
# EV 组合预设：模拟包围曝光的档位（P1-1）。±1EV 三张为默认（覆盖大多数
# 逆光场景且连拍快）；±1.5 五张留作极端光比场景。
EV_PRESETS = {
    "3张±1EV": (-1.0, 0.0, 1.0),
    "3张±2EV": (-2.0, 0.0, 2.0),
    "5张±1.5EV": (-1.5, -0.75, 0.0, 0.75, 1.5),
}
DEFAULT_EV_PRESET = "3张±1EV"

# 连拍间隔（秒）：拉出一点间隔让手持自然抖动有差异（ECC 有东西可对齐）
BURST_INTERVAL_S = 0.12

# ECC 对齐参数：EUCLIDEAN（旋转+平移）匹配手持抖动；50 次迭代 + 1e-4
# 收敛阈值在精度/耗时间平衡（720p 单对 ~20-40ms）
ECC_MAX_ITERS = 50
ECC_EPS = 1e-4
ECC_GAUSS_FILT = 5

# tonemap 显示参数（经验值：不过度压灰）
TONEMAP_DRAGO_PARAMS = dict(gamma=1.0, saturation=1.1)
TONEMAP_REINHARD_PARAMS = dict(gamma=1.0, intensity=0.0, light_adapt=0.8)


def exposure_lut(ev: float) -> np.ndarray:
    """EV 偏移 → 256 项 LUT（gamma 幂曲线模拟曝光偏移）。

    ev>0 提亮（模拟过曝）：指数 = 2^-ev < 1，高光平滑压缩而非线性硬 clip，
    保留了 Mertens 融合所需的高光层次；ev<0 压暗同理。
    """
    exponent = 2.0 ** (-float(ev))
    lut = (np.linspace(0.0, 1.0, 256) ** exponent) * 255.0
    return np.clip(np.round(lut), 0, 255).astype(np.uint8)


def simulate_exposure(frame_bgr: np.ndarray, ev: float) -> np.ndarray:
    """单帧施加 EV 偏移（LUT，O(1) 每像素）。"""
    return cv2.LUT(frame_bgr, exposure_lut(ev))


def align_to_reference(ref_bgr: np.ndarray, frame_bgr: np.ndarray
                       ) -> np.ndarray:
    """ECC 对齐 frame → ref（EUCLIDEAN 模型），返回对齐后的 BGR。

    ⚠️ 方向坑（实测）：findTransformECC(template, input, warp) 求出的
    warp 是 **template→input** 方向；把 input 拉回 template 坐标系必须
    取仿射逆（invertAffineTransform）。直接拿求解结果 warpAffine 会把
    帧推得更歪（合成平移实验：误用 before 13.3 → after 23.8，取逆
    13.3 → 1.6）。

    失败（完全不相似/纯色帧）时退回原帧——对齐是增强项，不是硬前提。
    灰度 float32 域上求解；gamma 模拟造成的是单调亮度映射，边缘结构
    不变，ECC（增强相关系数，对线性亮度变化不变）可正常收敛。
    """
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    ECC_MAX_ITERS, ECC_EPS)
        cv2.findTransformECC(ref_gray, gray, warp, cv2.MOTION_EUCLIDEAN,
                             criteria, None, ECC_GAUSS_FILT)
    except cv2.error:
        return frame_bgr
    h, w = ref_bgr.shape[:2]
    inv = cv2.invertAffineTransform(warp)   # template→input 的逆 = input→template
    return cv2.warpAffine(frame_bgr, inv, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def merge_exposure_stack(stack_bgr: list[np.ndarray]) -> np.ndarray:
    """Mertens 曝光融合：N 张对齐后的 BGR → uint8 融合图。"""
    if not stack_bgr:
        raise ValueError("空曝光栈")
    merged = cv2.createMergeMertens().process(stack_bgr)   # float32 [0,1]
    return np.clip(merged * 255.0, 0, 255).astype(np.uint8)


def tonemap_frame(merged_bgr: np.ndarray,
                  method: Optional[str] = None) -> np.ndarray:
    """对融合结果做显示用色调映射（P1-4，预览不刺眼的观感选项）。

    method: None 原样 / "drago" 柔和高光滚降 / "reinhard" 自适应局部对比。
    输入 uint8，先升到 float [0,1]（tonemap 的合法输入域），映射后回 uint8。

    ⚠️ 实测坑（实拍图复现）：Drago 是 log 域算法，输入含 0 值像素
    （Mertens 融合的欠曝黑区必然存在，实测 merged.min()==0）会产出
    NaN，cast uint8 后整图全黑（一张 720p 全黑 JPEG 恒为 ~15KB，
    两组实拍 final 大小一字节不差即此症状）。输入抬底 + 输出 NaN
    消毒双保险。
    """
    if not method:
        return merged_bgr
    f = merged_bgr.astype(np.float32) / 255.0
    f = np.maximum(f, 1e-4)                    # 防 log(0)
    if method == "drago":
        tm = cv2.createTonemapDrago(**TONEMAP_DRAGO_PARAMS)
    elif method == "reinhard":
        tm = cv2.createTonemapReinhard(**TONEMAP_REINHARD_PARAMS)
    else:
        raise ValueError(f"未知色调映射：{method!r}（应为 drago / reinhard）")
    out = tm.process(f)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    # tonemap 输出峰值不定（Reinhard 理论上到不了 1）：按峰值归一到全量程，
    # 下限 0.2 防纯色帧时噪声被过度拉伸
    peak = max(float(out.max()), 0.2)
    return np.clip(out / peak * 255.0, 0, 255).astype(np.uint8)


def hdr_pipeline(frames_raw: list[np.ndarray],
                 evs: tuple[float, ...] = EV_PRESETS[DEFAULT_EV_PRESET],
                 tonemap: Optional[str] = None,
                 ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """连拍帧 → HDR 成片。返回 (成片, 各 EV 曝光图列表, 融合原片)。

    frames_raw：连拍原始帧（BGR，未施加 EV）；数量须与 evs 一致。
    成片 = tonemap(merge)；融合原片 = merge 结果（tonemap 前），两者都
    存档便于"单张 vs 融合 vs 映射"对照（P1 验收要求）。
    """
    if len(frames_raw) != len(evs):
        raise ValueError(f"连拍 {len(frames_raw)} 张与 EV 档 {len(evs)} 个不一致")
    exposed = [simulate_exposure(f, ev) for f, ev in zip(frames_raw, evs)]
    # 以 0EV 帧（evs 中最接近 0 的）为参考对齐，几何基准最中性
    ref_i = int(np.argmin(np.abs(np.asarray(evs))))
    ref = exposed[ref_i]
    aligned = [align_to_reference(ref, img) for img in exposed]
    merged = merge_exposure_stack(aligned)
    return tonemap_frame(merged, tonemap), exposed, merged


# 中文字体候选路径（Pillow 画标签用；找不到就退回 ASCII，不影响产出）。
# OpenCV 的 Hershey 字体只有 ASCII 字形，中文会渲染成一串「?」。
_CJK_FONTS = (
    "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)
_LABEL_FONT: Optional[ImageFont.FreeTypeFont] = None
_LABEL_FONT_TRIED = False


def _label_font(size: int = 22):
    """懒加载中文字体；找不到可用字体返回 None（调用方退回 cv2 画 ASCII）。"""
    global _LABEL_FONT, _LABEL_FONT_TRIED
    if not _LABEL_FONT_TRIED:
        _LABEL_FONT_TRIED = True
        for path in _CJK_FONTS:
            if os.path.exists(path):
                try:
                    _LABEL_FONT = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
    return _LABEL_FONT


def _label_image(img: np.ndarray, text: str) -> np.ndarray:
    """在图上贴左上角标签（黑底白字）。用 Pillow 渲染中文字形。"""
    out = img.copy()
    font = _label_font()
    if font is None:
        # 无中文字体：退回 ASCII（标签会变「?」，但至少不崩）
        cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return out
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    box = draw.textbbox((0, 0), text, font=font)
    draw.rectangle((0, 0, box[2] + 16, box[3] + 10), fill=(0, 0, 0))
    draw.text((8, 5), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def build_hdr_comparison(single_bgr: np.ndarray, merged: np.ndarray,
                         final: np.ndarray, evs: tuple[float, ...]) -> np.ndarray:
    """「单张 vs 融合 vs 映射」横拼对照图（答辩素材，P1 验收要求）。"""
    panels = [single_bgr, merged, final]
    labels = [f"单张 0EV", "Mertens 融合", "融合+色调映射"]
    return np.hstack([_label_image(img, lab) for img, lab in zip(panels, labels)])
