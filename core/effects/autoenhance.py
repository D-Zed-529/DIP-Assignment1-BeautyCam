"""自适应画质优化 —— 经典 DIP 全时段画面校正（分区自动曝光 + CLAHE + 灰世界白平衡 + 饱和度）。

定位与低光增强（lowlight / lowlight_dnn）的分工：
  - 低光两档是"整帧亮度 < 60 的极端暗光"才触发的增强线；
  - 本效果面向**全时段**的常态问题：逆光人脸过暗、背景轻微过曝、整体
    偏色、画面发灰。两者可叠加 —— 本效果放链首，低光的暗光判定看到的
    是校正后的亮度，触发更准。

算法全部为经典数字图像处理（不新增任何模型）：
  1. 分区直方图统计：FaceMesh 轮廓掩膜（复用 beauty.face_oval_mask；
     美颜默认也在跑 faces 推理，与美颜同开时零额外推理成本）把画面
     分成人脸/背景两区，在 LAB 的 L 通道分别统计均值；
  2. 自动曝光 = 分区 gamma 校正：人脸目标亮度 135（正常皮肤口径）、
     背景目标 115；容差带内不校正（防常态抖动），gamma 裁剪防极端值。
     gamma 只作用于大尺度"照明层"（detail-preserving tone mapping）——
     局部对比度（眼窝/鼻影等 3D 明暗渐变）原样保留，提亮不把脸压成
     "面具"；脸/背景两个幂变换 LUT 按尺度自适应宽高斯羽化的软掩膜逐
     像素混合，过渡带无硬边；强逆光（背景比脸亮很多）时脸目标向背景
     回退，避免"亮脸贴亮底"的贴片感；
  3. 对比度 = CLAHE（限制对比度自适应直方图均衡，Zuiderveld 1994）
     作用于 L 通道；
  4. 色调 = 灰世界假设白平衡（Buchsbaum 1980）：在**背景区**估计通道
     增益 —— 人脸肤色的通道统计会系统性污染灰世界估计；
  5. 饱和度 = LAB 域就地缩放 a/b 通道（零颜色空间转换成本）；
  6. 时域稳定 = gamma 与白平衡增益做**参数级** EMA（平滑统计量而非
     图像，相机 AE 的标准做法）：逐帧直方图抖动不致画面闪烁。

无脸（检测丢失 / 人离开画面）时自动退化为全图按背景目标处理，效果不断档。
faces 推理每帧跑（inference_interval=1）：分区曝光贴脸是本效果的核心，
掩膜滞后一帧会在转头时造成曝光区域错位闪烁；统计量已有 EMA 防抖。
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from ..context import FaceInfo, FrameContext
from ..pipeline import Effect, NEED_FACES
from .beauty import face_oval_mask

# ------- 调参常量（集中顶部，中文注释） -------
FACE_TARGET_L = 135.0    # 人脸目标亮度（LAB-L）：正常皮肤口径（旧 150 吹成白纸，脸发"面具感"）
BG_TARGET_L = 115.0      # 背景目标亮度：中灰偏上（再高会把夜景硬拉成白天）
EXPO_TOL = 12.0          # 均值落在目标 ± 此带内不校正：常态画面防无谓抖动
GAMMA_MIN = 0.4          # gamma 裁剪下限（≤0.4 已是强提亮，再低噪声放大明显）
GAMMA_MAX = 2.5          # gamma 裁剪上限（压暗上限）
BACKLIGHT_DAMP_OFFSET = 30.0   # 逆光量（bg_mean - face_mean）超过此值开始回退脸目标
BACKLIGHT_DAMP_RANGE = 60.0    # 逆光量从 OFFSET 升到 OFFSET+RANGE 时脸目标线性回退到背景目标
PARTITION_FEATHER_RATIO = 0.02  # 分区混合羽化 sigma 占 min(w,h) 比例（旧 11px 高斯 ~6px 过渡带太硬）
PARTITION_FEATHER_MIN = 12.0    # 羽化 sigma 下限（px）
DETAIL_BASE_SIGMA_RATIO = 0.12  # "照明层"模糊 sigma 占 min(w,h) 比例（够大到盖过眼窝/鼻影等 3D 明暗，只把整体照明层留给 gamma）
DETAIL_BASE_MIN = 8.0           # 照明层模糊 sigma 下限（px）
DETAIL_RETAIN = 1.0             # 局部对比度保留比例（1.0 = 完全保留，只对 base 打 gamma）
DETAIL_BASE_DOWNSCALE = 8       # 照明层在 1/8 分辨率计算：base 是低频，大 sigma 高斯的成本按 D² 下降，且降采样本身已低通、缩小后的 sigma（86/D）才 ~11px，避开 OpenCV 大核高斯慢路径（实测 720p 全帧 sigma86 ≈430ms → 1/8 分辨率 ~0.9ms）
WB_GAIN_MIN = 0.8        # 白平衡增益裁剪：防个别帧统计把颜色拉飞
WB_GAIN_MAX = 1.25
WB_MIN_BG_RATIO = 0.05   # 背景像素占比低于此值时灰世界退回全图估计（脸占满屏）
CLAHE_CLIP_BASE = 0.5    # CLAHE clipLimit = BASE + SPAN × contrast 参数
CLAHE_CLIP_SPAN = 2.5
CLAHE_TILES = (8, 8)     # CLAHE 网格（8×8 为常用档，720p 下块感不明显）
SAT_SCALE_MAX = 0.35     # 饱和度=1.0 时的 a/b 缩放上限（1.35 倍，再高易色块）
FACE_MIN_PIXELS = 500    # 脸区像素低于此值当"无人脸"（掩膜过小则统计不可信）
SMOOTH_MAX = 0.95        # EMA 权重上限（=1 会永不收敛到新统计量）


# ------- 纯函数（可单测，不依赖状态） -------

def face_union_mask(frame_bgr: np.ndarray,
                    faces: Sequence[FaceInfo]) -> Optional[np.ndarray]:
    """全部人脸的轮廓掩膜并集（uint8 0~255，尺度自适应宽高斯羽化）。

    复用 beauty.face_oval_mask（FaceMesh 轮廓多边形 fillPoly + 11px 羽化），
    并集后再叠加一次宽高斯（sigma ≈ PARTITION_FEATHER_RATIO×min(w,h)）：
    分区曝光的脸/背景两套 gamma 在此软过渡带内平滑混合，脸缘不再有
    "贴图"硬边。无人脸返回 None（调用方走全图退化路径）。
    """
    if not faces:
        return None
    h, w = frame_bgr.shape[:2]
    acc: Optional[np.ndarray] = None
    for f in faces:
        lm = f.landmarks
        if lm is None or len(lm) == 0:
            continue
        m = face_oval_mask(frame_bgr, lm)
        acc = m if acc is None else np.maximum(acc, m)
    if acc is not None and acc.shape[:2] != (h, w):
        return None   # 防御：landmarks 与帧尺寸不符（理论不发生）
    # 分区混合专用宽羽化（不修改 face_oval_mask 本身 —— 美颜美白"仅脸部"仍要紧掩膜）。
    # 宽高斯在 1/4 分辨率算（掩膜是平滑 blob，视觉等价），避开全帧 sigma≈14
    # 大核高斯的慢路径（720p 实测 10.4ms → 0.3ms）。
    sigma = max(PARTITION_FEATHER_MIN, PARTITION_FEATHER_RATIO * min(h, w))
    sh, sw = max(h // 4, 1), max(w // 4, 1)
    small = cv2.resize(acc, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 4.0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def region_mean_l(l_ch: np.ndarray,
                  mask_u8: Optional[np.ndarray]) -> Optional[float]:
    """掩膜区域内 L 通道的均值 —— 用直方图口径计算（一次遍历）。

    mask_u8 非零处计入（OpenCV calcHist 语义，不加权）。区域像素数为 0
    返回 None。返回 float 是为了与 gamma_for_exposure 的连续数学对齐。
    """
    if mask_u8 is None:
        mask_u8 = None   # 全图
    hist = cv2.calcHist([l_ch], [0], mask_u8, [256], [0, 256])
    total = float(hist.sum())
    if total < 1.0:
        return None
    idx = np.arange(256, dtype=np.float32)
    return float(float((hist[:, 0] * idx).sum()) / total)


def gamma_for_exposure(mean_l: Optional[float],
                       target: float,
                       tol: float = EXPO_TOL) -> float:
    """由区域亮度均值求 gamma：使 (mean/255)^γ = target/255。

    mean 落在 target ± tol 容差带内返回 1.0（已达标不动，防常态抖动）；
    结果裁剪到 [GAMMA_MIN, GAMMA_MAX]。mean 为 None（区域空）也返回 1.0。
    """
    if mean_l is None:
        return 1.0
    if abs(mean_l - target) <= tol:
        return 1.0
    # 边界安全：mean=0 时 log(-inf) → γ→0；mean=255 时 log(0) → γ→inf。
    # 都交给裁剪兜底，绝不让 log 收到 0/负数产生 NaN（Drago 教训，AGENTS #17）
    m = min(max(mean_l, 1.0), 254.0) / 255.0
    t = min(max(target, 1.0), 254.0) / 255.0
    gamma = float(np.log(t) / np.log(m))
    return float(np.clip(gamma, GAMMA_MIN, GAMMA_MAX))


def face_target_backlight_damp(face_target: float,
                               bg_mean: Optional[float],
                               face_mean: Optional[float]) -> float:
    """逆光抑制：背景比脸亮很多时，人脸目标亮度向背景目标回退。

    逆光量 backlight = bg_mean - face_mean（>0 表示逆光）超过
    BACKLIGHT_DAMP_OFFSET 后线性回退：从 OFFSET 到 OFFSET+RANGE，
    脸目标从 face_target 平滑降到 BG_TARGET_L（强逆光下不再硬把脸
    拉亮，避免"亮脸贴亮底"的贴片感）。任一侧均值为 None（区域空/
    无人脸）时不做抑制。
    """
    if bg_mean is None or face_mean is None:
        return face_target
    backlight = bg_mean - face_mean
    if backlight <= BACKLIGHT_DAMP_OFFSET:
        return face_target
    t = min((backlight - BACKLIGHT_DAMP_OFFSET) / BACKLIGHT_DAMP_RANGE, 1.0)
    return float(BG_TARGET_L + (1.0 - t) * (face_target - BG_TARGET_L))


def gamma_lut(gamma: float) -> np.ndarray:
    """幂变换 LUT：lut[v] = round(255 × (v/255)^γ)，float32[256]。

    γ=1 时严格恒等（0..255 逐项相等），供"容差带内不动"的透传断言。
    """
    v = np.arange(256, dtype=np.float32) / 255.0
    return np.round(255.0 * np.power(v, gamma)).astype(np.float32)


def apply_partition_gamma(l_ch: np.ndarray, lut_face: np.ndarray,
                          lut_bg: np.ndarray,
                          mask01: Optional[np.ndarray]) -> np.ndarray:
    """分区 gamma 应用：脸区查 lut_face、背景查 lut_bg，软掩膜加权混合。

    mask01 为 float32 0~1 软掩膜（None = 全背景）。输出 uint8，形状同输入。
    """
    base = lut_bg[l_ch]
    if mask01 is None:
        return base.astype(np.uint8)
    out = base + (lut_face[l_ch] - base) * mask01
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def detail_preserving_gamma(l_ch: np.ndarray, lut_face: np.ndarray,
                            lut_bg: np.ndarray,
                            mask01: Optional[np.ndarray]) -> np.ndarray:
    """细节保留的分区 gamma：gamma 只打大尺度"照明层"，局部对比度原样保留。

    经典 detail-preserving tone mapping：base = blur(L)（大尺度照明）、
    detail = L - base（局部对比度，含 3D 明暗渐变 + 皮肤纹理），只对 base
    查 lut 曝光，再回加 detail。这样提亮不再把眼窝/鼻影等阴影压平（"面具
    感"主因）。对平坦区域 detail=0，退化为 apply_partition_gamma 的原行为。
    """
    h, w = l_ch.shape[:2]
    sigma = max(DETAIL_BASE_MIN, DETAIL_BASE_SIGMA_RATIO * min(h, w))
    # base 是低频照明层：整段（降采样 → 大 sigma 高斯 → 分区 LUT 混合 →
    # 上采样）都在 1/4 分辨率做，成本近似按 D² 下降（720p 大 sigma 全帧
    # 高斯 ~430ms → 本路径 ~5ms）。降采样本身是低通，视觉上等价于全分辨率
    # 大 sigma 高斯（beauty 磨皮的同款做法）。detail = L - base 会顺带把
    # 上采样残差也算进细节层，仍属"高于 base 尺度的结构"，保留无害。
    d = DETAIL_BASE_DOWNSCALE
    sh, sw = max(h // d, 1), max(w // d, 1)
    small = cv2.resize(l_ch, (sw, sh), interpolation=cv2.INTER_AREA)
    small_base = cv2.GaussianBlur(small, (0, 0), sigma / d)
    small_mask = (None if mask01 is None else
                  cv2.resize(mask01, (sw, sh), interpolation=cv2.INTER_AREA))
    small_base_new = apply_partition_gamma(small_base, lut_face, lut_bg, small_mask)
    # 曝光修正场（平滑）：out = l + (base_new - base)。修正场在低分辨率算好、
    # 只上采样一次，省掉两次全帧 base/base_new 上采样 + 全帧 detail 相减。
    small_corr = small_base_new.astype(np.float32) - small_base.astype(np.float32)
    corr = cv2.resize(small_corr, (w, h), interpolation=cv2.INTER_LINEAR)
    out = l_ch.astype(np.float32) + corr
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def gray_world_gains(frame_bgr: np.ndarray,
                     mask_u8: Optional[np.ndarray] = None
                     ) -> tuple[float, float, float]:
    """灰世界假设白平衡增益（BGR 顺序）：g_c = gray_mean / mean_c。

    mask_u8 非零处计入统计（典型用法：传"背景掩膜"，避开肤色污染）。
    增益归一到均值 1（整体曝光水平不漂移，只纠通道比例），并裁剪到
    [WB_GAIN_MIN, WB_GAIN_MAX]。背景像素占比过低或全黑帧返回 (1,1,1)。
    """
    if mask_u8 is not None:
        area = float(np.count_nonzero(mask_u8))
        if area < frame_bgr.shape[0] * frame_bgr.shape[1] * WB_MIN_BG_RATIO:
            mask_u8 = None   # 背景太小不可信，退回全图
    if mask_u8 is not None:
        means = [float(v) for v in cv2.mean(frame_bgr, mask=mask_u8)[:3]]
    else:
        means = [float(v) for v in frame_bgr.reshape(-1, 3).mean(axis=0)]
    gray = sum(means) / 3.0
    if gray < 1e-6:
        return (1.0, 1.0, 1.0)   # 全黑帧：无色彩信息可校
    gains = [gray / max(c, 1.0) for c in means]
    gains = [float(np.clip(g, WB_GAIN_MIN, WB_GAIN_MAX)) for g in gains]
    norm = sum(gains) / 3.0      # 归一均值 1：只动通道比例不动整体亮度
    if norm < 1e-6:
        return (1.0, 1.0, 1.0)
    return tuple(g / norm for g in gains)   # type: ignore[return-value]


def apply_wb(frame_bgr: np.ndarray,
             gains: tuple[float, float, float]) -> np.ndarray:
    """通道增益应用（BGR 顺序）。uint8 进出，饱和裁剪。"""
    g = np.asarray(gains, dtype=np.float32)
    out = frame_bgr.astype(np.float32) * g
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def scale_saturation_ab(a_ch: np.ndarray, b_ch: np.ndarray,
                        scale: float) -> tuple[np.ndarray, np.ndarray]:
    """LAB 饱和度缩放：a/b 以 128 为中心乘 scale（>1 增艳，=1 恒等）。

    int16 中转避免 uint8 减法下溢（a/b 可为 0~255 全域）。
    """
    if scale <= 1.0:
        return a_ch, b_ch
    a = np.clip((a_ch.astype(np.int16) - 128) * scale + 128,
                0, 255).astype(np.uint8)
    b = np.clip((b_ch.astype(np.int16) - 128) * scale + 128,
                0, 255).astype(np.uint8)
    return a, b


def ema_tuple(prev: Optional[tuple], new: tuple,
              smooth: float) -> tuple:
    """参数级 EMA：out = w·prev + (1-w)·new（逐分量）。prev 为 None 直接采新值。

    smooth 是"信历史"的权重（0 = 不平滑）。相机 AE 对统计量做这类
    平滑而非对图像做，是为了防直方图逐帧抖动导致画面亮度/色彩闪变。
    """
    if prev is None or len(prev) != len(new):
        return tuple(new)
    w = float(np.clip(smooth, 0.0, SMOOTH_MAX))
    return tuple(w * p + (1.0 - w) * n for p, n in zip(prev, new))


def clahe_apply(l_ch: np.ndarray, contrast: float) -> np.ndarray:
    """CLAHE 对比度增强（L 通道）。contrast 0~1 映射 clipLimit，≤0 恒等。"""
    c = float(np.clip(contrast, 0.0, 1.0))
    if c <= 0.0:
        return l_ch
    clip = CLAHE_CLIP_BASE + CLAHE_CLIP_SPAN * c
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=CLAHE_TILES).apply(l_ch)


# ------- 效果 -------

class AutoEnhanceEffect(Effect):
    """自适应画质优化：分区自动曝光 + CLAHE 对比度 + 灰世界白平衡 + 饱和度。

    参数：
      strength:      总强度 0~1（结果与原图混合，0 = 透传）
      face_exposure: 人脸曝光优先 0~1 —— 人脸目标亮度从背景目标 115 插值
                     到 150；0 = 两区同目标（全图统一曝光校正）
      contrast:      CLAHE 强度 0~1（映射 clipLimit 0.5~3.0）
      color:         白平衡强度 0~1（灰世界增益向 1 混合；0 = 不校色）
      saturation:    饱和度增强 0~1（a/b 缩放 1.0~1.35）
      smooth:        统计量时域 EMA 系数 0~0.95（0 = 逐帧即时）

    跨帧状态只有统计量 EMA（标量元组），无图像缓存：帧尺寸变化天然安全
    （仍实现 reset_temporal 在切换采集源时断开统计历史）。
    """

    name = "autoenhance"
    needs = frozenset({NEED_FACES})
    supports_gpu = True          # 张量快路径见 _torch_impl.autoenhance_process_t

    @staticmethod
    def default_params() -> dict:
        return {
            "strength": 0.8,
            "face_exposure": 0.5,
            "contrast": 0.3,
            "color": 0.5,
            "saturation": 0.0,
            "smooth": 0.8,
        }

    def process_gpu(self, frame_t, ctx: FrameContext):
        from ._torch_impl import autoenhance_process_t
        return autoenhance_process_t(frame_t, self, ctx)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ema_face: Optional[tuple] = None   # (gamma_face,)
        self._ema_bg: Optional[tuple] = None     # (gamma_bg,)
        self._ema_wb: Optional[tuple] = None     # (g_b, g_g, g_r)

    def reset_temporal(self) -> None:
        """清空统计量 EMA（切换采集源 / 逐张独立批跑前调用）。"""
        self._ema_face = None
        self._ema_bg = None
        self._ema_wb = None

    def set_enabled(self, on: bool) -> None:
        super().set_enabled(on)
        if on:
            self.reset_temporal()

    @property
    def stats(self) -> dict:
        """最近一帧的平滑统计量（GUI 状态显示 / 排障用，不进热路径）。"""
        return {
            "gamma_face": self._ema_face[0] if self._ema_face else 1.0,
            "gamma_bg": self._ema_bg[0] if self._ema_bg else 1.0,
            "wb_gains": self._ema_wb or (1.0, 1.0, 1.0),
        }

    def process(self, frame: np.ndarray, ctx: FrameContext) -> np.ndarray:
        p = self._p()
        if p["strength"] <= 0:
            return frame

        # ---- 人脸软掩膜（应用混合用）与二值脸区（统计用）----
        soft = face_union_mask(frame, ctx.faces)
        mask01: Optional[np.ndarray] = None
        face_bin: Optional[np.ndarray] = None
        if soft is not None and np.count_nonzero(soft >= 128) >= FACE_MIN_PIXELS:
            mask01 = soft.astype(np.float32) / 255.0
            face_bin = (soft >= 128).astype(np.uint8)

        # ---- 1) 白平衡（BGR 域，背景区估计，避开肤色污染）----
        out = frame
        if p["color"] > 0:
            bg_mask = None if face_bin is None else (
                (face_bin == 0).astype(np.uint8) * 255)
            gains = gray_world_gains(frame, bg_mask)
            self._ema_wb = ema_tuple(self._ema_wb, gains, float(p["smooth"]))
            k = float(p["color"])
            eff = tuple(1.0 + k * (g - 1.0) for g in self._ema_wb)
            out = apply_wb(out, eff)

        # ---- 2) LAB：分区自动曝光 + CLAHE + 饱和度 ----
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        face_mean = region_mean_l(l_ch, face_bin)
        bg_mean = region_mean_l(
            l_ch, None if face_bin is None
            else ((face_bin == 0).astype(np.uint8) * 255))

        face_target = BG_TARGET_L + float(p["face_exposure"]) * (
            FACE_TARGET_L - BG_TARGET_L)
        face_target = face_target_backlight_damp(face_target, bg_mean, face_mean)
        g_face = gamma_for_exposure(face_mean, face_target)
        g_bg = gamma_for_exposure(bg_mean, BG_TARGET_L)
        self._ema_face = ema_tuple(self._ema_face, (g_face,), float(p["smooth"]))
        self._ema_bg = ema_tuple(self._ema_bg, (g_bg,), float(p["smooth"]))
        gamma_f = self._ema_face[0]
        gamma_b = self._ema_bg[0]

        lut_face = gamma_lut(gamma_f)
        lut_bg = gamma_lut(gamma_b)
        l_new = detail_preserving_gamma(l_ch, lut_face, lut_bg, mask01)
        l_new = clahe_apply(l_new, float(p["contrast"]))
        a_ch, b_ch = scale_saturation_ab(
            a_ch, b_ch, 1.0 + SAT_SCALE_MAX * float(p["saturation"]))

        out = cv2.cvtColor(cv2.merge((l_new, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

        s = float(p["strength"])
        if s >= 1.0:
            return out
        return cv2.addWeighted(out, s, frame, 1.0 - s, 0)
