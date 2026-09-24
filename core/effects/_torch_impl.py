"""效果链的 PyTorch CUDA 实现（张量进出版）—— 各效果 process_gpu 的底座。

与各 CPU 参考实现（同模块内 numpy/cv2 版）的对应关系与口径差异：
  - 语义对齐：参数含义、掩膜管线、时域状态机与 CPU 版一一对应；
  - 数值允许小幅差异（float LAB vs cv2 8bit LUT-LAB、RGB 域双边 vs
    Lab 域双边等），视觉等价；CPU 版仍是"逐位等价"的基准与回退路径；
  - GPU 路径仅在 CUDA 可用时启用（Pipeline(use_gpu=True)；无 CUDA 时
    效果类自动走 CPU 版）。

帧流约定（2026-09 性能重构后）：本模块全部函数**张量进张量出**——
(1,3,H,W) uint8 BGR 设备张量输入、同布局输出，由 Pipeline 的 GPU 融合
路径串联：整条效果链只做一次帧上传/下载，效果之间零 PCIe 往返；
分割 alpha / 深度直接取 ctx 的张量形态（FrameContext 懒物化），同样
不经过 numpy。跨帧时域状态（EMA alpha、低光复用等）以设备张量持有。
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .. import gpuops as gpu
from ..gpuops import box_filter, gaussian_blur, guided_filter

# =====================================================================
# 公共：人脸轮廓掩膜（CPU fillPoly 半分辨率 + 上传）
# =====================================================================


def oval_mask_torch(h: int, w: int, landmarks_list: list[np.ndarray]
                    ) -> Optional[torch.Tensor]:
    """FaceMesh 轮廓多边形掩膜并集（fillPoly 在 CPU 半分辨率做，羽化上 GPU）。

    返回 (1,1,H,W) float [0,1]；无 landmarks 返回 None。
    """
    from ..infer import FACE_OVAL_IDS
    if not landmarks_list:
        return None
    half = (max(w // 2, 1), max(h // 2, 1))
    acc = np.zeros((half[1], half[0]), dtype=np.uint8)
    for lm in landmarks_list:
        pts = lm[FACE_OVAL_IDS][:, :2] * np.array(half)
        cv2.fillPoly(acc, [pts.astype(np.int32)], 255)
    acc = cv2.GaussianBlur(acc, (11, 11), 0)
    t = torch.from_numpy(acc.astype(np.float32) / 255.0)[None, None] \
        .to(gpu.device())
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)


def _quantize(f: torch.Tensor) -> torch.Tensor:
    """float [0,1] → uint8 张量（整链只在边界量化一次）。"""
    return (f.clamp_(0.0, 1.0) * 255.0).round_().to(torch.uint8)


# =====================================================================
# 美颜（BeautyEffect）
# =====================================================================

def beauty_smooth(f: torch.Tensor, mix: float,
                  diameter: int = 9, sigma: float = 60.0,
                  downscale: int = 4) -> torch.Tensor:
    """磨皮：低分辨率双边 → 上采样 → 混合（CPU 版 _smooth 的 GPU 对应）。

    GPU 版在 1/4 分辨率算（CPU 版是 1/2）：皮肤纹理是低频信号，上采样
    本身就是低通，视觉不可分；GPU 双边是移位叠加实现（每 tap ~6 个
    kernel），81 tap 在 1/2 分辨率 ~12ms / 1/3 ~7.5ms / 1/4 ~3.8ms。
    """
    if mix <= 0:
        return f
    h, w = f.shape[-2:]
    small = F.interpolate(f, size=(max(h // downscale, 1),
                                   max(w // downscale, 1)),
                          mode="area")
    small = gpu.bilateral_blur(small, diameter, sigma, sigma)
    smooth = F.interpolate(small, size=(h, w), mode="bilinear",
                           align_corners=False)
    return f * (1.0 - mix) + smooth * mix


def skin_mask(f_bgr: torch.Tensor) -> torch.Tensor:
    """YCrCb 肤色阈值掩膜（半分辨率计算），返回全分辨率 (1,1,H,W) float。

    f_bgr: BGR float [0,1]（全分辨率）。CPU 参考：beauty.get_skin_mask。
    """
    h, w = f_bgr.shape[-2:]
    small = F.interpolate(f_bgr, size=(max(h // 2, 1), max(w // 2, 1)),
                          mode="area")
    rgb = small.flip(1)
    ycrcb = gpu.rgb_to_ycrcb(rgb)                     # (1,3,h,w) [0,1] 域
    cr, cb = ycrcb[:, 1:2] * 255.0, ycrcb[:, 2:3] * 255.0
    m = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127))
    m8 = m.to(torch.uint8) * 255
    m8 = gpu.morph_op(m8, 5, dilate=False)            # 开运算（腐蚀+膨胀）
    m8 = gpu.morph_op(m8, 5, dilate=True)
    # cv2.GaussianBlur(7,7) 的隐式 σ = 0.3*((7-1)*0.5-1)+0.8 = 1.4
    m = gaussian_blur(m8.float() / 255.0, 1.4)
    return F.interpolate(m, size=(h, w), mode="bilinear",
                         align_corners=False)


def whiten_lab(f_bgr: torch.Tensor, alpha: torch.Tensor,
               strength: float) -> torch.Tensor:
    """LAB 美白：L += strength·alpha（cv2-L 域口径：物理 L 增量 = strength·100/255）。

    alpha 区域外逐位不变（按 alpha 混合回原帧，LAB 浮点往返误差不外泄）。
    """
    lab = gpu.rgb_to_lab(f_bgr.flip(1))
    lab[:, 0:1] = lab[:, 0:1] + strength * (100.0 / 255.0) * alpha
    out = gpu.lab_to_rgb(lab).flip(1)
    return f_bgr * (1.0 - alpha) + out * alpha


def _warp_roi(f: torch.Tensor, map_x: np.ndarray, map_y: np.ndarray,
              roi: tuple[int, int, int, int]) -> torch.Tensor:
    """cv2.remap(ROI 子图, map-roi 原点) 的 GPU 对应（REPLICATE 边界）。"""
    x1, y1, x2, y2 = roi
    dev = f.device
    mx = torch.from_numpy(map_x - np.float32(x1)).to(dev)
    my = torch.from_numpy(map_y - np.float32(y1)).to(dev)
    sub = f[:, :, y1:y2, x1:x2]
    warped = gpu.remap(sub, mx, my)
    out = f.clone()
    out[:, :, y1:y2, x1:x2] = warped
    return out


def beauty_slim_eyes(f_bgr: torch.Tensor, faces: list, params: dict,
                     w: int, h: int) -> torch.Tensor:
    """瘦脸（v5 液化）+ 大眼：位移场仍由 numpy 纯函数构建（毫秒级），
    重采样的 remap 上 GPU。全程 float 域（不做中间量化）。"""
    from .beauty import _slim_maps_core, estimate_yaw_deg, pose_gate
    f = f_bgr
    for face in faces:
        lm = face.landmarks
        yaw = estimate_yaw_deg(lm)
        if params["slim"] > 0:
            mx, my, roi = _slim_maps_core(w, h, lm, params["slim"],
                                          yaw_deg=yaw)
            if roi is not None:
                f = _warp_roi(f, mx, my, roi)
        if params["eye_enabled"] and params["eye_strength"] > 0:
            s = params["eye_strength"] * pose_gate(yaw)
            if s > 0:
                f = _enlarge_eyes_t(f, lm, s, w, h)
    return f


def _enlarge_eyes(f: torch.Tensor, landmarks: np.ndarray,
                  strength: float, w: int, h: int) -> torch.Tensor:
    """大眼（CPU enlarge_eyes 的 GPU 对应：局部 remap + 羽化混合，float 域）。"""
    from .beauty import EYE_MIN_CORNER_RATIO, EYE_RADIUS_FROM_CORNERS, \
        EYE_RADIUS_RATIO, LEFT_EYE_IDS, RIGHT_EYE_IDS
    out = f
    corner_pairs = {LEFT_EYE_IDS: (33, 133), RIGHT_EYE_IDS: (362, 263)}
    changed = False
    for eye_ids, (c_out, c_in) in corner_pairs.items():
        cx = float(np.mean(landmarks[list(eye_ids), 0]) * w)
        cy = float(np.mean(landmarks[list(eye_ids), 1]) * h)
        corner_dist = abs(float(landmarks[c_out, 0]) - float(landmarks[c_in, 0]))
        if corner_dist < EYE_MIN_CORNER_RATIO:
            continue
        r = int(min(EYE_RADIUS_FROM_CORNERS * corner_dist * w,
                    EYE_RADIUS_RATIO * w))
        if r < 4:
            continue
        x1, x2 = max(int(cx - r), 0), min(int(cx + r), w)
        y1, y2 = max(int(cy - r), 0), min(int(cy + r), h)
        if x2 <= x1 or y2 <= y1:
            continue
        xs, ys = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
        dx, dy = xs - cx, ys - cy
        dist = np.sqrt(dx * dx + dy * dy)
        scale = 1.0 - strength * np.clip(1.0 - dist / r, 0.0, 1.0)
        src_x = np.clip(cx + dx * scale, 0, w - 1).astype(np.float32)
        src_y = np.clip(cy + dy * scale, 0, h - 1).astype(np.float32)
        sub = out[:, :, y1:y2, x1:x2]
        roi = gpu.remap(sub, torch.from_numpy(src_x).to(f.device),
                        torch.from_numpy(src_y).to(f.device))
        weight = torch.from_numpy(
            np.clip(1.0 - dist / r, 0.0, 1.0)[None, None].astype(np.float32)
        ).to(f.device)
        out = out.clone()
        out[:, :, y1:y2, x1:x2] = roi * weight + out[:, :, y1:y2, x1:x2] \
            * (1.0 - weight)
        changed = True
    return out if changed else f


def _med3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """三元素中值（3 次 min/max，精确）。"""
    mn = torch.minimum(a, b)
    mx = torch.maximum(a, b)
    return torch.minimum(mx, torch.maximum(mn, c))


def beauty_finish(f: torch.Tensor) -> torch.Tensor:
    """收尾：3×3 中值去噪 + 锐化核 + 量化（CPU 版 medianBlur+filter2D 对应）。

    中值用"行 med3 → 列 med3"的可分离近似（12 次 min/max 元素算子）：
    torch.stack(9 层)+median 的直接写法在 720p float 上要 36ms（排序核
    在 (9,3,H,W) 大张量上无快速路径），可分离版 ~1.5ms 且去噪效果等价。
    """
    p = F.pad(f, (1, 1, 1, 1), mode="replicate")
    hh, ww = f.shape[-2:]
    win = [p[:, :, dy:dy + hh, dx:dx + ww]
           for dy in range(3) for dx in range(3)]
    rows = [_med3(win[0], win[1], win[2]),
            _med3(win[3], win[4], win[5]),
            _med3(win[6], win[7], win[8])]
    med = _med3(rows[0], rows[1], rows[2])
    c = med.shape[1]
    k = torch.tensor([0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
                     device=med.device).view(1, 1, 3, 3).expand(c, 1, 3, 3)
    sharp = F.conv2d(med, k.contiguous(), groups=c, padding=1)
    return _quantize(sharp)


def beauty_process_t(f_u8: torch.Tensor, ctx, p: dict) -> torch.Tensor:
    """BeautyEffect 的整链 GPU 版（uint8 张量进 / uint8 张量出，float 单域）。"""
    h, w = f_u8.shape[-2:]
    f = f_u8.float().div_(255.0)                    # BGR float [0,1]

    if p["smooth"] > 0:
        f = beauty_smooth(f, p["smooth"])
    if p["whiten"] > 0:
        alpha = skin_mask(f)
        if p["whiten_scope"] == "face" and ctx.faces:
            oval = oval_mask_torch(h, w, [fc.landmarks for fc in ctx.faces])
            if oval is not None:
                alpha = torch.minimum(alpha, oval)
        if float(alpha.max()) > 0.01:
            f = whiten_lab(f, alpha, p["whiten"])

    if ctx.faces:
        f = beauty_slim_eyes(f, ctx.faces, p, w, h)

    return beauty_finish(f) if p["finish"] else _quantize(f)


# =====================================================================
# 虚拟背景（SegmentEffect）
# =====================================================================

def sharpen_matte_t(alpha: torch.Tensor, contrast: float) -> torch.Tensor:
    """matte 对比度拉伸（CPU sharpen_matte 对应）。"""
    c = float(np.clip(contrast, 0.0, 1.0))
    if c <= 0.0:
        return alpha
    width = 1.0 - 0.8 * c
    lo = 0.5 - 0.5 * width
    return ((alpha - lo) / width).clamp_(0.0, 1.0)


def ema_matte_t(new: torch.Tensor, prev: Optional[torch.Tensor],
                smooth: float, motion_relief: float = 0.9
                ) -> tuple[torch.Tensor, bool]:
    """运动自适应 EMA（CPU ema_matte 对应）。返回 (alpha, 是否有时域历史)。"""
    if prev is None or prev.shape != new.shape or smooth <= 0.0:
        return new, False
    tw = 80, 45
    a = F.interpolate(new, size=tw, mode="area")
    b = F.interpolate(prev, size=tw, mode="area")
    motion = float((a - b).abs().mean())
    motion = min(motion / 0.03, 1.0)
    w = float(np.clip(smooth, 0.0, 1.0)) * (1.0 - motion_relief * motion)
    return prev * w + new * (1.0 - w), True


def refine_matte_t(alpha: torch.Tensor, f_bgr: torch.Tensor,
                   radius: int) -> torch.Tensor:
    """guided filter 边缘精修（半分辨率引导，CPU refine_matte 对应）。"""
    h, w = alpha.shape[-2:]
    half = (max(w // 2, 1), max(h // 2, 1))
    small = F.interpolate(f_bgr, size=half, mode="area")
    guide = gpu.rgb_to_gray(small.flip(1))
    a_small = F.interpolate(alpha, size=half, mode="area")
    out = guided_filter(guide, a_small, max(1, int(radius)))
    return F.interpolate(out, size=(h, w), mode="bilinear",
                         align_corners=False)


def blurred_background_t(f_bgr: torch.Tensor, strength: float) -> torch.Tensor:
    """背景虚化（大 sigma 走降采样路径，CPU _blurred 对应）。"""
    from .segment import BLUR_DOWNSCALE_AT, BLUR_SIGMA_MAX
    sigma = BLUR_SIGMA_MAX * float(np.clip(strength, 0.0, 1.0))
    if sigma < 0.5:
        return f_bgr
    h, w = f_bgr.shape[-2:]
    if sigma > BLUR_DOWNSCALE_AT:
        fac = max(1, int(sigma / BLUR_DOWNSCALE_AT))
        sw, sh = max(w // fac, 1), max(h // fac, 1)
        small = F.interpolate(f_bgr, size=(sh, sw), mode="area")
        small = gaussian_blur(small, sigma / fac)
        return F.interpolate(small, size=(h, w), mode="bilinear",
                             align_corners=False)
    return gaussian_blur(f_bgr, sigma)


def segment_process_t(f_u8: torch.Tensor, effect, ctx) -> torch.Tensor:
    """SegmentEffect.process 的 GPU 版（张量进出）。

    effect 为 SegmentEffect 实例（读 _p() 参数与跨帧 GPU 状态 _prev_alpha_t；
    CPU 侧 _prev_alpha 与之独立，切后端时自动断开）。
    alpha 取 ctx.person_alpha_t（引擎直供张量，无 numpy 中转）。
    """
    p = effect._p()
    h, w = f_u8.shape[-2:]

    # ---- alpha 状态机（GPU 张量域） ----
    raw_t = ctx.person_alpha_t
    if raw_t is not None:
        if raw_t.shape[-2:] != (h, w):
            raw_t = F.interpolate(raw_t, size=(h, w), mode="bilinear",
                                  align_corners=False)
            effect._prev_alpha_t = None
        raw_t = sharpen_matte_t(raw_t, p["matte_contrast"])
        raw_t, _ = ema_matte_t(raw_t, effect._prev_alpha_t,
                               float(p["smooth"]))
        effect._prev_alpha_t = raw_t
    elif effect._prev_alpha_t is not None \
            and effect._prev_alpha_t.shape[-2:] != (h, w):
        effect._prev_alpha_t = None

    alpha = effect._prev_alpha_t
    if alpha is None:
        return f_u8                # 还没有任何掩膜：透传（CPU 版同语义）
    if float(alpha.mean()) < float(p["min_person_ratio"]):
        return f_u8                # 人像占比过低：透传

    if int(p["edge_shift"]):
        alpha = gpu.morph_op((alpha * 255).to(torch.uint8),
                             2 * abs(int(p["edge_shift"])) + 1,
                             dilate=int(p["edge_shift"]) < 0).float() / 255.0

    f = f_u8.float().div_(255.0)   # BGR float
    if p["refine"]:
        alpha = refine_matte_t(alpha, f, int(round(float(p["feather"]))))
    else:
        alpha = gaussian_blur(alpha, float(p["feather"]) / 2.0)
    alpha = alpha.clamp(0.0, 1.0)
    effect._last_alpha_t = alpha   # 排障口径（debug_alpha 懒物化 numpy）

    bg = segment_background_t(effect, p, f)
    out = f * alpha + bg * (1.0 - alpha)
    return _quantize(out)


def segment_background_t(effect, p: dict,
                         f_bgr: torch.Tensor) -> torch.Tensor:
    """三档背景（blur/image/color）的 GPU 版（背景张量带缓存）。"""
    from .segment import MODE_BLUR, MODE_COLOR, MODE_IMAGE, parse_hex_color
    h, w = f_bgr.shape[-2:]
    mode = p["mode"]
    if mode == MODE_BLUR:
        return blurred_background_t(f_bgr, float(p["strength"]))
    if mode == MODE_IMAGE:
        img = effect._cached((str(p["bg_path"]), w, h),
                             lambda: effect._load_image(str(p["bg_path"]),
                                                        w, h))
        if img is not None:
            key = ("gpu", str(p["bg_path"]), w, h)
            hit = effect._bg_gpu_cache.get(key)
            if hit is None:
                hit = gpu.upload_frame(img).float().div_(255.0)
                effect._bg_gpu_cache[key] = hit
            return hit
    color_key = ("gpu-color", str(p["bg_color"]), w, h)
    hit = effect._bg_gpu_cache.get(color_key)
    if hit is None:
        bgr = torch.tensor(parse_hex_color(str(p["bg_color"])),
                           device=f_bgr.device, dtype=torch.float32) / 255.0
        hit = bgr.view(1, 3, 1, 1).expand(1, 3, h, w).contiguous()
        effect._bg_gpu_cache[color_key] = hit
    return hit


# =====================================================================
# 自适应画质（AutoEnhanceEffect）
# =====================================================================

def autoenhance_process_t(f_u8: torch.Tensor, effect, ctx) -> torch.Tensor:
    """AutoEnhanceEffect.process 的 GPU 版（张量进出）。"""
    from .autoenhance import (BG_TARGET_L, CLAHE_TILES, FACE_MIN_PIXELS,
                              FACE_TARGET_L, SAT_SCALE_MAX, WB_GAIN_MAX,
                              WB_GAIN_MIN, ema_tuple, gamma_for_exposure,
                              gamma_lut)
    p = effect._p()
    if p["strength"] <= 0:
        return f_u8
    h, w = f_u8.shape[-2:]
    dev = gpu.device()
    f = f_u8.float().div_(255.0)     # BGR

    # 人脸软掩膜（fillPoly 在 CPU 半分辨率，羽化上 GPU；只需帧形状）
    mask01_t: Optional[torch.Tensor] = None
    face_bin_t: Optional[torch.Tensor] = None
    if ctx.faces:
        soft = oval_mask_torch(h, w, [fc.landmarks for fc in ctx.faces])
        if soft is not None and int((soft >= 0.5).sum()) >= FACE_MIN_PIXELS:
            mask01_t = soft
            face_bin_t = (soft >= 0.5).float()

    # ---- 1) 灰世界白平衡（背景区估计） ----
    out = f
    if p["color"] > 0:
        if face_bin_t is not None:
            bg_bin = 1.0 - face_bin_t
            bg_count = bg_bin.sum()
            bg_means = (out * bg_bin).sum(dim=(2, 3)) / bg_count.clamp_min(1.0)
            means = torch.where(bg_count >= 0.05 * h * w,
                                bg_means, out.mean(dim=(2, 3)))
        else:
            means = out.mean(dim=(2, 3))
        # 背景面积判断留在 GPU，三个通道均值一次性回传。
        means = means[0].detach().cpu().numpy()        # BGR（3 标量）
        gray = float(means.mean())
        if gray < 1e-6:
            gains = [1.0, 1.0, 1.0]
        else:
            gains = [gray / max(float(c), 1.0 / 255.0) for c in means]
            gains = [float(np.clip(g, WB_GAIN_MIN, WB_GAIN_MAX)) for g in gains]
            norm = sum(gains) / 3.0
            gains = [g / norm for g in gains]
        effect._ema_wb = ema_tuple(effect._ema_wb, tuple(gains),
                                   float(p["smooth"]))
        k = float(p["color"])
        eff = torch.tensor([1.0 + k * (g - 1.0) for g in effect._ema_wb],
                           device=dev).view(1, 3, 1, 1)
        out = out * eff

    # ---- 2) LAB：分区 gamma 曝光 + CLAHE + 饱和度 ----
    lab = gpu.rgb_to_lab(out.flip(1))                   # RGB
    L = lab[:, 0:1]                                     # 物理域 [0,100]

    L8 = (L / 100.0 * 255.0).round_().clamp_(0, 255).to(torch.uint8)
    face_mean = bg_mean = None
    if face_bin_t is not None:
        bgm = 1.0 - face_bin_t
        # 人脸/背景面积与亮度和打包下载，避免逐标量触发多次 D2H 同步。
        stats = torch.stack((face_bin_t.sum(), bgm.sum(),
                             (L * face_bin_t).sum(), (L * bgm).sum()))
        n_f, n_b, l_f, l_b = stats.detach().cpu().numpy()
        face_mean = float(l_f / n_f) if n_f > 0 else None
        bg_mean = float(l_b / n_b) if n_b > 0 else None
    else:
        bg_mean = float(L.mean())

    face_target = BG_TARGET_L + float(p["face_exposure"]) * (
        FACE_TARGET_L - BG_TARGET_L)
    g_face = gamma_for_exposure(face_mean, face_target)
    g_bg = gamma_for_exposure(bg_mean, BG_TARGET_L)
    effect._ema_face = ema_tuple(effect._ema_face, (g_face,), float(p["smooth"]))
    effect._ema_bg = ema_tuple(effect._ema_bg, (g_bg,), float(p["smooth"]))
    lut_f = torch.from_numpy(gamma_lut(effect._ema_face[0]))[None].to(dev)
    lut_b = torch.from_numpy(gamma_lut(effect._ema_bg[0]))[None].to(dev)
    v = L8[:, 0].long()                                # (1,H,W) 量化 L
    Lq = lut_b[0][v]
    if mask01_t is not None:
        Lq = Lq + (lut_f[0][v] - lut_b[0][v]) * mask01_t

    if p["contrast"] > 0:
        L8_new = gpu.clahe_lut(Lq.round_().clamp_(0, 255).to(torch.uint8),
                               CLAHE_TILES,
                               0.5 + 2.5 * float(p["contrast"]))
    else:
        L8_new = Lq

    L_new = L8_new.float() / 255.0 * 100.0
    sat = 1.0 + SAT_SCALE_MAX * float(p["saturation"])
    if sat > 1.0:
        lab = torch.cat([L_new,
                         lab[:, 1:2] * sat,
                         lab[:, 2:3] * sat], dim=1)
    else:
        lab = torch.cat([L_new, lab[:, 1:2], lab[:, 2:3]], dim=1)
    enhanced = gpu.lab_to_rgb(lab).flip(1)              # → BGR

    s = float(p["strength"])
    if s >= 1.0:
        return _quantize(enhanced)
    return _quantize(f * (1.0 - s) + enhanced * s)


# =====================================================================
# 低光（启发式 + SCI/Retinexformer）
# =====================================================================

def lowlight_heuristic_t(f_u8: torch.Tensor, p: dict,
                         is_dark_out: list) -> torch.Tensor:
    """LowLightEffect.process 的 GPU 版（线性增益 + Y 直方图均衡）。"""
    from .lowlight import GAIN_ALPHA, GAIN_BETA
    f = f_u8.float().div_(255.0)     # BGR
    h, w = f.shape[-2:]
    small = F.interpolate(f, size=(max(h // 4, 1), max(w // 4, 1)), mode="area")
    brightness = float(gpu.rgb_to_gray(small.flip(1)).mean()) * 255.0
    is_dark_out[0] = brightness < p["threshold"]
    if p["auto"] and not is_dark_out[0]:
        return f_u8
    if p["strength"] <= 0:
        return f_u8

    enhanced = (f * GAIN_ALPHA + GAIN_BETA / 255.0).clamp_(0.0, 1.0)
    rgb = enhanced.flip(1)
    ycrcb = gpu.rgb_to_ycrcb(rgb)
    y8 = (ycrcb[:, 0:1] * 255.0).round_().clamp_(0, 255).to(torch.uint8)
    lut = gpu.equalize_lut(y8)
    y_new = lut[0][y8[:, 0].long()] / 255.0
    ycrcb = torch.cat([y_new, ycrcb[:, 1:2], ycrcb[:, 2:3]], dim=1)
    enhanced = gpu.ycrcb_to_rgb(ycrcb).flip(1)
    s = float(p["strength"])
    return _quantize(enhanced * s + f * (1.0 - s))


def lowlight_dnn_blend_t(f_u8: torch.Tensor, session, p: dict,
                         out_state: dict) -> torch.Tensor:
    """LowLightDnnEffect 的 GPU 版：增强前向 + 隔帧复用 + 强度混合全在 GPU。

    out_state: {'last_dark': bool, 'last_enhanced_t': tensor|None,
                'frame_seen': int} 由效果实例持有（跨帧 GPU 状态）。
    会话无张量接口（如 SCI 的 ONNX 会话）时抛 NotImplementedError，
    由 Pipeline 落回 CPU 路径。
    """
    if not hasattr(session, "enhance_tensor"):
        raise NotImplementedError("低光会话无张量接口（ONNX 会话走 CPU）")
    f = f_u8.float().div_(255.0)     # BGR
    h, w = f.shape[-2:]
    small = F.interpolate(f, size=(max(h // 4, 1), max(w // 4, 1)),
                          mode="area")
    brightness = float(gpu.rgb_to_gray(small.flip(1)).mean()) * 255.0
    out_state["last_dark"] = brightness < p["threshold"]
    if p["auto"] and not out_state["last_dark"]:
        out_state["last_enhanced_t"] = None
        return f_u8
    if p["strength"] <= 0:
        return f_u8

    interval = max(1, int(p["infer_interval"]))
    due = out_state["frame_seen"] % interval == 0
    out_state["frame_seen"] += 1
    last = out_state["last_enhanced_t"]
    if due or last is None or last.shape[-2:] != (h, w):
        enhanced = session.enhance_tensor(f)            # (1,3,H,W) float BGR
        out_state["last_enhanced_t"] = enhanced
    else:
        enhanced = last
    s = float(p["strength"])
    return _quantize(enhanced * s + f * (1.0 - s))


# =====================================================================
# 深度渐进虚化（BokehEffect）
# =====================================================================

def _smoothstep_t(x: torch.Tensor) -> torch.Tensor:
    return x * x * (3.0 - 2.0 * x)


def _focus_depth_t(depth: torch.Tensor,
                   alpha: Optional[torch.Tensor]):
    """焦平面深度（张量标量）：人像区中位（有掩膜），否则高分位。"""
    from .bokeh import FOCUS_PERCENTILE, PERSON_ALPHA_MIN
    if alpha is not None and float(alpha.mean()) >= PERSON_ALPHA_MIN:
        m = alpha > 0.6
        if int(m.sum()) > 64:
            return depth[m].median()
    return torch.quantile(depth.flatten().float(),
                          FOCUS_PERCENTILE / 100.0)


def bokeh_process_t(f_u8: torch.Tensor, effect, ctx, p: dict
                    ) -> torch.Tensor:
    """BokehEffect 的 GPU 版（CPU depth_blur 的张量对应）。

    金字塔在 1/4 分辨率上构建（模糊是低频），K 档 sigma 阶梯按"相邻档
    线性插值"退化为 Σ_k clamp(1−|t−k|,0,1)·level_k 的加权叠加（与 CPU
    版 floor/lerp 数学等价）。深度来自 ctx.depth_t（引擎直供张量）或
    效果的隔帧缓存（inference_interval>1 时中间帧复用）。
    """
    from .bokeh import FOCUS_RANGE, PYRAMID_LEVELS, PYR_FACTOR, SIGMA_MAX
    sigma_max = SIGMA_MAX * float(np.clip(p["strength"], 0.0, 1.0))
    if sigma_max < 0.5:
        return f_u8

    depth = ctx.depth_t
    if depth is None:
        depth = getattr(effect, "_depth_cache_t", None)
    else:
        effect._depth_cache_t = depth
    if depth is None:
        return f_u8           # 无深度输入（透传；告警由 CPU 语义分支负责）

    alpha = ctx.person_alpha_t if p.get("use_matte") else None
    focus = _focus_depth_t(depth, alpha)
    rng = max(float(p["range"]), 1e-3)

    f = f_u8.float().div_(255.0)
    h, w = f.shape[-2:]
    hw, hh = max(w // PYR_FACTOR, 1), max(h // PYR_FACTOR, 1)
    small = F.interpolate(f, size=(hh, hw), mode="area")
    depth_s = F.interpolate(depth, size=(hh, hw), mode="bilinear",
                            align_corners=False)
    if alpha is not None and alpha.shape[-2:] != (h, w):
        alpha = F.interpolate(alpha, size=(h, w), mode="bilinear",
                              align_corners=False)

    # 逐像素目标 sigma（1/4 分辨率域）
    dist_s = (depth_s - focus).abs() / rng
    smap_s = sigma_max * _smoothstep_t(dist_s.clamp(0.0, 1.0))

    # sigma 阶梯金字塔 + 相邻档加权叠加
    ladder = np.linspace(0.0, sigma_max, PYRAMID_LEVELS)
    t = smap_s / max(sigma_max, 1e-3) * (PYRAMID_LEVELS - 1)
    blended = small * (1.0 - t.clamp(0.0, 1.0))   # k=0 权重（t∈[0,1] 时）
    for k, s in enumerate(ladder[1:], start=1):
        sig = max(s / PYR_FACTOR, 0.5)
        w_k = (1.0 - (t - float(k)).abs()).clamp_(0.0, 1.0)
        blended = blended + gaussian_blur(small, sig) * w_k
    bokeh = F.interpolate(blended, size=(h, w), mode="bilinear",
                          align_corners=False)

    # 清晰权重：焦平面高斯衰减；人像区强制清晰
    dist = (depth - focus).abs() / rng
    w_clear = torch.exp(-dist * dist)
    if alpha is not None:
        w_clear = torch.maximum(w_clear, alpha)
    out = f * w_clear + bokeh * (1.0 - w_clear)
    return _quantize(out)
