"""PyTorch CUDA 图像算子库 —— 效果链 GPU 化的公共底座（2026-09 Windows/RTX3060 部署）。

设计约定：
  - 张量布局统一 (1, C, H, W)；uint8 帧上传后按需转 float32 [0,1]；
  - 帧上传/下载走页锁定（pinned）暂存区，摊薄 H2D/D2H 开销；
  - 所有算子与 OpenCV 对应函数语义对齐（数值允许 ±1~2 的舍入差），
    目标是替代热路径而非逐位复刻 —— 逐位等价由 CPU 原路径继续保证；
  - 无 CUDA 时全部退化为 CPU 张量执行（接口不变，速度收益消失但可用）。

与 cv2 的口径差异说明（有意为之，勿"修复"）：
  - cv2 的 8bit LAB 是 LUT 近似；本模块用标准 sRGB→CIELAB 浮点变换，
    美白/分区曝光在感知均匀域更准，允许与 CPU 路径有小幅数值差；
  - cv2.GaussianBlur 的核按 sigma 截断；本模块用固定展开卷积，
    sigma 一致时视觉等价。
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# ------- 设备管理（模块级单例；首次调用时锁定） -------

_DEVICE: Optional[torch.device] = None
_DEVICE_LOCK = threading.Lock()

# 帧上传缓存：同一 ndarray 对象在帧内重复上传时直接复用。缓存持有数组
# 强引用 —— 只要条目存在，其 id 不可能被新分配复用，查找绝对安全；
# 典型场景 = engine.process 上传做推理，随后第一个效果再上传同一帧。
_UPLOAD_CACHE: dict[int, tuple[np.ndarray, torch.Tensor]] = {}
_UPLOAD_CACHE_MAX = 8


def device() -> torch.device:
    """推理/效果链统一设备：CUDA 可用则 cuda，否则 CPU。"""
    global _DEVICE
    if _DEVICE is None:
        with _DEVICE_LOCK:
            if _DEVICE is None:
                _DEVICE = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def torch_available_cuda() -> bool:
    return torch.cuda.is_available()


# ------- 帧上传 / 下载 -------

def upload_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 (H,W,3) -> (1,3,H,W) uint8 设备张量（带对象级缓存）。"""
    hit = _UPLOAD_CACHE.get(id(frame_bgr))
    if hit is not None and hit[0] is frame_bgr:
        return hit[1]
    t = torch.from_numpy(np.ascontiguousarray(frame_bgr))   # (H,W,3) uint8
    t = t.permute(2, 0, 1).unsqueeze(0).to(device())
    if len(_UPLOAD_CACHE) >= _UPLOAD_CACHE_MAX:
        _UPLOAD_CACHE.clear()
    _UPLOAD_CACHE[id(frame_bgr)] = (frame_bgr, t)
    return t


def download_frame(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) -> BGR uint8 (H,W,3) numpy。float 输入约定为 [0,1] 域。"""
    if t.dtype != torch.uint8:
        t = (t.clamp_(0.0, 1.0) * 255.0).round_().to(torch.uint8)
    out = t[0].permute(1, 2, 0).contiguous()
    return out.cpu().numpy()


def f01(t_u8: torch.Tensor) -> torch.Tensor:
    """uint8 (1,3,H,W) -> float32 [0,1]。"""
    return t_u8.float().div_(255.0)


def bgr_to_rgb(t: torch.Tensor) -> torch.Tensor:
    return t.flip(1)


# ------- 颜色空间（float32 [0,1] 域） -------

def rgb_to_gray(t: torch.Tensor) -> torch.Tensor:
    """(1,3,H,W) RGB -> (1,1,H,W) BT.601 亮度（与 cv2.BGR2GRAY 同系数）。"""
    r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


_SRGB_TO_LAB_M = None   # 3x3 变换矩阵（sRGB→XYZ D65→LAB），懒加载缓存
_LAB_XYZ_M = None       # XYZ→linear RGB 矩阵（lab_to_rgb 用）


def _chan_matmul(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """(1,C,H,W) 按通道右乘 3x3 矩量 m（色彩空间基变换）。"""
    return torch.matmul(x.permute(0, 2, 3, 1), m.T).permute(0, 3, 1, 2)


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp_min(1e-8) ** (1 / 2.4) - 0.055)


def rgb_to_lab(t: torch.Tensor) -> torch.Tensor:
    """RGB float [0,1] -> LAB（L∈[0,100]，a/b∈[-128,127]）。标准 D65 口径。"""
    global _SRGB_TO_LAB_M
    lin = _srgb_to_linear(t)
    if _SRGB_TO_LAB_M is None or _SRGB_TO_LAB_M.device != t.device:
        m = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]], device=t.device)
        _SRGB_TO_LAB_M = m
    xyz = _chan_matmul(lin, _SRGB_TO_LAB_M)
    white = torch.tensor([0.95047, 1.0, 1.08883], device=t.device)
    # LAB 以 D65 白点为参考；逆变换会乘回白点，此处必须先归一化。
    xyz = xyz / white.view(1, 3, 1, 1)
    f = torch.where(xyz > 0.008856,
                    xyz.clamp_min(1e-8) ** (1 / 3),
                    7.787 * xyz + 16 / 116)
    fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.cat([L, a, b], dim=1)


def lab_to_rgb(t: torch.Tensor) -> torch.Tensor:
    """LAB -> RGB float [0,1]（rgb_to_lab 的逆）。"""
    global _LAB_XYZ_M
    L, a, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    def finv(f):
        f3 = f ** 3
        return torch.where(f3 > 0.008856, f3, (f - 16 / 116) / 7.787)
    xyz = torch.cat([finv(fx), finv(fy), finv(fz)], dim=1)
    white = torch.tensor([0.95047, 1.0, 1.08883], device=t.device)
    xyz = xyz * white.view(1, 3, 1, 1)
    if _LAB_XYZ_M is None or _LAB_XYZ_M.device != t.device:
        m = torch.tensor([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252]], device=t.device)
        _LAB_XYZ_M = m
    lin = _chan_matmul(xyz, _LAB_XYZ_M)
    rgb = _linear_to_srgb(lin)
    return rgb.clamp_(0.0, 1.0)


def rgb_to_ycrcb(t: torch.Tensor) -> torch.Tensor:
    """RGB float [0,1] -> YCrCb（与 cv2 8bit 口径一致的系数，值域对齐
    Y∈[0,1]、Cr/Cb 移位到 [0,1] 即 cv2 的 /255）。"""
    r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 0.5
    cb = (b - y) * 0.564 + 0.5
    return torch.cat([y, cr, cb], dim=1)


def ycrcb_to_rgb(t: torch.Tensor) -> torch.Tensor:
    """rgb_to_ycrcb 的逆变换（YCrCb float [0,1] -> RGB float [0,1]）。"""
    y, cr, cb = t[:, 0:1], t[:, 1:2] - 0.5, t[:, 2:3] - 0.5
    # 正变换已乘 0.713 / 0.564，逆变换必须除回，避免色度被再次压缩。
    r = y + cr / 0.713
    b = y + cb / 0.564
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return torch.cat([r, g, b], dim=1).clamp_(0.0, 1.0)


# ------- 滤波 -------

def box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """均值滤波（reflect 边界，与 cv2.boxFilter 对齐）。x: (1,C,H,W)。"""
    k = 2 * radius + 1
    if k <= 1:
        return x
    pad = F.pad(x, (radius,) * 4, mode="reflect")
    c = x.shape[1]
    kernel = torch.full((c, 1, k, k), 1.0 / (k * k), device=x.device,
                        dtype=x.dtype)
    return F.conv2d(pad, kernel, groups=c)


def _gaussian_kernel1d(sigma: float, device) -> torch.Tensor:
    radius = max(int(3.0 * sigma + 0.5), 1)
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    return k / k.sum(), radius


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """可分离高斯（reflect 边界）。x: (1,C,H,W)。"""
    if sigma <= 0:
        return x
    k, r = _gaussian_kernel1d(sigma, x.device)
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, len(k)).contiguous()
    ky = k.view(1, 1, -1, 1).expand(c, 1, len(k), 1).contiguous()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), ky, groups=c)
    return x


def guided_filter(guide: torch.Tensor, src: torch.Tensor, radius: int,
                  eps: float = 1e-4) -> torch.Tensor:
    """引导滤波（He et al. 2010），guide/src: (1,1,H,W) float。"""
    mean_g = box_filter(guide, radius)
    mean_s = box_filter(src, radius)
    cov = box_filter(guide * src, radius) - mean_g * mean_s
    var = box_filter(guide * guide, radius) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_s - a * mean_g
    return box_filter(a, radius) * guide + box_filter(b, radius)


def bilateral_blur(x: torch.Tensor, diameter: int, sigma_color: float,
                   sigma_space: float) -> torch.Tensor:
    """双边滤波：小图用邻域展开换速度，大图回退逐位叠加。

    直径与两个 sigma 与 cv2.bilateralFilter(d, sc, ss) 同参语义；
    颜色域按各通道联合欧氏距离（与 cv2 的 Lab 域近似有差，磨皮用途
    视觉等价）。预览磨皮在 1/4 分辨率运行，展开的 81 邻域张量约占
    32~56MB，消除逐位累加的数百个小 kernel 启动。大图限制邻域张量
    不超过 256MB，防止拍照全分辨率或以后复用时显存骤增。
    """
    if diameter <= 1:
        return x
    r = diameter // 2
    dev = x.device
    k = 2 * r + 1
    ys, xs = torch.meshgrid(
        torch.arange(-r, r + 1, device=dev, dtype=torch.float32),
        torch.arange(-r, r + 1, device=dev, dtype=torch.float32),
        indexing="ij")
    w_space = torch.exp(-(xs ** 2 + ys ** 2) / (2 * sigma_space ** 2))
    w_space = w_space.flatten()          # 按窗口序线性索引（(dy,dx) 行序）
    inv2sc2 = 1.0 / (2 * sigma_color ** 2 / (255.0 ** 2) * (x.shape[1]))
    patch_bytes = x.numel() * k * k * x.element_size()
    if x.is_cuda and patch_bytes <= 256 * 1024 * 1024:
        n, c, h, w = x.shape
        patches = F.unfold(F.pad(x, (r, r, r, r), mode="reflect"),
                           kernel_size=k).view(n, c, k * k, h, w)
        color = torch.exp(
            -(patches - x.unsqueeze(2)).square().sum(1, keepdim=True)
            * inv2sc2)
        weights = color * w_space.view(1, 1, k * k, 1, 1)
        return ((patches * weights).sum(2)
                / weights.sum(2).clamp_min(1e-12))

    # 显存预算外沿用旧版移位累加，避免 im2col 巨型临时张量。
    acc = torch.zeros_like(x)
    wacc = torch.zeros_like(x[:, :1])
    padded = F.pad(x, (r, r, r, r), mode="reflect")
    idx = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            shifted = padded[:, :, r + dy:r + dy + x.shape[2],
                             r + dx:r + dx + x.shape[3]]
            w_color = torch.exp(
                -((x - shifted) ** 2).sum(1, keepdim=True) * inv2sc2)
            w = w_space[idx] * w_color
            acc = acc + shifted * w
            wacc = wacc + w
            idx += 1
    return acc / wacc.clamp_min(1e-12)


# ------- 采样 / 变形 -------

def remap(t: torch.Tensor, map_x: torch.Tensor, map_y: torch.Tensor,
          border: str = "replicate") -> torch.Tensor:
    """cv2.remap 的 GPU 版。t: (1,C,H,W)；map_x/map_y: (H,W) float 绝对
    采样坐标（与 cv2.remap(map_x, map_y) 同口径）。"""
    h, w = t.shape[-2:]
    gx = map_x / max(w - 1, 1) * 2 - 1
    gy = map_y / max(h - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)   # (1,H,W,2)
    mode = "bilinear"
    pad_mode = {"replicate": "border", "constant": "zeros",
                "reflect": "reflection"}[border]
    return F.grid_sample(t, grid, mode=mode, padding_mode=pad_mode,
                         align_corners=True)


def resize(t: torch.Tensor, size_wh: tuple[int, int],
           area: bool = True) -> torch.Tensor:
    """双线性/区域下采样 resize 到 (W,H)。"""
    mode = "area" if area else "bilinear"
    return F.interpolate(t, size=size_wh, mode=mode,
                         align_corners=None if mode == "area" else False,
                         antialias=False)


# ------- LUT / 直方图 -------

def apply_lut(t_u8: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """对 (1,C,H,W) uint8 按通道应用 LUT。lut: (C,256) 或 (1,256) float。"""
    c = t_u8.shape[1]
    if lut.shape[0] == 1 and c > 1:
        lut = lut.expand(c, -1)
    lut = lut.to(t_u8.device)
    out = torch.empty_like(t_u8)
    for ch in range(c):
        out[:, ch] = lut[t_u8[:, ch].long()]
    return out


def equalize_lut(t_u8_gray: torch.Tensor) -> torch.Tensor:
    """均衡化 LUT（输入 (1,1,H,W) uint8），返回 (1,256) float32 LUT。

    cv2.equalizeHist 口径：lut[i] = round((cdf[i] - cdf_min)
    / (N - cdf_min) * 255)，cdf_min 取首个非零 bin 的累计数。
    """
    hist = torch.histc(t_u8_gray.float().flatten(), bins=256, min=0, max=255)
    cdf = torch.cumsum(hist, 0)
    nz = torch.nonzero(hist)
    cdf_min = float(cdf[int(nz[0].item())]) if len(nz) else 0.0
    denom = (cdf[-1] - cdf_min).clamp_min(1.0)
    lut = torch.round((cdf - cdf_min) / denom * 255.0).clamp(0, 255)
    return lut.unsqueeze(0)


def clahe_lut(l_ch_u8: torch.Tensor, tiles_yx: tuple[int, int],
              clip_limit: float) -> torch.Tensor:
    """CLAHE（Zuiderveld 1994）：分块直方图 → 限幅重分配 → 块间双线性 LUT 插值。

    输入 (1,1,H,W) uint8，输出同形状 uint8。与 cv2.createCLAHE 语义一致
    （tile 直方图按 bin 均匀重分配，像素取四邻 tile LUT 的双线性混合）。
    """
    h, w = l_ch_u8.shape[-2:]
    ty, tx = tiles_yx
    dev = l_ch_u8.device
    # 补到整块（右/下边界补边复制），避免 unfold 丢像素
    ph = (ty - h % ty) % ty
    pw = (tx - w % tx) % tx
    if ph or pw:
        x = F.pad(l_ch_u8.float(), (0, pw, 0, ph), mode="replicate")
    else:
        x = l_ch_u8.float()
    th, tw = x.shape[-2] // ty, x.shape[-1] // tx
    tiles = x.view(1, 1, ty, th, tx, tw).permute(2, 4, 0, 1, 3, 5) \
        .reshape(ty * tx, 1, th, tw)
    # 全部 tile 的直方图：一次 bincount（tile_id*256 + 灰度值）
    q = tiles.long().flatten()                        # (T*tile_pixels,)
    tile_ids = torch.arange(ty * tx, device=dev).view(-1, 1)
    tile_ids = tile_ids.expand(ty * tx, th * tw).flatten() * 256
    hist = torch.bincount(tile_ids + q,
                          minlength=ty * tx * 256).view(ty * tx, 256)
    n = th * tw
    # 限幅重分配（与 cv2 相同：超限部分均匀摊回所有 bin）
    excess = (hist - clip_limit * n).clamp_min(0)
    hist = hist.clamp_max(int(clip_limit * n))
    hist += (excess.sum(1, keepdim=True) // 256).long()
    cdf = torch.cumsum(hist, 1)
    luts = (cdf - 0.5) / n * 255.0                    # (T,256) —— cv2 同式
    luts = luts.clamp(0, 255)
    # 每像素四邻 tile LUT 双线性混合
    ys = torch.arange(h, device=dev, dtype=torch.float32) + 0.5
    xs = torch.arange(w, device=dev, dtype=torch.float32) + 0.5
    fy = ys / th - 0.5                                # tile 中心坐标系
    fx = xs / tw - 0.5
    y0 = torch.floor(fy).clamp(0, ty - 1).long()
    x0 = torch.floor(fx).clamp(0, tx - 1).long()
    wy = (fy - y0.float()).clamp(0, 1)
    wx = (fx - x0.float()).clamp(0, 1)
    y1 = (y0 + 1).clamp(0, ty - 1)
    x1 = (x0 + 1).clamp(0, tx - 1)
    v = l_ch_u8[0, 0].long()                          # (H,W)
    t00 = y0[:, None] * tx + x0[None, :]
    t01 = y0[:, None] * tx + x1[None, :]
    t10 = y1[:, None] * tx + x0[None, :]
    t11 = y1[:, None] * tx + x1[None, :]
    wy = wy[:, None]                                  # (H,1) / (1,W) 广播
    wx = wx[None, :]
    val = (luts[t00, v] * (1 - wy) * (1 - wx)
           + luts[t01, v] * (1 - wy) * wx
           + luts[t10, v] * wy * (1 - wx)
           + luts[t11, v] * wy * wx)
    return val.round_().clamp_(0, 255).to(torch.uint8).unsqueeze(0).unsqueeze(0)


# ------- 形态学（小核，滑窗 max/min） -------

def morph_op(mask_u8: torch.Tensor, k: int, dilate: bool) -> torch.Tensor:
    """椭圆近似（用方形窗口）腐蚀/膨胀，k 为核边长。mask: (1,1,H,W) uint8。"""
    r = k // 2
    if r <= 0:
        return mask_u8
    f = mask_u8.float()
    win = 2 * r + 1
    # unfold 实现滑窗 max/min
    unf = f.unfold(2, win, 1).unfold(3, win, 1)        # (1,1,H',W',k,k)
    out = unf.amax(dim=(-2, -1)) if dilate else unf.amin(dim=(-2, -1))
    # 边界补齐：腐蚀用 replicate 最小等价——直接把边界 r 圈置 0/255 再覆盖
    out = out.unsqueeze(0) if out.dim() == 3 else out
    if out.shape[-2:] != mask_u8.shape[-2:]:
        pad_val = 255.0 if dilate else 0.0
        out = F.pad(out, (0, mask_u8.shape[-1] - out.shape[-1],
                          0, mask_u8.shape[-2] - out.shape[-2]),
                    value=pad_val)
    return out.to(torch.uint8)


# ------- 合成 -------

def blend_over(t: torch.Tensor, bg: torch.Tensor,
               alpha: torch.Tensor) -> torch.Tensor:
    """out = t*alpha + bg*(1-alpha)。t/bg: (1,3,H,W) float；alpha: (1,1,H,W)。"""
    return t * alpha + bg * (1.0 - alpha)


def sync() -> None:
    if device().type == "cuda":
        torch.cuda.synchronize()
