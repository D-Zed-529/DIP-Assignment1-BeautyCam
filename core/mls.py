"""MLS 变形（Moving Least Squares, Schaefer et al. 2006）。

复数形式实现（2D 点视为复数），粗网格求解 + 双线性上采样成稠密
位移场，供 cv2.remap 使用——纯 numpy 下 720p 人脸 ROI 实时可行。

两种变体（f(v) = A·(v − p*) + q*，差别只在系数 A）：
  - similarity：A = a（最优旋转+均匀缩放），控制点处精确跟随。
    瘦脸这类"下颌整体向中心收"的收缩运动必须用它：刚体变体禁
    缩放，收缩会被模型本身对抗（实测跟随率仅 ~0.25，观感偏弱）。
  - rigid：A = a/|a|（只留旋转，禁缩放）。适合"平移/旋转"型控制
    点布局（拖拽、对齐类），收缩型布局下幅度严重衰减。

参考：
  - Schaefer et al. "Image Deformation Using Moving Least Squares", 2006
  - numpy 参考实现 github.com/Jarvis73/Moving-Least-Squares
"""

from __future__ import annotations

import cv2
import numpy as np

GRID_STEP = 6        # 粗网格步长（像素）：求解点数 ≈ 面积/36，再上采样
WEIGHT_ALPHA = 1.5   # 权重指数：w = 1/|v-p|^{2α}（论文常用 1~2）
MIN_DIST2 = 4.0      # 权重距离平方下限（2px），防控制点处奇异


def identity_maps(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    mx = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    my = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return mx, my


def upsample_displacement_dense(dx: np.ndarray, dy: np.ndarray,
                                x1: int, y1: int, grid_step: int,
                                x2: int, y2: int
                                ) -> tuple[np.ndarray, np.ndarray]:
    """粗网格位移场 → ROI 内稠密采样图（不做全帧恒等图分配）。

    返回 (fx, fy)：shape (y2-y1, x2-x1) 的绝对采样坐标图（float32），
    可直接用于 ROI 子图的 cv2.remap（需减去 ROI 原点偏移）。
    节点对齐约定见 upsample_displacement_field 的 docstring。
    """
    dense_w = x2 - x1
    dense_h = y2 - y1
    need_c = int(np.ceil((dense_w - 1) / grid_step)) + 1
    need_r = int(np.ceil((dense_h - 1) / grid_step)) + 1
    pad_c = max(need_c - dx.shape[1], 0)
    pad_r = max(need_r - dx.shape[0], 0)
    dx = np.pad(dx, ((0, pad_r), (0, pad_c)))
    dy = np.pad(dy, ((0, pad_r), (0, pad_c)))
    gx_p = x1 + np.arange(dx.shape[1], dtype=np.float64) * grid_step
    gy_p = y1 + np.arange(dx.shape[0], dtype=np.float64) * grid_step
    fx = gx_p.astype(np.float32)[None, :] + dx.astype(np.float32)
    fy = gy_p.astype(np.float32)[:, None] + dy.astype(np.float32)
    kx = np.clip(np.arange(dense_w, dtype=np.float32) / grid_step,
                 0.0, dx.shape[1] - 1)
    ky = np.clip(np.arange(dense_h, dtype=np.float32) / grid_step,
                 0.0, dx.shape[0] - 1)
    kk_x, kk_y = np.meshgrid(kx, ky)
    up_kwargs = dict(interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    return (cv2.remap(fx, kk_x, kk_y, **up_kwargs),
            cv2.remap(fy, kk_x, kk_y, **up_kwargs))


def upsample_displacement_field(dx: np.ndarray, dy: np.ndarray,
                                x1: int, y1: int, grid_step: int,
                                h: int, w: int,
                                x2: int | None = None,
                                y2: int | None = None
                                ) -> tuple[np.ndarray, np.ndarray]:
    """粗网格位移场 → 全帧稠密 cv2.remap 采样图（MLS 与液化共用例程）。

    dx/dy: 网格节点上的采样图偏移 (gh,gw)（map = v + offset，节点绝对
    位置 = (x1+j·step, y1+i·step)；注意这是 remap 逆映射的采样偏移，
    与内容位移反号）；
    ROI = (x1, y1, x2, y2)（None 时取图像右/下边缘），ROI 外严格恒等。
    稠密像素 j 必须严格采样网格节点 j/grid_step（不能用 cv2.resize：其
    像素中心约定会系统性偏移 ~0.5 个节点，grid_step=6 时即 2.5px 内容
    平移，锚点钉扎失效）。尾部覆盖不到 ROI 边缘时用"虚拟节点"补齐
    （位置前进、位移=0）：若直接 clip 到末节点，map 的绝对位置在尾部
    被冻结，位移会变成 末节点位置−x 的线性负斜坡（ROI 边缘假漂移）。
    """
    fx_up, fy_up = upsample_displacement_dense(
        dx, dy, x1, y1, grid_step,
        w if x2 is None else x2, h if y2 is None else y2)
    map_x, map_y = identity_maps(h, w)
    map_x[y1:y1 + fx_up.shape[0], x1:x1 + fx_up.shape[1]] = fx_up
    map_y[y1:y1 + fy_up.shape[0], x1:x1 + fy_up.shape[1]] = fy_up
    return map_x, map_y


def mls_similarity_maps(h: int, w: int, P: np.ndarray, Q: np.ndarray,
                        grid_step: int = GRID_STEP,
                        roi: tuple[int, int, int, int] | None = None,
                        feather: float = 0.25
                        ) -> tuple[np.ndarray, np.ndarray]:
    """MLS 相似变体（旋转+均匀缩放）：控制点精确跟随，瘦脸推荐档。"""
    return _mls_maps(h, w, P, Q, rigid=False, grid_step=grid_step,
                     roi=roi, feather=feather)


def mls_rigid_maps(h: int, w: int, P: np.ndarray, Q: np.ndarray,
                   grid_step: int = GRID_STEP,
                   roi: tuple[int, int, int, int] | None = None,
                   feather: float = 0.25
                   ) -> tuple[np.ndarray, np.ndarray]:
    """MLS 刚体变体（只旋转，禁缩放）：适合平移/拖拽类控制布局。"""
    return _mls_maps(h, w, P, Q, rigid=True, grid_step=grid_step,
                     roi=roi, feather=feather)


def _mls_maps(h: int, w: int, P: np.ndarray, Q: np.ndarray, *,
              rigid: bool,
              grid_step: int,
              roi: tuple[int, int, int, int] | None,
              feather: float
              ) -> tuple[np.ndarray, np.ndarray]:
    """求解 MLS 变形的稠密采样场（供 cv2.remap 直接使用）。

    语义约定：内容从 P 移动到 Q（remap 的采样图是逆映射，内部已对调
    求解，调用方按"内容 P→Q"理解即可）。P==Q 的点即锚点（不动）。
    P: (n,2) 原位置；Q: (n,2) 目标位置（像素坐标）。
    roi: (x1, y1, x2, y2) 求解区域；None 时取控制点包围盒外扩 2×网格步长。
    feather: ROI 外圈比例，位移场在羽化带内余弦衰减到 0（消接缝，
             并保证 ROI 外严格恒等）。
    返回 (map_x, map_y)：float32 (h, w)。

    数学（复数形式，每输出点 v）：
      w_i  = 1/max(|v-p_i|^{2α}, ε)
      p*   = Σw p / Σw，q* = Σw q / Σw
      a    = Σ w q̂_i conj(p̂_i) / Σ w |p̂_i|²    （加权最小二乘相似变换）
      刚体：f(v) = (a/|a|)·(v − p*) + q*
    """
    P = np.asarray(P, np.float64)
    Q = np.asarray(Q, np.float64)
    if len(P) == 0 or np.allclose(P, Q):
        return identity_maps(h, w)

    if roi is None:
        pad = 2 * grid_step
        x1 = int(max(P[:, 0].min() - pad, 0))
        y1 = int(max(P[:, 1].min() - pad, 0))
        x2 = int(min(P[:, 0].max() + pad + 1, w))
        y2 = int(min(P[:, 1].max() + pad + 1, h))
    else:
        x1, y1, x2, y2 = roi
    if x2 - x1 < 4 or y2 - y1 < 4:
        return identity_maps(h, w)

    # ---- 粗网格上求解（注意：求逆映射，P/Q 对调） ----
    gx, gy = np.meshgrid(
        np.arange(x1, x2, grid_step, dtype=np.float64),
        np.arange(y1, y2, grid_step, dtype=np.float64))
    v = (gx + 1j * gy).ravel()                       # (G,) 输出点（复数）

    pc = Q[:, 0] + 1j * Q[:, 1]                      # 目标位置作为"源控制点"
    qc = P[:, 0] + 1j * P[:, 1]                      # 原位置作为"映射目标"

    # 权重 (G, n)
    d = np.abs(v[:, None] - pc[None, :])
    wgt = 1.0 / np.maximum(d ** (2.0 * WEIGHT_ALPHA), MIN_DIST2)

    wsum = wgt.sum(axis=1)
    p_star = (wgt @ pc) / wsum                       # (G,)
    q_star = (wgt @ qc) / wsum

    phat = pc[None, :] - p_star[:, None]             # (G, n)
    qhat = qc[None, :] - q_star[:, None]

    mu = (wgt * (phat * np.conj(phat)).real).sum(axis=1)
    a = (wgt * qhat * np.conj(phat)).sum(axis=1) / np.maximum(mu, 1e-9)
    if rigid:
        a = a / np.maximum(np.abs(a), 1e-9)          # 刚体：只留旋转

    f = a * (v - p_star) + q_star                    # (G,) 采样位置

    # ---- 位移场余弦羽化到 ROI 边界（消接缝，ROI 外严格恒等） ----
    dx = (f.real - v.real).reshape(gx.shape)
    dy = (f.imag - v.imag).reshape(gy.shape)
    if feather > 0:
        gh, gwd = gx.shape
        ramp_w = max(int(gwd * feather), 1)
        ramp_h = max(int(gh * feather), 1)
        # 水平/垂直双侧余弦坡
        wx = np.ones(gwd)
        wx[:ramp_w] = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp_w) / ramp_w)
        wx[-ramp_w:] = wx[:ramp_w][::-1]
        wy = np.ones(gh)
        wy[:ramp_h] = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp_h) / ramp_h)
        wy[-ramp_h:] = wy[:ramp_h][::-1]
        window = wy[:, None] * wx[None, :]
        dx = dx * window
        dy = dy * window

    # ---- 粗网格位移场上采样成稠密场（公共例程，节点对齐约定见其 docstring） ----
    return upsample_displacement_field(dx, dy, x1, y1, grid_step, h, w, x2, y2)
