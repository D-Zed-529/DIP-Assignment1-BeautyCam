"""液化变形：紧支撑笔刷 + Shepard 归一化叠加（瘦脸 v5 求解核）。

与 MLS（core/mls.py，全局最小二乘）互补的关键点变形方案。v4 实测问题：
MLS 是全局光滑场，动点与大量锚点互相拉锯——动点处实际位移被稀释、
方向被周边锚点掰歪，strength=1.0 时下巴被挤成尖锥、嘴角被斜向拖拽。

v5 采用美颜/液化工具的行业做法（局部平移液化，liquify笔刷语义）：

    D(v) = Σ_k w_k·φ_k(v)·d_k / max(Σ_k w_k·φ_k(v), 1)

    φ = (1−t²)³ (t<1，否则 0)，d_k = 动点位移命令（方向 = 下颌轮廓
    法向）；w_k·φ_k·d_k = 0 的点为"保护点"（嘴角/下巴/眼等），只进
    分母、把附近场平滑拉向 0——Shepard 凸平均下只衰减、不振荡。

设计取舍（v5 迭代中用真图验证）：
  - 曾尝试"动点精确插值 + 锚点零位移"的 RBF 插值（解 Φc=d）：当动点
    命令与邻近锚点约束强冲突时（如贴着嘴角的下颌动点），光滑核插值
    在两组约束间振铃过冲（Runge 现象，实测探针位移由 +9.9px 翻成
    −4.4px）——弃用。
  - Shepard 归一化叠加是位移命令的凸加权平均：场处处有界（不超过最
    大命令）、同向笔刷之间形成平滑平台（脸颊整体内收而非贴线窄带）、
    支撑边界外严格为 0（φ 紧支撑且 C²），无需求解线性方程组。
  - 五官保护：真图实测 R=0.30×脸宽 的笔刷会盖到嘴角（嘴角被拖
    12.5px），保护点权值（3×嘴角等）把五官附近场平滑压向 0，且不
    引入任何过冲。
  - 眉线以上另加竖直保护窗（见 core/effects/beauty.py），比特级恒等。

粗网格求值 + 双线性上采样成稠密采样图（节点对齐约定与 core/mls.py
一致，共用 upsample_displacement_field）。
"""

from __future__ import annotations

import numpy as np

from .mls import (GRID_STEP, identity_maps, upsample_displacement_dense,
                  upsample_displacement_field)


def brush_falloff(t: np.ndarray) -> np.ndarray:
    """液化笔刷衰减核 φ(t) = (1−t²)³，t = |v−p|/R，t≥1 时严格为 0。

    C² 紧支撑：场在支撑边界处位移及其一、二阶导均为 0，无接缝；
    峰值平台比 Wendland C2 平缓，脸颊主体能整体被拖动。
    """
    t = np.asarray(t)
    out = np.zeros_like(t)
    m = t < 1.0
    out[m] = (1.0 - t[m] ** 2) ** 3
    return out


def rbf_liquify_maps(h: int, w: int,
                     P: np.ndarray, D: np.ndarray, radius: float,
                     weights: np.ndarray | None = None,
                     grid_step: int = GRID_STEP,
                     return_roi: bool = False,
                     ) -> tuple:
    """液化笔刷叠加位移场 → cv2.remap 采样图。

    P: (K,2) 控制点（像素坐标）；D: (K,2) 内容位移命令（保护点为 0，
    只进分母起衰减作用）；radius: 笔刷半径（像素，全点统一）；
    weights: (K,) 点权（默认全 1，保护点 >1）。
    默认返回 (map_x, map_y)：全帧采样图，支撑并集之外严格恒等。

    return_roi=True 时返回 (map_x, map_y, roi)：map 仅 ROI 尺寸（ROI =
    (x1,y1,x2,y2)，控制点包围盒外扩一个支撑半径），roi 为 None 表示
    无变形。热路径（瘦脸）用这一档：720p 下全帧恒等图的分配+填充+
    全帧 remap 约 5ms，脸围盒只占画面一角时可省 3~4ms；场在 ROI 边界
    严格为 0（φ 紧支撑），ROI 内 remap 与全帧 remap 逐位一致（唯一
    差异是 ROI 触帧边时的 REPLICATE 边缘，与全帧版行为相同）。
    """
    P = np.asarray(P, np.float64)
    D = np.asarray(D, np.float64)
    if len(P) == 0 or not np.any(D):
        return (None, None, None) if return_roi else identity_maps(h, w)
    if weights is None:
        weights = np.ones(len(P))
    weights = np.asarray(weights, np.float64)

    # ---- 求解域：控制点包围盒外扩一个支撑半径（场外严格为 0） ----
    x1 = int(max(P[:, 0].min() - radius, 0))
    y1 = int(max(P[:, 1].min() - radius, 0))
    x2 = int(min(P[:, 0].max() + radius + 1, w))
    y2 = int(min(P[:, 1].max() + radius + 1, h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return (None, None, None) if return_roi else identity_maps(h, w)

    if return_roi:
        # ROI 各边再外扩 1px：remap 边界像素的双线性插值需要界外 1px
        # 邻居，子图裁掉后 REPLICATE 会顶替成边缘像素、与全帧 remap 差
        # 1px 混合。外扩带在支撑边界之外（场恒为 0 → map 恒等），多算
        # 的这圈是恒等区，代价可忽略。
        x1, y1 = max(x1 - 1, 0), max(y1 - 1, 0)
        x2, y2 = min(x2 + 1, w), min(y2 + 1, h)

    # ---- 粗网格求值：Shepard 归一化叠加 ----
    gx, gy = np.meshgrid(
        np.arange(x1, x2, grid_step, dtype=np.float64),
        np.arange(y1, y2, grid_step, dtype=np.float64))
    gdx = gx[..., None] - P[None, None, :, 0]
    gdy = gy[..., None] - P[None, None, :, 1]
    wgt = brush_falloff(np.hypot(gdx, gdy) / radius) * weights  # (Gh,Gw,K)
    wsum = wgt.sum(axis=2)                               # (Gh,Gw)
    denom = np.maximum(wsum, 1.0)
    # remap 是逆映射：内容位移为 D ⇒ 采样图偏移为 −D
    #（v + D 处的内容来自 v，即采样坐标 = v − D）
    dx = -(wgt @ D[:, 0]) / denom                        # (Gh,Gw)
    dy = -(wgt @ D[:, 1]) / denom

    if return_roi:
        mx, my = upsample_displacement_dense(dx, dy, x1, y1, grid_step,
                                             x2, y2)
        return mx, my, (x1, y1, x2, y2)
    return upsample_displacement_field(dx, dy, x1, y1, grid_step, h, w,
                                       x2, y2)
