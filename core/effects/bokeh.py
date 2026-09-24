"""深度渐进虚化（P3-4，算力升级后补上）—— 单反镜头式背景虚化。

与 SegmentEffect 的均匀背景虚化的区别：模糊量由**深度**决定（近清远糊），
焦平面对齐人物（有人像掩膜时取人像区中位深度），远处比近处更糊 ——
"背景虚化"模式是它的零阶特例（背景恒定最大 sigma）。

实现（纯 cv2，720p 实测 ~6ms）：
  1. ctx.depth（Depth Anything V2 相对深度，值越大越近）；
  2. 焦平面 focus = 人像区中位深度（无掩膜时取 85 分位——最近的主体）；
  3. 逐像素目标 sigma = sigma_max · smoothstep(|depth − focus| / range)；
  4. 半分辨率高斯金字塔（K 档 sigma 阶梯）+ 逐像素相邻档线性插值 ——
     连续深度→连续模糊的标准近似，避免逐像素可变核卷积的天价；
  5. 合成：清晰权重 w = exp(−(dist/range)²)，人像掩膜区强制 w=1
     （use_matte 且 ctx.person_alpha 可用时）。
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..context import FrameContext
from ..pipeline import Effect, NEED_DEPTH, NEED_SEGMENTATION

# ------- 调参常量 -------
SIGMA_MAX = 18.0        # strength=1.0 时的最大高斯 sigma（px，全分辨率口径）
FOCUS_RANGE = 0.35      # 焦平面外深度差达到它即满糊（相对深度域）
PYRAMID_LEVELS = 5      # sigma 阶梯档数（0 ~ sigma_max 均分）
PYR_FACTOR = 4          # 金字塔计算域下采样倍数（模糊是低频，1/4 无损且快）
FOCUS_PERCENTILE = 85   # 无掩膜时的焦平面深度分位（最近的主体）
PERSON_ALPHA_MIN = 0.2  # 人像 alpha 均值低于它视为"没有人"，退回分位焦平面
# 深度推理隔帧间隔：深度是低频信号（DepthSession 内还有 min/max 的时域
# EMA），DA-v2 前向（CUDA Graph 后 ~8ms）隔 3 帧跑一次、中间帧复用上一次
# 深度图，均摊 ~3ms；间隔过大在快速运动场景会有 1~2 帧的虚化滞后。
DEPTH_INTERVAL = 3


def smoothstep(x: np.ndarray) -> np.ndarray:
    """Hermite 平滑阶跃（0→1），输入已裁剪到 [0,1]。"""
    return x * x * (3.0 - 2.0 * x)


def focus_depth(depth: np.ndarray,
                alpha: Optional[np.ndarray]) -> float:
    """焦平面深度：人像区中位（有掩膜），否则高分位（最近主体）。"""
    if alpha is not None and float(alpha.mean()) >= PERSON_ALPHA_MIN:
        m = alpha > 0.6
        if m.sum() > 64:
            return float(np.median(depth[m]))
    return float(np.percentile(depth, FOCUS_PERCENTILE))


def blur_sigma_map(depth: np.ndarray, focus: float,
                   sigma_max: float, rng: float) -> np.ndarray:
    """逐像素目标 sigma（全分辨率 float32）。"""
    dist = np.abs(depth.astype(np.float32) - np.float32(focus)) \
        / max(rng, 1e-3)
    return (sigma_max * smoothstep(np.clip(dist, 0.0, 1.0))).astype(np.float32)


def depth_blur(frame: np.ndarray, depth: np.ndarray,
               alpha: Optional[np.ndarray], strength: float,
               rng: float = FOCUS_RANGE) -> np.ndarray:
    """主处理：帧 + 深度 → 渐进虚化帧（纯函数，可单测）。"""
    sigma_max = SIGMA_MAX * float(np.clip(strength, 0.0, 1.0))
    if sigma_max < 0.5:
        return frame
    h, w = frame.shape[:2]
    focus = focus_depth(depth, alpha)
    smap = blur_sigma_map(depth, focus, sigma_max, rng)

    fac = PYR_FACTOR
    hw, hh = max(w // fac, 1), max(h // fac, 1)
    small = cv2.resize(frame, (hw, hh), interpolation=cv2.INTER_AREA)
    smap_s = cv2.resize(smap, (hw, hh), interpolation=cv2.INTER_LINEAR)

    # sigma 阶梯金字塔 + 相邻档插值
    ladder = np.linspace(0.0, sigma_max, PYRAMID_LEVELS)
    levels = [small]
    for s in ladder[1:]:
        sig = max(s / fac, 0.5)          # 半分辨率域的等效 sigma
        levels.append(cv2.GaussianBlur(small, (0, 0), sig))
    t = np.clip(smap_s / max(sigma_max, 1e-3) * (PYRAMID_LEVELS - 1),
                0, PYRAMID_LEVELS - 1)
    i0 = np.floor(t).astype(np.int32)
    i1 = np.minimum(i0 + 1, PYRAMID_LEVELS - 1)
    f = (t - i0)[..., None].astype(np.float32)
    stacked = np.stack(levels).astype(np.float32)   # (K,h,w,3)
    yy, xx = np.indices(smap_s.shape)
    blended = (stacked[i0, yy, xx] * (1.0 - f)
               + stacked[i1, yy, xx] * f)
    bokeh = cv2.resize(np.clip(blended, 0, 255).astype(np.uint8), (w, h),
                       interpolation=cv2.INTER_LINEAR)

    # 清晰权重：焦平面高斯衰减；人像区强制清晰
    dist = np.abs(depth.astype(np.float32) - np.float32(focus)) \
        / max(rng, 1e-3)
    w_clear = np.exp(-dist * dist)
    if alpha is not None:
        w_clear = np.maximum(w_clear, alpha)
    out = frame.astype(np.float32) * w_clear[..., None] \
        + bokeh.astype(np.float32) * (1.0 - w_clear)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


class BokehEffect(Effect):
    """深度渐进虚化（近清远糊，焦平面对齐人物）。

    参数：
      strength:  虚化强度 0~1（映射最大 sigma）
      range:     焦外深度范围（相对深度域，越小虚化过渡越陡）
      use_matte: 用人像掩膜锁定人物清晰（需分割推理配合）
    """

    name = "bokeh"
    supports_gpu = True          # 张量快路径见 _torch_impl.bokeh_process_t

    @staticmethod
    def default_params() -> dict:
        return {"strength": 0.7, "range": FOCUS_RANGE, "use_matte": True}

    @property
    def needs(self) -> frozenset[str]:
        p = self._p()
        base = {NEED_DEPTH}
        if p.get("use_matte"):
            base.add(NEED_SEGMENTATION)
        return frozenset(base)

    def inference_interval(self, need: str) -> int:
        """深度隔帧推理（DEPTH_INTERVAL），分割掩膜仍每帧（时域敏感）。"""
        if need == NEED_DEPTH:
            return DEPTH_INTERVAL
        return 1

    def reset_temporal(self) -> None:
        """清空深度隔帧缓存（切换采集源 / 逐张独立批跑前调用）。"""
        self._depth_cache = None
        self._depth_cache_t = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._warned_no_depth = False
        self._depth_cache: Optional[np.ndarray] = None      # CPU 路径缓存
        self._depth_cache_t = None                          # GPU 路径缓存

    def process_gpu(self, frame_t, ctx: FrameContext):
        from ._torch_impl import bokeh_process_t
        return bokeh_process_t(frame_t, self, ctx, self._p())

    def process(self, frame: np.ndarray, ctx: FrameContext) -> np.ndarray:
        if ctx.depth is None:
            # 隔帧推理的中间帧：复用上一帧深度（inference_interval>1 时
            # ctx.depth 为 None 属正常，不算"无深度输入"）
            if self._depth_cache is None:
                if not self._warned_no_depth:
                    self._warned_no_depth = True
                    print("[深度虚化] 无深度输入（需 torch 后端 + Depth Anything "
                          "V2 权重），效果透传。")
                return frame
            depth = self._depth_cache
        else:
            depth = ctx.depth
            self._depth_cache = depth
        p = self._p()
        alpha = ctx.person_alpha if p["use_matte"] else None
        return depth_blur(frame, depth, alpha, float(p["strength"]),
                          float(p["range"]))
