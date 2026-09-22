"""低光增强效果 —— 一期启发式的迁移版。

Phase 2 将替换为 SCI / Zero-DCE++ ONNX 深度模型（≤480p 推理 + 上采样 +
时域平滑），本模块保留启发式作为回退与"经典 vs 深度"对比基线。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..camera import LOW_LIGHT_THRESHOLD, mean_brightness
from ..context import FrameContext
from ..pipeline import Effect

# ------- 调参常量（一期口径） -------
GAIN_ALPHA = 1.3       # 线性增益斜率
GAIN_BETA = 20.0       # 线性增益偏置


class LowLightEffect(Effect):
    """启发式弱光增强：线性增益 + YCrCb 亮度通道直方图均衡。

    参数：
      auto:      True 时仅当整帧亮度低于阈值才增强（默认）
      threshold: 自动触发的亮度阈值
      strength:  增强强度 0~1（结果与原图混合）
    """

    name = "lowlight"

    @staticmethod
    def default_params() -> dict:
        return {
            "auto": True,
            "threshold": LOW_LIGHT_THRESHOLD,
            "strength": 1.0,
        }

    @property
    def is_dark(self) -> bool:
        """最近一帧是否判定为弱光（供 GUI 状态显示）。"""
        return self._last_dark

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_dark = False

    def process(self, frame: np.ndarray, ctx: FrameContext) -> np.ndarray:
        p = self._p()
        brightness = mean_brightness(frame)
        self._last_dark = brightness < p["threshold"]
        if p["auto"] and not self._last_dark:
            return frame
        if p["strength"] <= 0:
            return frame
        enhanced = cv2.convertScaleAbs(frame, alpha=GAIN_ALPHA, beta=GAIN_BETA)
        ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        s = float(p["strength"])
        return cv2.addWeighted(enhanced, s, frame, 1.0 - s, 0)
