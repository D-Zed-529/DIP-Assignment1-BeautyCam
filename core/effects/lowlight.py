"""低光增强效果 —— 启发式基线 + SCI 深度模型（Phase 2）。

两档实现共存，GUI/CLI 可切换（"经典 vs 深度"对比线，延续本项目套路）：

  - LowLightEffect   一期启发式（线性增益 + YCrCb 直方图均衡）：无依赖、
                     速度稳定，作为回退与客观对比基线（P5-3 评测用）。
  - LowLightDnnEffect SCI（Self-Calibrated Illumination, CVPR 2022）
                     ONNX 推理：54KB、固定 512×512 输入、三档强度
                     （easy/medium/difficult），CoreML EP 实测 1.6ms/次
                     （CPU 8.1ms）。P2-1 选型记录见 core/infer.py 模块头。

深度档降载策略（PLAN §3.3）：
  - 推理在 512×512（模型固定尺寸，小于 720p）；
  - infer_interval 隔帧推理（默认 2）：中间帧复用上一帧增强结果——
    相邻帧内容位移极小，视觉等价；相机快速平移时仅 1 帧滞后；
  - 亮度自动触发（与启发式同口径 mean_brightness < threshold）。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..camera import LOW_LIGHT_THRESHOLD, mean_brightness
from ..context import FrameContext
from ..infer import LOWLIGHT_DEFAULT_LEVEL, LOWLIGHT_LEVELS
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


class LowLightDnnEffect(Effect):
    """SCI 深度弱光增强（Phase 2 主力档）。

    参数：
      auto:          True 时仅当整帧亮度低于阈值才增强（默认）
      threshold:     自动触发的亮度阈值（与启发式同口径）
      strength:      增强强度 0~1（增强结果与原图混合）
      level:         SCI 强度档 easy / medium / difficult（换模型文件）
      infer_interval: 推理间隔帧数（默认 2：隔帧推理 + 复用上一帧结果）
    """

    name = "lowlight_dnn"
    needs = frozenset()          # 不依赖 MediaPipe 推理结果

    @staticmethod
    def default_params() -> dict:
        return {
            "auto": True,
            "threshold": LOW_LIGHT_THRESHOLD,
            "strength": 1.0,
            "level": LOWLIGHT_DEFAULT_LEVEL,
            "infer_interval": 2,
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._params["level"] not in LOWLIGHT_LEVELS:
            raise ValueError(f"未知 SCI 强度档：{self._params['level']}")
        self._last_dark = False
        self._frame_seen = 0
        self._session = None       # 懒加载（缺权重时降级透传并告警一次）
        self._session_level: str | None = None
        self._warned = False
        self._last_enhanced: np.ndarray | None = None   # 隔帧复用（原尺寸 BGR）

    @property
    def is_dark(self) -> bool:
        return self._last_dark

    @property
    def provider(self) -> str:
        """实际生效的执行后端（无会话时空串；GUI/答辩性能素材用）。"""
        return self._session.provider if self._session is not None else ""

    def inference_interval(self, need: str) -> int:
        # ONNX 推理不走 MediaPipe 聚合（needs 为空），这里仅作自声明，
        # 供 GUI 显示与 CLI 评测读取口径。
        return max(1, int(self._p()["infer_interval"]))

    def reset_temporal(self) -> None:
        """清空跨帧状态（切换采集源 / 逐张独立批跑前调用）。"""
        self._last_enhanced = None
        self._frame_seen = 0

    def set_enabled(self, on: bool) -> None:
        super().set_enabled(on)
        if on:
            self.reset_temporal()

    def _get_session(self):
        """懒加载/换档 SCI 会话；失败（缺权重/缺 onnxruntime）告警一次并透传。"""
        level = self._p()["level"]
        if self._session is not None and self._session_level == level:
            return self._session
        try:
            from ..infer import get_lowlight_session
            self._session = get_lowlight_session(level)
            self._session_level = level
        except Exception as exc:   # noqa: BLE001 —— 缺权重属预期部署形态
            self._session = None
            if not self._warned:
                self._warned = True
                print(f"[低光 SCI] 会话不可用，效果透传：{exc}")
        return self._session

    def process(self, frame: np.ndarray, ctx: FrameContext) -> np.ndarray:
        p = self._p()
        brightness = mean_brightness(frame)
        self._last_dark = brightness < p["threshold"]
        if p["auto"] and not self._last_dark:
            self._last_enhanced = None   # 退出暗光：断开复用链，避免闪旧帧
            return frame
        if p["strength"] <= 0:
            return frame

        h, w = frame.shape[:2]
        interval = max(1, int(p["infer_interval"]))
        due = (self._frame_seen % interval == 0)
        self._frame_seen += 1

        if due or self._last_enhanced is None \
                or self._last_enhanced.shape[:2] != (h, w):
            session = self._get_session()
            if session is None:
                return frame
            # 512×512 推理 → resize 回原尺寸（纵横比畸变只存在于推理域）
            rgb = session.enhance(frame)
            out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)   # float32 [0,1]
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
            out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
            self._last_enhanced = out
        else:
            out = self._last_enhanced

        s = float(p["strength"])
        return cv2.addWeighted(out, s, frame, 1.0 - s, 0)
