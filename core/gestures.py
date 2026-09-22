"""手势/笑脸判定与自动拍照状态机（纯函数，可单测）。

一期问题修复说明：
  - 一期用帧计数（v_sign_frames >= 10）近似持续时长，帧率波动时判定不稳；
    v2 改为墙钟时间持续判定（AutoCaptureState）。
  - 一期 v_sign_frames / smile_start_time 缺 global 声明导致计数失效的
    bug 在 v2 架构下不复存在（状态封装在类实例里，无全局变量）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .context import FaceInfo, HandInfo

# ------- 调参常量（模块顶部集中，中文注释） -------
V_SIGN_HOLD = 1.0        # 剪刀手需持续的秒数（保持一期体验）
V_SIGN_ANGLE_MIN = 15.0  # 食指与中指夹角下限（度）
V_SIGN_ANGLE_MAX = 65.0  # 夹角上限（度）
SMILE_HOLD = 0.5         # 笑脸需持续的秒数
SMILE_SCORE_THR = 0.45   # blendshapes 微笑置信度阈值
SMILE_MOUTH_GAP = 0.012  # 回退口径：上下唇距（归一化 y 差）阈值（一期口径）
TRIGGER_COOLDOWN = 2.0   # 两次自动拍照的最小间隔秒数


def _straight_up(tip_y: float, base_y: float) -> bool:
    """指尖高于指根（图像 y 向下增长）视为伸直（一期口径）。"""
    return tip_y < base_y


def is_v_sign(hands: list[HandInfo], frame_w: int, frame_h: int) -> bool:
    """稳定剪刀手：仅食指+中指伸直，其余弯曲；两指夹角 15°~65°。

    角度在像素空间计算（与一期 640×480 口径一致的行为）；hands 为空返回 False。
    """
    for hand in hands:
        lm = hand.landmarks          # (21, 3) 归一化
        px = lm[:, 0] * frame_w      # 像素坐标
        py = lm[:, 1] * frame_h

        straight = {
            "idx": _straight_up(py[8], py[5]),
            "mid": _straight_up(py[12], py[9]),
            "rin": _straight_up(py[16], py[13]),
            "pnk": _straight_up(py[20], py[17]),
        }
        # 仅食指+中指伸直
        if not (straight["idx"] and straight["mid"]):
            continue
        if straight["rin"] or straight["pnk"]:
            continue

        vec_idx = np.array([px[8] - px[5], py[8] - py[5]])
        vec_mid = np.array([px[12] - px[9], py[12] - py[9]])
        norm = np.linalg.norm(vec_idx) * np.linalg.norm(vec_mid)
        if norm < 1e-6:
            continue
        cos_angle = np.clip(np.dot(vec_idx, vec_mid) / norm, -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cos_angle)))
        if V_SIGN_ANGLE_MIN < angle < V_SIGN_ANGLE_MAX:
            return True
    return False


def is_smiling(face: FaceInfo) -> bool:
    """笑脸判定：优先 blendshapes 微笑置信度，缺失时回退嘴部张合。"""
    if face.smile is not None:
        return face.smile > SMILE_SCORE_THR
    lm = face.landmarks
    if len(lm) <= 14:
        return False
    # 13=上唇内侧点，14=下唇内侧点（一期口径：y 差 > 阈值算张嘴/笑）
    return float(lm[14, 1] - lm[13, 1]) > SMILE_MOUTH_GAP


def any_smiling(faces: list[FaceInfo]) -> bool:
    return any(is_smiling(f) for f in faces)


class AutoCaptureState:
    """自动拍照触发状态机（V 手势 / 笑脸）。

    由相机工作线程每帧调用 update()；时间源由调用方传入（time.time()），
    便于单测用假时钟。返回触发的触发器名（"v_sign" / "smile"）或 None。
    """

    def __init__(self, v_hold: float = V_SIGN_HOLD,
                 smile_hold: float = SMILE_HOLD,
                 cooldown: float = TRIGGER_COOLDOWN):
        self.v_hold = v_hold
        self.smile_hold = smile_hold
        self.cooldown = cooldown
        self._v_since: Optional[float] = None       # V 手势开始时刻
        self._smile_since: Optional[float] = None   # 笑脸开始时刻
        self._last_fire: float = -1e9               # 上次触发时刻

    def update(self, now: float, v_sign: bool, smiling: bool) -> Optional[str]:
        trigger: Optional[str] = None

        # V 手势持续判定
        if v_sign:
            if self._v_since is None:
                self._v_since = now
            if now - self._v_since >= self.v_hold:
                trigger = "v_sign"
        else:
            self._v_since = None

        # 笑脸持续判定
        if smiling:
            if self._smile_since is None:
                self._smile_since = now
            if now - self._smile_since >= self.smile_hold:
                trigger = trigger or "smile"
        else:
            self._smile_since = None

        # 冷却 + 触发后复位持续计时
        if trigger is not None:
            if now - self._last_fire < self.cooldown:
                return None
            self._last_fire = now
            self._v_since = None
            self._smile_since = None
        return trigger

    def reset(self) -> None:
        self._v_since = None
        self._smile_since = None
