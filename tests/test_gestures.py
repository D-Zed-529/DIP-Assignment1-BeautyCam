"""手势/笑脸判定与自动拍照状态机单测（合成关键点，不依赖模型）。"""

from __future__ import annotations

import unittest

import numpy as np

from core.context import FaceInfo, HandInfo
from core.gestures import (
    AutoCaptureState, is_smiling, is_v_sign, any_smiling,
    SMILE_MOUTH_GAP, V_SIGN_ANGLE_MAX, V_SIGN_ANGLE_MIN,
)

W, H = 1280, 720   # 测试像素坐标口径


def make_hand(fingers: dict[str, bool], spread_deg: float = 30.0,
              origin: tuple[float, float] = (0.5, 0.7)) -> HandInfo:
    """构造 21 点手部关键点。

    指根（5/9/13/17）横向排布在 origin；伸直的手指从指根竖直向上
    （长度 0.25 归一化）；弯曲的手指指尖在指根下方 0.02。食指/中指
    按 spread_deg/2 各向外/内倾斜，模拟 V 字夹角。
    """
    lm = np.zeros((21, 3), np.float32)
    lm[0] = (origin[0], origin[1] + 0.12, 0)          # 手腕
    bases = {"idx": 5, "mid": 9, "rin": 13, "pnk": 17}
    offsets = {"idx": -0.04, "mid": 0.0, "rin": 0.04, "pnk": 0.08}
    half = np.radians(spread_deg) / 2
    for name, base in bases.items():
        bx = origin[0] + offsets[name]
        lm[base] = (bx, origin[1], 0)
        tip = base + 3
        if fingers.get(name, False):
            tilt = -half if name == "idx" else (half if name == "mid" else 0.0)
            dx = 0.25 * np.sin(tilt)
            dy = -0.25 * np.cos(tilt)
            lm[tip] = (bx + dx, origin[1] + dy, 0)
        else:
            lm[tip] = (bx, origin[1] + 0.02, 0)
        # 中间关节（6/7 等）取指根与指尖中点占位（不参与判定）
        mid_pt = ((lm[base][0] + lm[tip][0]) / 2,
                  (lm[base][1] + lm[tip][1]) / 2, 0)
        lm[base + 1] = mid_pt
        lm[base + 2] = mid_pt
    return HandInfo(landmarks=lm, handedness="Right")


def make_face(mouth_gap: float = 0.03, smile: float | None = None) -> FaceInfo:
    lm = np.zeros((468, 3), np.float32)
    lm[13] = (0.5, 0.5, 0)                     # 上唇内侧
    lm[14] = (0.5, 0.5 + mouth_gap, 0)         # 下唇内侧
    return FaceInfo(landmarks=lm, box=(0.3, 0.2, 0.7, 0.8), smile=smile)


class TestVSign(unittest.TestCase):
    def test_v_sign_detected(self):
        hand = make_hand({"idx": True, "mid": True}, spread_deg=30.0)
        self.assertTrue(is_v_sign([hand], W, H))

    def test_fist_not_detected(self):
        hand = make_hand({"idx": False, "mid": False})
        self.assertFalse(is_v_sign([hand], W, H))

    def test_open_palm_not_detected(self):
        # 四指全伸 -> 无名指/小指也伸直，被排除
        hand = make_hand({"idx": True, "mid": True, "rin": True, "pnk": True})
        self.assertFalse(is_v_sign([hand], W, H))

    def test_angle_out_of_band_not_detected(self):
        # 夹角 ~90°（远超 65° 上限）
        hand = make_hand({"idx": True, "mid": True}, spread_deg=90.0)
        self.assertFalse(is_v_sign([hand], W, H))

    def test_angle_in_band_boundary(self):
        # 20° 在带内
        hand = make_hand({"idx": True, "mid": True}, spread_deg=20.0)
        self.assertTrue(is_v_sign([hand], W, H))

    def test_empty_hands(self):
        self.assertFalse(is_v_sign([], W, H))
        empty = HandInfo(landmarks=np.zeros((21, 3), np.float32))
        # 全零关键点：食指/中指"伸直"判定不成立（y 不小于）
        self.assertFalse(is_v_sign([empty], W, H))


class TestSmile(unittest.TestCase):
    def test_blendshape_smile(self):
        self.assertTrue(is_smiling(make_face(smile=0.8)))
        self.assertFalse(is_smiling(make_face(smile=0.1)))

    def test_mouth_gap_fallback(self):
        gap = SMILE_MOUTH_GAP + 0.005
        self.assertTrue(is_smiling(make_face(mouth_gap=gap)))
        self.assertFalse(is_smiling(make_face(mouth_gap=SMILE_MOUTH_GAP / 2)))

    def test_any_smiling(self):
        faces = [make_face(mouth_gap=0.001), make_face(mouth_gap=0.05)]
        self.assertTrue(any_smiling(faces))
        self.assertFalse(any_smiling([make_face(mouth_gap=0.001)]))


class TestAutoCaptureState(unittest.TestCase):
    def test_v_sign_hold_triggers(self):
        st = AutoCaptureState()
        # 前 0.9s 不触发
        self.assertIsNone(st.update(0.0, True, False))
        self.assertIsNone(st.update(0.5, True, False))
        self.assertIsNone(st.update(0.9, True, False))
        # 满 1.0s 触发
        self.assertEqual(st.update(1.0, True, False), "v_sign")

    def test_interrupted_hold_resets(self):
        st = AutoCaptureState()
        st.update(0.0, True, False)
        st.update(0.5, False, False)     # 中断，持续计时清零
        # 1.4s 重新出现 V 手势：从 1.4 起算，仅持续 0.1s 不触发
        self.assertIsNone(st.update(1.4, True, False))
        # 持续到 2.4s 满足 1.0s，触发
        self.assertEqual(st.update(2.4, True, False), "v_sign")

    def test_smile_hold_triggers(self):
        st = AutoCaptureState()
        self.assertIsNone(st.update(0.0, False, True))
        self.assertEqual(st.update(0.6, False, True), "smile")

    def test_cooldown_blocks(self):
        st = AutoCaptureState()
        st.update(0.0, False, True)
        self.assertEqual(st.update(0.5, False, True), "smile")
        # 冷却期内（距上次触发 <2s）V 手势即使满足持续条件也不触发
        self.assertIsNone(st.update(1.4, True, False))   # 开始保持 V
        self.assertIsNone(st.update(2.4, True, False))   # 满 1s 但冷却差 0.1s
        # 冷却结束后立即触发（持续条件已满足）
        self.assertEqual(st.update(2.5, True, False), "v_sign")

    def test_no_signal_no_trigger(self):
        st = AutoCaptureState()
        for t in np.arange(0, 5, 0.1):
            self.assertIsNone(st.update(float(t), False, False))


if __name__ == "__main__":
    unittest.main()
