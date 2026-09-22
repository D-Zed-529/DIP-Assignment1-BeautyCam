"""美颜纯函数单测（静态合成图，不依赖摄像头/模型）。

运行：.venv/bin/python -m unittest discover tests -v
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from core.effects.beauty import (
    BeautyEffect, JAW_LEFT_IDS, JAW_RIGHT_IDS, enlarge_eyes, face_oval_mask,
    get_skin_mask, slim_face, whitening,
)
from core.context import FaceInfo, FrameContext
from core.infer import FACE_OVAL_IDS as OVAL_IDS

# 基础色（BGR）：肤色/蓝背景均在/均不在 YCrCb 肤色窗内（一期阈值）
SKIN_BGR = np.array([120, 150, 190], np.int16)
BG_BGR = np.array([180, 90, 30], np.int16)


def synth_face_frame(w=640, h=480, seed=42) -> np.ndarray:
    """合成"肤色椭圆脸 + 蓝背景"测试图。

    叠加 ±12 噪声：既满足肤色阈值窗，又提供纹理，让位移/放大类
    断言能观测到像素变化（纯色块位移后外观不变）。
    """
    rng = np.random.default_rng(seed)
    noise = rng.integers(-12, 13, (h, w, 3), dtype=np.int16)
    frame = np.clip(BG_BGR[None, None, :] + noise, 0, 255).astype(np.uint8)
    face = np.zeros((h, w), np.uint8)
    cv2.ellipse(face, (w // 2, h // 2), (w // 5, h // 3), 0, 0, 360, 255, -1)
    face_part = np.clip(SKIN_BGR[None, None, :] + noise, 0, 255).astype(np.uint8)
    return np.where(face[..., None] > 0, face_part, frame)


def synth_face_landmarks(frame: np.ndarray) -> np.ndarray:
    """合成 468 点归一化关键点：轮廓椭圆 + 下颌两侧点。"""
    h, w = frame.shape[:2]
    lm = np.zeros((468, 3), np.float32)
    cx, cy, rx, ry = w / 2, h / 2, w / 5, h / 3
    t = np.linspace(0, 2 * np.pi, len(OVAL_IDS), endpoint=False)
    for idx, ang in zip(OVAL_IDS, t):
        lm[idx] = (cx + rx * np.cos(ang)) / w, (cy + ry * np.sin(ang)) / h, 0.0
    reserved = set(OVAL_IDS) | set(JAW_LEFT_IDS) | set(JAW_RIGHT_IDS)
    for i in range(468):
        if i not in reserved:
            lm[i] = cx / w, cy / h, 0.0
    # 下颌两侧：左颊 x < 中心，右颊 x > 中心，y 在下半段
    left_xs = np.linspace(cx - rx, cx - rx * 0.2, len(JAW_LEFT_IDS))
    right_xs = np.linspace(cx + rx * 0.2, cx + rx, len(JAW_RIGHT_IDS))
    ys = np.linspace(cy + ry * 0.55, cy + ry * 0.95, len(JAW_LEFT_IDS))
    for idx, x, y in zip(JAW_LEFT_IDS, left_xs, ys):
        lm[idx] = x / w, y / h, 0.0
    for idx, x, y in zip(JAW_RIGHT_IDS, right_xs, ys):
        lm[idx] = x / w, y / h, 0.0
    return lm


class TestSkinMask(unittest.TestCase):
    def test_skin_detected_background_rejected(self):
        frame = synth_face_frame()
        mask = get_skin_mask(frame)
        self.assertEqual(mask.max(), 255)
        h, w = frame.shape[:2]
        # 脸中心应为肤色（取 5×5 均值，容忍噪声）
        self.assertGreater(mask[h // 2 - 2:h // 2 + 3,
                                w // 2 - 2:w // 2 + 3].mean(), 128)
        # 背景角落不应算肤色
        self.assertLess(mask[:10, :10].max(), 128)

    def test_mask_shape_matches(self):
        frame = synth_face_frame()
        self.assertEqual(get_skin_mask(frame).shape, frame.shape[:2])


class TestWhitening(unittest.TestCase):
    def test_whiten_increases_brightness_in_mask(self):
        frame = synth_face_frame()
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (w // 2, h // 2), 40, 255, -1)
        out = whitening(frame, mask, strength=20)
        before = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[
            h // 2 - 3:h // 2 + 4, w // 2 - 3:w // 2 + 4].mean()
        after = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[
            h // 2 - 3:h // 2 + 4, w // 2 - 3:w // 2 + 4].mean()
        self.assertGreater(after, before)
        # 掩膜外仅允许 LAB 往返 ±2 的量化误差
        self.assertTrue(np.allclose(out[5, 5], frame[5, 5], atol=2))


class TestFaceOvalMask(unittest.TestCase):
    def test_oval_mask_covers_center_not_corner(self):
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        mask = face_oval_mask(frame, lm)
        h, w = frame.shape[:2]
        self.assertGreater(int(mask[h // 2, w // 2]), 100)
        self.assertEqual(mask[5, 5], 0)


class TestEnlargeEyes(unittest.TestCase):
    def test_zero_strength_identity(self):
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        out = enlarge_eyes(frame, lm, strength=0.0)
        self.assertTrue(np.array_equal(out, frame))

    def test_enlarge_changes_eye_region_only(self):
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        for eye_ids in ((33, 133, 159, 145), (362, 263, 386, 374)):
            for i in eye_ids:
                lm[i, 0], lm[i, 1] = 0.5, 0.5
        out = enlarge_eyes(frame, lm, strength=0.3)
        h, w = frame.shape[:2]
        eye_roi = np.s_[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10]
        self.assertFalse(np.array_equal(out[eye_roi], frame[eye_roi]))
        # 远离眼睛的区域不动
        self.assertTrue(np.array_equal(out[:40, :40], frame[:40, :40]))


class TestSlimFace(unittest.TestCase):
    def test_zero_strength_identity(self):
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        out = slim_face(frame, lm, strength=0.0)
        self.assertTrue(np.array_equal(out, frame))

    def test_slim_pulls_jaw_toward_center(self):
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        out = slim_face(frame, lm, strength=1.0)
        h, w = frame.shape[:2]
        left_jaw = lm[JAW_LEFT_IDS]
        y_probe = int(left_jaw[:, 1].mean() * h)
        x_probe = int(left_jaw[:, 0].mean() * w)
        jaw_roi = np.s_[y_probe - 3:y_probe + 3, x_probe - 3:x_probe + 3]
        self.assertFalse(np.array_equal(out[jaw_roi], frame[jaw_roi]))
        # 图像顶部远端不变
        self.assertTrue(np.array_equal(out[:30, :30], frame[:30, :30]))


class TestBeautyEffect(unittest.TestCase):
    def _ctx_with_face(self, frame):
        lm = synth_face_landmarks(frame)
        info = FaceInfo(landmarks=lm, box=(0.3, 0.2, 0.7, 0.8))
        return FrameContext(width=frame.shape[1], height=frame.shape[0],
                            faces=[info])

    def test_disabled_passthrough(self):
        frame = synth_face_frame()
        eff = BeautyEffect(enabled=False)
        out = eff.process(frame.copy(), self._ctx_with_face(frame))
        self.assertTrue(np.array_equal(out, frame))

    def test_enabled_changes_frame(self):
        frame = synth_face_frame()
        eff = BeautyEffect()
        out = eff.process(frame.copy(), self._ctx_with_face(frame))
        self.assertFalse(np.array_equal(out, frame))
        self.assertEqual(out.shape, frame.shape)

    def test_no_face_still_smooths(self):
        frame = synth_face_frame()
        eff = BeautyEffect()
        ctx = FrameContext(width=frame.shape[1], height=frame.shape[0])
        out = eff.process(frame.copy(), ctx)
        self.assertFalse(np.array_equal(out, frame))

    def test_unknown_param_rejected(self):
        eff = BeautyEffect()
        with self.assertRaises(ValueError):
            eff.set_params(no_such_param=1)

    def test_param_roundtrip(self):
        eff = BeautyEffect()
        eff.set_params(whiten=25.0)
        self.assertEqual(eff.get_params()["whiten"], 25.0)
        # 其余参数不受影响
        self.assertEqual(eff.get_params()["smooth"],
                         BeautyEffect.default_params()["smooth"])


if __name__ == "__main__":
    unittest.main()
