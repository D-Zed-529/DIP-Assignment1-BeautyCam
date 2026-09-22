"""美颜纯函数单测（静态合成图，不依赖摄像头/模型）。

运行：.venv/bin/python -m unittest discover tests -v
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from core.effects.beauty import (
    BeautyEffect, JAW_LEFT_IDS, JAW_RIGHT_IDS, enlarge_eyes, estimate_yaw_deg,
    face_oval_mask, get_skin_mask, pose_gate, slim_face, whitening,
)
import core.effects.beauty as B
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
    """合成 468 点归一化关键点（解剖一致版）。

    FACE_OVAL 序从额顶 (10) 起沿轮廓一圈，每个 id 依序间隔 10°、起点在
    正上方（θ=270°，图像 y 向下）。下颌链是轮廓子序列，自然落在下弧，
    轮廓多边形保持凸——瘦脸的"内外"判定依赖这一点。
    """
    import math
    h, w = frame.shape[:2]
    lm = np.zeros((468, 3), np.float32)
    cx, cy, rx, ry = w / 2, h / 2, w / 5, h / 3
    for k, idx in enumerate(OVAL_IDS):
        th = math.radians((270.0 + 10.0 * k) % 360.0)
        lm[idx] = (cx + rx * np.cos(th)) / w, (cy + ry * np.sin(th)) / h, 0.0
    reserved = set(OVAL_IDS) | set(JAW_LEFT_IDS) | set(JAW_RIGHT_IDS)
    for i in range(468):
        if i not in reserved:
            lm[i] = cx / w, cy / h, 0.0   # 其余点放脸中心
    # 眼/嘴锚点（瘦脸竖直渐变窗依赖）：下眼睑与下唇
    lm[145] = 0.42, 0.45, 0.0
    lm[374] = 0.58, 0.45, 0.0
    lm[14] = 0.50, 0.66, 0.0
    return lm


def yawed_face_landmarks(frame: np.ndarray, squeeze: float) -> np.ndarray:
    """模拟侧脸：把左半张脸的水平坐标向中线压缩（透视缩短）。

    squeeze=0 完全不动；squeeze=0.35 表示左侧压缩 35%（左半为远端）。
    同时作用于轮廓点与下颌链，保持几何一致。
    """
    lm = synth_face_landmarks(frame).copy()
    cx = 0.5
    left = lm[:, 0] < cx
    lm[left, 0] = cx - (cx - lm[left, 0]) * (1.0 - squeeze)
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
    def _spread_eyes(self, lm):
        # 眼角保持间距（外角 0.45 / 内角 0.55），上下睑略偏移
        for eye_ids in ((33, 133, 159, 145), (362, 263, 386, 374)):
            outer, inner, top, bottom = eye_ids
            lm[outer] = (0.45, 0.50, 0)
            lm[inner] = (0.55, 0.50, 0)
            lm[top] = (0.50, 0.48, 0)
            lm[bottom] = (0.50, 0.52, 0)
        return lm

    def test_zero_strength_identity(self):
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        out = enlarge_eyes(frame, lm, strength=0.0)
        self.assertTrue(np.array_equal(out, frame))

    def test_enlarge_changes_eye_region_only(self):
        frame = synth_face_frame()
        lm = self._spread_eyes(synth_face_landmarks(frame))
        out = enlarge_eyes(frame, lm, strength=0.3)
        h, w = frame.shape[:2]
        eye_roi = np.s_[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10]
        self.assertFalse(np.array_equal(out[eye_roi], frame[eye_roi]))
        # 远离眼睛的区域不动
        self.assertTrue(np.array_equal(out[:40, :40], frame[:40, :40]))

    def test_disabled_at_large_yaw(self):
        frame = synth_face_frame()
        lm = self._spread_eyes(synth_face_landmarks(frame))
        out = enlarge_eyes(frame, lm, strength=0.3, yaw_deg=40.0)
        self.assertTrue(np.array_equal(out, frame))

    def test_squint_eye_skipped(self):
        """透视塌缩的眼睛（眼角距过小）不做变形。"""
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        # 左眼中心 0.30、右眼中心 0.70，外角/内角各偏 0.05（正常间距）
        for eye_ids, c in (((33, 133, 159, 145), 0.30),
                           ((362, 263, 386, 374), 0.70)):
            outer, inner, top, bottom = eye_ids
            lm[outer] = (c - 0.05, 0.50, 0)
            lm[inner] = (c + 0.05, 0.50, 0)
            lm[top] = (c, 0.48, 0)
            lm[bottom] = (c, 0.52, 0)
        # 左眼外角贴到内角旁：眼角距 0.005 < 0.02，跳过该眼
        lm[33] = (0.345, 0.50, 0)
        out = enlarge_eyes(frame, lm, strength=0.4)
        h, w = frame.shape[:2]
        left_roi = np.s_[h // 2 - 8:h // 2 + 8, int(0.24 * w):int(0.36 * w)]
        self.assertTrue(np.array_equal(out[left_roi], frame[left_roi]))
        right_roi = np.s_[h // 2 - 8:h // 2 + 8, int(0.64 * w):int(0.76 * w)]
        self.assertFalse(np.array_equal(out[right_roi], frame[right_roi]))


class TestSlimFace(unittest.TestCase):
    def test_zero_strength_identity(self):
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        out = slim_face(frame, lm, strength=0.0)
        self.assertTrue(np.array_equal(out, frame))

    def test_slim_pulls_jaw_toward_center(self):
        """方向回归：左下颌处的标记块应向脸中心（右）移动，而非外扩。

        标记经 alpha 混合后不再是纯绿，按"绿色主导"检测（G 高、R/B 低）。
        """
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        h, w = frame.shape[:2]
        # 在左下颌链中点放一个纯绿标记块
        chain = lm[JAW_LEFT_IDS]
        mx = int(chain[:, 0].mean() * w)
        my = int(chain[:, 1].mean() * h)
        cv2.rectangle(frame, (mx - 4, my - 4), (mx + 4, my + 4), (0, 255, 0), -1)

        def marker_centroid(img):
            m = (img[:, :, 1] > 200) & (img[:, :, 0] < 80) & (img[:, :, 2] < 80)
            self.assertGreater(int(m.sum()), 20)   # 标记可检出
            return float(np.nonzero(m)[1].mean())

        before = marker_centroid(frame)
        out = slim_face(frame, lm, strength=1.0)
        after = marker_centroid(out)
        self.assertGreater(after, before + 3.0)   # 明显右移（内收）

    def test_slim_leaves_forehead_untouched(self):
        """瘦脸上界在眼线渐变：额头/头顶/太阳穴不应被位移。"""
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        out = slim_face(frame, lm, strength=1.0)
        h = frame.shape[0]
        top = np.s_[:int(h * 0.45), :]   # 眼线（0.45h）以上区域
        self.assertTrue(np.array_equal(out[top], frame[top]))


class TestWhitenScope(unittest.TestCase):
    """美白范围：默认全身肤色；whiten_scope=face 时才限定脸部。"""

    def _frame_with_body_skin(self):
        frame = synth_face_frame()
        # 左侧背景铺一块纯"身体皮肤"色（模拟脖子/手臂），远离脸部椭圆
        frame[:, :60] = np.uint8(SKIN_BGR)
        return frame

    def test_default_scope_whitens_body_skin(self):
        frame = self._frame_with_body_skin()
        eff = BeautyEffect(params={"smooth": 0.0, "whiten": 20.0})
        ctx = FrameContext(width=frame.shape[1], height=frame.shape[0])  # 无脸
        out = eff.process(frame.copy(), ctx)
        before = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, 5:55].mean()
        after = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[:, 5:55].mean()
        self.assertGreater(after, before + 3.0)

    def test_face_scope_leaves_body_skin(self):
        frame = self._frame_with_body_skin()
        eff = BeautyEffect(params={"smooth": 0.0, "whiten": 20.0,
                                   "whiten_scope": "face"})
        lm = synth_face_landmarks(frame)
        ctx = FrameContext(width=frame.shape[1], height=frame.shape[0],
                           faces=[FaceInfo(landmarks=lm, box=(0.3, 0.2, 0.7, 0.8))])
        out = eff.process(frame.copy(), ctx)
        # 身体皮肤区域（取内部，避开边缘与收尾锐化的边界行）亮度不变
        body = np.s_[5:-5, 5:55]
        before = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[body].mean()
        after = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[body].mean()
        self.assertAlmostEqual(after, before, delta=1.0)


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


class TestPoseGate(unittest.TestCase):
    """侧脸防伪影：头姿估计、门控与远端下颌链跳过。"""

    def test_estimate_yaw_symmetric_zero(self):
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        self.assertAlmostEqual(estimate_yaw_deg(lm), 0.0, delta=2.0)

    def test_estimate_yaw_asymmetric(self):
        frame = synth_face_frame()
        # 左半压缩 35% → 远端在左，yaw 约 (1-0.65)/(1+0.65)*90 ≈ 19°
        lm = yawed_face_landmarks(frame, squeeze=0.35)
        yaw = estimate_yaw_deg(lm)
        self.assertGreater(yaw, B.FAR_SIDE_SKIP_DEG)
        self.assertLess(yaw, B.YAW_ZERO_DEG)

    def test_pose_gate_ramp(self):
        self.assertEqual(pose_gate(0.0), 1.0)
        self.assertEqual(pose_gate(-B.YAW_FULL_DEG), 1.0)
        mid = (B.YAW_FULL_DEG + B.YAW_ZERO_DEG) / 2
        self.assertAlmostEqual(pose_gate(mid), 0.5, places=6)
        self.assertEqual(pose_gate(B.YAW_ZERO_DEG), 0.0)
        self.assertEqual(pose_gate(80.0), 0.0)

    def test_slim_disabled_at_large_yaw(self):
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        out = slim_face(frame, lm, strength=1.0, yaw_deg=40.0)
        self.assertTrue(np.array_equal(out, frame))

    def test_slim_far_side_suppressed(self):
        """侧脸（中等偏航）时远端（塌缩侧）位移远小于近端（真实轮廓侧）。

        远端链被跳过后，仅剩近端宽内侧场的跨中线残余影响——语义是
        "远端不再被直接液化"，而非逐位不变。
        """
        frame = synth_face_frame()
        lm = yawed_face_landmarks(frame, squeeze=0.35)
        yaw = estimate_yaw_deg(lm)
        self.assertGreater(abs(yaw), B.FAR_SIDE_SKIP_DEG)
        h, w = frame.shape[:2]

        def marker_centroid(img, x, y, win=70):
            """只在与预期位置 win 半径内找红色标记，避免串扰。"""
            m = (img[:, :, 2] > 200) & (img[:, :, 1] < 80) & (img[:, :, 0] < 80)
            yy, xx = np.nonzero(m)
            sel = (np.abs(yy - y) < win) & (np.abs(xx - x) < win)
            self.assertGreater(int(sel.sum()), 10)
            return float(xx[sel].mean())

        far_chain = lm[JAW_LEFT_IDS]
        near_chain = lm[JAW_RIGHT_IDS]
        fx = int(far_chain[:, 0].mean() * w)
        fy = int(far_chain[:, 1].mean() * h)
        nx = int(near_chain[:, 0].mean() * w)
        ny = int(near_chain[:, 1].mean() * h)
        for x, y in ((fx, fy), (nx, ny)):
            cv2.rectangle(frame, (x - 4, y - 4), (x + 4, y + 4), (0, 0, 255), -1)
        out = slim_face(frame, lm, strength=1.0)
        far_shift = abs(marker_centroid(out, fx, fy) - fx)     # 朝中心（右）
        near_shift = abs(nx - marker_centroid(out, nx, ny))    # 朝中心（左）
        self.assertGreater(near_shift, 8.0)            # 近端明显内收
        self.assertLess(far_shift, near_shift / 3.0)   # 远端被显著抑制

    def test_slim_cheek_bulk_moves(self):
        """脸颊主体（轮廓内侧 ~40px）应整体内收，而非只有贴线窄管在动。"""
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        h, w = frame.shape[:2]
        chain = lm[JAW_LEFT_IDS]
        cx = int(chain[:, 0].mean() * w)
        cy = int(chain[:, 1].mean() * h)
        # 沿"链中点 → 椭圆中心"方向内移 40px（脸颊内部）
        ctr = np.array([w / 2, h / 2])
        p = np.array([cx, cy], float)
        d = 40.0 * (ctr - p) / np.linalg.norm(ctr - p)
        probe_x, probe_y = int(p[0] + d[0]), int(p[1] + d[1])
        cv2.rectangle(frame, (probe_x - 4, probe_y - 4),
                      (probe_x + 4, probe_y + 4), (0, 0, 255), -1)
        out = slim_face(frame, lm, strength=1.0)
        m = (out[:, :, 2] > 200) & (out[:, :, 1] < 80) & (out[:, :, 0] < 80)
        yy, xx = np.nonzero(m)
        sel = (np.abs(yy - probe_y) < 90) & (np.abs(xx - probe_x) < 90)
        self.assertGreater(int(sel.sum()), 10)
        after = float(xx[sel].mean())
        self.assertGreater(after, probe_x + 8.0)   # 明显向中心移动

    def test_slim_near_side_still_works(self):
        """侧脸时近端（真实轮廓一侧）仍正常内收。"""
        frame = synth_face_frame()
        lm = yawed_face_landmarks(frame, squeeze=0.35)
        h, w = frame.shape[:2]
        near_chain = lm[JAW_RIGHT_IDS]
        nx = int(near_chain[:, 0].mean() * w)
        ny = int(near_chain[:, 1].mean() * h)
        cv2.rectangle(frame, (nx - 4, ny - 4), (nx + 4, ny + 4), (0, 0, 255), -1)

        def red_centroid(img):
            m = (img[:, :, 2] > 200) & (img[:, :, 1] < 80) & (img[:, :, 0] < 80)
            self.assertGreater(int(m.sum()), 10)
            return float(np.nonzero(m)[1].mean())

        before = red_centroid(frame)
        out = slim_face(frame, lm, strength=1.0)
        after = red_centroid(out)
        self.assertLess(after, before - 1.0)   # 向左（中线方向）移动


if __name__ == "__main__":
    unittest.main()
