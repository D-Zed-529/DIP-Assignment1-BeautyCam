"""美颜纯函数单测（静态合成图，不依赖摄像头/模型）。

运行：.venv/bin/python -m unittest discover tests -v
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from core.effects.beauty import (
    BeautyEffect, JAW_LEFT_IDS, JAW_RIGHT_IDS, enlarge_eyes, estimate_yaw_deg,
    face_oval_mask, get_skin_mask, person_region_mask, person_soft_mask_u8,
    pose_gate, slim_face, slim_face_maps, whitening,
)
from core.mls import identity_maps
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
    # 眼/眉/鼻/嘴锚点（MLS 锚点与竖直窗依赖，须真实分布而非堆在中心）：
    lm[33] = 0.435, 0.42, 0.0    # 左眼外角
    lm[133] = 0.465, 0.42, 0.0   # 左眼内角
    lm[362] = 0.535, 0.42, 0.0   # 右眼内角
    lm[263] = 0.565, 0.42, 0.0   # 右眼外角
    lm[145] = 0.45, 0.45, 0.0    # 左下眼睑
    lm[374] = 0.55, 0.45, 0.0    # 右下眼睑
    lm[105] = 0.44, 0.36, 0.0    # 左眉
    lm[334] = 0.56, 0.36, 0.0    # 右眉
    lm[1] = 0.50, 0.52, 0.0      # 鼻尖
    lm[61] = 0.46, 0.63, 0.0     # 左嘴角
    lm[291] = 0.54, 0.63, 0.0    # 右嘴角
    lm[14] = 0.50, 0.66, 0.0     # 下唇
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
        """极端偏航大眼仍工作但被门控衰减（不等于全强度效果）。"""
        frame = synth_face_frame()
        lm = self._spread_eyes(synth_face_landmarks(frame))
        out_extreme = enlarge_eyes(frame, lm, strength=0.3,
                                   yaw_deg=B.YAW_ZERO_DEG)
        out_frontal = enlarge_eyes(frame, lm, strength=0.3, yaw_deg=0.0)
        self.assertFalse(np.array_equal(out_extreme, frame))
        self.assertFalse(np.array_equal(out_extreme, out_frontal))

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
        """瘦脸位移场钉住额头：眉线以上严格恒等（保护窗），眼下过渡带 ≤3.5px。

        MLS 是全局光滑场，眉线以上由竖直余弦保护窗乘到严格 0（比特级
        不动）；眉线到眼线的过渡带允许小幅光滑泄漏（硬置零会在眼线处
        产生可见接缝）。断言位移场量而非像素值——高对比边缘处的像素
        值差是重采样放大，不代表可见位移。
        """
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        h, w = frame.shape[:2]
        map_x, map_y = slim_face_maps(w, h, lm, strength=1.0)
        id_x, id_y = identity_maps(h, w)
        disp = np.hypot(map_x - id_x, map_y - id_y)
        brow_top = int(min(lm[105, 1], lm[334, 1]) * h)
        # 眉线以上：保护窗严格恒等（比特级）
        self.assertTrue(np.array_equal(map_x[:brow_top, :], id_x[:brow_top, :]))
        self.assertTrue(np.array_equal(map_y[:brow_top, :], id_y[:brow_top, :]))
        # 眉线→眼线过渡带：光滑泄漏上限（亚视觉）
        self.assertLessEqual(float(disp[:int(h * 0.45), :].max()), 3.5)


class TestWhitenScope(unittest.TestCase):
    """美白范围：默认"全身肤色"约束在人像区域（脸框扩展），背景同色不误白；
    whiten_scope=face 时仅脸部椭圆。"""

    BOX = (0.3, 0.2, 0.7, 0.8)   # 与 synth_face_landmarks 的脸椭圆一致

    def _paint_skin(self, frame, x1, x2, y1, y2):
        frame[y1:y2, x1:x2] = np.uint8(SKIN_BGR)
        return frame

    def _ctx(self, frame):
        lm = synth_face_landmarks(frame)
        return FrameContext(width=frame.shape[1], height=frame.shape[0],
                            faces=[FaceInfo(landmarks=lm, box=self.BOX)])

    def _gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def test_skin_scope_whitens_neck_below_face(self):
        """默认档：脸正下方的脖子（人像区域内）肤色应被美白。"""
        frame = synth_face_frame()
        self._paint_skin(frame, 280, 360, 410, 455)   # 脸椭圆底(y≈400)下方
        eff = BeautyEffect(params={"smooth": 0.0, "whiten": 20.0})
        out = eff.process(frame.copy(), self._ctx(frame))
        before = self._gray(frame)[412:453, 282:358].mean()
        after = self._gray(out)[412:453, 282:358].mean()
        self.assertGreater(after, before + 3.0)

    def test_skin_scope_ignores_background_skin(self):
        """默认档：远离人脸的同色背景不应被美白。"""
        frame = synth_face_frame()
        self._paint_skin(frame, 0, 40, 0, 40)   # 左上角（人像区域外）
        eff = BeautyEffect(params={"smooth": 0.0, "whiten": 20.0})
        out = eff.process(frame.copy(), self._ctx(frame))
        before = self._gray(frame)[2:38, 2:38].mean()
        after = self._gray(out)[2:38, 2:38].mean()
        self.assertAlmostEqual(after, before, delta=1.0)

    def test_face_scope_leaves_neck(self):
        """仅脸部档：脖子（脸椭圆下方）不应被美白。"""
        frame = synth_face_frame()
        self._paint_skin(frame, 280, 360, 410, 455)   # 脖子
        eff = BeautyEffect(params={"smooth": 0.0, "whiten": 20.0,
                                   "whiten_scope": "face"})
        out = eff.process(frame.copy(), self._ctx(frame))
        before = self._gray(frame)[412:453, 282:358].mean()
        after = self._gray(out)[412:453, 282:358].mean()
        self.assertAlmostEqual(after, before, delta=1.0)

    def test_no_face_no_whiten(self):
        """无人脸时不美白（避免整帧乱白肤色背景）。"""
        frame = synth_face_frame()
        self._paint_skin(frame, 0, 40, 0, 40)   # 背景肤色
        eff = BeautyEffect(params={"smooth": 0.0, "whiten": 20.0})
        ctx = FrameContext(width=frame.shape[1], height=frame.shape[0])  # 无脸
        out = eff.process(frame.copy(), ctx)
        before = self._gray(frame)[2:38, 2:38].mean()
        after = self._gray(out)[2:38, 2:38].mean()
        self.assertAlmostEqual(after, before, delta=1.0)

    def test_person_region_geometry(self):
        """person_region_mask：覆盖脸下方（脖子/肩），排除远处角落。"""
        frame = synth_face_frame()
        mask = person_region_mask(frame, self._ctx(frame).faces)
        h, w = frame.shape[:2]
        self.assertEqual(mask.shape, (h, w))
        # 脸正下方（脖子，y≈440）在人像区域内
        self.assertGreater(int(mask[440, 320]), 200)
        # 远处左上角在人像区域外
        self.assertLess(int(mask[10, 10]), 64)

    def test_skin_scope_uses_segmentation_over_box(self):
        """有分割软掩膜时用它（比脸框扩展更紧）圈人：框内但掩膜外的背景肤色不白。"""
        frame = synth_face_frame()
        self._paint_skin(frame, 100, 150, 200, 250)   # 脸左侧：框内、椭圆外
        eff = BeautyEffect(params={"smooth": 0.0, "whiten": 20.0})
        ctx = self._ctx(frame)
        h, w = frame.shape[:2]
        alpha = np.zeros((h, w), np.float32)
        cv2.ellipse(alpha, (w // 2, h // 2), (w // 5, h // 3), 0, 0, 360, 1.0, -1)
        ctx.person_alpha = alpha          # 只圈脸椭圆的分割软掩膜
        out = eff.process(frame.copy(), ctx)
        before = self._gray(frame)[206:244, 106:144].mean()
        after = self._gray(out)[206:244, 106:144].mean()
        self.assertAlmostEqual(after, before, delta=1.0)

    def test_person_soft_mask_ghost_floor_removed(self):
        """多分类模型的背景鬼影地板（~0.08）应被清零，避免背景轻微带白。"""
        h, w = 100, 100
        alpha = np.full((h, w), 0.08, np.float32)   # 背景地板
        alpha[40:60, 40:60] = 0.9                   # 前景
        out = person_soft_mask_u8(alpha, None)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(int(out[10, 10]), 0)       # 地板被清零
        self.assertGreater(int(out[50, 50]), 200)   # 前景保留
        # 无 alpha 退化到硬掩膜
        mask = np.full((h, w), 255, np.uint8)
        self.assertIs(person_soft_mask_u8(None, mask), mask)
        # 都没有返回 None
        self.assertIsNone(person_soft_mask_u8(None, None))


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
        self.assertGreater(yaw, 10.0)
        self.assertLess(yaw, B.YAW_ZERO_DEG)

    def test_pose_gate_ramp(self):
        self.assertEqual(pose_gate(0.0), 1.0)
        self.assertEqual(pose_gate(-B.YAW_FULL_DEG), 1.0)
        mid = (B.YAW_FULL_DEG + B.YAW_ZERO_DEG) / 2
        # 衰减区间中点：从 1.0 线性降到下限 0.5 的过程值
        self.assertAlmostEqual(pose_gate(mid),
                               1.0 - 0.5 * (1.0 - B.POSE_GATE_MIN), places=6)
        self.assertAlmostEqual(pose_gate(B.YAW_ZERO_DEG), B.POSE_GATE_MIN)
        self.assertEqual(pose_gate(120.0), B.POSE_GATE_MIN)

    def test_slim_chin_also_lifted(self):
        """瘦脸时下巴尖也应上收（"下巴也瘦一点"），水平方向不外扩。"""
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        h, w = frame.shape[:2]
        map_x, map_y = slim_face_maps(w, h, lm, strength=1.0)
        xi, yi = int(lm[152, 0] * w), int(lm[152, 1] * h)
        dx = float(xi - map_x[yi, xi])
        dy = float(yi - map_y[yi, xi])
        self.assertLess(dy, -3.0)     # 内容上移 = 下巴收短
        self.assertLess(abs(dx), 2.0)  # 不应水平外扩（防尖锥回归）

    def test_slim_half_floor_at_extreme_yaw(self):
        """极端偏航不再关死（"打一半"）：仍有效果，但明显弱于正脸。"""
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        out_extreme = slim_face(frame, lm, strength=1.0,
                                yaw_deg=B.YAW_ZERO_DEG)
        out_frontal = slim_face(frame, lm, strength=1.0, yaw_deg=0.0)
        self.assertFalse(np.array_equal(out_extreme, frame))    # 仍有作用
        self.assertFalse(np.array_equal(out_extreme, out_frontal))  # 但被衰减

    def test_slim_far_side_suppressed(self):
        """侧脸（中重度偏航）时远端（塌缩侧）位移远小于近端（真实轮廓侧）。

        远端链按塌缩比连续衰减后仍保留少量幅度（无硬切），仅剩近端
        场的跨中线残余影响——语义是"远端不再被明显液化"，而非逐位不变。
        """
        frame = synth_face_frame()
        lm = yawed_face_landmarks(frame, squeeze=0.35)
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

    def test_slim_far_side_continuous_not_hard_cut(self):
        """中度侧脸（squeeze=0.3，塌缩比 ≈0.7）远端应保留部分幅度。

        回归"硬切"语义：旧版超过 15° 直接整条跳过远端链（跳变），
        现版按塌缩比连续过渡——中度侧脸远端幅度应明显非零、且仍小于
        近端（不对称收缩）。
        """
        frame = synth_face_frame()
        lm = yawed_face_landmarks(frame, squeeze=0.30)
        h, w = frame.shape[:2]

        def marker_centroid(img, x, y, win=70):
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
        far_shift = abs(marker_centroid(out, fx, fy) - fx)
        near_shift = abs(nx - marker_centroid(out, nx, ny))
        self.assertGreater(far_shift, 1.5)             # 远端仍有作用（无硬切）
        self.assertLess(far_shift, near_shift * 0.8)   # 但仍弱于近端

    def test_slim_cheek_bulk_moves(self):
        """脸颊主体（轮廓内侧 ~40px）应整体内收，而非只有贴线窄管在动。

        断言位移场量：探针处内容位移沿"指向脸中心"方向 ≥3.5px@strength=1.0。
        标记质心法在强压缩场（探针处应变 ~100%，标记块被压扁）里读数
        会被不对称压缩拉低，场量才是直接、确定性的证据。注意合成脸很
        小（探针距嘴角保护点仅 ~36px，嘴角权值 5 的保护场会平滑衰减
        探针位移；真图脸颊主体位移比例远高于此，效果以真图 A/B 为准），
        该断言用于守住"脸颊整体内收"的下限与方向，不做效果上限。
        """
        frame = synth_face_frame()
        lm = synth_face_landmarks(frame)
        h, w = frame.shape[:2]
        map_x, map_y = slim_face_maps(w, h, lm, strength=1.0)
        chain = lm[JAW_LEFT_IDS]
        ctr = np.array([w / 2, h / 2])
        p = chain.mean(axis=0)[:2] * np.array([w, h], np.float32)
        d = 40.0 * (ctr - p) / np.linalg.norm(ctr - p)
        probe = p + d
        xi, yi = int(probe[0]), int(probe[1])
        disp = np.array([xi - map_x[yi, xi], yi - map_y[yi, xi]])
        inward = float(disp @ (ctr - probe) / np.linalg.norm(ctr - probe))
        self.assertGreater(inward, 3.5)   # 明显向中心移动

        # 像素级方向冒烟：标记块整体仍应向中心移动（阈值放宽到 >1px）
        frame2 = synth_face_frame()
        probe_x, probe_y = xi, yi
        cv2.rectangle(frame2, (probe_x - 4, probe_y - 4),
                      (probe_x + 4, probe_y + 4), (0, 0, 255), -1)
        out = slim_face(frame2, lm, strength=1.0)
        m = (out[:, :, 2] > 200) & (out[:, :, 1] < 80) & (out[:, :, 0] < 80)
        yy, xx = np.nonzero(m)
        sel = (np.abs(yy - probe_y) < 90) & (np.abs(xx - probe_x) < 90)
        self.assertGreater(int(sel.sum()), 10)
        self.assertGreater(float(xx[sel].mean()), probe_x + 1.0)

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


class TestPerfEquivalence(unittest.TestCase):
    """性能优化（ROI remap / 半分辨率）与原实现的数值等价性。"""

    def test_slim_roi_remap_bitwise_equal(self):
        """瘦脸 ROI remap 必须与全帧 remap 逐位一致（仅性能优化，不改结果）。"""
        frame = synth_face_frame(w=320, h=240, seed=7)
        lm = synth_face_landmarks(frame)
        map_x, map_y = slim_face_maps(320, 240, lm, strength=1.0, yaw_deg=0.0)
        full = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        roi_out = slim_face(frame, lm, strength=1.0, yaw_deg=0.0)
        self.assertTrue(np.array_equal(full, roi_out))

    def test_whitening_bbox_matches_masked_region(self):
        """美白 boundingRect 路径：盒外严格不变、盒内确实提亮。"""
        frame = synth_face_frame(w=320, h=240, seed=9)
        mask = np.zeros(frame.shape[:2], np.uint8)
        mask[40:160, 60:220] = 200       # 任意矩形软掩膜
        out = whitening(frame, mask, strength=25.0)
        self.assertTrue(np.array_equal(out[:40], frame[:40]))      # 盒外不变
        lab_in = cv2.cvtColor(frame[41:159, 61:219], cv2.COLOR_BGR2LAB)
        lab_out = cv2.cvtColor(out[41:159, 61:219], cv2.COLOR_BGR2LAB)
        self.assertGreater(float(lab_out[:, :, 0].mean()),
                           float(lab_in[:, :, 0].mean()))          # L 通道提亮

    def test_skin_mask_halfres_shape_and_semantics(self):
        """肤色掩膜半分辨率路径：形状不变、肤色区命中、背景区 miss。"""
        frame = synth_face_frame(w=320, h=240, seed=3)
        mask = get_skin_mask(frame)
        self.assertEqual(mask.shape, frame.shape[:2])
        h, w = mask.shape
        self.assertGreater(float(mask[h // 2, w // 2]) / 255.0, 0.5)   # 脸中心
        self.assertLess(float(mask[10, 10]) / 255.0, 0.1)              # 角落背景
