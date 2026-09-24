"""torch 推理引擎（TorchInferenceEngine）单测 —— CUDA 迁移（2026-09）。

需要 torch + CUDA + models/torch/*.ts + RVM/DA-v2 权重；任一缺失自动跳过
（纯 CPU / 未转换环境仍可用 mediapipe 路径，见 test_infer.py）。

数值一致性的完整校准记录见 scripts/calibrate_torch.py 与
core/infer_torch.py 模块头（对 mediapipe：landmark Δ≈0.002、alpha
MAE≈0.004、blendshapes 逐名一致）。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from core.infer import SEGMENTER_INTERVAL, torch_backend_ready

_SAMPLE = (Path(__file__).resolve().parent.parent
           / "assets" / "samples" / "portrait1.jpg")

RVM_READY = torch_backend_ready() and (
    Path(__file__).resolve().parent.parent
    / "models" / "torch" / "rvm.ts").exists()


def _sample_720p():
    img = cv2.imread(str(_SAMPLE))
    if img is None:
        return None
    h, w = img.shape[:2]
    # 等比缩到 720 高（不畸变：FaceMesh 对纵横比敏感，AGENTS.md 坑 #14）
    scale = 720 / h
    return cv2.resize(img, (round(w * scale), 720))


@unittest.skipUnless(torch_backend_ready() and _SAMPLE.exists(),
                     "torch 后端未就绪（CUDA / TorchScript 缺失）")
class TestTorchEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.infer_torch import TorchInferenceEngine
        cls.engine = TorchInferenceEngine()
        cls.img = _sample_720p()
        if cls.img is None:
            raise unittest.SkipTest("样例图无法读取")

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_faces_on_sample(self):
        ctx = self.engine.process(self.img, faces=True, hands=True,
                                  segmentation=False)
        self.assertEqual(len(ctx.faces), 1, "样例单人照应恰好检到 1 张脸")
        f = ctx.faces[0]
        self.assertEqual(f.landmarks.shape, (478, 3))   # 含 iris 10 点
        # 微笑置信度来自 HUND blendshapes（原始 0~1）
        self.assertIsNotNone(f.smile)
        self.assertGreaterEqual(f.smile, 0.0)
        self.assertLessEqual(f.smile, 1.0)
        # 人脸框在画面中部（归一化坐标）
        x1, y1, x2, y2 = f.box
        self.assertLess(x1, 0.5)
        self.assertGreater(x2, 0.5)

    def test_tracking_stable_no_duplicates(self):
        """同帧重复：跟踪不应复制轨迹（检测刷新去重），人脸数恒定。"""
        counts = [len(self.engine.process(self.img, faces=True).faces)
                  for _ in range(15)]
        self.assertEqual(set(counts), {1},
                         f"人脸数应恒为 1，实际 {counts}")

    def test_no_false_face_on_background(self):
        """纯背景帧不得检出人脸（presence = sigmoid(Identity_1) 过滤幻影）。"""
        bg = np.zeros((720, 1280, 3), np.uint8)
        bg[:] = (60, 120, 160)
        ctx = self.engine.process(bg, faces=True)
        self.assertEqual(len(ctx.faces), 0)

    def test_depth_shape_and_range(self):
        try:
            ctx = self.engine.process(self.img, faces=False, hands=False,
                                      depth=True)
        except FileNotFoundError:
            self.skipTest("Depth Anything V2 权重缺失")
        self.assertIsNotNone(ctx.depth)
        self.assertEqual(ctx.depth.shape, (720, self.img.shape[1]))
        self.assertEqual(ctx.depth.dtype, np.float32)
        self.assertGreaterEqual(float(ctx.depth.min()), 0.0)
        self.assertLessEqual(float(ctx.depth.max()), 1.0)

    def test_parallel_models_match_sequential(self):
        """同帧人脸与分割并发不应改变关键点或 RVM 掩膜。"""
        original = self.engine.parallel_inference
        try:
            self.engine.reset_temporal()
            self.engine.parallel_inference = False
            serial = self.engine.process(
                self.img, faces=True, hands=False, segmentation=True,
                blendshapes=False)
            self.engine.reset_temporal()
            self.engine.parallel_inference = True
            parallel = self.engine.process(
                self.img, faces=True, hands=False, segmentation=True,
                blendshapes=False)
            self.assertEqual(len(serial.faces), len(parallel.faces))
            for left, right in zip(serial.faces, parallel.faces):
                np.testing.assert_allclose(left.landmarks, right.landmarks,
                                           atol=1e-5)
            np.testing.assert_allclose(serial.person_alpha,
                                       parallel.person_alpha, atol=1e-5)
        finally:
            self.engine.parallel_inference = original
            self.engine.reset_temporal()


@unittest.skipUnless(RVM_READY and _SAMPLE.exists(), "RVM 权重缺失")
class TestRvmSegmentation(unittest.TestCase):
    """RVM 掩膜的探针式语义复核（与 mediapipe 分割器同一套探针）。"""

    @classmethod
    def setUpClass(cls):
        from core.infer_torch import TorchInferenceEngine
        img = _sample_720p()
        if img is None:
            raise unittest.SkipTest("样例图无法读取")
        cls.h, cls.w = img.shape[:2]
        cls.engine = TorchInferenceEngine(segmenter_model="rvm")
        # RVM 需要几帧收敛循环状态
        for _ in range(3):
            cls.ctx = cls.engine.process(img, faces=False, hands=False,
                                         segmentation=True)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_rvm_spec_and_interval(self):
        self.assertEqual(self.engine.segmenter_model, "rvm")
        self.assertEqual(SEGMENTER_INTERVAL["rvm"], 1)

    def test_person_is_foreground(self):
        alpha = self.ctx.person_alpha
        fg = np.zeros(alpha.shape, bool)
        fg[int(self.h * 0.42):int(self.h * 0.97),
           int(self.w * 0.42):int(self.w * 0.58)] = True
        self.assertGreater(float(alpha[fg].mean()), 0.8,
                           "人像探针区 alpha 应接近 1")
        self.assertLess(float(alpha[:40].mean()), 0.2,
                        "画面上边缘必为背景")
        self.assertGreater(float(alpha.mean()), 0.05)
        self.assertLess(float(alpha.mean()), 0.95)


if __name__ == "__main__":
    unittest.main()
