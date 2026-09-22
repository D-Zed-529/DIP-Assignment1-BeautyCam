"""推理引擎单测（需 models/ 下的权重；缺失时自动跳过）。"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from core.infer import MODELS_DIR, InferenceEngine, _landmarks_to_face_info


def models_present() -> bool:
    return all((MODELS_DIR / f).exists() for f in (
        "face_landmarker.task", "hand_landmarker.task",
        "selfie_multiclass_256x256.tflite"))


@unittest.skipUnless(models_present(), "models/ 权重缺失，先跑 scripts/download_models.py")
class TestInferenceEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = InferenceEngine()

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def _frame(self):
        img = np.full((720, 1280, 3), 128, np.uint8)
        cv2.ellipse(img, (640, 360), (160, 220), 0, 0, 360,
                    (200, 180, 160), -1)
        return img

    def test_process_returns_valid_ctx(self):
        ctx = self.engine.process(self._frame(), faces=True, hands=True)
        self.assertEqual((ctx.width, ctx.height), (1280, 720))
        self.assertIsInstance(ctx.faces, list)
        self.assertIsInstance(ctx.hands, list)
        # VIDEO 模式时间戳单调递增
        ctx2 = self.engine.process(self._frame(), faces=False, hands=False)
        self.assertGreater(ctx2.timestamp_ms, ctx.timestamp_ms)

    def test_segmentation_mask(self):
        ctx = self.engine.process(self._frame(), faces=False, hands=False,
                                  segmentation=True)
        self.assertIsNotNone(ctx.person_mask)
        self.assertEqual(ctx.person_mask.shape, (720, 1280))
        self.assertIn(ctx.person_mask.dtype, (np.uint8,))
        self.assertTrue(set(np.unique(ctx.person_mask)) <= {0, 255})

    def test_face_info_box_from_landmarks(self):
        lm = np.zeros((468, 3), np.float32)
        lm[:, 0] = 0.4
        lm[:, 1] = 0.2
        lm[152] = (0.4, 0.8, 0)   # 下巴最低点（152 是轮廓点，统一 x=0.4）
        info = _landmarks_to_face_info(lm)
        x1, y1, x2, y2 = info.box
        self.assertAlmostEqual(x1, 0.4, places=5)
        self.assertAlmostEqual(x2, 0.4, places=5)
        # 下扩 8%：0.8 + 0.08*0.6 ≈ 0.848
        self.assertAlmostEqual(y2, 0.848, delta=0.005)


if __name__ == "__main__":
    unittest.main()
