"""推理引擎单测（需 models/ 下的权重；缺失时自动跳过）。"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from core.context import FrameContext
from core.infer import (
    MODELS_DIR, SEGMENTER_SPECS, TORCH_ONLY_SEGMENTERS, InferenceEngine,
    _landmarks_to_face_info, border_foreground_ratio, category_to_person_mask,
    foreground_from_confidence,
)

# 探针法语义复核用的真实半身像样本（入库的小体积样例）
_SAMPLE = (Path(__file__).resolve().parent.parent
           / "assets" / "samples" / "portrait1.jpg")


def models_present() -> bool:
    return all((MODELS_DIR / f).exists() for f in (
        "face_landmarker.task", "hand_landmarker.task",
        "selfie_segmenter.tflite", "selfie_multiclass_256x256.tflite"))


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
        # 软掩膜与硬掩膜同尺寸、同为前景语义（Phase 3）
        self.assertIsNotNone(ctx.person_alpha)
        self.assertEqual(ctx.person_alpha.shape, (720, 1280))
        self.assertEqual(ctx.person_alpha.dtype, np.float32)
        self.assertGreaterEqual(float(ctx.person_alpha.min()), 0.0)
        self.assertLessEqual(float(ctx.person_alpha.max()), 1.0)

    def test_no_segmentation_leaves_alpha_none(self):
        ctx = self.engine.process(self._frame(), faces=False, hands=False)
        self.assertIsNone(ctx.person_mask)
        self.assertIsNone(ctx.person_alpha)

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


class TestSegmenterSpecs(unittest.TestCase):
    """分割模型约定表的纯函数级校验（不需要权重）。"""

    def test_two_models_have_opposite_conventions(self):
        """两个模型的类别编码与置信图极性必须相反 —— 抄反了会导致静默掩膜反转。"""
        b = SEGMENTER_SPECS["selfie_segmenter_binary"]
        m = SEGMENTER_SPECS["selfie_segmenter"]
        self.assertNotEqual(b["person_is_zero"], m["person_is_zero"])
        self.assertNotEqual(b["alpha_invert"], m["alpha_invert"])

    def test_rvm_spec_exists_and_direct_alpha(self):
        """RVM（torch 后端）直接输出前景 alpha：无类别图概念、不取反。"""
        spec = SEGMENTER_SPECS["rvm"]
        self.assertFalse(spec["person_is_zero"])
        self.assertFalse(spec["alpha_invert"])

    def test_category_to_person_mask(self):
        cat_binary = np.array([[0, 255], [255, 0]], np.uint8)
        mask = category_to_person_mask(cat_binary, person_is_zero=True)
        self.assertEqual(mask.tolist(), [[True, False], [False, True]])
        cat_multi = np.array([[0, 3], [5, 0]], np.uint8)
        mask = category_to_person_mask(cat_multi, person_is_zero=False)
        self.assertEqual(mask.tolist(), [[False, True], [True, False]])

    def test_foreground_from_confidence(self):
        conf = np.array([[0.9, 0.1]], np.float32)
        # 二元：conf[0] 直接就是前景概率
        np.testing.assert_allclose(
            foreground_from_confidence(conf, alpha_invert=False),
            [[0.9, 0.1]], atol=1e-6)
        # 多分类：conf[0] 是背景概率，要取反
        np.testing.assert_allclose(
            foreground_from_confidence(conf, alpha_invert=True),
            [[0.1, 0.9]], atol=1e-6)

    def test_border_foreground_ratio(self):
        alpha = np.zeros((40, 60), np.float32)
        alpha[10:30, 20:40] = 1.0
        self.assertAlmostEqual(border_foreground_ratio(alpha, margin=5), 0.0)
        # 整幅为前景 -> 边框也高（这正是"疑似反相"的判据）
        self.assertAlmostEqual(
            border_foreground_ratio(np.ones((40, 60), np.float32), margin=5), 1.0)

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            InferenceEngine(segmenter_model="不存在")

    def test_torch_only_model_rejected_by_mediapipe_engine(self):
        """mediapipe 后端不能执行 RVM（TorchScript），必须显式报错而非静默。"""
        with self.assertRaises(ValueError):
            InferenceEngine(segmenter_model="rvm")


@unittest.skipUnless(models_present() and _SAMPLE.exists(),
                     "models/ 或样例图缺失")
class TestSegmenterSemantics(unittest.TestCase):
    """用**与掩膜语义无关的探针**复核两个模型的实际约定。

    这是防"掩膜整体反相"的关键回归测试：图像四角必是背景、画面中下部中心
    必是人（head-and-shoulders 构图），与任何类别/置信图约定都无关。
    只做形状与范围断言的话，全幅 255 的反相掩膜也能通过。
    """

    H, W = 720, 1280

    @classmethod
    def setUpClass(cls):
        img = cv2.imread(str(_SAMPLE))
        if img is None:
            raise unittest.SkipTest("样例图无法读取")
        cls.img = cv2.resize(img, (cls.W, cls.H))
        cls.fg_probe = np.zeros((cls.H, cls.W), bool)
        cls.fg_probe[300:700, 560:720] = True      # 画面中下部中心：头/躯干
        cls.border = np.zeros((cls.H, cls.W), bool)
        cls.border[:40] = cls.border[-40:] = True
        cls.border[:, :40] = cls.border[:, -40:] = True

    def _ctx(self, key: str) -> FrameContext:
        engine = InferenceEngine(segmenter_model=key)
        try:
            return engine.process(self.img, faces=False, hands=False,
                                  segmentation=True)
        finally:
            engine.close()

    def test_person_is_foreground_in_both_models(self):
        # RVM 是 torch-only（mediapipe 引擎跑不了），只在 mediapipe 可执行
        # 的模型上探针复核；torch 引擎的 RVM 探针见 tests/test_infer_torch.py
        for key in [k for k in SEGMENTER_SPECS
                    if k not in TORCH_ONLY_SEGMENTERS]:
            with self.subTest(model=key):
                ctx = self._ctx(key)
                alpha = ctx.person_alpha
                self.assertGreater(
                    float(alpha[self.fg_probe].mean()), 0.8,
                    f"{key}: 人像探针区 alpha 应接近 1（否则掩膜整体反相）")
                self.assertLess(
                    float(alpha[:40].mean()), 0.2,
                    f"{key}: 画面上边缘必为背景，alpha 应接近 0")
                # 硬掩膜与软掩膜的语义必须一致
                self.assertGreater(
                    float((ctx.person_mask > 0)[self.fg_probe].mean()), 0.8,
                    f"{key}: person_mask 在人像探针区应为 255")
                self.assertLess(
                    float((ctx.person_mask > 0)[:40].mean()), 0.2,
                    f"{key}: person_mask 在画面上边缘应为 0")
                # 掩膜必须同时含前景与背景（全 0 或全 255 都是坏结果）
                ratio = float(alpha.mean())
                self.assertGreater(ratio, 0.05)
                self.assertLess(ratio, 0.95)


if __name__ == "__main__":
    unittest.main()
