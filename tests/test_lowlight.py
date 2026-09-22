"""低光增强单测：启发式基线 + SCI 深度档（缺权重/缺 onnxruntime 自动跳过端到端）。"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from core.context import FrameContext
from core.effects.lowlight import LowLightDnnEffect, LowLightEffect
from core.infer import (
    LOWLIGHT_INPUT_SIZE, LOWLIGHT_LEVELS, LOWLIGHT_DEFAULT_LEVEL,
    MODELS_DIR, postprocess_sci, preprocess_sci,
)


def dark_frame(w=320, h=240, seed=11) -> np.ndarray:
    """合成暗帧（噪声 + 低亮度），保证 <60 触发增强。"""
    rng = np.random.default_rng(seed)
    return (rng.integers(8, 40, (h, w, 3))).astype(np.uint8)


def bright_frame(w=320, h=240) -> np.ndarray:
    return np.full((h, w, 3), 200, np.uint8)


class TestHeuristic(unittest.TestCase):
    def test_auto_skips_bright(self):
        """auto 模式下亮帧必须原样返回（对象同一）。"""
        eff = LowLightEffect(params={"auto": True})
        f = bright_frame()
        out = eff.process(f, FrameContext())
        self.assertIs(out, f)

    def test_dark_enhanced(self):
        eff = LowLightEffect(params={"auto": True})
        f = dark_frame()
        out = eff.process(f, FrameContext())
        self.assertGreater(float(out.mean()), float(f.mean()))
        self.assertTrue(eff.is_dark)

    def test_zero_strength_passthrough(self):
        eff = LowLightEffect(params={"auto": False, "strength": 0.0})
        f = dark_frame()
        self.assertIs(eff.process(f, FrameContext()), f)


def dnn_ready() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return (MODELS_DIR / LOWLIGHT_LEVELS[LOWLIGHT_DEFAULT_LEVEL]).exists()


@unittest.skipUnless(dnn_ready(), "缺 onnxruntime 或 models/ SCI 权重")
class TestSciPreprocess(unittest.TestCase):
    def test_preprocess_shape_and_range(self):
        f = dark_frame()
        x = preprocess_sci(f)
        self.assertEqual(x.shape, (1, 3, LOWLIGHT_INPUT_SIZE, LOWLIGHT_INPUT_SIZE))
        self.assertEqual(x.dtype, np.float32)
        self.assertGreaterEqual(float(x.min()), 0.0)
        self.assertLessEqual(float(x.max()), 1.0)

    def test_postprocess_range(self):
        out = postprocess_sci(np.random.rand(1, 3, 8, 8).astype(np.float32) * 3 - 1)
        self.assertEqual(out.shape, (8, 8, 3))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)


@unittest.skipUnless(dnn_ready(), "缺 onnxruntime 或 models/ SCI 权重")
class TestSciEffect(unittest.TestCase):
    def test_dark_enhanced_and_bright_passthrough(self):
        eff = LowLightDnnEffect(params={"auto": True})
        f = dark_frame()
        out = eff.process(f.copy(), FrameContext())
        self.assertGreater(float(out.mean()), float(f.mean()))
        self.assertTrue(eff.is_dark)
        bright = bright_frame()
        self.assertIs(eff.process(bright, FrameContext()), bright)
        self.assertIn(eff.provider, ("CoreMLExecutionProvider",
                                     "CPUExecutionProvider"))

    def test_interval_reuse_same_shape(self):
        """隔帧推理：中间帧复用上一帧结果，输出形状保持原帧尺寸。"""
        eff = LowLightDnnEffect(params={"auto": False, "infer_interval": 2})
        f = dark_frame()
        outs = [eff.process(f.copy(), FrameContext()) for _ in range(3)]
        for o in outs:
            self.assertEqual(o.shape, f.shape)

    def test_reset_temporal_breaks_reuse(self):
        eff = LowLightDnnEffect(params={"auto": False, "infer_interval": 4})
        eff.process(dark_frame().copy(), FrameContext())
        eff.reset_temporal()
        self.assertIsNone(eff._last_enhanced)

    def test_unknown_level_rejected(self):
        with self.assertRaises(ValueError):
            LowLightDnnEffect(params={"level": "extreme"})


if __name__ == "__main__":
    unittest.main()
