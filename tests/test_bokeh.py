"""深度渐进虚化（BokehEffect / core/effects/bokeh.py）单测 —— 纯函数，不依赖模型。"""

from __future__ import annotations

import unittest

import numpy as np

from core.effects.bokeh import (
    BokehEffect, blur_sigma_map, depth_blur, focus_depth, smoothstep,
)
from core.context import FrameContext
from core.pipeline import NEED_DEPTH, NEED_SEGMENTATION


def _depth_scene(h=240, w=320):
    """合成深度：中下部一个"人"（近，值 0.9），四周渐远（0.2）。"""
    depth = np.full((h, w), 0.2, np.float32)
    depth[int(h * 0.35):, int(w * 0.35):int(w * 0.65)] = 0.9
    return depth


def _alpha_scene(h=240, w=320):
    a = np.zeros((h, w), np.float32)
    a[int(h * 0.35):, int(w * 0.35):int(w * 0.65)] = 1.0
    return a


class TestPureFunctions(unittest.TestCase):
    def test_smoothstep(self):
        self.assertEqual(smoothstep(np.float32(0.0)), 0.0)
        self.assertEqual(smoothstep(np.float32(1.0)), 1.0)
        self.assertAlmostEqual(float(smoothstep(np.float32(0.5))), 0.5,
                               places=6)
        # 单调
        xs = np.linspace(0, 1, 11, dtype=np.float32)
        ys = smoothstep(xs)
        self.assertTrue(bool(np.all(np.diff(ys) >= 0)))

    def test_focus_depth_prefers_person_region(self):
        d = _depth_scene()
        alpha = _alpha_scene()
        # 有人像掩膜：焦平面取人像区中位（0.9）
        self.assertGreater(focus_depth(d, alpha), 0.8)
        # 无掩膜：85 分位（最近的主体同样 0.9）
        self.assertGreater(focus_depth(d, None), 0.8)
        # 掩膜占比过低（空掩膜）：退回分位口径
        self.assertGreater(focus_depth(d, np.zeros_like(alpha)), 0.8)

    def test_blur_sigma_map(self):
        d = _depth_scene()
        smap = blur_sigma_map(d, focus=0.9, sigma_max=18.0, rng=0.35)
        # 焦平面上几乎不糊
        self.assertLess(float(smap[int(200), int(160)]), 1.0)
        # 远处满糊
        self.assertGreater(float(smap[10, 10]), 15.0)
        self.assertLessEqual(float(smap.max()), 18.0)

    def test_depth_blur_identity_at_zero_strength(self):
        frame = (np.random.default_rng(0).random((120, 160, 3))
                 * 255).astype(np.uint8)
        out = depth_blur(frame, _depth_scene(120, 160), None, strength=0.0)
        np.testing.assert_array_equal(out, frame)

    def test_depth_blur_blurs_far_keeps_person(self):
        frame = np.full((240, 320, 3), 200, np.uint8)
        # 远处撒高频纹理，人区保持平滑 —— 虚化后远处方差应显著低于原值
        rng = np.random.default_rng(1)
        noisy = frame.copy()
        noisy[:60] = rng.integers(0, 255, size=(60, 320, 3), dtype=np.uint8)
        depth = _depth_scene()
        alpha = _alpha_scene()
        out = depth_blur(noisy, depth, alpha, strength=1.0)
        far_var = float(out[:40].std())
        person_mean = float(out[int(240 * 0.5):int(240 * 0.9),
                                int(320 * 0.4):int(320 * 0.6)].std())
        orig_far = float(noisy[:40].std())
        self.assertLess(far_var, orig_far * 0.6,
                        "远处高频应被显著抹平")
        self.assertLess(person_mean, far_var + 40.0)   # 人区不比远处更糊


class TestBokehEffect(unittest.TestCase):
    def test_needs_include_depth_and_optional_segmentation(self):
        eff = BokehEffect()
        self.assertIn(NEED_DEPTH, eff.needs)
        self.assertIn(NEED_SEGMENTATION, eff.needs)   # 默认 use_matte=True
        eff.set_params(use_matte=False)
        self.assertIn(NEED_DEPTH, eff.needs)
        self.assertNotIn(NEED_SEGMENTATION, eff.needs)

    def test_passthrough_without_depth(self):
        eff = BokehEffect()
        frame = np.full((60, 80, 3), 128, np.uint8)
        ctx = FrameContext(width=80, height=60)   # depth=None
        out = eff.process(frame, ctx)
        np.testing.assert_array_equal(out, frame)

    def test_process_with_depth(self):
        eff = BokehEffect(params={"use_matte": False})
        frame = np.full((120, 160, 3), 120, np.uint8)
        ctx = FrameContext(width=160, height=120,
                           depth=_depth_scene(120, 160))
        out = eff.process(frame, ctx)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
