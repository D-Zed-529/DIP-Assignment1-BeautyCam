"""背景替换 / 人像虚化效果单测（不依赖摄像头 / GUI / 模型权重）。

覆盖：matte 对比度拉伸 · 时域 EMA · guided filter 边缘保持 · 背景 cover 缩放 ·
纯色解析 · 掩膜收缩膨胀 · 合成边界情形 · SegmentEffect 状态机与参数校验。
"""

from __future__ import annotations

import os
import tempfile
import unittest

import cv2
import numpy as np

from core.context import FrameContext
from core.effects.segment import (
    MODE_BLUR, MODE_COLOR, MODE_IMAGE, SegmentEffect, blend_over, cover_resize,
    ema_matte, guided_filter, load_image, parse_hex_color, refine_matte,
    sharpen_matte, shift_edges, soften_matte,
)


def ctx_with_alpha(alpha: np.ndarray) -> FrameContext:
    h, w = alpha.shape[:2]
    ctx = FrameContext(width=w, height=h)
    ctx.person_alpha = alpha.astype(np.float32)
    return ctx


def solid_frame(h: int = 60, w: int = 80, value: int = 200) -> np.ndarray:
    return np.full((h, w, 3), value, np.uint8)


def centered_blob(h: int = 60, w: int = 80) -> np.ndarray:
    """中间一块前景、四周背景的合成 alpha（用于效果级测试）。"""
    a = np.zeros((h, w), np.float32)
    a[h // 4:3 * h // 4, w // 4:3 * w // 4] = 1.0
    return a


class TestMatteStretch(unittest.TestCase):
    def test_zero_contrast_is_identity(self):
        a = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
        self.assertTrue(np.array_equal(sharpen_matte(a, 0.0), a))

    def test_does_not_mutate_input(self):
        a = np.full((10, 10), 0.5, np.float32)
        original = a.copy()
        sharpen_matte(a, 0.6)
        self.assertTrue(np.array_equal(a, original))

    def test_widens_separation_between_fg_and_bg(self):
        """核心作用：把"糊"的置信度拉开（多分类模型不做这步背景纯度是 0%）。

        只断言**分离度变大**这个本质性质，不假设窗口具体位置。
        """
        a = np.array([[0.35, 0.5, 0.65]], np.float32)
        before = float(a[0, 2] - a[0, 0])
        out = sharpen_matte(a, 0.6)
        self.assertGreater(float(out[0, 2] - out[0, 0]), before)
        self.assertAlmostEqual(out[0, 1], 0.5, places=5)   # 中点不动

    def test_max_contrast_pushes_toward_binary(self):
        a = np.array([[0.35, 0.65]], np.float32)
        out = sharpen_matte(a, 1.0)
        self.assertLess(float(out[0, 0]), 0.01)
        self.assertGreater(float(out[0, 1]), 0.99)

    def test_monotonic_and_bounded(self):
        a = np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16)
        out = sharpen_matte(a, 0.8)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)
        self.assertTrue(np.all(np.diff(out.ravel()) >= -1e-7))


class TestEmaMatte(unittest.TestCase):
    def test_no_history_returns_new(self):
        a = np.full((8, 8), 0.5, np.float32)
        self.assertIs(ema_matte(a, None, 0.7), a)

    def test_shape_change_returns_new(self):
        new = np.full((8, 8), 0.5, np.float32)
        prev = np.full((4, 4), 0.5, np.float32)
        self.assertIs(ema_matte(new, prev, 0.7), new)

    def test_zero_smooth_returns_new(self):
        new = np.full((8, 8), 0.5, np.float32)
        self.assertIs(ema_matte(new, np.zeros((8, 8), np.float32), 0.0), new)

    def test_static_scene_trusts_history(self):
        """静止（仅 1px 抖动）时强烈信任历史：结果应更靠近旧值而非新值。"""
        prev = np.zeros((180, 320), np.float32)
        prev[40:140, 80:240] = 1.0
        nudged = np.zeros((180, 320), np.float32)
        nudged[40:140, 81:241] = 1.0        # 1px 平移 ≈ 静止
        out = ema_matte(nudged, prev, 0.9)
        self.assertLess(float(np.abs(out - prev).mean()),
                        float(np.abs(out - nudged).mean()))

    def test_motion_reduces_smoothing(self):
        """大运动时更信新帧 —— 否则轮廓会拖影（运动自适应 EMA 的目的）。"""
        prev = np.zeros((180, 320), np.float32)
        prev[40:140, 40:160] = 1.0
        nudged = np.zeros((180, 320), np.float32)
        nudged[40:140, 41:161] = 1.0        # 静止
        moved = np.zeros((180, 320), np.float32)
        moved[40:140, 160:280] = 1.0        # 横向大位移

        out_still = ema_matte(nudged, prev, 0.9)
        out_moved = ema_matte(moved, prev, 0.9)
        # 静止时更贴近旧值；大运动时更贴近新值
        self.assertLess(float(np.abs(out_still - prev).mean()),
                        float(np.abs(out_moved - prev).mean()))
        self.assertLess(float(np.abs(out_moved - moved).mean()),
                        float(np.abs(out_still - moved).mean()))


class TestGuidedFilter(unittest.TestCase):
    def test_constant_guide_returns_src_mean(self):
        guide = np.full((40, 40), 0.5, np.float32)
        src = np.full((40, 40), 0.3, np.float32)
        out = guided_filter(guide, src, 4)
        np.testing.assert_allclose(out, 0.3, atol=1e-3)

    def test_denoises_flat_regions(self):
        rng = np.random.default_rng(0)
        guide = np.full((60, 60), 0.4, np.float32)
        src = (0.5 + 0.1 * rng.standard_normal((60, 60))).astype(np.float32)
        out = guided_filter(guide, src, 4)
        self.assertLess(float(out.std()), float(src.std()))

    def test_snaps_to_guide_edge(self):
        """引导图有竖直边缘时输出边缘应更陡 —— 这是掩膜能吸附到真实边缘的依据。

        用**最大梯度**衡量陡峭度，而不是"过渡带像素数"：锐化会在陡跳两侧留下
        浅裙边，反而增加中间调像素计数，用它当指标会得出相反的错误结论。
        """
        guide = np.zeros((60, 120), np.float32)
        guide[:, 60:] = 1.0
        src = cv2.GaussianBlur(guide, (0, 0), 8.0)   # 糊掉的"掩膜"（约 26px 过渡带）
        out = guided_filter(guide, src, 3)

        def max_grad(a):
            return float(np.abs(np.diff(a[30])).max())
        self.assertGreater(max_grad(out), 2.0 * max_grad(src))
        # 输出仍是合法的 alpha（引导滤波不改变值域）
        self.assertGreaterEqual(float(out.min()), -1e-3)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-3)

    def test_refine_matte_keeps_shape_and_range(self):
        frame = solid_frame(60, 80)
        a = centered_blob(60, 80)
        out = refine_matte(a, frame, 3)
        self.assertEqual(out.shape, (60, 80))
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(float(out.min()), -0.05)
        self.assertLessEqual(float(out.max()), 1.05)


class TestCoverResize(unittest.TestCase):
    def test_exact_output_shape(self):
        img = np.zeros((100, 200, 3), np.uint8)
        for w, h in ((50, 50), (200, 100), (33, 77)):
            self.assertEqual(cover_resize(img, w, h).shape, (h, w, 3))

    @staticmethod
    def _marker_centroid(out: np.ndarray) -> tuple[float, float]:
        """亮块的质心 (x, y)，用于几何校验（单像素标记会被缩放核平均掉）。"""
        ys, xs = np.nonzero(out[..., 0] > 100)
        return float(xs.mean()), float(ys.mean())

    def test_wide_source_crops_sides_without_distortion(self):
        """2:1 的图放进 1:1：应裁剪两侧而非拉伸（标记块保持在原归一化位置）。"""
        img = np.zeros((100, 200, 3), np.uint8)
        img[45:55, 45:55] = 255            # 源图归一化位置 x≈0.25, y≈0.5
        out = cover_resize(img, 50, 50)
        cx, cy = self._marker_centroid(out)
        # 缩放 0.5 后裁剪 x∈[25,75]：源 x=50 映射到输出 x=0
        self.assertLess(cx, 6.0)
        self.assertAlmostEqual(cy, 25.0, delta=3.0)   # 纵向不发生位移

    def test_center_is_preserved(self):
        img = np.zeros((100, 200, 3), np.uint8)
        img[45:55, 95:105] = 255           # 源图正中
        out = cover_resize(img, 50, 50)
        cx, cy = self._marker_centroid(out)
        self.assertAlmostEqual(cx, 25.0, delta=3.0)
        self.assertAlmostEqual(cy, 25.0, delta=3.0)

    def test_degenerate_input_returns_blank(self):
        out = cover_resize(np.zeros((0, 0, 3), np.uint8), 8, 6)
        self.assertEqual(out.shape, (6, 8, 3))


class TestParseHexColor(unittest.TestCase):
    def test_parses_to_bgr(self):
        # #FF0000 = 红 -> BGR (0, 0, 255)
        self.assertEqual(parse_hex_color("#FF0000"), (0, 0, 255))
        self.assertEqual(parse_hex_color("00FF00"), (0, 255, 0))

    def test_invalid_falls_back(self):
        for bad in ("", "#12345", "zzzzzz", "#GGGGGG", None):
            self.assertEqual(parse_hex_color(bad), (113, 110, 60))


class TestShiftEdges(unittest.TestCase):
    def setUp(self):
        self.a = np.zeros((40, 40), np.float32)
        self.a[10:30, 10:30] = 1.0

    def test_zero_is_identity(self):
        self.assertIs(shift_edges(self.a, 0), self.a)

    def test_positive_shrinks(self):
        self.assertLess(shift_edges(self.a, 2).sum(), self.a.sum())

    def test_negative_expands(self):
        self.assertGreater(shift_edges(self.a, -2).sum(), self.a.sum())


class TestBlendOver(unittest.TestCase):
    def test_alpha_one_is_frame(self):
        frame = solid_frame(value=200)
        bg = solid_frame(value=0)
        ones = np.ones(frame.shape[:2], np.float32)
        self.assertTrue(np.array_equal(blend_over(frame, bg, ones), frame))

    def test_alpha_zero_is_background(self):
        """a≡0 时整帧应变成背景（人像占比兜底之外，合成本身必须是精确的）。"""
        frame = solid_frame(value=200)
        bg = solid_frame(value=0)
        self.assertTrue(
            np.array_equal(blend_over(frame, bg, np.zeros(frame.shape[:2], np.float32)), bg))

    def test_half_alpha_is_midpoint(self):
        frame = solid_frame(value=200)
        bg = solid_frame(value=100)
        out = blend_over(frame, bg, np.full(frame.shape[:2], 0.5, np.float32))
        np.testing.assert_allclose(out, 150, atol=1)

    def test_accepts_uint8_alpha(self):
        frame = solid_frame(value=200)
        bg = solid_frame(value=0)
        out = blend_over(frame, bg, np.full(frame.shape[:2], 255, np.uint8))
        self.assertTrue(np.array_equal(out, frame))

    def test_shape_and_dtype_preserved(self):
        frame = solid_frame(60, 80)
        out = blend_over(frame, solid_frame(60, 80), centered_blob(60, 80))
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, np.uint8)


class TestSoftening(unittest.TestCase):
    def test_zero_radius_is_identity(self):
        a = centered_blob()
        self.assertIs(soften_matte(a, 0), a)

    def test_soften_reduces_transition_sharpness(self):
        a = centered_blob(60, 80)
        out = soften_matte(a, 2)
        self.assertGreater(float(out[15, 20]), 0.0)   # 边界外被晕开
        self.assertEqual(out.shape, a.shape)


class TestSegmentEffect(unittest.TestCase):
    def setUp(self):
        self.frame = solid_frame(60, 80)
        self.eff = SegmentEffect(enabled=True)

    def tearDown(self):
        self.eff.reset_temporal()

    # ------- 兜底路径 -------

    def test_passthrough_without_any_alpha(self):
        """从未拿到过掩膜（未启用分割 / 模型缺失）时原样返回。"""
        ctx = FrameContext(width=80, height=60)
        out = self.eff.process(self.frame.copy(), ctx)
        self.assertTrue(np.array_equal(out, self.frame))

    def test_falls_back_to_hard_mask(self):
        """只有 person_mask 时也要能工作（回退口径）。"""
        ctx = FrameContext(width=80, height=60)
        ctx.person_mask = (centered_blob(60, 80) * 255).astype(np.uint8)
        self.eff.set_params(mode=MODE_COLOR, bg_color="#FF0000")
        out = self.eff.process(self.frame.copy(), ctx)
        self.assertFalse(np.array_equal(out, self.frame))

    def test_min_person_ratio_guard(self):
        """虚化模式下掩膜全 0 时原样输出。"""
        ctx = ctx_with_alpha(np.zeros((60, 80), np.float32))
        out = self.eff.process(self.frame.copy(), ctx)
        self.assertTrue(np.array_equal(out, self.frame))

    def test_color_mode_replaces_empty_mask(self):
        """无人像时仍应显示纯色背景，不能被虚化模式的保护分支拦截。"""
        self.eff.set_params(mode=MODE_COLOR, bg_color="#00B140")
        ctx = ctx_with_alpha(np.zeros((60, 80), np.float32))
        out = self.eff.process(self.frame.copy(), ctx)
        self.assertTrue(np.all(out == np.array([64, 177, 0], np.uint8)))

    def test_image_mode_replaces_empty_mask(self):
        """无人像时也应显示选定的背景图片。"""
        self.eff.set_params(mode=MODE_IMAGE, bg_path="fixture")
        self.eff._load_image = lambda path, w, h: np.full((h, w, 3), 17, np.uint8)
        ctx = ctx_with_alpha(np.zeros((60, 80), np.float32))
        out = self.eff.process(self.frame.copy(), ctx)
        self.assertTrue(np.all(out == 17))

    def test_disabled_effect_is_passthrough(self):
        self.eff.set_enabled(False)
        ctx = ctx_with_alpha(centered_blob(60, 80))
        self.assertTrue(np.array_equal(self.eff.process(self.frame.copy(), ctx),
                                       self.frame))

    # ------- 三种模式 -------

    def test_color_mode_replaces_background(self):
        self.eff.set_params(mode=MODE_COLOR, bg_color="#FF0000")
        out = self.eff.process(self.frame.copy(), ctx_with_alpha(centered_blob(60, 80)))
        self.assertEqual(int(out[0, 0, 2]), 255)      # 角落 -> 红
        self.assertEqual(int(out[0, 0, 0]), 0)
        self.assertEqual(int(out[30, 40, 0]), 200)    # 中心 -> 原帧灰度

    def test_blur_mode_zero_strength_keeps_frame(self):
        """虚化强度 0 时背景就是原帧，合成结果应等于原帧。"""
        self.eff.set_params(mode=MODE_BLUR, strength=0.0)
        out = self.eff.process(self.frame.copy(), ctx_with_alpha(centered_blob(60, 80)))
        self.assertTrue(np.array_equal(out, self.frame))

    def test_blur_mode_changes_frame(self):
        self.eff.set_params(mode=MODE_BLUR, strength=1.0)
        ctx = ctx_with_alpha(centered_blob(60, 80))
        self.eff.process(self.frame.copy(), ctx)
        # 纯色帧模糊后不变，改用带纹理的帧验证确实做了处理
        noisy = np.random.default_rng(0).integers(
            0, 255, (60, 80, 3), dtype=np.uint8)
        out = self.eff.process(noisy.copy(), ctx)
        self.assertLess(float(out.std()), float(noisy.std()))

    def test_image_mode_loads_and_covers(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bg.png")
            cv2.imwrite(path, np.full((200, 100, 3), (0, 0, 255), np.uint8))
            self.eff.set_params(mode=MODE_IMAGE, bg_path=path)
            out = self.eff.process(self.frame.copy(),
                                   ctx_with_alpha(centered_blob(60, 80)))
            self.assertEqual(int(out[0, 0, 2]), 255)   # 背景图是纯红

    def test_image_mode_falls_back_to_color_when_missing(self):
        """背景图缺失时退回纯色，而不是整帧变黑。"""
        self.eff.set_params(mode=MODE_IMAGE, bg_path="/不存在的路径.jpg",
                            bg_color="#00FF00")
        out = self.eff.process(self.frame.copy(),
                               ctx_with_alpha(centered_blob(60, 80)))
        self.assertEqual(int(out[0, 0, 1]), 255)       # 绿
        self.assertGreater(int(out[0, 0].sum()), 0)    # 不是黑

    def test_image_mode_reads_non_ascii_path(self):
        """非 ASCII 路径必须能读 —— 内置背景图是中文名。

        cv2.imread 在 Windows 上对中文路径静默返回 None，会让整个内置图库
        失效（表现为"选了图没反应"），且不报错。用 np.fromfile+imdecode 才稳。
        """
        with tempfile.TemporaryDirectory() as d:
            # 用 PNG（无损）避免 JPEG 往返把 255 变成 254 干扰断言
            path = os.path.join(d, "中文背景名.png")
            ok, buf = cv2.imencode(".png", np.full((40, 60, 3), (0, 0, 255),
                                                  np.uint8))
            self.assertTrue(ok)
            with open(path, "wb") as f:
                f.write(buf.tobytes())
            loaded = load_image(path)
            self.assertIsNotNone(loaded, "中文路径读取失败")
            self.eff.set_params(mode=MODE_IMAGE, bg_path=path)
            out = self.eff.process(self.frame.copy(),
                                   ctx_with_alpha(centered_blob(60, 80)))
            self.assertEqual(int(out[0, 0, 2]), 255)   # 背景图是纯红

    def test_load_image_tolerates_bad_input(self):
        self.assertIsNone(load_image(""))
        self.assertIsNone(load_image("/不存在/的/路径.jpg"))
        with tempfile.TemporaryDirectory() as d:
            junk = os.path.join(d, "垃圾文件.jpg")
            with open(junk, "wb") as f:
                f.write(b"not an image")
            self.assertIsNone(load_image(junk))       # 不抛异常

    def test_background_cache_keyed_by_size(self):
        """尺寸变化必须重算背景（否则 cv2 会因尺寸不符报错）。"""
        self.eff.set_params(mode=MODE_COLOR, bg_color="#FF0000")
        self.eff.process(self.frame.copy(), ctx_with_alpha(centered_blob(60, 80)))
        small = solid_frame(30, 40)
        out = self.eff.process(small.copy(), ctx_with_alpha(centered_blob(30, 40)))
        self.assertEqual(out.shape, (30, 40, 3))

    def test_alpha_resized_when_frame_size_changes(self):
        """采集源尺寸变化（相机 -> 视频文件）时 alpha 应被重采样而非报错。"""
        ctx = ctx_with_alpha(centered_blob(60, 80))
        self.eff.set_params(mode=MODE_COLOR, bg_color="#FF0000")
        self.eff.process(self.frame.copy(), ctx)
        out = self.eff.process(solid_frame(30, 40).copy(), ctx)
        self.assertEqual(out.shape, (30, 40, 3))

    # ------- 状态机 / 参数 -------

    def test_reuses_alpha_on_frames_without_new_data(self):
        """隔帧推理的中间帧（ctx 无 alpha）应复用上一帧结果。"""
        self.eff.set_params(mode=MODE_COLOR, bg_color="#FF0000")
        self.eff.process(self.frame.copy(), ctx_with_alpha(centered_blob(60, 80)))
        out = self.eff.process(self.frame.copy(), FrameContext(width=80, height=60))
        self.assertFalse(np.array_equal(out, self.frame))

    def test_reenable_resets_stale_state(self):
        ctx = ctx_with_alpha(centered_blob(60, 80))
        self.eff.process(self.frame.copy(), ctx)
        self.assertIsNotNone(self.eff._prev_alpha)
        self.eff.set_enabled(False)
        self.eff.set_enabled(True)
        self.assertIsNone(self.eff._prev_alpha)
        self.assertIsNone(self.eff._bg_key)

    def test_inference_interval_follows_param(self):
        self.eff.set_params(infer_interval=4)
        self.assertEqual(self.eff.inference_interval("segmentation"), 4)
        self.eff.set_params(infer_interval=1)
        self.assertEqual(self.eff.inference_interval("segmentation"), 1)
        self.assertEqual(self.eff.inference_interval("faces"), 1)

    def test_needs_segmentation(self):
        self.assertIn("segmentation", self.eff.needs)

    def test_unknown_param_rejected(self):
        with self.assertRaises(ValueError):
            self.eff.set_params(不存在的参数=1)

    def test_unknown_mode_rejected(self):
        """非法模式若被静默接受会悄悄按纯色处理，必须在写入时就报错。"""
        for bad in ("blurr", "", None, "IMAGE"):
            with self.assertRaises(ValueError):
                self.eff.set_params(mode=bad)
        with self.assertRaises(ValueError):
            SegmentEffect(enabled=True, params={"mode": "nope"})

    def test_unknown_param_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            SegmentEffect(enabled=True, params={"nope": 1})

    def test_defaults_are_within_documented_ranges(self):
        p = SegmentEffect.default_params()
        self.assertIn(p["mode"], (MODE_BLUR, MODE_IMAGE, MODE_COLOR))
        self.assertGreaterEqual(p["strength"], 0.0)
        self.assertLessEqual(p["strength"], 1.0)
        self.assertGreaterEqual(p["matte_contrast"], 0.0)
        self.assertLessEqual(p["matte_contrast"], 1.0)


if __name__ == "__main__":
    unittest.main()
