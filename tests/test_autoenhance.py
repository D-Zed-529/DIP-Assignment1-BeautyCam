"""自适应画质优化单测：纯函数 + 合成图效果（不依赖模型权重与摄像头）。

合成 landmarks 说明：face_oval_mask 只取 landmarks[FACE_OVAL_IDS] 的
前两列（归一化 x, y）画轮廓多边形，其余点位不参与 —— 测试里用参数化
椭圆填充这 36 个索引即可，无需真实 FaceMesh 输出。
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from core.context import FaceInfo, FrameContext
from core.effects.autoenhance import (
    BG_TARGET_L, FACE_TARGET_L, AutoEnhanceEffect, apply_partition_gamma,
    apply_wb, clahe_apply, detail_preserving_gamma, ema_tuple,
    face_target_backlight_damp, face_union_mask, gamma_for_exposure,
    gamma_lut, gray_world_gains, region_mean_l, scale_saturation_ab,
)
from core.infer import FACE_OVAL_IDS

W, H = 320, 240


def ellipse_landmarks(cx=0.25, cy=0.5, rx=0.15, ry=0.3,
                      seed=7) -> np.ndarray:
    """合成 468 点 landmarks：FACE_OVAL_IDS 的 36 个位序排成椭圆，其余零。

    椭圆放在左半中心（backlit_frame 的暗区），分区统计才互不污染。
    """
    lm = np.zeros((468, 3), dtype=np.float32)
    t = np.linspace(0, 2 * np.pi, len(FACE_OVAL_IDS), endpoint=False)
    lm[FACE_OVAL_IDS, 0] = cx + rx * np.cos(t)
    lm[FACE_OVAL_IDS, 1] = cy + ry * np.sin(t)
    lm[FACE_OVAL_IDS, 2] = 0.0
    return lm


def ctx_with_face(seed=7) -> FrameContext:
    return FrameContext(
        width=W, height=H,
        faces=[FaceInfo(landmarks=ellipse_landmarks(seed=seed),
                        box=(0.1, 0.2, 0.4, 0.8))])


class TestGamma(unittest.TestCase):
    def test_dark_lifts(self):
        """均值低于目标 → γ<1（提亮），且幂变换确实把均值推向目标。"""
        g = gamma_for_exposure(80.0, FACE_TARGET_L)
        self.assertLess(g, 1.0)
        lifted = 255.0 * (80.0 / 255.0) ** g
        self.assertAlmostEqual(lifted, FACE_TARGET_L, delta=1.0)

    def test_bright_presses(self):
        g = gamma_for_exposure(200.0, BG_TARGET_L)
        self.assertGreater(g, 1.0)

    def test_tolerance_band_is_identity(self):
        self.assertEqual(gamma_for_exposure(FACE_TARGET_L + 5.0,
                                            FACE_TARGET_L), 1.0)
        self.assertEqual(gamma_for_exposure(BG_TARGET_L - 10.0,
                                            BG_TARGET_L), 1.0)

    def test_none_and_extremes_no_nan(self):
        """区域空 / 全黑 / 全白：不 NaN、不越界（AGENTS #17 的 Drago 教训）。"""
        self.assertEqual(gamma_for_exposure(None, 150.0), 1.0)
        for mean in (0.0, 255.0):
            g = gamma_for_exposure(mean, 150.0)
            self.assertTrue(np.isfinite(g))
            self.assertGreaterEqual(g, 0.4)
            self.assertLessEqual(g, 2.5)

    def test_lut_identity_and_monotonic(self):
        lut = gamma_lut(1.0)
        np.testing.assert_array_equal(lut, np.arange(256))
        for g in (0.5, 1.7):
            lut = gamma_lut(g)
            self.assertTrue(np.all(np.diff(lut) >= 0), "LUT 必须单调")
            self.assertEqual(lut[0], 0.0)
            self.assertEqual(lut[-1], 255.0)


class TestRegionStats(unittest.TestCase):
    def test_masked_mean(self):
        l = np.zeros((H, W), np.uint8)
        l[:, :W // 2] = 60
        l[:, W // 2:] = 180
        mask = np.zeros((H, W), np.uint8)
        mask[:, :W // 2] = 255
        self.assertAlmostEqual(region_mean_l(l, mask), 60.0, delta=0.5)
        self.assertAlmostEqual(region_mean_l(l, None), 120.0, delta=0.5)

    def test_empty_region_returns_none(self):
        l = np.full((H, W), 90, np.uint8)
        mask = np.zeros((H, W), np.uint8)
        self.assertIsNone(region_mean_l(l, mask))


class TestPartitionGamma(unittest.TestCase):
    def test_partition_applies_different_luts(self):
        """同亮度两区 + 左半掩膜：掩膜内按人脸 LUT（目标高 → 更亮）。"""
        l = np.full((100, 100), 60, np.uint8)
        mask = np.zeros((100, 100), np.float32)
        mask[:, :50] = 1.0
        lut_f = gamma_lut(gamma_for_exposure(60.0, FACE_TARGET_L))
        lut_b = gamma_lut(gamma_for_exposure(60.0, BG_TARGET_L))
        out = apply_partition_gamma(l, lut_f, lut_b, mask)
        # 期望值与实现同解析式（γ 可能被 GAMMA_MIN/MAX 裁剪，按返回值算）
        exp_f = 255.0 * (60.0 / 255.0) ** gamma_for_exposure(60.0, FACE_TARGET_L)
        exp_b = 255.0 * (60.0 / 255.0) ** gamma_for_exposure(60.0, BG_TARGET_L)
        self.assertGreater(exp_f, exp_b)   # 人脸目标更高 → 掩膜内更亮
        self.assertAlmostEqual(float(out[:, :50].mean()), exp_f, delta=2.0)
        self.assertAlmostEqual(float(out[:, 50:].mean()), exp_b, delta=2.0)

    def test_no_mask_uses_bg_lut(self):
        """过亮值 200 → 目标 115 的 γ 超上限被裁到 GAMMA_MAX，按裁剪后映射。"""
        l = np.full((10, 10), 200, np.uint8)
        lut_b = gamma_lut(gamma_for_exposure(200.0, BG_TARGET_L))
        out = apply_partition_gamma(l, gamma_lut(1.0), lut_b, None)
        expect = 255.0 * (200.0 / 255.0) ** 2.5   # GAMMA_MAX 裁剪值
        self.assertAlmostEqual(float(out.mean()), expect, delta=1.5)

    def test_identity_luts_pass_through(self):
        l = np.arange(256, dtype=np.uint8).reshape(1, 256)
        out = apply_partition_gamma(l, gamma_lut(1.0), gamma_lut(1.0), None)
        np.testing.assert_array_equal(out, l)


class TestWhiteBalance(unittest.TestCase):
    def test_color_cast_direction(self):
        """偏红帧 → R 增益 < 1、B/G 增益 > 1（增益是 BGR 顺序）。"""
        frame = np.zeros((50, 50, 3), np.uint8)
        frame[..., 0] = 80    # B
        frame[..., 1] = 80    # G
        frame[..., 2] = 180   # R 偏高
        g_b, g_g, g_r = gray_world_gains(frame)
        self.assertLess(g_r, 1.0)
        self.assertGreater(g_b, 1.0)
        self.assertGreater(g_g, 1.0)

    def test_gray_frame_identity(self):
        frame = np.full((50, 50, 3), 110, np.uint8)
        self.assertEqual(gray_world_gains(frame), (1.0, 1.0, 1.0))

    def test_black_frame_safe(self):
        frame = np.zeros((50, 50, 3), np.uint8)
        self.assertEqual(gray_world_gains(frame), (1.0, 1.0, 1.0))

    def test_masked_estimation_ignores_face_region(self):
        """掩膜圈定纯灰区估计（盖住偏色区）→ 增益应接近 1。

        mask 语义是"非零处计入"：这里下半置 255 = 只统计下半纯灰区。
        """
        frame = np.full((50, 50, 3), 110, np.uint8)
        frame[:25, :, 2] = 200   # 上半偏红（当作"脸"）
        mask = np.zeros((50, 50), np.uint8)
        mask[25:, :] = 255       # 只统计下半（灰）
        gains = gray_world_gains(frame, mask)
        for g in gains:
            self.assertAlmostEqual(g, 1.0, delta=0.02)

    def test_apply_wb_stays_uint8(self):
        frame = np.full((10, 10, 3), 250, np.uint8)
        out = apply_wb(frame, (1.2, 1.25, 0.8))
        self.assertEqual(out.dtype, np.uint8)
        self.assertLessEqual(int(out.max()), 255)
        self.assertEqual(int(out[..., 2].max()), 200)   # 250×0.8


class TestClaheAndSaturation(unittest.TestCase):
    def test_clahe_zero_is_identity(self):
        l = np.arange(256, dtype=np.uint8).reshape(1, 256).repeat(8, 0)
        self.assertIs(clahe_apply(l, 0.0), l)

    def test_clahe_increases_local_contrast(self):
        """低对比图（窄直方图）经 CLAHE 后灰度范围显著拉宽。"""
        rng = np.random.default_rng(3)
        l = rng.integers(110, 130, (H, W)).astype(np.uint8)
        out = clahe_apply(l, 0.6)
        self.assertGreater(int(out.max()) - int(out.min()),
                           int(l.max()) - int(l.min()))

    def test_saturation_scaling(self):
        """a/b 缩放后彩度（a/b 离 128 的偏差）增大，=1 恒等。"""
        rng = np.random.default_rng(5)
        a = rng.integers(98, 158, (20, 20)).astype(np.uint8)
        b = rng.integers(98, 158, (20, 20)).astype(np.uint8)
        a2, b2 = scale_saturation_ab(a, b, 1.3)
        self.assertGreater(float(np.abs(a2.astype(int) - 128).mean()
                                 + np.abs(b2.astype(int) - 128).mean()),
                           float(np.abs(a.astype(int) - 128).mean()
                                 + np.abs(b.astype(int) - 128).mean()))
        a1, b1 = scale_saturation_ab(a, b, 1.0)
        self.assertIs(a1, a)
        self.assertIs(b1, b)
        # 边界不越界
        self.assertEqual(a2.dtype, np.uint8)


class TestEmaAndReset(unittest.TestCase):
    def test_ema_converges_and_none_adopts(self):
        self.assertEqual(ema_tuple(None, (1.5,), 0.8), (1.5,))
        cur = (1.0,)
        for _ in range(50):
            cur = ema_tuple(cur, (2.0,), 0.8)
        self.assertAlmostEqual(cur[0], 2.0, delta=0.01)

    def test_reset_clears_stats(self):
        eff = AutoEnhanceEffect()
        eff.process(backlit_frame(), ctx_with_face())
        self.assertIsNotNone(eff._ema_bg)
        eff.reset_temporal()
        self.assertIsNone(eff._ema_bg)
        self.assertIsNone(eff._ema_wb)


def backlit_frame(seed=1) -> np.ndarray:
    """合成逆光帧：左半（人脸区）暗 55、右半（背景）亮 210。"""
    f = np.zeros((H, W, 3), np.uint8)
    rng = np.random.default_rng(seed)
    f[:, :W // 2] = (rng.integers(48, 64, (H, W // 2, 3))).astype(np.uint8)
    f[:, W // 2:] = (rng.integers(200, 220, (H, W // 2, 3))).astype(np.uint8)
    return f


class TestEffectEndToEnd(unittest.TestCase):
    def test_zero_strength_passthrough(self):
        eff = AutoEnhanceEffect(params={"strength": 0.0})
        f = backlit_frame()
        self.assertIs(eff.process(f, FrameContext(width=W, height=H)), f)

    def test_backlit_face_lifted_background_pressed(self):
        """逆光帧 + 人脸掩膜在左半：脸核心 L 大幅上升、亮背景 L 向目标回落。

        评估在 LAB 的 L 通道（算法作用域）且按掩膜分区：椭圆只盖左半
        的一部分，拿"整个左半"当脸区会被背景 LUT 拖低均值。
        """
        eff = AutoEnhanceEffect(params={"strength": 1.0, "contrast": 0.0,
                                        "color": 0.0, "saturation": 0.0})
        f = backlit_frame()
        ctx = ctx_with_face()
        soft = face_union_mask(f, ctx.faces)
        face_zone = soft > 200                     # 脸核心（羽化带内）
        bg_zone = np.zeros((H, W), bool)
        bg_zone[:, int(W * 0.8):] = True           # 右侧纯背景（掩膜外）
        l_before = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[:, :, 0]
        out = eff.process(f.copy(), ctx)
        l_after = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 0]
        self.assertGreater(float(l_after[face_zone].mean()),
                           float(l_before[face_zone].mean()) + 50)
        self.assertLess(float(l_after[bg_zone].mean()),
                        float(l_before[bg_zone].mean()))

    def test_no_face_degrades_to_global(self):
        """无人脸时不报错，全图按背景目标校正：整图 L 均值朝 115 移动。"""
        eff = AutoEnhanceEffect(params={"strength": 1.0, "contrast": 0.0,
                                        "color": 0.0, "saturation": 0.0,
                                        "smooth": 0.0})
        f = backlit_frame()
        out = eff.process(f.copy(), FrameContext(width=W, height=H))
        self.assertEqual(out.shape, f.shape)
        self.assertEqual(out.dtype, np.uint8)
        l_after = float(cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 0].mean())
        l_before = float(cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[:, :, 0].mean())
        self.assertLess(abs(l_after - BG_TARGET_L),
                        abs(l_before - BG_TARGET_L))

    def test_face_exposure_priority_lifts_face_more(self):
        """face_exposure 插值方向：非逆光场景下 0.7 比 0 对脸核心的提亮更强。

        用均匀暗帧（脸/背景同亮，逆光量≈0 不触发逆光抑制），隔离
        face_exposure 的插值方向；逆光场景另有 test_backlight_damp 覆盖。
        """
        f = np.full((H, W, 3), 60, np.uint8)
        ctx = ctx_with_face()
        face_zone = face_union_mask(f, ctx.faces) > 200

        def run(fx: float) -> float:
            eff = AutoEnhanceEffect(params={
                "strength": 1.0, "contrast": 0.0, "color": 0.0,
                "saturation": 0.0, "smooth": 0.0, "face_exposure": fx})
            out = eff.process(f.copy(), ctx)
            return float(cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 0][face_zone].mean())

        self.assertGreater(run(0.7), run(0.0) + 5)

    def test_smooth_reduces_flicker(self):
        """统计量 EMA：场景突变后第二帧的校正量小于不平滑的第一帧响应差。"""
        eff = AutoEnhanceEffect(params={"strength": 1.0, "contrast": 0.0,
                                        "color": 0.0, "saturation": 0.0,
                                        "smooth": 0.9})
        bright = np.full((H, W, 3), 200, np.uint8)
        eff.process(bright.copy(), ctx_with_face())   # 建立亮场景统计
        dark = np.full((H, W, 3), 60, np.uint8)
        out1 = eff.process(dark.copy(), ctx_with_face())   # 突变第一帧
        g1 = eff.stats["gamma_bg"]
        eff2 = AutoEnhanceEffect(params={"strength": 1.0, "contrast": 0.0,
                                         "color": 0.0, "saturation": 0.0,
                                         "smooth": 0.0})
        eff2.process(bright.copy(), ctx_with_face())
        eff2.process(dark.copy(), ctx_with_face())
        g2 = eff2.stats["gamma_bg"]
        # 平滑版第一帧的 γ 还拖着亮场景的历史（更接近 1），即时版已到目标 γ
        self.assertGreater(g1, g2)

    def test_face_union_mask_pixels(self):
        f = np.zeros((H, W, 3), np.uint8)
        m = face_union_mask(f, ctx_with_face().faces)
        self.assertIsNotNone(m)
        self.assertEqual(m.shape, (H, W))
        # 椭圆中心（左半中央）在掩膜内、远处角落不在
        self.assertGreater(int(m[int(H * 0.5), int(W * 0.25)]), 200)
        self.assertEqual(int(m[5, 5]), 0)
        self.assertEqual(int(m[int(H * 0.5), int(W * 0.9)]), 0)
        self.assertIsNone(face_union_mask(f, []))

    def test_unknown_param_rejected(self):
        with self.assertRaises(ValueError):
            AutoEnhanceEffect(params={"nope": 1.0})

    def test_needs_faces_declared(self):
        from core.pipeline import NEED_FACES
        self.assertIn(NEED_FACES, AutoEnhanceEffect.needs)


class TestMaskFeelImprovements(unittest.TestCase):
    """针对"面具感"的三处改进的回归单测：宽羽化 / 细节保留 / 逆光抑制。"""

    def test_union_mask_wide_feather(self):
        """宽羽化：椭圆边缘外 ~10px 处 mask 值落在 (0,255) 中间（过渡带变宽）。

        旧实现只用 face_oval_mask 的 11px 高斯（sigma≈2），10px 外几乎为 0；
        现在并集后叠加尺度自适应宽高斯（sigma≥12），该处应有显著羽化值。
        椭圆右边界顶点在 (0.4W, 0.5H)，10px 外即 (0.4W+10, 0.5H)。
        """
        f = np.zeros((H, W, 3), np.uint8)
        m = face_union_mask(f, ctx_with_face().faces)
        v = float(m[int(H * 0.5), int(W * 0.4) + 10])
        self.assertGreater(v, 10.0)     # 非零 → 羽化带已扩到 10px 外
        self.assertLess(v, 240.0)       # 非满 → 仍在过渡带内

    def test_detail_preserving_keeps_local_contrast(self):
        """细节保留：gamma 提亮只打大尺度 base，局部明暗起伏不被压平。

        直接用 γ=0.5（强提亮）验证纯函数差异：同一张含高频明暗起伏的 L，
        apply_partition_gamma 把局部对比度压到 ≈局部 LUT 斜率倍（<1），
        detail_preserving_gamma 回加 detail 后几乎原样保留（"面具感"的修复）。
        """
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        ripple = 20.0 * np.sin(2 * np.pi * xx / 4.0)   # 波长 4px，远小于 base sigma
        l_ch = np.clip(128.0 + ripple, 0, 255).astype(np.uint8)

        lut = gamma_lut(0.5)          # γ=0.5：强提亮，局部对比度被压
        mask = np.ones((H, W), np.float32)
        old_out = apply_partition_gamma(l_ch, lut, lut, mask)
        new_out = detail_preserving_gamma(l_ch, lut, lut, mask)

        def detail_std(x: np.ndarray) -> float:
            base = cv2.GaussianBlur(x, (0, 0), 8)
            return float((x.astype(np.float32) - base.astype(np.float32)).std())

        c_before = detail_std(l_ch)
        c_old = detail_std(old_out)
        c_new = detail_std(new_out)
        self.assertLess(c_old, 0.85 * c_before)     # 老路径被 gamma 明显压平
        self.assertGreater(c_new, 0.9 * c_before)   # 新路径几乎原样保留
        self.assertGreater(c_new, c_old)

    def test_backlight_damp_retreats_face_target(self):
        """逆光抑制：背景比脸亮很多时，脸目标回退到背景目标附近。"""
        ft = FACE_TARGET_L
        # 无逆光（差值 ≤30）：目标不变
        self.assertAlmostEqual(face_target_backlight_damp(ft, 100.0, 90.0), ft)
        # 强逆光（差值 155）：完全回退到背景目标
        self.assertAlmostEqual(
            face_target_backlight_damp(ft, 210.0, 55.0), BG_TARGET_L)
        # 部分逆光（差值 60 → t=0.5）：回退到两者之间
        mid = face_target_backlight_damp(ft, 140.0, 80.0)
        self.assertGreater(mid, BG_TARGET_L)
        self.assertLess(mid, ft)
        # 区域空（None）：不做抑制
        self.assertAlmostEqual(face_target_backlight_damp(ft, None, 80.0), ft)
        self.assertAlmostEqual(face_target_backlight_damp(ft, 140.0, None), ft)


if __name__ == "__main__":
    unittest.main()
