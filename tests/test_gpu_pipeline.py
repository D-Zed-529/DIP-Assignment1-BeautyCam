"""GPU 融合链（Pipeline use_gpu=True）与 FrameContext 张量承载的单测。

需要 torch + CUDA + 权重就绪；任一缺失自动跳过（口径同 test_infer_torch）。
覆盖：
  - FrameContext 张量/ numpy 双形态懒互转与缓存；
  - GPU 融合管线的端到端形状/类型/数值合理性（美颜/虚化/深度虚化全开）；
  - BokehEffect 深度隔帧缓存（interval=3 时中间帧复用）；
  - SegmentEffect GPU 状态在 reset_temporal 后正确断开。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from core.infer import torch_backend_ready
from core.pipeline import Effect, Pipeline

_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE = _ROOT / "assets" / "samples" / "portrait1.jpg"

READY = torch_backend_ready() and _SAMPLE.exists()


class _GpuAddEffect(Effect):
    name = "gpu_add_test"
    supports_gpu = True

    @staticmethod
    def default_params() -> dict:
        return {}

    def process(self, frame, ctx):
        raise AssertionError("应走 GPU 路径")

    def process_gpu(self, frame_t, ctx):
        return frame_t + 20


class _CpuAddEffect(Effect):
    name = "cpu_add_test"

    @staticmethod
    def default_params() -> dict:
        return {}

    def process(self, frame, ctx):
        if not np.all(frame == 30):
            raise AssertionError("CPU 插件没有收到前一个 GPU 效果的结果")
        return frame + 5


class TestGpuCpuBoundary(unittest.TestCase):
    def test_cpu_effect_receives_latest_gpu_frame(self):
        from core.context import FrameContext
        frame = np.full((4, 5, 3), 10, np.uint8)
        pipeline = Pipeline([_GpuAddEffect(), _CpuAddEffect()], use_gpu=True)
        out = pipeline.process(frame, FrameContext(width=5, height=4))
        self.assertTrue(np.all(out == 35))

    def test_color_background_with_empty_matte(self):
        """CUDA 效果链的纯色替换不依赖最低人像面积。"""
        from core.context import FrameContext
        from core.effects.segment import MODE_COLOR, SegmentEffect
        frame = np.full((12, 16, 3), 200, np.uint8)
        ctx = FrameContext(width=16, height=12,
                           person_alpha=np.zeros((12, 16), np.float32))
        effect = SegmentEffect(params={"mode": MODE_COLOR, "bg_color": "#00B140",
                                       "refine": False, "feather": 0.0})
        out = Pipeline([effect], use_gpu=True).process(frame, ctx)
        self.assertTrue(np.all(out == np.array([64, 177, 0], np.uint8)))

    def test_gpu_autoenhance_without_face(self):
        """无人脸时 GPU 自动曝光也须返回与输入同尺寸的 BGR 帧。"""
        from core.context import FrameContext
        from core.effects.autoenhance import AutoEnhanceEffect

        frame = np.full((64, 96, 3), 70, np.uint8)
        effect = AutoEnhanceEffect(params={"color": 0.0, "contrast": 0.0,
                                           "smooth": 0.0})
        out = Pipeline([effect], use_gpu=True).process(
            frame, FrameContext(width=96, height=64))
        self.assertEqual(out.shape, frame.shape)
        self.assertGreater(float(out.mean()), float(frame.mean()))


def _sample_720p():
    img = cv2.imread(str(_SAMPLE))
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = 720 / h
    return cv2.resize(img, (round(w * scale), 720))


class TestFrameContextDualForm(unittest.TestCase):
    def test_tensor_alpha_lazy_numpy(self):
        """张量承载 → numpy 懒物化并缓存（D2H 只发生一次）。"""
        import torch
        from core.context import FrameContext
        ctx = FrameContext(width=8, height=6)
        t = torch.rand(1, 1, 6, 8, device="cuda" if torch.cuda.is_available()
                       else "cpu")
        ctx.person_alpha_t = t
        a1 = ctx.person_alpha
        a2 = ctx.person_alpha
        self.assertIs(a1, a2, "numpy 物化结果应缓存")
        self.assertEqual(a1.shape, (6, 8))
        self.assertTrue(np.allclose(a1, t[0, 0].cpu().numpy(), atol=1e-6))

    def test_numpy_depth_lazy_tensor(self):
        """numpy 承载 → 张量懒上传（mediapipe 引擎路径同款语义）。"""
        import torch
        from core.context import FrameContext
        ctx = FrameContext(width=8, height=6)
        d = np.random.rand(6, 8).astype(np.float32)
        ctx.depth = d
        t = ctx.depth_t
        self.assertEqual(tuple(t.shape), (1, 1, 6, 8))
        self.assertTrue(torch.allclose(t[0, 0].cpu(), torch.from_numpy(d)))

    def test_mask_derived_from_alpha(self):
        import torch
        from core.context import FrameContext
        ctx = FrameContext(width=4, height=4)
        t = torch.zeros(1, 1, 4, 4)
        t[..., :2] = 1.0        # 左半前景
        ctx.person_alpha_t = t
        m = ctx.person_mask
        self.assertEqual(set(np.unique(m)), {0, 255})
        self.assertEqual(m[:, :2].max(), 255)
        self.assertEqual(m[:, 2:].max(), 0)


@unittest.skipUnless(READY, "torch 后端未就绪（CUDA / 权重缺失）")
class TestGpuFusedPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.effects.beauty import BeautyEffect
        from core.effects.bokeh import BokehEffect
        from core.effects.segment import SegmentEffect
        from core.infer import get_engine
        cls.engine = get_engine()
        if not str(cls.engine.backend_name).startswith("torch"):
            raise unittest.SkipTest("当前引擎非 torch 后端")
        cls.beauty = BeautyEffect()
        cls.segment = SegmentEffect(params={"smooth": 0.0})
        cls.bokeh = BokehEffect()
        cls.pipe_gpu = Pipeline([cls.beauty, cls.segment, cls.bokeh],
                                use_gpu=True)
        cls.pipe_cpu = Pipeline([cls.beauty, cls.segment, cls.bokeh],
                                use_gpu=False)
        cls.img = _sample_720p()
        if cls.img is None:
            raise unittest.SkipTest("样例图无法读取")

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_end_to_end_shape_and_sanity(self):
        ctx = self.engine.process(self.img, faces=True, hands=False,
                                  segmentation=True, depth=True,
                                  blendshapes=False)
        out = self.pipe_gpu.process(self.img.copy(), ctx)
        self.assertEqual(out.shape, self.img.shape)
        self.assertEqual(out.dtype, np.uint8)
        # 处理过的帧与原图应有差异但不至于全帧崩坏
        diff = np.abs(out.astype(int) - self.img.astype(int)).mean()
        self.assertGreater(diff, 1.0)
        self.assertLess(out.mean(), 255.0)

    def test_bokeh_depth_interval_reuse(self):
        """深度隔帧（interval=3）：中间帧 ctx.depth 为 None 时复用缓存。"""
        for i in range(3):
            needs = self.pipe_gpu.infer_needs_for(i)
            want_depth = "depth" in needs
            if i == 0:
                self.assertTrue(want_depth)
            else:
                self.assertFalse(want_depth, "间隔内不应重复推理深度")
            ctx = self.engine.process(
                self.img, faces=True, hands=False,
                segmentation="segmentation" in needs,
                depth=want_depth, blendshapes=False)
            out = self.bokeh.process(self.img.copy(), ctx)
            self.assertEqual(out.shape, self.img.shape)
        self.assertIsNotNone(self.bokeh._depth_cache,
                             "中间帧应复用深度缓存")

    def test_gpu_segment_reset_temporal(self):
        ctx = self.engine.process(self.img, faces=False, hands=False,
                                  segmentation=True)
        self.pipe_gpu.process(self.img.copy(), ctx)
        self.assertIsNotNone(self.segment._prev_alpha_t)
        self.segment.reset_temporal()
        self.assertIsNone(self.segment._prev_alpha_t)
        self.assertEqual(self.segment._bg_gpu_cache, {})

    def test_gpu_enlarge_eyes_matches_cpu(self):
        """大眼启用后 CUDA 路径应执行局部 remap，与 CPU 参考接近。"""
        from core.effects.beauty import BeautyEffect
        ctx = self.engine.process(self.img, faces=True, hands=False,
                                  segmentation=False, blendshapes=False)
        if not ctx.faces:
            self.skipTest("样例图未检测到人脸")
        effect = BeautyEffect(params={"smooth": 0.0, "whiten": 0.0,
                                      "slim": 0.0, "eye_enabled": True,
                                      "eye_strength": 0.5, "finish": False})
        gpu_out = Pipeline([effect], use_gpu=True).process(self.img.copy(), ctx)
        cpu_out = effect.process(self.img.copy(), ctx)
        self.assertGreater(np.count_nonzero(np.abs(gpu_out.astype(int)
                                                   - self.img.astype(int)) > 5), 100)
        self.assertLess(np.abs(gpu_out.astype(int) - cpu_out.astype(int)).mean(),
                        0.1)

    def test_gpu_autoenhance_matches_cpu_exposure(self):
        """实时 GPU 曝光与 CPU 分区曝光保持相近亮度，并走通有脸路径。"""
        from core.effects.autoenhance import AutoEnhanceEffect

        ctx = self.engine.process(self.img, faces=True, hands=False,
                                  segmentation=False, blendshapes=False)
        if not ctx.faces:
            self.skipTest("样例图未检测到人脸")
        params = {"strength": 1.0, "color": 0.0, "contrast": 0.0,
                  "saturation": 0.0, "smooth": 0.0}
        gpu_eff = AutoEnhanceEffect(params=params)
        cpu_eff = AutoEnhanceEffect(params=params)
        gpu_out = Pipeline([gpu_eff], use_gpu=True).process(self.img.copy(), ctx)
        cpu_out = cpu_eff.process(self.img.copy(), ctx)
        self.assertEqual(gpu_out.shape, self.img.shape)
        self.assertLess(abs(float(gpu_out.mean()) - float(cpu_out.mean())), 8.0)

    def test_gpu_whitening_respects_empty_person_matte(self):
        """人像掩膜全空时，GPU 美白不得改变肤色背景。"""
        import torch
        from core.effects.beauty import BeautyEffect

        ctx = self.engine.process(self.img, faces=True, hands=False,
                                  segmentation=False, blendshapes=False)
        if not ctx.faces:
            self.skipTest("样例图未检测到人脸")
        h, w = self.img.shape[:2]
        ctx.person_alpha_t = torch.zeros(1, 1, h, w, device="cuda")
        effect = BeautyEffect(params={"smooth": 0.0, "whiten": 20.0,
                                      "slim": 0.0, "eye_enabled": False,
                                      "finish": False})
        out = Pipeline([effect], use_gpu=True).process(self.img.copy(), ctx)
        self.assertLess(np.abs(out.astype(int) - self.img.astype(int)).mean(),
                        0.01)


if __name__ == "__main__":
    unittest.main()
