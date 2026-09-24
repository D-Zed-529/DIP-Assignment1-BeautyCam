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
from core.pipeline import Pipeline

_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE = _ROOT / "assets" / "samples" / "portrait1.jpg"

READY = torch_backend_ready() and _SAMPLE.exists()


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


if __name__ == "__main__":
    unittest.main()
