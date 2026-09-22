"""管线与采集源单测（不依赖摄像头/GUI/模型）。"""

from __future__ import annotations

import os
import tempfile
import threading
import unittest

import cv2
import numpy as np

from core.camera import ImageSequenceSource, VideoFileSource, mean_brightness
from core.context import FrameContext
from core.effects.beauty import BeautyEffect
from core.effects.lowlight import LowLightEffect
from core.pipeline import NEED_FACES, Effect, Pipeline


class RecEffect(Effect):
    """记录调用顺序的桩效果。"""

    def __init__(self, name: str, needs=()):
        super().__init__()
        self.name = name
        self.needs = frozenset(needs)
        self.calls: list[str] = []

    @staticmethod
    def default_params() -> dict:
        return {}

    def process(self, frame, ctx):
        self.calls.append(self.name)
        return frame


class TestPipeline(unittest.TestCase):
    def test_order_preserved(self):
        a, b, c = RecEffect("a"), RecEffect("b"), RecEffect("c")
        pipe = Pipeline([a, b, c])
        ctx = FrameContext()
        pipe.process(np.zeros((4, 4, 3), np.uint8), ctx)
        self.assertEqual([e.calls for e in (a, b, c)],
                         [["a"], ["b"], ["c"]])

    def test_disabled_skipped(self):
        a, b = RecEffect("a"), RecEffect("b")
        pipe = Pipeline([a, b])
        pipe.set_enabled("b", False)
        pipe.process(np.zeros((4, 4, 3), np.uint8), FrameContext())
        self.assertEqual(a.calls, ["a"])
        self.assertEqual(b.calls, [])

    def test_duplicate_name_rejected(self):
        with self.assertRaises(ValueError):
            Pipeline([RecEffect("x"), RecEffect("x")])

    def test_infer_needs_union_of_enabled(self):
        a = RecEffect("a", needs={NEED_FACES})
        b = RecEffect("b", needs={"segmentation"})
        pipe = Pipeline([a, b])
        self.assertEqual(pipe.infer_needs(), {NEED_FACES, "segmentation"})
        pipe.set_enabled("b", False)
        self.assertEqual(pipe.infer_needs(), {NEED_FACES})
        pipe.set_enabled("a", False)
        self.assertEqual(pipe.infer_needs(), set())

    def test_concurrent_param_updates(self):
        """GUI 线程狂改参数、工作线程狂读 —— 无异常且值合法（竞态回归测试）。"""
        eff = BeautyEffect()
        pipe = Pipeline([eff])
        stop = threading.Event()
        errors: list[Exception] = []

        def writer():
            try:
                for i in range(2000):
                    pipe.set_params("beauty", whiten=float(i % 31),
                                    slim=(i % 101) / 101.0)
            except Exception as e:  # pragma: no cover
                errors.append(e)
            stop.set()

        t = threading.Thread(target=writer)
        t.start()
        frame = np.random.randint(0, 255, (48, 64, 3), np.uint8)
        ctx = FrameContext(width=64, height=48)
        while not stop.is_set():
            pipe.process(frame, ctx)
        t.join()
        self.assertEqual(errors, [])

    def test_unknown_effect_params_rejected(self):
        pipe = Pipeline([RecEffect("a")])
        with self.assertRaises(ValueError):
            pipe.set_params("a", bogus=1)
        with self.assertRaises(KeyError):
            pipe.set_params("no_such_effect", x=1)


class ThrottledEffect(RecEffect):
    """按固定间隔声明推理需求的桩效果（模拟重推理隔帧降载）。"""

    def __init__(self, name: str, needs, interval: int, enabled: bool = True):
        super().__init__(name, needs)
        self._interval = interval
        self.set_enabled(enabled)

    def inference_interval(self, need: str) -> int:
        return self._interval


class TestInferNeedsThrottle(unittest.TestCase):
    """infer_needs_for 的隔帧节流（Phase 3 分割 / Phase 2 低光重推理降载）。"""

    def test_default_interval_runs_every_frame(self):
        pipe = Pipeline([RecEffect("a", needs={NEED_FACES})])
        for i in range(5):
            self.assertEqual(pipe.infer_needs_for(i), {NEED_FACES})

    def test_interval_keeps_phase(self):
        pipe = Pipeline([ThrottledEffect("a", {"segmentation"}, interval=4)])
        hits = ["segmentation" in pipe.infer_needs_for(i) for i in range(8)]
        self.assertEqual(hits, [True, False, False, False,
                                True, False, False, False])

    def test_first_frame_always_runs_everything(self):
        """首帧必跑全量 —— 否则刚开效果会出现"拿不到掩膜"的空窗。"""
        pipe = Pipeline([ThrottledEffect("a", {"segmentation"}, interval=7)])
        self.assertIn("segmentation", pipe.infer_needs_for(0))

    def test_infer_needs_equals_frame_zero(self):
        """infer_needs() 保留原语义（全量并集），供不关心隔帧的调用方使用。"""
        pipe = Pipeline([
            ThrottledEffect("a", {"segmentation"}, interval=4),
            RecEffect("b", needs={NEED_FACES}),
        ])
        self.assertEqual(pipe.infer_needs(), pipe.infer_needs_for(0))
        self.assertEqual(pipe.infer_needs(), {"segmentation", NEED_FACES})

    def test_disabled_effect_not_counted(self):
        pipe = Pipeline([ThrottledEffect("a", {"segmentation"}, interval=1,
                                          enabled=False)])
        self.assertEqual(pipe.infer_needs_for(0), set())
        self.assertEqual(pipe.infer_needs_for(4), set())

    def test_union_across_effects_with_different_intervals(self):
        """同一 need 被多个效果需要时取并：任一效果要求本帧跑就跑。"""
        pipe = Pipeline([
            ThrottledEffect("slow", {"segmentation"}, interval=4),
            ThrottledEffect("fast", {"segmentation"}, interval=2),
        ])
        hits = ["segmentation" in pipe.infer_needs_for(i) for i in range(5)]
        self.assertEqual(hits, [True, False, True, False, True])

    def test_zero_or_negative_interval_clamped_to_one(self):
        """非法间隔不应导致除零或永久跳过推理。"""
        pipe = Pipeline([ThrottledEffect("a", {"segmentation"}, interval=0)])
        for i in range(3):
            self.assertIn("segmentation", pipe.infer_needs_for(i))
        pipe2 = Pipeline([ThrottledEffect("b", {"segmentation"}, interval=-3)])
        self.assertIn("segmentation", pipe2.infer_needs_for(1))

    def test_real_segment_effect_declares_interval(self):
        from core.effects.segment import SegmentEffect
        eff = SegmentEffect(enabled=True, params={"infer_interval": 3})
        pipe = Pipeline([eff])
        hits = ["segmentation" in pipe.infer_needs_for(i) for i in range(6)]
        self.assertEqual(hits, [True, False, False, True, False, False])
        self.assertEqual(eff.inference_interval("faces"), 1)


class TestLowLightEffect(unittest.TestCase):
    def test_dark_frame_enhanced(self):
        frame = np.full((60, 80, 3), 30, np.uint8)   # 暗帧
        self.assertLess(mean_brightness(frame), 60)
        eff = LowLightEffect(enabled=True)
        out = eff.process(frame.copy(), FrameContext(width=80, height=60))
        self.assertGreater(mean_brightness(out), mean_brightness(frame))

    def test_bright_frame_passthrough_when_auto(self):
        frame = np.full((60, 80, 3), 200, np.uint8)
        eff = LowLightEffect(enabled=True)   # auto 默认 True
        out = eff.process(frame.copy(), FrameContext(width=80, height=60))
        self.assertTrue(np.array_equal(out, frame))

    def test_force_mode_applies_on_bright_frame(self):
        frame = np.full((60, 80, 3), 200, np.uint8)
        eff = LowLightEffect(enabled=True, params={"auto": False})
        out = eff.process(frame.copy(), FrameContext(width=80, height=60))
        self.assertFalse(np.array_equal(out, frame))


class TestMeanBrightness(unittest.TestCase):
    def test_uniform_frame(self):
        frame = np.full((10, 10, 3), 100, np.uint8)
        self.assertAlmostEqual(mean_brightness(frame), 100.0)


class TestImageSequenceSource(unittest.TestCase):
    def test_reads_all_in_order(self):
        with tempfile.TemporaryDirectory() as td:
            for i in range(3):
                cv2.imwrite(os.path.join(td, f"img_{i}.jpg"),
                            np.full((20, 30, 3), 50 + i, np.uint8))
            src = ImageSequenceSource(directory=td)
            self.assertTrue(src.open())
            for expect in range(3):
                ret, frame = src.read()
                self.assertTrue(ret)
                self.assertEqual(int(frame[0, 0, 0]), 50 + expect)
            ret, frame = src.read()
            self.assertFalse(ret)
            src.release()

    def test_skips_corrupt_file(self):
        with tempfile.TemporaryDirectory() as td:
            good = os.path.join(td, "good.jpg")
            cv2.imwrite(good, np.zeros((20, 30, 3), np.uint8))
            bad = os.path.join(td, "bad.jpg")
            with open(bad, "w") as f:
                f.write("not an image")
            src = ImageSequenceSource(directory=td)
            src.open()
            ret, frame = src.read()
            self.assertTrue(ret)
            self.assertEqual(frame.shape, (20, 30, 3))
            ret, _ = src.read()
            self.assertFalse(ret)


class TestVideoFileSource(unittest.TestCase):
    def _make_video(self, path: str, n=5):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                            30, (32, 24))
        for i in range(n):
            w.write(np.full((24, 32, 3), 40 * i, np.uint8))
        w.release()

    def test_reads_all_then_eof(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "v.mp4")
            self._make_video(path)
            src = VideoFileSource(path)
            self.assertTrue(src.open())
            count = 0
            while True:
                ret, _ = src.read()
                if not ret:
                    break
                count += 1
            self.assertGreaterEqual(count, 4)
            src.release()

    def test_loop(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "v.mp4")
            self._make_video(path, n=3)
            src = VideoFileSource(path, loop=True)
            src.open()
            total = 0
            for _ in range(10):
                ret, _ = src.read()
                if ret:
                    total += 1
            self.assertEqual(total, 10)   # 循环读不完
            src.release()


if __name__ == "__main__":
    unittest.main()
