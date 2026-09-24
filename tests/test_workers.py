"""相机线程的按需推理调度测试（不打开摄像头）。"""

import unittest
from unittest.mock import patch

import numpy as np

from core.effects.beauty import BeautyEffect
from core.effects.bokeh import BokehEffect
from core.effects.segment import SegmentEffect
from core.context import FrameContext
from core.gestures import AutoCaptureState
from core.pipeline import Pipeline
from gui.workers import CameraWorker, TRIGGER_SMILE, TRIGGER_V_SIGN


class _RecordingEngine:
    backend_name = "torch:cuda"

    def __init__(self):
        self.kwargs = None

    def process(self, frame, **kwargs):
        self.kwargs = kwargs
        return None


class TestWorkerInferenceNeeds(unittest.TestCase):
    def setUp(self):
        self.engine = _RecordingEngine()
        self.worker = CameraWorker(
            object(), Pipeline([BeautyEffect()]), self.engine, triggers=set())
        self.frame = np.zeros((16, 16, 3), dtype=np.uint8)

    def test_default_beauty_only_runs_faces(self):
        self.worker._infer(self.frame, 0)
        self.assertTrue(self.engine.kwargs["faces"])
        self.assertFalse(self.engine.kwargs["hands"])
        self.assertFalse(self.engine.kwargs["blendshapes"])

    def test_v_sign_only_adds_hands(self):
        self.worker.triggers = {TRIGGER_V_SIGN}
        self.worker._infer(self.frame, 0)
        self.assertTrue(self.engine.kwargs["faces"])
        self.assertTrue(self.engine.kwargs["hands"])
        self.assertFalse(self.engine.kwargs["blendshapes"])

    def test_smile_only_adds_blendshapes(self):
        self.worker.triggers = {TRIGGER_SMILE}
        self.worker._infer(self.frame, 0)
        self.assertTrue(self.engine.kwargs["faces"])
        self.assertFalse(self.engine.kwargs["hands"])
        self.assertTrue(self.engine.kwargs["blendshapes"])

    def test_full_load_uses_serial_cuda_and_restores_light_load(self):
        self.engine.parallel_inference = True
        pipe = Pipeline([BeautyEffect(), SegmentEffect(enabled=True),
                         BokehEffect(enabled=True)])
        worker = CameraWorker(object(), pipe, self.engine,
                              triggers={TRIGGER_V_SIGN, TRIGGER_SMILE})
        worker._infer(self.frame, 0)
        self.assertFalse(self.engine.parallel_inference)
        self.assertTrue(self.engine.kwargs["hands"])
        self.assertTrue(self.engine.kwargs["depth"])
        pipe.set_enabled("segment", False)
        pipe.set_enabled("bokeh", False)
        worker.triggers = set()
        worker._infer(self.frame, 1)
        self.assertTrue(self.engine.parallel_inference)


class _OneFrameSource:
    continuous = False

    def __init__(self):
        self.frame = np.zeros((16, 16, 3), dtype=np.uint8)

    def read(self):
        if self.frame is None:
            return False, None
        frame, self.frame = self.frame, None
        return True, frame


class _ContextEngine(_RecordingEngine):
    def process(self, frame, **kwargs):
        self.kwargs = kwargs
        return FrameContext(width=frame.shape[1], height=frame.shape[0])


class TestWorkerAutoCapture(unittest.TestCase):
    def _run_trigger(self, trigger, detector_name):
        engine = _ContextEngine()
        worker = CameraWorker(_OneFrameSource(), Pipeline([]), engine,
                              triggers=set(), process_scale=1.0)
        # 模拟相机启动后用户才勾选触发器。
        worker.triggers = {trigger}
        worker._auto_state = AutoCaptureState(v_hold=0, smile_hold=0,
                                               cooldown=0)
        saved = []
        worker._save_photo = lambda frame, name: saved.append(name)
        with patch(f"gui.workers.{detector_name}", return_value=True):
            worker._loop()
        self.assertEqual(saved, [trigger])
        return engine.kwargs

    def test_v_sign_after_start_captures(self):
        kwargs = self._run_trigger(TRIGGER_V_SIGN, "is_v_sign")
        self.assertTrue(kwargs["hands"])

    def test_smile_after_start_captures(self):
        kwargs = self._run_trigger(TRIGGER_SMILE, "any_smiling")
        self.assertTrue(kwargs["faces"])
        self.assertTrue(kwargs["blendshapes"])


if __name__ == "__main__":
    unittest.main()
