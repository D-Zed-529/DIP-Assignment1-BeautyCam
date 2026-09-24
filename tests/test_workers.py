"""相机线程的按需推理调度测试（不打开摄像头）。"""

import unittest

import numpy as np

from core.effects.beauty import BeautyEffect
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


if __name__ == "__main__":
    unittest.main()
