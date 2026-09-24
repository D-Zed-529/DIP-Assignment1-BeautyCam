"""实时相机的最新帧采集测试（使用假设备，不占用摄像头）。"""

import queue
import time
import unittest
from unittest.mock import patch

import numpy as np

from core.camera import LiveCamera


class _FakeCapture:
    def __init__(self):
        self.frames = queue.Queue()
        self.frames.put(np.zeros((2, 2, 3), dtype=np.uint8))
        self.closed = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def read(self):
        try:
            return True, self.frames.get(timeout=0.05)
        except queue.Empty:
            return False, None

    def release(self):
        self.closed = True


class TestLiveCameraLatestFrame(unittest.TestCase):
    def test_skips_stale_frames(self):
        fake = _FakeCapture()
        with patch("core.camera.cv2.VideoCapture", return_value=fake):
            camera = LiveCamera(mirror=False)
            self.assertTrue(camera.open())
            try:
                for value in (1, 2, 3):
                    fake.frames.put(np.full((2, 2, 3), value, dtype=np.uint8))
                deadline = time.monotonic() + 1.0
                while camera._latest_seq < 4 and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertGreaterEqual(camera._latest_seq, 4)
                ok, frame = camera.read()
                self.assertTrue(ok)
                self.assertEqual(int(frame[0, 0, 0]), 3)
                camera.READ_WAIT_S = 0.02
                self.assertEqual(camera.read(), (False, None))
            finally:
                camera.release()
            self.assertTrue(fake.closed)


if __name__ == "__main__":
    unittest.main()
