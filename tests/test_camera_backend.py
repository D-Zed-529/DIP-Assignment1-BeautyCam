"""实时摄像头在不同系统下选择正确的 OpenCV 后端。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import cv2

from core.camera import LiveCamera


class TestLiveCameraBackend(unittest.TestCase):
    def test_windows_prefers_directshow(self):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, None)
        with patch("core.camera.platform.system", return_value="Windows"), \
             patch("core.camera.cv2.VideoCapture", return_value=cap) as open_cap:
            camera = LiveCamera(index=0)
            self.assertTrue(camera.open())
            open_cap.assert_called_once_with(0, cv2.CAP_DSHOW)
            camera.release()

    def test_windows_falls_back_when_directshow_unavailable(self):
        first, second = MagicMock(), MagicMock()
        first.isOpened.return_value = False
        second.isOpened.return_value = True
        second.read.return_value = (True, None)
        with patch("core.camera.platform.system", return_value="Windows"), \
             patch("core.camera.cv2.VideoCapture", side_effect=[first, second]) as open_cap:
            camera = LiveCamera(index=0)
            self.assertTrue(camera.open())
            self.assertEqual(open_cap.call_args_list[0].args, (0, cv2.CAP_DSHOW))
            self.assertEqual(open_cap.call_args_list[1].args, (0, cv2.CAP_ANY))
            first.release.assert_called_once()
            camera.release()


if __name__ == "__main__":
    unittest.main()
