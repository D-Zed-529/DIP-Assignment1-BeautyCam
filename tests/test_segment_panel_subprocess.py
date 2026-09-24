"""Qt 面板检查在独立进程运行，避免与 MediaPipe 的测试共享运行时。"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest


class TestSegmentPanelSubprocess(unittest.TestCase):
    def test_background_controls(self):
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "tests.segment_panel_check", "-q"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
