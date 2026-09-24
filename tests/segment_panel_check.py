"""背景面板联动独立检查（需单独进程运行，避免 Qt 与推理测试共享进程）。"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from core.effects.segment import MODE_BLUR, MODE_IMAGE, SegmentEffect, load_image  # noqa: E402
from core.pipeline import Pipeline  # noqa: E402
from demos.faceswap.effect import FaceSwapEffect  # noqa: E402
from gui.panels import FaceSwapPanel, SegmentPanel  # noqa: E402


class TestSegmentPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.effect = SegmentEffect(enabled=False)
        self.panel = SegmentPanel(Pipeline([self.effect]))

    def tearDown(self):
        self.panel.close()

    def test_initialization_does_not_enable_effect(self):
        self.assertFalse(self.effect.enabled)
        self.assertEqual(self.effect.get_params()["mode"], MODE_BLUR)

    def test_image_mode_selects_default_background_and_enables(self):
        self.panel.cmb_mode.setCurrentIndex(self.panel.cmb_mode.findData(MODE_IMAGE))
        p = self.effect.get_params()
        self.assertTrue(self.effect.enabled)
        self.assertEqual(p["mode"], MODE_IMAGE)
        self.assertIsNotNone(load_image(p["bg_path"]))
        self.assertIn("当前背景", self.panel.lbl_bg_status.text())

    def test_gallery_selection_enables_effect(self):
        self.panel.cmb_mode.setCurrentIndex(self.panel.cmb_mode.findData(MODE_IMAGE))
        self.panel.chk_enabled.setChecked(False)
        item = self.panel.lst_bg.item(1)
        self.assertIsNotNone(item)
        self.panel._pick_gallery_bg(item)
        self.assertTrue(self.effect.enabled)
        self.assertEqual(self.effect.get_params()["bg_path"],
                         item.data(Qt.ItemDataRole.UserRole))


class TestFaceSwapPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.effect = FaceSwapEffect(enabled=False)
        self.panel = FaceSwapPanel(Pipeline([self.effect]))

    def tearDown(self):
        self.panel.close()

    def test_enable_requires_consent_and_source(self):
        self.assertGreaterEqual(self.panel.lst_faces.count(), 4)
        self.assertFalse(self.panel.chk_enabled.isEnabled())
        self.panel.chk_consent.setChecked(True)
        self.assertFalse(self.panel.chk_enabled.isEnabled())
        self.panel._pick_gallery_source(self.panel.lst_faces.item(0))
        self.assertTrue(self.panel.chk_enabled.isEnabled())
        self.assertTrue(self.effect.enabled)
        self.panel.chk_consent.setChecked(False)
        self.assertFalse(self.effect.enabled)


if __name__ == "__main__":
    unittest.main()
