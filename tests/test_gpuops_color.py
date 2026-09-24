"""GPU 算子使用的 LAB 颜色空间数值回归。"""

import unittest

import torch

from core.gpuops import lab_to_rgb, rgb_to_lab, rgb_to_ycrcb, ycrcb_to_rgb


class TestLabColorConversion(unittest.TestCase):
    def test_neutral_gray_has_no_chroma(self):
        levels = torch.linspace(0, 1, 256).view(1, 1, 1, 256)
        gray = levels.expand(1, 3, 1, 256)
        lab = rgb_to_lab(gray)
        self.assertLess(float(lab[:, 1:].abs().max()), 0.002)

    def test_color_gradient_round_trip(self):
        levels = torch.linspace(0, 1, 256).view(1, 1, 1, 256)
        rgb = torch.cat([levels, levels.square(), 1.0 - levels], dim=1)
        restored = lab_to_rgb(rgb_to_lab(rgb))
        self.assertLess(float((restored - rgb).abs().max()), 0.001)

    def test_ycrcb_color_round_trip(self):
        levels = torch.linspace(0, 1, 256).view(1, 1, 1, 256)
        rgb = torch.cat([levels, 1.0 - levels, levels.square()], dim=1)
        restored = ycrcb_to_rgb(rgb_to_ycrcb(rgb))
        self.assertLess(float((restored - rgb).abs().max()), 0.001)


if __name__ == "__main__":
    unittest.main()
