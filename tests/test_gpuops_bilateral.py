"""向量化双边滤波与逐位参考路径的数值一致性。"""

import unittest

import torch

from core.gpuops import bilateral_blur


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 对照两个实现")
class TestBilateralVectorized(unittest.TestCase):
    def test_matches_reference(self):
        torch.manual_seed(17)
        image = torch.rand(1, 3, 16, 20)
        reference = bilateral_blur(image, 9, 60.0, 60.0)
        actual = bilateral_blur(image.cuda(), 9, 60.0, 60.0).cpu()
        self.assertLess(float((actual - reference).abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
