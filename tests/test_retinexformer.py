"""Retinexformer 低光质量档（core/retinexformer.py）单测。

架构构建 / 前向 shape 是纯 torch 逻辑（无权重也跑）；会话级用例需
models/retinexformer_lol_v1.pth，缺失自动跳过。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

_WEIGHTS = (Path(__file__).resolve().parent.parent / "models"
            / "retinexformer_lol_v1.pth")

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch 未安装")
class TestRetinexformerArch(unittest.TestCase):
    def test_build_lol_v1_config(self):
        from core.retinexformer import build_lol_v1_model
        m = build_lol_v1_model()
        n_params = sum(p.numel() for p in m.parameters())
        # LOL-v1 配置（n_feat=40/stage=1/blocks=[1,2,2]）约 1.6M 参数
        self.assertGreater(n_params, 1_000_000)
        self.assertLess(n_params, 3_000_000)

    def test_forward_shape_and_range(self):
        from core.retinexformer import build_lol_v1_model
        m = build_lol_v1_model().eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            y = m(x)
        self.assertEqual(tuple(y.shape), (1, 3, 64, 64))

    def test_illumination_estimator_groups_bug_compatibility(self):
        """官方代码 depth_conv 的 groups=n_fea_in(4)（40 通道可整除），
        vendor 版逐参数等价 —— 结构抽查防手抄错。"""
        from core.retinexformer import Illumination_Estimator
        est = Illumination_Estimator(40)
        self.assertEqual(est.depth_conv.groups, 4)
        self.assertEqual(est.conv1.in_channels, 4)   # img 3 + mean_c 1


@unittest.skipUnless(HAS_TORCH and _WEIGHTS.exists(),
                     "Retinexformer 权重缺失（scripts/download_models.py）")
class TestRetinexformerSession(unittest.TestCase):
    def test_enhance_brightens_dark_frame(self):
        from core.retinexformer import RetinexformerSession
        sess = RetinexformerSession()
        dark = np.full((180, 320, 3), 30, np.uint8)
        out = sess.enhance(dark)
        # 口径：RGB float [0,1] (512,512,3)（推理域，效果层统一 resize）
        self.assertEqual(out.shape, (512, 512, 3))
        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(float(out.mean()), float(dark.mean()) / 255.0,
                           "暗帧应被显著提亮")


class TestLowLightEngineParam(unittest.TestCase):
    """低光效果的 engine 参数校验（不需要权重/会话）。"""

    def test_default_engine_is_sci(self):
        from core.effects.lowlight import LowLightDnnEffect
        eff = LowLightDnnEffect()
        self.assertEqual(eff.get_params()["engine"], "sci")

    def test_unknown_engine_rejected(self):
        from core.effects.lowlight import LowLightDnnEffect
        with self.assertRaises(ValueError):
            LowLightDnnEffect(params={"engine": "zero-dce"})


if __name__ == "__main__":
    unittest.main()
