"""Retinexformer 低光增强（质量档，ICCV 2023，LOL-v1 PSNR 25.16dB）。

架构 vendor 自官方 caiyuanhao1998/Retinexformer（Apache-2.0 with
common clause? —— 官方仓库 License: Apache 2.0；本文件仅作课程项目
研究用途，权重与代码版权归原作者）。改动：
  - 去掉 einops 依赖（rearrange 手写等价 permute/reshape）；
  - 去掉 pdb 调试 import；
  - checkpoint 加载兼容 raw state_dict 与 basicsr 包装两种格式。

配置 = Options/RetinexFormer_LOL_v1.yml：
  in/out 3 通道、n_feat=40、stage=1、num_blocks=[1,2,2]。
注意力为 d×d 线性复杂度（k@q^T 在特征维而非 token 维），512×512 推理
在 6GB 卡上无压力（这是选它做质量档的关键原因之一）。

会话：RetinexformerSession（接口对齐 infer.LowLightSession——
enhance(frame_bgr) 返回 RGB float [0,1]），fp16 autocast 加速，
权重 models/retinexformer_lol_v1.pth（scripts/download_models.py
经 gdown 从官方 Drive 文件夹拉取）。
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpuops import device, upload_frame

WEIGHTS_PATH = (Path(__file__).resolve().parent.parent / "models"
                / "retinexformer_lol_v1.pth")

# 推理尺寸（官方训练 patch 128；低光增强是低频任务，512 足够且保细节）
INFERENCE_SIZE = 512


# ------- vendored 架构（RetinexFormer_arch.py 清理版） -------

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in trunc_normal_.")
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def _rearrange_bhd(t: torch.Tensor, heads: int) -> torch.Tensor:
    """einops 'b n (h d) -> b h n d' 的等价实现。t: (b, n, h*d)。"""
    b, n, hd = t.shape
    d = hd // heads
    return t.view(b, n, heads, d).permute(0, 2, 1, 3)


class _GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, 1, bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, 5, padding=2,
                                    bias=True, groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, 1, bias=True)

    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)
        x = torch.cat([img, mean_c], dim=1)
        x_1 = self.conv1(x)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class IG_MSA(nn.Module):
    """Illumination-Guided Multi-head Self-Attention（d×d 线性复杂度）。"""

    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            _GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans
        q, k, v, illu_attn = map(
            lambda t: _rearrange_bhd(t, self.num_heads),
            (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v = v * illu_attn
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_c + out_p


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            _GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False,
                      groups=dim * mult),
            _GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class _PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class IGAB(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                _PreNorm(dim, FeedForward(dim=dim)),
            ]))

    def forward(self, x, illu_fea):
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2,
                 num_blocks=(2, 4, 4)):
        super().__init__()
        self.dim = dim
        self.level = level

        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim,
                     heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
            ]))
            dim_level *= 2

        self.bottleneck = IGAB(dim=dim_level, dim_head=dim,
                               heads=dim_level // dim,
                               num_blocks=num_blocks[-1])

        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(dim=dim_level // 2,
                     num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                     heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            _no_grad_trunc_normal_(m.weight, mean=0., std=.02, a=-2., b=2.)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        fea = self.embedding(x)

        fea_encoder = []
        illu_fea_list = []
        for (block, fea_down, illu_down) in self.encoder_layers:
            fea = block(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = fea_down(fea)
            illu_fea = illu_down(illu_fea)

        fea = self.bottleneck(fea, illu_fea)

        for i, (fea_up, fusion, block) in enumerate(self.decoder_layers):
            fea = fea_up(fea)
            fea = fusion(torch.cat(
                [fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = block(fea, illu_fea)

        return self.mapping(fea) + x


class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2,
                 num_blocks=(1, 1, 1)):
        super().__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels,
                                 dim=n_feat, level=level,
                                 num_blocks=num_blocks)

    def forward(self, img):
        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        return self.denoiser(input_img, illu_fea)


class RetinexFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3,
                 num_blocks=(1, 1, 1)):
        super().__init__()
        self.stage = stage
        modules_body = [
            RetinexFormer_Single_Stage(
                in_channels=in_channels, out_channels=out_channels,
                n_feat=n_feat, level=2, num_blocks=num_blocks)
            for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return self.body(x)


def build_lol_v1_model() -> RetinexFormer:
    """官方 LOL-v1 配置（Options/RetinexFormer_LOL_v1.yml）。"""
    return RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                         num_blocks=[1, 2, 2])


# ------- 会话 -------

class RetinexformerSession:
    """低光增强质量档会话（接口对齐 LowLightSession.enhance）。

    2026-09 性能重构：前向捕获为 CUDA Graph（数百小算子的 eager 图在
    WDDM 下被启动开销淹没，67ms → 图回放 ~20ms 量级），并新增
    enhance_tensor 张量接口供 GPU 效果链零拷贝衔接。
    """

    def __init__(self, weights: Path | str = WEIGHTS_PATH,
                 inference_size: int = INFERENCE_SIZE):
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"Retinexformer 权重缺失：{weights}，请先运行 "
                f"python scripts/download_models.py")
        self.size = inference_size
        self.model = build_lol_v1_model()
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            for wrap in ("params_network_g", "params", "state_dict"):
                if wrap in state and isinstance(state[wrap], dict):
                    state = state[wrap]      # basicsr 训练包装（三选一）
                    break
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(device()).eval()
        self.provider = f"Retinexformer-{device().type.upper()}"
        self._graph = None

    def _forward(self, x):
        use_amp = device().type == "cuda"
        with torch.no_grad(), torch.autocast("cuda", enabled=use_amp):
            return self.model(x)

    def _forward_graphed(self, x):
        if self._graph is None:
            from .cudagraph import GraphedCall
            self._graph = GraphedCall(self._forward, "retinexformer", [x])
        return self._graph(x)

    def enhance_tensor(self, f_bgr01: torch.Tensor) -> torch.Tensor:
        """GPU 张量版：BGR float [0,1] (1,3,H,W) → 增强结果（同域同尺寸）。"""
        import torch.nn.functional as F
        x = F.interpolate(f_bgr01.flip(1), size=(self.size, self.size),
                          mode="area")
        out = self._forward_graphed(x).float()
        up = F.interpolate(out, size=f_bgr01.shape[-2:], mode="bilinear",
                           align_corners=False)
        return up.flip(1).clamp_(0.0, 1.0)

    def enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        """整帧增强（方形推理域），返回 RGB float32 (size,size,3) [0,1]。

        口径与 infer.LowLightSession / LowLightSessionTorch.enhance 完全
        一致（推理域输出，由效果层统一 resize 回原尺寸）。
        """
        import torch.nn.functional as F
        t = upload_frame(frame_bgr).float().div_(255.0).flip(1)  # RGB
        x = F.interpolate(t, size=(self.size, self.size), mode="area")
        out = self._forward_graphed(x)
        return out[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
