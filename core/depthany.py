"""Depth Anything V2 深度估计会话（P3-4 深度渐进虚化的推理后端）。

模型：depth-anything/Depth-Anything-V2-Small-hf（ViT-S 编码 + DPT 头，
HuggingFace transformers 格式，本地权重 models/hf/depth-anything-v2-small-hf，
由 scripts/download_models.py 拉取）。算力升级（RTX 3060 / CUDA）后启用
—— 2026-09 之前 PLAN 里标记"进阶可选"的 P3-4 就此补上。

设计：
  - transformers 懒加载（无 transformers/权重的环境其余功能不受影响）；
  - fp16 autocast（CUDA）加速；推理尺寸 392（518 是官方默认，392 对
    虚化用途足够且更快，参数可调）；
  - 2026-09 性能重构：预处理全部在 GPU 张量域完成（原 HF processor 在
    CPU 上做 resize + 归一化，外加 cvtColor/上传共 ~8ms），前向捕获为
    CUDA Graph（transformers 前向数百小算子，WDDM 启动开销同样致命），
    min/max 时域 EMA 用 GPU 标量张量维护（免去逐帧 float() 同步）；
  - 输出 (h, w) float32 **相对深度**（Depth Anything 口径：值越大越近），
    已缩放到 [0,1]（按本帧 min/max，时域 EMA 平滑 min/max 防闪）。

预处理口径与 HF processor 数值等价（低频信号，uint8/float 的插值舍入差
不影响虚化用途）：resize(392, bilinear) → (x − 0.5) / 0.5。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np

HF_MODEL_DIR = (Path(__file__).resolve().parent.parent / "models" / "hf"
                / "depth-anything-v2-small-hf")

# 推理输入边长（ViT 需要 14 的倍数；392 = 28 patch × 14）
DEFAULT_INFERENCE_SIZE = 392

# min/max 归一化的时域 EMA 系数（越大越稳；0 = 每帧独立归一化）
NORM_EMA = 0.9


class DepthSession:
    """Depth Anything V2 Small 单例会话（线程懒加载 + 时域归一化状态）。"""

    def __init__(self, model_dir: Path | str = HF_MODEL_DIR,
                 inference_size: int = DEFAULT_INFERENCE_SIZE,
                 use_graph: bool = True):
        self.model_dir = Path(model_dir)
        self.size = inference_size
        if not (self.model_dir / "model.safetensors").exists():
            raise FileNotFoundError(
                f"深度模型缺失：{self.model_dir}，请先运行 "
                f"python scripts/download_models.py")
        import torch
        from transformers import AutoModelForDepthEstimation
        self._torch = torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForDepthEstimation.from_pretrained(
            str(self.model_dir), local_files_only=True).to(dev).eval()
        self.device = dev
        self.provider = f"torch:{dev}"
        self._dmin = None   # 时域 min/max（GPU 标量张量，归一化用）
        self._dmax = None
        self._lock = threading.Lock()
        self._graph = None
        self._use_graph = use_graph and dev == "cuda"
        if self._use_graph:
            # 首次多模型推理会在独立 CUDA stream 中并发执行。Graph 捕获必须
            # 在提交这些任务前完成，否则别的模型发 kernel 会使捕获失败，
            # 甚至让本次 CUDA 上下文不可再用。
            from .cudagraph import GraphedCall
            sample = torch.zeros((1, 3, self.size, self.size), device=dev)
            self._graph = GraphedCall(self._forward, "depth_anything", [sample])

    def reset(self) -> None:
        """清空时域归一化状态（切换采集源时调用）。"""
        self._dmin = self._dmax = None

    # ------- 前向（CUDA Graph 化） -------

    def _forward(self, x):
        """(1,3,s,s) 归一化输入 → predicted_depth（ autocast fp16）。"""
        torch = self._torch
        with torch.no_grad(), torch.autocast(
                self.device, enabled=self.device == "cuda"):
            out = self.model(pixel_values=x)
        return out.predicted_depth

    def _forward_graphed(self, x):
        if self._graph is None:
            from .cudagraph import GraphedCall
            self._graph = GraphedCall(self._forward, "depth_anything", [x])
        return self._graph(x)

    # ------- 张量域主入口 -------

    def estimate_t(self, rgb_t, out_wh: tuple[int, int]):
        """GPU 张量主入口：RGB float [0,1] (1,3,H,W) → (1,1,h,w) [0,1]。

        全程驻留设备（预处理/归一化/回投都是张量算子），供引擎与 GPU
        效果链零拷贝衔接。
        """
        torch = self._torch
        import torch.nn.functional as F
        with self._lock:
            x = F.interpolate(rgb_t, size=(self.size, self.size),
                              mode="bilinear", align_corners=False)
            x = (x - 0.5) / 0.5
            if self._use_graph:
                disp = self._forward_graphed(x).float()
            else:
                disp = self._forward(x).float()
            if disp.dim() == 4:
                disp = disp[0, 0]
            elif disp.dim() == 3:
                disp = disp[0]
            # min/max 归一化（时域 EMA 防画面亮度/场景微变带来的闪动；
            # 用 GPU 标量张量维护，免逐帧 D2H 同步）
            dmin = disp.min()
            dmax = disp.max()
            if self._dmin is None or self._dmax is None:
                self._dmin, self._dmax = dmin, dmax
            else:
                a = NORM_EMA
                self._dmin = a * self._dmin + (1 - a) * dmin
                self._dmax = a * self._dmax + (1 - a) * dmax
            span = (self._dmax - self._dmin).clamp_min(1e-6)
            norm = (disp - self._dmin) / span
            depth = F.interpolate(norm[None, None], size=(out_wh[1], out_wh[0]),
                                  mode="bilinear", align_corners=False)
            return depth.clamp_(0, 1)

    def estimate(self, frame_bgr: np.ndarray,
                 out_wh: tuple[int, int]) -> np.ndarray:
        """numpy 兼容入口：整帧相对深度 (h, w) float32 [0,1]（值越大越近）。"""
        import cv2
        from .gpuops import device
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = self._torch.from_numpy(rgb).permute(2, 0, 1)[None] \
            .to(device()).float().div_(255.0)
        return self.estimate_t(t, out_wh)[0, 0].cpu().numpy() \
            .astype(np.float32)


_session: Optional[DepthSession] = None
_session_lock = threading.Lock()


def get_depth_session() -> DepthSession:
    """模块级单例（线程安全）。"""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = DepthSession()
    return _session
