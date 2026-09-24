"""CUDA Graph 回放封装 —— 消除 Windows WDDM 下小算子图的启动开销。

背景（2026-09 性能剖析，scripts/profile_stages.py）：tflite 逐算子转换出的
TorchScript 图（face_landmarks 256² 等）每帧前向要发起数百个微型 kernel；
Linux 上每次启动 ~3μs 无感，Windows WDDM 却要 ~15-25μs，单次前向被启动
开销堆到 54ms（face_landmarks）/17ms（face_detector）——GPU 本身几乎空闲。
CUDA Graph 把整段前向捕获成一张图，逐帧只需一次 `graph.replay()`，
启动开销归零，剩余耗时 ≈ 真实 GPU 计算（实测 face_landmarks 54→~5ms）。

使用约束（本仓库内的调用方都已满足）：
  - 输入形状/类型/设备必须静态（每个模型固定输入尺寸，正好如此）；
  - 前向内部不能有依赖张量**数值**的 CPU 分支（`.item()`/`if tensor`）——
    解码/门限判断都在捕获之外做；
  - 输入张量每次 copy_ 进静态缓冲；输出每次克隆返回（静态输出区会被
    下一次 replay 覆写，调用方若跨帧持有输出必须拿到独立副本——RVM 的
    循环状态就是这样回灌的）。

无 CUDA / 捕获失败时自动退化为直调（行为不变，只是慢）。
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Sequence

import torch

_GRAPH_WARNED: set[str] = set()
_LOCK = threading.Lock()


def _clone_out(out: Any) -> Any:
    """结构保持地克隆捕获输出的张量叶节点。"""
    if isinstance(out, torch.Tensor):
        return out.clone()
    if isinstance(out, (list, tuple)):
        chained = [_clone_out(o) for o in out]
        return type(out)(chained) if isinstance(out, tuple) else chained
    if isinstance(out, dict):
        return {k: _clone_out(v) for k, v in out.items()}
    return out


class GraphedCall:
    """fn(*tensor_args) 的 CUDA Graph 化调用器（fn 须无数值依赖分支）。

    首次 __call__ 前需已构造；构造即完成预热 + 捕获（约多花 3 次前向）。
    """

    def __init__(self, fn: Callable[..., Any], name: str,
                 sample_args: Sequence[torch.Tensor], warmup: int = 3):
        self._fn = fn
        self.name = name
        self._static_in = [a.detach().clone() for a in sample_args]
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_out: Any = None
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(warmup):
                out = fn(*self._static_in)
        torch.cuda.current_stream().wait_stream(stream)
        try:
            with torch.no_grad():
                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph):
                    self._static_out = fn(*self._static_in)
        except Exception as exc:   # noqa: BLE001 —— 捕获失败退直调，不阻塞功能
            self._graph = None
            with _LOCK:
                if name not in _GRAPH_WARNED:
                    _GRAPH_WARNED.add(name)
                    print(f"[CUDA Graph] {name} 捕获失败，退化为直调：{exc}")

    def __call__(self, *args: torch.Tensor) -> Any:
        if self._graph is None:
            with torch.no_grad():
                return _clone_out(self._fn(*args))
        # Graph 可能在 GUI 的 inference_mode 中首次创建，静态缓冲随之
        # 成为 inference tensor；离线脚本随后直调时仍需允许写入该缓冲。
        with torch.inference_mode():
            for buf, a in zip(self._static_in, args):
                buf.copy_(a, non_blocking=True)
            self._graph.replay()
            return _clone_out(self._static_out)


def graph_call(fn: Callable[..., Any], name: str,
               sample_args: Sequence[torch.Tensor],
               enabled: bool = True) -> Callable[..., Any]:
    """便捷工厂：CUDA 可用且 enabled 时返回 GraphedCall，否则原样返回 fn。

    sample_args 只取形状/类型/设备做静态缓冲模板（值会被覆写）。
    """
    if not (enabled and torch.cuda.is_available()):
        return fn
    return GraphedCall(fn, name, sample_args)
