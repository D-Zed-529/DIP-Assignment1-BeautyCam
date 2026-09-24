"""效果链：有序 Effect 列表 + 线程安全参数更新。

线程模型：GUI 线程写（set_params / set_enabled），相机工作线程读（process）。
参数更新用"整字典替换"而非原地改键，读取方拿到的是不可变快照，无需加锁
拷贝 —— 避免边跑边改的竞态（PLAN §3.2）。
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .context import FrameContext

# 效果可声明的推理需求（决定 infer.process 每帧跑哪些模型）
NEED_FACES = "faces"
NEED_HANDS = "hands"
NEED_SEGMENTATION = "segmentation"
NEED_DEPTH = "depth"


class Effect(ABC):
    """效果基类：每个效果是 core/effects/ 下一个模块里的一个类。

    子类需提供 name、default_params()、process(frame, ctx)。
    可选 GPU 快路径：supports_gpu=True 且实现 process_gpu(frame_t, ctx)
    （(1,3,H,W) uint8 BGR 设备张量进出）。Pipeline(use_gpu=True) 时，
    连续的 GPU 效果共享同一次帧上传/下载（整链常驻显存，见
    core/effects/_torch_impl.py 模块头）；process_gpu 抛 NotImplementedError
    时该效果自动回退 CPU 路径（如低光 SCI 的 ONNX 会话）。
    """

    name: str = "base"
    # 该效果启用时需要的推理结果，如 {NEED_FACES}
    needs: frozenset[str] = frozenset()
    # 是否具备 GPU 快路径（类级声明；运行期不可用由 process_gpu 自行回退）
    supports_gpu = False

    def __init__(self, enabled: bool = True, params: Optional[dict] = None):
        self._lock = threading.Lock()
        self._enabled = enabled
        merged = dict(self.default_params())
        if params:
            unknown = set(params) - set(merged)
            if unknown:
                raise ValueError(f"效果 {self.name} 未知参数: {unknown}")
            merged.update(params)
        self._params = merged

    # ------- 子类接口 -------

    @staticmethod
    @abstractmethod
    def default_params() -> dict:
        """参数默认值字典（GUI 滑杆范围据此对齐）。"""

    @abstractmethod
    def process(self, frame: np.ndarray, ctx: FrameContext) -> np.ndarray:
        """处理一帧（BGR），返回处理后的帧。ctx 为本帧共享推理结果。"""

    def process_gpu(self, frame_t, ctx: FrameContext):
        """GPU 快路径：(1,3,H,W) uint8 BGR 设备张量进出。

        默认不支持（supports_gpu=False 的效果不会被调到这里）；具备快
        路径但在当前参数/会话下不可用（如低光 SCI 走 ONNX 会话）时抛
        NotImplementedError，Pipeline 会下载回 CPU 走 process()。
        """
        raise NotImplementedError

    def inference_interval(self, need: str) -> int:
        """该效果对某类推理的调用间隔（帧数）。1 = 每帧都跑；N = 每 N 帧跑一次。

        用于重推理降载（PLAN §3.3）：代价高的模型隔帧跑，中间帧由效果自身的
        时域状态复用上次结果（如 SegmentEffect 复用上一帧的 alpha）。
        声明为方法而非类属性，便于按参数（如所选的模型）动态决定。
        """
        return 1

    def reset_temporal(self) -> None:
        """清空跨帧状态（切换采集源、逐张独立批跑之前调用）。

        无状态的效果不必实现（默认空实现）。切换采集源时尺寸/场景突变，
        残留的时域状态（缓存的掩膜、背景图等）必须作废，否则会闪一帧旧结果
        甚至因尺寸不符让帧循环抛异常。
        """

    # ------- 参数读写（线程安全） -------

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, on: bool) -> None:
        with self._lock:
            self._enabled = on

    def get_params(self) -> dict:
        """参数快照（浅拷贝），供 GUI 读回显。"""
        with self._lock:
            return dict(self._params)

    def set_params(self, **kwargs) -> None:
        unknown = set(kwargs) - set(self.default_params())
        if unknown:
            raise ValueError(f"效果 {self.name} 未知参数: {unknown}")
        with self._lock:
            merged = dict(self._params)
            merged.update(kwargs)
            self._params = merged   # 整字典替换：读方快照一致性

    def _p(self) -> dict:
        """工作线程内部取参数快照（process 里调用）。"""
        with self._lock:
            return self._params


class Pipeline:
    """有序效果链。process 前先由调用方（worker/CLI）统一跑推理填充 ctx。

    use_gpu=True 时启用效果链 GPU 融合：连续的 supports_gpu 效果在同一次
    帧上传/下载里直通（frame_t 在效果间以设备张量传递，见 Effect.process_gpu），
    遇 CPU 效果自动落回 numpy。默认 False（CPU 路径是逐位基准，测试/评测
    默认走它）。
    """

    def __init__(self, effects: list[Effect], use_gpu: bool = False):
        self._effects: dict[str, Effect] = {}
        for e in effects:
            if e.name in self._effects:
                raise ValueError(f"效果重名: {e.name}")
            self._effects[e.name] = e
        self._order: list[str] = [e.name for e in effects]
        self._use_gpu = bool(use_gpu)

    # ------- 帧处理 -------

    def process(self, frame: np.ndarray, ctx: FrameContext) -> np.ndarray:
        if not self._use_gpu:
            for name in self._order:
                e = self._effects[name]
                if not e.enabled:
                    continue
                frame = e.process(frame, ctx)
            return frame
        from .gpuops import download_frame, upload_frame
        pending = None              # 当前帧的 GPU 形态（无则 frame 为准）
        for name in self._order:
            e = self._effects[name]
            if not e.enabled:
                continue
            if e.supports_gpu:
                if pending is None:
                    pending = upload_frame(frame)
                try:
                    pending = e.process_gpu(pending, ctx)
                    continue
                except NotImplementedError:
                    pass             # 该效果当前不可 GPU 化：落回 CPU
                frame = download_frame(pending)
                pending = None
            frame = e.process(frame, ctx)
        if pending is not None:
            frame = download_frame(pending)
        return frame

    # ------- 推理需求聚合 -------

    def infer_needs(self) -> set[str]:
        """所有启用中效果的推理需求并集。

        等价于 infer_needs_for(0)（第 0 帧必跑），保留此签名给不关心隔帧的调用方。
        """
        return self.infer_needs_for(0)

    def infer_needs_for(self, frame_index: int) -> set[str]:
        """本帧实际要跑的推理需求（已计入各效果的 inference_interval 隔帧降载）。

        frame_index 从 0 起；0 % N == 0 恒成立，因此首帧一定跑全量推理，
        不会出现"刚打开效果却拿不到掩膜"的空窗。
        """
        needs: set[str] = set()
        for name in self._order:
            e = self._effects[name]
            if not e.enabled:
                continue
            for need in e.needs:
                interval = max(1, int(e.inference_interval(need)))
                if frame_index % interval == 0:
                    needs.add(need)
        return needs

    def reset_temporal(self) -> None:
        """清空全部效果的跨帧状态（切换采集源 / 逐张独立批跑前调用）。"""
        for name in self._order:
            self._effects[name].reset_temporal()

    # ------- 参数/开关（GUI 线程调用） -------

    def set_enabled(self, name: str, on: bool) -> None:
        self._effects[name].set_enabled(on)

    def set_params(self, name: str, **kwargs) -> None:
        self._effects[name].set_params(**kwargs)

    def get_effect(self, name: str) -> Optional[Effect]:
        return self._effects.get(name)

    def effects(self) -> list[Effect]:
        return [self._effects[n] for n in self._order]
