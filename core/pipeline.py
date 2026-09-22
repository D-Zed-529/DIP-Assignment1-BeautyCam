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


class Effect(ABC):
    """效果基类：每个效果是 core/effects/ 下一个模块里的一个类。

    子类需提供 name、default_params()、process(frame, ctx)。
    """

    name: str = "base"
    # 该效果启用时需要的推理结果，如 {NEED_FACES}
    needs: frozenset[str] = frozenset()

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
    """有序效果链。process 前先由调用方（worker/CLI）统一跑推理填充 ctx。"""

    def __init__(self, effects: list[Effect]):
        self._effects: dict[str, Effect] = {}
        for e in effects:
            if e.name in self._effects:
                raise ValueError(f"效果重名: {e.name}")
            self._effects[e.name] = e
        self._order: list[str] = [e.name for e in effects]

    # ------- 帧处理 -------

    def process(self, frame: np.ndarray, ctx: FrameContext) -> np.ndarray:
        for name in self._order:
            e = self._effects[name]
            if not e.enabled:
                continue
            frame = e.process(frame, ctx)
        return frame

    # ------- 推理需求聚合 -------

    def infer_needs(self) -> set[str]:
        """所有启用中效果的推理需求并集（worker 与手势需求做 OR）。"""
        needs: set[str] = set()
        for name in self._order:
            e = self._effects[name]
            if e.enabled:
                needs |= set(e.needs)
        return needs

    # ------- 参数/开关（GUI 线程调用） -------

    def set_enabled(self, name: str, on: bool) -> None:
        self._effects[name].set_enabled(on)

    def set_params(self, name: str, **kwargs) -> None:
        self._effects[name].set_params(**kwargs)

    def get_effect(self, name: str) -> Optional[Effect]:
        return self._effects.get(name)

    def effects(self) -> list[Effect]:
        return [self._effects[n] for n in self._order]
