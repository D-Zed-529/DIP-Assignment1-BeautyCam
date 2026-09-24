"""FrameContext — 每帧共享的推理结果。

infer.py 每帧只跑一次推理，把人脸/手部/分割结果放进 FrameContext，
各效果（effects）从 ctx 读取，禁止效果内部重复起会话推理。

2026-09 性能重构：person_alpha / person_mask / depth 支持双形态承载 ——
torch 引擎直接存 GPU 张量（`*_t`），numpy 消费方首次访问时才物化并缓存
（GPU→CPU 一次，之后复用）；反向（mediapipe 引擎存 numpy，GPU 效果取
`*_t` 时才上传）同理。整条 GPU 效果链因此可以全程不落回 CPU。
属性赋值兼容两种类型：`ctx.person_alpha = np.ndarray` 或 `= torch.Tensor`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:                       # torch 是硬依赖，但 mediapipe 回退环境可能未装
    import torch as _torch
except ImportError:        # pragma: no cover - 仅无 torch 环境回退用
    _torch = None


@dataclass
class FaceInfo:
    """一张人脸的推理结果。

    landmarks: (468, 3) 归一化坐标（x, y, z），MediaPipe FaceMesh 口径
    box:      (x1, y1, x2, y2) 归一化人脸框，由轮廓关键点导出（已含下巴）
    smile:    微笑置信度 0~1（FaceBlendshapes 提供；为 None 时走嘴部张合回退判定）
    """

    landmarks: np.ndarray
    box: tuple[float, float, float, float]
    smile: Optional[float] = None


@dataclass
class HandInfo:
    """一只手的推理结果。landmarks: (21, 3) 归一化坐标。"""

    landmarks: np.ndarray
    # "Left"/"Right"（MediaPipe 按图像内容判定；预览画面已镜像时左右相反）
    handedness: str = ""


def _device_of(t):
    return t.device if _torch is not None and isinstance(t, _torch.Tensor) \
        else None


class FrameContext:
    """一帧的全部共享推理结果。

    构造参数与旧 dataclass 版完全兼容（关键字传入）；person_alpha /
    person_mask / depth 的读写见模块头说明（numpy 与张量懒互转）。
    """

    def __init__(self, frame_id: int = 0, timestamp_ms: int = 0,
                 width: int = 0, height: int = 0,
                 faces: Optional[list[FaceInfo]] = None,
                 hands: Optional[list[HandInfo]] = None,
                 person_alpha=None, person_mask=None, depth=None):
        self.frame_id = frame_id
        self.timestamp_ms = timestamp_ms
        self.width = width
        self.height = height
        self.faces: list[FaceInfo] = faces if faces is not None else []
        self.hands: list[HandInfo] = hands if hands is not None else []
        # 双形态缓存（同侧赋值会使另一侧失效）
        self._alpha_np: Optional[np.ndarray] = None
        self._alpha_t = None
        self._mask_np: Optional[np.ndarray] = None
        self._depth_np: Optional[np.ndarray] = None
        self._depth_t = None
        if person_alpha is not None:
            self.person_alpha = person_alpha
        if person_mask is not None:
            self.person_mask = person_mask
        if depth is not None:
            self.depth = depth

    # ------- 便捷查询 -------

    @property
    def has_face(self) -> bool:
        return len(self.faces) > 0

    @property
    def has_hand(self) -> bool:
        return len(self.hands) > 0

    def face_boxes_px(self) -> list[tuple[int, int, int, int]]:
        """所有人脸框转像素坐标 (x1, y1, x2, y2)。"""
        out = []
        for f in self.faces:
            x1, y1, x2, y2 = f.box
            out.append(
                (
                    int(max(x1 * self.width, 0)),
                    int(max(y1 * self.height, 0)),
                    int(min(x2 * self.width, self.width - 1)),
                    int(min(y2 * self.height, self.height - 1)),
                )
            )
        return out

    # ------- person_alpha：float32 (h, w) 0~1 前景概率 -------

    def _is_tensor(self, v) -> bool:
        return _torch is not None and isinstance(v, _torch.Tensor)

    @property
    def person_alpha(self) -> Optional[np.ndarray]:
        if self._alpha_np is None and self._alpha_t is not None:
            t = self._alpha_t.detach()
            if t.dim() == 4:
                t = t[0, 0]
            self._alpha_np = t.float().clamp(0.0, 1.0).cpu().numpy()
        return self._alpha_np

    @person_alpha.setter
    def person_alpha(self, v) -> None:
        if self._is_tensor(v):
            self._alpha_t = v.detach()
            self._alpha_np = None
        else:
            self._alpha_np = v
            self._alpha_t = None

    @property
    def person_alpha_t(self):
        """(1,1,h,w) float [0,1] GPU 张量视图（numpy 侧赋值时懒上传）。"""
        if self._alpha_t is None and self._alpha_np is not None and \
                _torch is not None:
            from .gpuops import device
            self._alpha_t = _torch.from_numpy(
                np.ascontiguousarray(self._alpha_np, dtype=np.float32)
            )[None, None].to(device())
        return self._alpha_t

    @person_alpha_t.setter
    def person_alpha_t(self, v) -> None:
        self.person_alpha = v   # 张量走同一路赋值逻辑

    # ------- person_mask：uint8 (h, w)，前景=255 背景=0 -------

    @property
    def person_mask(self) -> Optional[np.ndarray]:
        if self._mask_np is None and self._alpha_t is not None:
            # 由前景 alpha 阈值化派生（与 RVM/二元引擎口径一致：
            # person ⇔ alpha > 0.5；多分类的类别图引擎侧直接给 numpy）
            t = self._alpha_t.detach()
            if t.dim() == 4:
                t = t[0, 0]
            self._mask_np = ((t > 0.5).to(_torch.uint8) * 255).cpu().numpy()
        return self._mask_np

    @person_mask.setter
    def person_mask(self, v) -> None:
        if self._is_tensor(v):
            self._mask_np = ((v.detach() > 0).to(_torch.uint8) * 255
                             ).cpu().numpy()
        else:
            self._mask_np = v

    # ------- depth：float32 (h, w) [0,1] 相对深度（值越大越近） -------

    @property
    def depth(self) -> Optional[np.ndarray]:
        if self._depth_np is None and self._depth_t is not None:
            t = self._depth_t.detach()
            if t.dim() == 4:
                t = t[0, 0]
            self._depth_np = t.float().cpu().numpy()
        return self._depth_np

    @depth.setter
    def depth(self, v) -> None:
        if self._is_tensor(v):
            self._depth_t = v.detach()
            self._depth_np = None
        else:
            self._depth_np = v
            self._depth_t = None

    @property
    def depth_t(self):
        """(1,1,h,w) float [0,1] GPU 张量视图（numpy 侧赋值时懒上传）。"""
        if self._depth_t is None and self._depth_np is not None and \
                _torch is not None:
            from .gpuops import device
            self._depth_t = _torch.from_numpy(
                np.ascontiguousarray(self._depth_np, dtype=np.float32)
            )[None, None].to(device())
        return self._depth_t

    @depth_t.setter
    def depth_t(self, v) -> None:
        self.depth = v
