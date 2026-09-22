"""FrameContext — 每帧共享的推理结果。

infer.py 每帧只跑一次推理，把人脸/手部/分割结果放进 FrameContext，
各效果（effects）从 ctx 读取，禁止效果内部重复起会话推理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


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


@dataclass
class FrameContext:
    """一帧的全部共享推理结果。"""

    frame_id: int = 0
    timestamp_ms: int = 0
    width: int = 0
    height: int = 0
    faces: list[FaceInfo] = field(default_factory=list)
    hands: list[HandInfo] = field(default_factory=list)
    # (h, w) uint8，前景=255 背景=0；未启用分割时为 None（Phase 3 使用）
    person_mask: Optional[np.ndarray] = None

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
