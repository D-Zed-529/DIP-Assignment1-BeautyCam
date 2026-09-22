"""采集源抽象：实时相机 / 视频文件 / 图片序列。

统一的 read() -> (ret, frame_bgr) 接口，供 GUI 工作线程与 headless 评测
CLI（scripts/run_pipeline.py）共用；弱光检测口径（灰度均值）也统一在这里。
"""

from __future__ import annotations

import glob
import os
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

# 预览固定分辨率：避免尺寸波动导致闪烁（一期经验）
CAP_WIDTH = 1280
CAP_HEIGHT = 720

# 弱光判定阈值：整帧灰度均值低于该值视为弱光（一期口径，Phase 2 将与
# 亮度自动开关协同/替换）
LOW_LIGHT_THRESHOLD = 60.0


def mean_brightness(frame_bgr: np.ndarray) -> float:
    """整帧灰度均值 —— 全项目统一的弱光检测口径。"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


class CameraSource(ABC):
    """采集源基类。帧一律为 BGR ndarray。"""

    @abstractmethod
    def open(self) -> bool:
        """打开源，返回是否成功。"""

    @abstractmethod
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """读取一帧，返回 (成功标志, BGR 帧)。"""

    @abstractmethod
    def release(self) -> None:
        """释放资源。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """源描述（用于日志/状态栏）。"""

    def __enter__(self) -> "CameraSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class LiveCamera(CameraSource):
    """实时摄像头。固定 1280×720，默认镜像（自拍语义）。"""

    def __init__(self, index: int = 0, mirror: bool = True,
                 width: int = CAP_WIDTH, height: int = CAP_HEIGHT):
        self.index = index
        self.mirror = mirror
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        # AVFoundation 后端显式指定，避免 OpenCV 误选其他后端
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # 曝露参数保持自动：macOS 上手动曝光不可控（见 PLAN §2 绕坑决策），
        # 一期曾设置的 CAP_PROP_EXPOSURE 在 AVFoundation 下无效且可能干扰
        return True

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def name(self) -> str:
        return f"摄像头#{self.index}"


class VideoFileSource(CameraSource):
    """视频文件源（评测/演示用）。可选循环回放。"""

    def __init__(self, path: str, loop: bool = False):
        self.path = path
        self.loop = loop
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.path)
        return self._cap.isOpened()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        if not ret:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            if not ret:
                return False, None
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 30.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0


class ImageSequenceSource(CameraSource):
    """图片序列源（评测批跑用）：目录（按文件名排序）或显式路径列表。"""

    EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, paths=None, directory: Optional[str] = None):
        if paths:
            self.paths = list(paths)
        elif directory:
            self.paths = sorted(
                p for p in glob.glob(os.path.join(directory, "*"))
                if p.lower().endswith(self.EXTS)
            )
        else:
            raise ValueError("paths 与 directory 必须提供其一")
        self._idx = 0

    def open(self) -> bool:
        self._idx = 0
        return len(self.paths) > 0

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._idx >= len(self.paths):
            return False, None
        path = self.paths[self._idx]
        self._idx += 1
        frame = cv2.imread(path)
        if frame is None:
            # 单张损坏图跳过，继续取下一张
            return self.read()
        return True, frame

    def release(self) -> None:
        pass

    @property
    def name(self) -> str:
        return f"图片序列×{len(self.paths)}"

    @property
    def current_path(self) -> Optional[str]:
        """最近一次 read 成功的图片路径（输出文件命名用）。"""
        return self.paths[self._idx - 1] if 0 < self._idx <= len(self.paths) else None
