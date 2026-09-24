"""采集源抽象：实时相机 / 视频文件 / 图片序列。

统一的 read() -> (ret, frame_bgr) 接口，供 GUI 工作线程与 headless 评测
CLI（scripts/run_pipeline.py）共用；弱光检测口径（灰度均值）也统一在这里。
"""

from __future__ import annotations

import glob
import os
import sys
import threading
import time
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
    """整帧灰度均值 —— 全项目统一的弱光检测口径。

    大帧用 1/4 边长缩略图估计（INTER_AREA 是区域均值，均值统计几乎
    不变；弱光判定阈值 60 本身就是粗粒度开关）：720p 实测 0.4ms → 0.1ms。
    """
    h, w = frame_bgr.shape[:2]
    if h >= 4 and w >= 4:
        frame_bgr = cv2.resize(frame_bgr, (w // 4, h // 4),
                               interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


class CameraSource(ABC):
    """采集源基类。帧一律为 BGR ndarray。"""

    # 实时源（相机）读帧失败时重试而非结束；文件/序列源读尽即结束
    continuous: bool = False

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


def _camera_backend() -> int:
    """按平台选 OpenCV 采集后端。

    Windows 用 CAP_DSHOW（比默认 MSMF 打开快、枚举稳）；macOS 显式
    AVFoundation（误选其他后端会打不开，见 LiveCamera 类注释）；
    其余平台走 OpenCV 默认。
    """
    if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def scan_cameras(max_index: int = 4, warmup_reads: int = 2) -> list[dict]:
    """扫描本机可用摄像头，返回按索引升序的设备列表。

    每项：{"index": int, "width": int, "height": int}。逐个打开 + 试读
    一帧判定可用（isOpened 为 True 但出不了帧的虚拟设备会被过滤掉）。
    扫描是阻塞操作（每设备 open ~0.3-1s），GUI 里由按钮触发，不要在
    启动时自动全扫。max_index=4 已覆盖内置 + 外接的常见形态。
    """
    found: list[dict] = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, _camera_backend())
        ok = cap.isOpened()
        w = h = 0
        if ok:
            got = False
            for _ in range(warmup_reads):
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    got = True
                    break
            ok = got
        cap.release()
        if ok:
            found.append({"index": idx, "width": int(w), "height": int(h)})
    return found


class LiveCamera(CameraSource):
    """实时摄像头。固定 1280×720，默认镜像（自拍语义）。

    macOS 坑（本机实测）：
      - 摄像头权限（TCC）未授予宿主终端应用时 isOpened() 直接 False，
        stderr 出现 "not authorized to capture video"（打开失败的主因）；
      - 授权后首次 open 也可能初始化不完全，需要重试几次；
      - 前几帧常为空（AVFoundation 会话启动中），read() 需预热。
    Windows：后端走 DSHOW（见 _camera_backend），打开/枚举比 MSMF 稳。
    """

    # open() 失败后的重试次数与间隔（授权弹窗确认存在竞态窗口）
    OPEN_RETRIES = 3
    OPEN_RETRY_DELAY = 0.8
    WARMUP_READS = 5          # open 后预读帧数（丢掉启动空帧）
    READ_WAIT_S = 0.3         # 采集线程暂时无帧时，消费端单次等待上限
    continuous = True          # 实时源：偶发空帧重试而非结束

    def __init__(self, index: int = 0, mirror: bool = True,
                 width: int = CAP_WIDTH, height: int = CAP_HEIGHT):
        self.index = index
        self.mirror = mirror
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_stop = threading.Event()
        self._frame_ready = threading.Condition()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_seq = 0
        self._read_seq = 0
        self.last_error: str = ""

    def open(self) -> bool:
        backend = _camera_backend()
        for attempt in range(self.OPEN_RETRIES):
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                cap.release()
                self.last_error = (
                    f"摄像头#{self.index} 打不开（第 {attempt + 1} 次）。"
                    "Windows 常见原因：设备被占用或索引不对（点「扫描摄像头」"
                    "看可用列表）；macOS 常见原因：未授权（系统设置 → 隐私与"
                    "安全性 → 摄像头，勾选运行 Python 的终端应用后重试）。")
                time.sleep(self.OPEN_RETRY_DELAY)
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # 后端支持时限制驱动缓冲，防止处理慢时排队旧帧；不支持则由下方
            # 独立采集线程持续读取并覆盖旧帧。
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # 预热：会话启动中前几帧为空
            first_frame = None
            for _ in range(self.WARMUP_READS):
                ret, first_frame = cap.read()
                if ret:
                    break
                time.sleep(0.1)
            self._cap = cap
            with self._frame_ready:
                self._latest_frame = first_frame if ret else None
                self._latest_seq = 1 if ret else 0
                self._read_seq = 0
            self._capture_stop.clear()
            self._capture_thread = threading.Thread(
                target=self._capture_loop, args=(cap,), daemon=True,
                name=f"BeautyCam-摄像头{self.index}")
            self._capture_thread.start()
            self.last_error = ""
            return True
        return False

    def _capture_loop(self, cap: cv2.VideoCapture) -> None:
        """始终读取设备新帧，消费端只领取最新一张。"""
        while not self._capture_stop.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            with self._frame_ready:
                self._latest_frame = frame
                self._latest_seq += 1
                self._frame_ready.notify_all()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        with self._frame_ready:
            fresh = self._frame_ready.wait_for(
                lambda: self._latest_seq > self._read_seq
                or self._capture_stop.is_set(), timeout=self.READ_WAIT_S)
            if not fresh or self._latest_seq <= self._read_seq:
                return False, None
            frame = self._latest_frame
            self._read_seq = self._latest_seq
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._capture_stop.set()
            with self._frame_ready:
                self._frame_ready.notify_all()
            if self._capture_thread is not None:
                self._capture_thread.join(timeout=1.0)
                self._capture_thread = None
            self._cap.release()
            self._cap = None
            with self._frame_ready:
                self._latest_frame = None

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
