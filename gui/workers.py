"""相机工作线程（QThread）。

线程纪律（AGENTS.md / PLAN §3.3）：本线程只做 采集 → 推理 → 效果链，
**永不触碰 GUI 控件**；帧/照片/状态一律经信号 emit 到主线程。
参数更新走 pipeline.set_params()（内部锁保护）。

一期两个 bug 的根治：
  - open_camera() 启动了两个 update_camera 线程 → v2 只有一个 worker 实例，
    start 前保证未运行。
  - v_sign_frames / smile_start_time 缺 global → 状态封装在
    AutoCaptureState 实例里，无全局变量。
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.camera import CameraSource
from core.gestures import AutoCaptureState, any_smiling, is_v_sign
from core.infer import InferenceEngine
from core.pipeline import NEED_FACES, NEED_HANDS, Pipeline

# 自动拍照的触发器名（与 GUI 复选框一一对应）
TRIGGER_MANUAL, TRIGGER_V_SIGN, TRIGGER_SMILE = "manual", "v_sign", "smile"

PHOTOS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "photos")

FPS_SMOOTHING = 0.9   # FPS 指数平滑系数


class CameraWorker(QThread):
    """采集 + 推理 + 效果链的工作线程。"""

    frame_ready = Signal(object)        # np.ndarray（BGR，处理后）
    photo_saved = Signal(str)           # 照片文件路径
    fps_changed = Signal(float)
    face_count_changed = Signal(int)    # 每帧人脸数（状态栏显示）
    status_message = Signal(str)
    source_finished = Signal()          # 视频/图片序列读完后发出
    failed = Signal(str)

    def __init__(self, source: CameraSource, pipeline: Pipeline,
                 engine: InferenceEngine,
                 triggers: Optional[set[str]] = None,
                 parent=None):
        super().__init__(parent)
        self.source = source
        self.pipeline = pipeline
        self.engine = engine
        # 启用的自动触发器集合（GUI 可随时改，集合赋值本身原子）
        self.triggers: set[str] = triggers if triggers is not None else set()
        self._stop_flag = False
        self._capture_request = False    # 手动拍照请求（跨线程标志）
        self._auto_state = AutoCaptureState()
        self._fps = 0.0
        self._last_frame_time = time.time()
        os.makedirs(PHOTOS_DIR, exist_ok=True)

    # ------- 外部控制（GUI 线程调用） -------

    def request_capture(self) -> None:
        """请求手动拍照（下一帧生效）。"""
        self._capture_request = True

    def stop(self) -> None:
        self._stop_flag = True
        self.wait(5000)

    # ------- 主循环 -------

    def run(self) -> None:
        if not self.source.open():
            self.failed.emit(f"无法打开采集源：{self.source.name}")
            return
        self.status_message.emit(f"已打开 {self.source.name}")
        try:
            self._loop()
        except Exception as exc:  # noqa: BLE001 —— 帧循环兜底，不让线程静默死掉
            self.failed.emit(f"相机线程异常：{exc}")
        finally:
            self.source.release()

    def _loop(self) -> None:
        while not self._stop_flag:
            ret, frame = self.source.read()
            if not ret or frame is None:
                self.source_finished.emit()
                break

            # 统一推理：效果链需求 ∨ 手势触发需求
            needs = self.pipeline.infer_needs()
            gesture_on = bool(self.triggers)
            if gesture_on:
                needs |= {NEED_FACES, NEED_HANDS}
            ctx = self.engine.process(
                frame,
                faces=NEED_FACES in needs,
                hands=NEED_HANDS in needs,
                segmentation="segmentation" in needs,
            )

            frame = self.pipeline.process(frame, ctx)
            self.face_count_changed.emit(len(ctx.faces))

            # 自动触发判定（时间持续 + 冷却，见 core/gestures）
            now = time.time()
            trigger = self._auto_state.update(
                now,
                v_sign=(TRIGGER_V_SIGN in self.triggers
                        and is_v_sign(ctx.hands, ctx.width, ctx.height)),
                smiling=(TRIGGER_SMILE in self.triggers and any_smiling(ctx.faces)),
            )
            if self._capture_request:
                self._capture_request = False
                trigger = TRIGGER_MANUAL
            if trigger is not None:
                self._save_photo(frame, trigger)

            # 帧发出（拷贝，防 QImage 读到复用缓冲）
            self.frame_ready.emit(np.ascontiguousarray(frame))

            # FPS 统计（指数平滑）
            t = time.time()
            inst = 1.0 / max(t - self._last_frame_time, 1e-6)
            if self._fps:
                self._fps = FPS_SMOOTHING * self._fps + (1 - FPS_SMOOTHING) * inst
            else:
                self._fps = inst
            self._last_frame_time = t
            self.fps_changed.emit(self._fps)

    def _save_photo(self, frame: np.ndarray, trigger: str) -> None:
        filename = os.path.join(
            PHOTOS_DIR,
            f"{trigger}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
        ok = cv2.imwrite(filename, frame,
                         [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            self.photo_saved.emit(filename)
            label = {"v_sign": "V手势", "smile": "笑脸", "manual": "手动"}[trigger]
            self.status_message.emit(f"📸 已保存（{label}）")
