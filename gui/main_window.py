"""PySide6 主窗口：视频区 / 效果控制面板 / 拍照预览条。

运行：python -m gui.main_window（需摄像头权限 + 本地 GUI）。
无头环境只能验证 import 与纯函数逻辑（AGENTS.md 验证方式）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget,
)

from core.camera import LiveCamera, VideoFileSource
from core.effects.beauty import BeautyEffect
from core.effects.lowlight import LowLightEffect
from core.infer import get_engine
from core.pipeline import Pipeline
from gui.panels import BeautyPanel, CapturePanel, LowLightPanel
from gui.workers import CameraWorker, PHOTOS_DIR

RECENT_PHOTOS = 4          # 预览条缩略图数量（一期口径）
THUMB_SIZE = (152, 100)    # 缩略图尺寸
VIDEO_VIEW = (960, 540)    # 视频显示区逻辑尺寸


def ndarray_to_pixmap(frame_bgr: np.ndarray, size: tuple[int, int]) -> QPixmap:
    """BGR ndarray -> 等比缩放 QPixmap。"""
    h, w = frame_bgr.shape[:2]
    img = QImage(frame_bgr.data, w, h, 3 * w, QImage.Format.Format_BGR888)
    pix = QPixmap.fromImage(img)
    return pix.scaled(*size, Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.SmoothTransformation)


class PhotoDialog(QDialog):
    """点击缩略图后的放大预览（一期 show_large_image 迁移）。"""

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("照片预览")
        img = cv2.imread(path)
        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        screen = self.screen().availableGeometry() if self.screen() else None
        max_w = int(screen.width() * 0.8) if screen else 1200
        max_h = int(screen.height() * 0.85) if screen else 800
        pix = pix.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
        lbl = QLabel()
        lbl.setPixmap(pix)
        lbl.setStyleSheet("background-color:#000;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(lbl)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BeautyCam v2")
        self.resize(1280, 760)

        # 管线：低光增强 → 美颜（HDR 为拍照模式，不进逐帧链）
        self.pipeline = Pipeline([
            LowLightEffect(enabled=False),
            BeautyEffect(enabled=True),
        ])
        self.engine = get_engine()
        self.worker: Optional[CameraWorker] = None
        self._last_faces = 0

        self._thumbs: list[QLabel] = []
        self._recent_paths: list[str] = []
        self._build_ui()
        self._load_recent_photos()

    # ---------------- UI 构建 ----------------

    def _build_ui(self) -> None:
        central = QWidget(objectName="root")
        self.setCentralWidget(central)
        root_lay = QVBoxLayout(central)

        top = QHBoxLayout()
        root_lay.addLayout(top, stretch=1)

        self.video_label = QLabel("尚未开启相机")
        self.video_label.setObjectName("videoLabel")
        self.video_label.setFixedSize(*VIDEO_VIEW)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignTop)

        top.addWidget(self._build_control_panel())

        # 拍照预览条
        preview = QGroupBox("拍照预览（点击放大）")
        strip = QHBoxLayout(preview)
        for i in range(RECENT_PHOTOS):
            lbl = QLabel()
            lbl.setObjectName("thumbLabel")
            lbl.setFixedSize(*THUMB_SIZE)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setCursor(Qt.CursorShape.PointingHandCursor)
            lbl.mousePressEvent = lambda e, idx=i: self._open_thumb(idx)
            self._thumbs.append(lbl)
            strip.addWidget(lbl)
        strip.addStretch(1)
        open_album = QPushButton("打开相册")
        open_album.clicked.connect(self._open_album)
        strip.addWidget(open_album, alignment=Qt.AlignmentFlag.AlignBottom)
        root_lay.addWidget(preview)

        self.statusBar().showMessage("就绪 —— 选择采集源并点击「开始」")

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(320)
        lay = QVBoxLayout(panel)
        lay.setSpacing(10)

        # ---- 采集源 ----
        src_box = QGroupBox("采集源")
        src_lay = QGridLayout(src_box)
        self.cam_index = QSpinBox()
        self.cam_index.setRange(0, 9)
        src_lay.addWidget(QLabel("摄像头"), 0, 0)
        src_lay.addWidget(self.cam_index, 0, 1)
        btn_video = QPushButton("选择视频文件…")
        btn_video.clicked.connect(self._pick_video)
        src_lay.addWidget(btn_video, 1, 0, 1, 2)
        self.btn_toggle = QPushButton("开始")
        self.btn_toggle.setObjectName("dangerBtn")
        self.btn_toggle.clicked.connect(self._toggle_camera)
        src_lay.addWidget(self.btn_toggle, 2, 0, 1, 2)
        self.selected_video: Optional[str] = None
        lay.addWidget(src_box)

        self.beauty_panel = BeautyPanel(self.pipeline)
        lay.addWidget(self.beauty_panel)
        self.lowlight_panel = LowLightPanel(self.pipeline)
        lay.addWidget(self.lowlight_panel)
        self.capture_panel = CapturePanel(self._manual_capture)
        lay.addWidget(self.capture_panel)

        lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        return scroll

    # ---------------- 相机控制 ----------------

    def _toggle_camera(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self._stop_worker()
            return
        if self.selected_video:
            source = VideoFileSource(self.selected_video)
        else:
            source = LiveCamera(index=self.cam_index.value(), mirror=True)
        # 一期双线程 bug 根治：单 worker 实例，先建后启动
        self.worker = CameraWorker(
            source, self.pipeline, self.engine,
            triggers=self.capture_panel.triggers())
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.photo_saved.connect(self._on_photo_saved)
        self.worker.fps_changed.connect(self._on_fps)
        self.worker.face_count_changed.connect(self._on_face_count)
        self.worker.status_message.connect(self.statusBar().showMessage)
        self.worker.source_finished.connect(self._on_source_finished)
        self.worker.failed.connect(self._on_worker_failed)
        self.btn_toggle.setText("停止")
        self.capture_panel.btn_capture.setEnabled(True)
        self.worker.start()

    def _stop_worker(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.btn_toggle.setText("开始")
        self.capture_panel.btn_capture.setEnabled(False)
        self.video_label.setText("相机已停止")
        self.statusBar().showMessage("相机已停止")

    # ---------------- 信号槽（主线程） ----------------

    def _on_frame(self, frame: np.ndarray) -> None:
        self.video_label.setPixmap(ndarray_to_pixmap(frame, VIDEO_VIEW))

    def _on_face_count(self, n: int) -> None:
        self._last_faces = n

    def _on_fps(self, fps: float) -> None:
        lowlight = self.pipeline.get_effect("lowlight")
        dark = " · 暗光增强中" if (lowlight.enabled and lowlight.is_dark) else ""
        self.statusBar().showMessage(f"{fps:5.1f} fps · 人脸 {self._last_faces}{dark}")

    def _on_photo_saved(self, path: str) -> None:
        QApplication.beep()   # 快门提示音（一期 root.bell 等价物）
        self._recent_paths.insert(0, path)
        self._recent_paths = self._recent_paths[:RECENT_PHOTOS]
        self._refresh_thumbs()

    def _on_source_finished(self) -> None:
        self.statusBar().showMessage("素材播放完毕")
        self._stop_worker()

    def _on_worker_failed(self, msg: str) -> None:
        QMessageBox.warning(self, "错误", msg)
        self._stop_worker()

    # ---------------- 预览条 ----------------

    def _load_recent_photos(self) -> None:
        if not os.path.isdir(PHOTOS_DIR):
            return
        files = [os.path.join(PHOTOS_DIR, f)
                 for f in os.listdir(PHOTOS_DIR)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files.sort(key=os.path.getmtime, reverse=True)
        self._recent_paths = files[:RECENT_PHOTOS]
        self._refresh_thumbs()

    def _refresh_thumbs(self) -> None:
        for i, lbl in enumerate(self._thumbs):
            if i < len(self._recent_paths):
                img = cv2.imread(self._recent_paths[i])
                if img is not None:
                    lbl.setPixmap(ndarray_to_pixmap(img, THUMB_SIZE))
                    continue
            lbl.clear()

    def _open_thumb(self, idx: int) -> None:
        if idx < len(self._recent_paths):
            PhotoDialog(self._recent_paths[idx], self).exec()

    def _open_album(self) -> None:
        os.makedirs(PHOTOS_DIR, exist_ok=True)
        subprocess.Popen(["open", PHOTOS_DIR])

    # ---------------- 其他 ----------------

    def _pick_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频 (*.mp4 *.mov *.avi *.mkv)")
        if path:
            self.selected_video = path
            self.statusBar().showMessage(f"已选视频：{os.path.basename(path)}")

    def _manual_capture(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_capture()
        # 触发器复选框变化同步给 worker（collection 赋值原子，无需锁）
        if self.worker is not None:
            self.worker.triggers = self.capture_panel.triggers()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._manual_capture()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._stop_worker()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    qss_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theme.qss")
    with open(qss_path, encoding="utf-8") as f:
        app.setStyleSheet(f.read())
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
