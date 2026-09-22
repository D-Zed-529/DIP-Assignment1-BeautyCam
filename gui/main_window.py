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
from core.effects.autoenhance import AutoEnhanceEffect
from core.effects.beauty import BeautyEffect
from core.effects.lowlight import LowLightDnnEffect, LowLightEffect
from core.effects.segment import SegmentEffect
from core.infer import SEGMENTER_INTERVAL, get_engine
from core.pipeline import Pipeline
from gui.panels import (
    AutoEnhancePanel, BeautyPanel, CapturePanel, HdrPanel, LowLightPanel,
    SegmentPanel,
)
from gui.workers import CameraWorker, DISPLAY_SIZE, PHOTOS_DIR

RECENT_PHOTOS = 4          # 预览条缩略图数量（一期口径）
THUMB_SIZE = (152, 100)    # 缩略图尺寸
VIDEO_VIEW = DISPLAY_SIZE  # 视频显示区逻辑尺寸（与 worker 预缩放一致）


def ndarray_to_pixmap(frame_bgr: np.ndarray, size: tuple[int, int]) -> QPixmap:
    """BGR ndarray -> 等比缩放 QPixmap（缩略图等小图用；视频帧走零缩放路径）。"""
    h, w = frame_bgr.shape[:2]
    img = QImage(frame_bgr.data, w, h, 3 * w, QImage.Format.Format_BGR888)
    pix = QPixmap.fromImage(img)
    return pix.scaled(*size, Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.SmoothTransformation)


def frame_to_pixmap(frame_bgr: np.ndarray) -> QPixmap:
    """worker 预缩放后的帧 -> QPixmap（零缩放，主线程只剩包装开销）。"""
    h, w = frame_bgr.shape[:2]
    img = QImage(frame_bgr.data, w, h, 3 * w, QImage.Format.Format_BGR888)
    return QPixmap.fromImage(img)


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
        self.setWindowTitle("BeautyCam v2 — 多效果实时相机")
        self.resize(1360, 800)

        # 管线：自适应画质（链首，先校正曝光/色调，低光的暗光判定看到的是
        # 校正后的亮度）→ 低光（启发式/SCI 互斥，默认都关）→ 美颜 → 虚化/替换。
        # 顺序对齐 PLAN §3.2；HDR 是拍照模式不进链（gui/workers.py 连拍）。
        self.pipeline = Pipeline([
            AutoEnhanceEffect(enabled=False),
            LowLightEffect(enabled=False),
            LowLightDnnEffect(enabled=False),
            BeautyEffect(enabled=True),
            SegmentEffect(enabled=False),
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
        root_lay.setSpacing(10)

        # ---- 顶部 header：标题 + 实时状态徽章 ----
        header = QWidget(objectName="header")
        hlay = QHBoxLayout(header)
        title = QLabel("BeautyCam v2")
        title.setObjectName("appTitle")
        subtitle = QLabel("DIP 课程项目 · 美颜 / 虚化 / HDR / 低光增强")
        subtitle.setObjectName("appSubtitle")
        hlay.addWidget(title)
        hlay.addWidget(subtitle)
        hlay.addStretch(1)
        self.badge_dark = QLabel("暗光")
        self.badge_dark.setObjectName("badgeWarn")
        self.badge_dark.setToolTip("低光增强生效中（亮度 < 阈值）")
        self.badge_dark.hide()
        self.badge_faces = QLabel("人脸 0")
        self.badge_faces.setObjectName("badge")
        self.badge_fps = QLabel("-- fps")
        self.badge_fps.setObjectName("badge")
        for b in (self.badge_dark, self.badge_faces, self.badge_fps):
            hlay.addWidget(b)
        root_lay.addWidget(header)

        top = QHBoxLayout()
        top.setSpacing(10)
        root_lay.addLayout(top, stretch=1)

        self.video_label = QLabel("尚未开启相机")
        self.video_label.setObjectName("videoLabel")
        self.video_label.setFixedSize(*VIDEO_VIEW)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignTop)

        top.addWidget(self._build_control_panel())

        # ---- 拍照预览条 ----
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
        panel.setFixedWidth(330)
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
        self.btn_toggle.setObjectName("primaryBtn")
        self.btn_toggle.clicked.connect(self._toggle_camera)
        src_lay.addWidget(self.btn_toggle, 2, 0, 1, 2)
        self.selected_video: Optional[str] = None
        lay.addWidget(src_box)

        self.beauty_panel = BeautyPanel(self.pipeline)
        lay.addWidget(self.beauty_panel)
        self.autoenhance_panel = AutoEnhancePanel(self.pipeline)
        lay.addWidget(self.autoenhance_panel)
        self.lowlight_panel = LowLightPanel(self.pipeline)
        lay.addWidget(self.lowlight_panel)
        self.segment_panel = SegmentPanel(self.pipeline, self._on_model_change)
        lay.addWidget(self.segment_panel)
        self.hdr_panel = HdrPanel(self._hdr_capture)
        lay.addWidget(self.hdr_panel)
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
        self.btn_toggle.setObjectName("dangerBtn")
        self.btn_toggle.style().unpolish(self.btn_toggle)
        self.btn_toggle.style().polish(self.btn_toggle)
        self.capture_panel.btn_capture.setEnabled(True)
        self.hdr_panel.set_running(True)
        self.worker.start()

    def _stop_worker(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.btn_toggle.setText("开始")
        self.btn_toggle.setObjectName("primaryBtn")
        self.btn_toggle.style().unpolish(self.btn_toggle)
        self.btn_toggle.style().polish(self.btn_toggle)
        self.capture_panel.btn_capture.setEnabled(False)
        self.hdr_panel.set_running(False)
        self.video_label.setText("相机已停止")
        self.badge_fps.setText("-- fps")
        self.statusBar().showMessage("相机已停止")

    # ---------------- 信号槽（主线程） ----------------

    def _on_frame(self, frame: np.ndarray) -> None:
        # worker 已按显示尺寸预缩放：主线程零缩放，只做 QImage 包装
        self.video_label.setPixmap(frame_to_pixmap(frame))

    def _on_face_count(self, n: int) -> None:
        self._last_faces = n
        self.badge_faces.setText(f"人脸 {n}")

    def _on_fps(self, fps: float) -> None:
        self.badge_fps.setText(f"{fps:5.1f} fps")
        # 档位变化才切换样式（每帧 polish/unpolish 会让主线程做无谓的重绘）
        tier = ("badgeOk" if fps >= 24 else
                "badgeWarn" if fps >= 14 else "badgeBad")
        if tier != self.badge_fps.objectName():
            self.badge_fps.setObjectName(tier)
            self.badge_fps.style().unpolish(self.badge_fps)
            self.badge_fps.style().polish(self.badge_fps)
        low_heur = self.pipeline.get_effect("lowlight")
        low_dnn = self.pipeline.get_effect("lowlight_dnn")
        dark = ((low_heur.enabled and low_heur.is_dark)
                or (low_dnn.enabled and low_dnn.is_dark))
        self.badge_dark.setVisible(dark)

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

    def _hdr_capture(self, ev_preset: str, tonemap: object) -> None:
        """HDR 连拍入口（HdrPanel 回调；tonemap None=直出 Mertens）。"""
        if self.worker is None or not self.worker.isRunning():
            self.statusBar().showMessage("先「开始」相机再连拍 HDR")
            return
        self.worker.triggers = self.capture_panel.triggers()
        self.worker.request_hdr_capture(ev_preset, tonemap)

    def _on_model_change(self, key: str) -> None:
        """切换分割模型：会话重建交给工作线程，同时同步隔帧间隔。

        间隔必须跟着模型走 —— 二元 13ms 可每帧跑，多分类 155ms 必须隔帧，
        否则三开会从 14fps 掉到 7fps 以下。
        """
        interval = SEGMENTER_INTERVAL[key]
        self.pipeline.set_params("segment", infer_interval=interval)
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_segmenter_model(key)
            self.statusBar().showMessage(
                f"分割模型将切换为 {key}（推理间隔 {interval} 帧）")
        else:
            # 没在跑就不用担心撞上推理，直接切
            self.engine.set_segmenter_model(key)
            self.statusBar().showMessage(
                f"分割模型已切换为 {key}（推理间隔 {interval} 帧）")

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
