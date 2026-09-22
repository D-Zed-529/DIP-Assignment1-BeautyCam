"""效果控制面板（QGroupBox 集合）：开关 + 滑杆。

每个面板只做两件事：读 pipeline 状态初始化控件、把控件变化写回
pipeline.set_params/set_enabled（线程安全）。不接触工作线程。
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QGridLayout, QGroupBox, QLabel, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

from core.pipeline import Pipeline


class SliderRow(QWidget):
    """标签 + 滑杆 + 数值显示一行。on_change(实数值) 回调写回参数。"""

    def __init__(self, label: str, lo: float, hi: float, value: float,
                 on_change: Callable[[float], None],
                 scale: float = 100.0, parent=None):
        super().__init__(parent)
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel(label), 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(lo * scale), int(hi * scale))
        self.slider.setValue(int(value * scale))
        self.value_label = QLabel(f"{value:g}")
        self.value_label.setFixedWidth(32)
        lay.addWidget(self.slider, 0, 1)
        lay.addWidget(self.value_label, 0, 2)
        self.slider.valueChanged.connect(
            lambda v: (self.value_label.setText(f"{v / scale:g}"),
                       on_change(v / scale)))


class BeautyPanel(QGroupBox):
    """美颜：总开关 + 磨皮/美白/瘦脸/大眼滑杆。"""

    def __init__(self, pipeline: Pipeline, parent=None):
        super().__init__("美颜", parent)
        self.pipeline = pipeline
        effect = pipeline.get_effect("beauty")
        p = effect.get_params()

        grid = QGridLayout(self)
        self.chk_enabled = QCheckBox("启用美颜")
        self.chk_enabled.setChecked(effect.enabled)
        self.chk_enabled.toggled.connect(
            lambda on: pipeline.set_enabled("beauty", on))
        grid.addWidget(self.chk_enabled, 0, 0, 1, 3)

        grid.addWidget(SliderRow("磨皮", 0.0, 1.0, p["smooth"],
                                 self._set("smooth")), 1, 0, 1, 3)
        grid.addWidget(SliderRow("美白", 0.0, 30.0, p["whiten"],
                                 self._set("whiten"), scale=1.0), 2, 0, 1, 3)
        self.chk_face_only = QCheckBox("美白仅限脸部")
        self.chk_face_only.setChecked(p["whiten_scope"] == "face")
        self.chk_face_only.setToolTip(
            "默认全身肤色美白（含脖子/手臂等皮肤）；勾选后仅脸部（一期口径）")
        self.chk_face_only.toggled.connect(
            lambda on: self.pipeline.set_params(
                "beauty", whiten_scope="face" if on else "skin"))
        grid.addWidget(self.chk_face_only, 3, 0, 1, 3)
        grid.addWidget(SliderRow("瘦脸", 0.0, 1.0, p["slim"],
                                 self._set("slim")), 4, 0, 1, 3)

        self.chk_eye = QCheckBox("大眼")
        self.chk_eye.setChecked(p["eye_enabled"])
        self.chk_eye.toggled.connect(self._set("eye_enabled"))
        grid.addWidget(self.chk_eye, 5, 0)
        grid.addWidget(SliderRow("大眼强度", 0.0, 0.5, p["eye_strength"],
                                 self._set("eye_strength")), 5, 1, 1, 2)

    def _set(self, key: str):
        return lambda v: self.pipeline.set_params("beauty", **{key: v})


class LowLightPanel(QGroupBox):
    """低光增强（启发式版，Phase 2 换深度模型）。"""

    def __init__(self, pipeline: Pipeline, parent=None):
        super().__init__("低光增强（启发式）", parent)
        self.pipeline = pipeline
        effect = pipeline.get_effect("lowlight")
        v = QVBoxLayout(self)
        self.chk_enabled = QCheckBox("启用")
        self.chk_enabled.setChecked(effect.enabled)
        self.chk_enabled.toggled.connect(
            lambda on: pipeline.set_enabled("lowlight", on))
        self.chk_auto = QCheckBox("仅暗光时自动增强（灰度均值 < 60）")
        self.chk_auto.setChecked(effect.get_params()["auto"])
        self.chk_auto.toggled.connect(
            lambda on: pipeline.set_params("lowlight", auto=on))
        v.addWidget(self.chk_enabled)
        v.addWidget(self.chk_auto)


class CapturePanel(QGroupBox):
    """拍照：手动按钮 + V 手势 / 笑脸自动触发。"""

    def __init__(self, on_manual: Callable[[], None], parent=None):
        super().__init__("拍照", parent)
        v = QVBoxLayout(self)
        self.btn_capture = QPushButton("📸 拍照（空格）")
        self.btn_capture.setEnabled(False)
        self.btn_capture.clicked.connect(on_manual)
        self.chk_vsign = QCheckBox("V 手势自动拍照（持续 1s）")
        self.chk_vsign.setChecked(True)
        self.chk_smile = QCheckBox("笑脸自动拍照（持续 0.5s）")
        self.chk_smile.setChecked(True)
        for w in (self.btn_capture, self.chk_vsign, self.chk_smile):
            v.addWidget(w)

    def triggers(self) -> set[str]:
        from gui.workers import TRIGGER_SMILE, TRIGGER_V_SIGN
        out = set()
        if self.chk_vsign.isChecked():
            out.add(TRIGGER_V_SIGN)
        if self.chk_smile.isChecked():
            out.add(TRIGGER_SMILE)
        return out
