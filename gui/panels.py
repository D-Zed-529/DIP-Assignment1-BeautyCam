"""效果控制面板（QGroupBox 集合）：开关 + 滑杆。

每个面板只做两件事：读 pipeline 状态初始化控件、把控件变化写回
pipeline.set_params/set_enabled（线程安全）。不接触工作线程。
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QPushButton, QSlider, QVBoxLayout,
    QWidget,
)

from core.effects.segment import (
    MODE_BLUR, MODE_COLOR, MODE_IMAGE, list_backgrounds, load_image,
)
from core.infer import DEFAULT_SEGMENTER, SEGMENTER_SPECS
from core.pipeline import Pipeline

# 背景图库缩略图尺寸
BG_THUMB = (72, 41)

# 纯色背景预设：(显示名, 十六进制)。绿幕色便于后续做抠像演示
COLOR_PRESETS = [
    ("低饱和青灰", "#3C6E71"),
    ("纯白", "#FFFFFF"),
    ("浅灰", "#D8D8D6"),
    ("天蓝", "#BBD3E0"),
    ("深灰蓝", "#2E3A45"),
    ("绿幕", "#00B140"),
]

# 分割模型下拉：(显示名, 模型键)。用于答辩时的"质量 vs 速度"对比
SEGMENTER_CHOICES = [
    ("二元·快（13ms）", "selfie_segmenter_binary"),
    ("多分类·慢（155ms）", "selfie_segmenter"),
]


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
        grid.addWidget(SliderRow("瘦脸", 0.0, 1.0, p["slim"],
                                 self._set("slim")), 3, 0, 1, 3)

        self.chk_eye = QCheckBox("大眼")
        self.chk_eye.setChecked(p["eye_enabled"])
        self.chk_eye.toggled.connect(self._set("eye_enabled"))
        grid.addWidget(self.chk_eye, 4, 0)
        grid.addWidget(SliderRow("大眼强度", 0.0, 0.5, p["eye_strength"],
                                 self._set("eye_strength")), 4, 1, 1, 2)

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


class SegmentPanel(QGroupBox):
    """人像虚化 / 背景替换（Phase 3）：模式 + 背景选择 + 质量滑杆。

    控件只做两件事：读 pipeline 状态初始化、把变化写回 set_params / set_enabled。
    分割模型的切换要动推理会话，通过 on_model_change 回调交给主窗口，由工作线程
    在帧循环顶部消费（GUI 线程绝不直接重建会话）。
    """

    def __init__(self, pipeline: Pipeline,
                 on_model_change: Optional[Callable[[str], None]] = None,
                 parent=None):
        super().__init__("人像虚化 / 背景替换", parent)
        self.pipeline = pipeline
        self._on_model_change = on_model_change
        effect = pipeline.get_effect("segment")
        p = effect.get_params()

        lay = QVBoxLayout(self)
        self.chk_enabled = QCheckBox("启用")
        self.chk_enabled.setChecked(effect.enabled)
        self.chk_enabled.toggled.connect(
            lambda on: pipeline.set_enabled("segment", on))
        lay.addWidget(self.chk_enabled)

        # ---- 模式 ----
        grid = QGridLayout()
        grid.addWidget(QLabel("模式"), 0, 0)
        self.cmb_mode = QComboBox()
        for label, value in (("背景虚化", MODE_BLUR),
                             ("换成背景图", MODE_IMAGE),
                             ("纯色背景", MODE_COLOR)):
            self.cmb_mode.addItem(label, value)
        self.cmb_mode.setCurrentIndex(
            max(0, self.cmb_mode.findData(p["mode"])))
        self.cmb_mode.currentIndexChanged.connect(self._mode_changed)
        grid.addWidget(self.cmb_mode, 0, 1)
        lay.addLayout(grid)

        # ---- 背景图库 ----
        self.lbl_gallery = QLabel("背景图库")
        lay.addWidget(self.lbl_gallery)
        self.lst_bg = QListWidget()
        self.lst_bg.setViewMode(QListWidget.ViewMode.IconMode)
        self.lst_bg.setIconSize(QPixmap(*BG_THUMB).size())
        self.lst_bg.setGridSize(self.lst_bg.iconSize())
        self.lst_bg.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.lst_bg.setFixedHeight(3 * (BG_THUMB[1] + 14))
        self.lst_bg.setMovement(QListWidget.Movement.Static)
        self.lst_bg.itemClicked.connect(self._pick_gallery_bg)
        lay.addWidget(self.lst_bg)
        self._load_gallery()

        btn_choose = QPushButton("选择图片…")
        btn_choose.clicked.connect(self._pick_file_bg)
        lay.addWidget(btn_choose)

        # ---- 纯色 ----
        self.lbl_color = QLabel("纯色")
        lay.addWidget(self.lbl_color)
        self.cmb_color = QComboBox()
        for label, value in COLOR_PRESETS:
            self.cmb_color.addItem(label, value)
        self.cmb_color.setCurrentIndex(
            max(0, self.cmb_color.findData(p["bg_color"])))
        self.cmb_color.currentIndexChanged.connect(
            lambda i: self._set(bg_color=self.cmb_color.itemData(i)))
        lay.addWidget(self.cmb_color)

        # ---- 质量滑杆 ----
        self.row_strength = SliderRow(
            "虚化强度", 0.0, 1.0, p["strength"], self._set("strength"))
        lay.addWidget(self.row_strength)
        self.row_contrast = SliderRow(
            "matte 对比度", 0.0, 1.0, p["matte_contrast"],
            self._set("matte_contrast"))
        lay.addWidget(self.row_contrast)
        self.row_feather = SliderRow(
            "边缘羽化", 0.0, 12.0, p["feather"], self._set("feather"), scale=1.0)
        lay.addWidget(self.row_feather)
        self.row_smooth = SliderRow(
            "时域平滑", 0.0, 0.95, p["smooth"], self._set("smooth"))
        lay.addWidget(self.row_smooth)

        self.chk_refine = QCheckBox("边缘精修（引导滤波）")
        self.chk_refine.setChecked(p["refine"])
        self.chk_refine.toggled.connect(self._set("refine"))
        lay.addWidget(self.chk_refine)

        # ---- 分割模型（答辩对比用） ----
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("分割模型"))
        self.cmb_model = QComboBox()
        for label, value in SEGMENTER_CHOICES:
            self.cmb_model.addItem(label, value)
        self.cmb_model.setCurrentIndex(
            max(0, self.cmb_model.findData(DEFAULT_SEGMENTER)))
        self.cmb_model.currentIndexChanged.connect(self._model_changed)
        mrow.addWidget(self.cmb_model)
        lay.addLayout(mrow)

        self._mode_changed()   # 按当前模式初始化控件可见性

    # ------- 控件 ↔ 参数 -------

    def _set(self, key: str = "", **extra):
        def apply(value=None):
            kwargs = dict(extra)
            if key:
                kwargs[key] = value
            self.pipeline.set_params("segment", **kwargs)
        return apply

    def _mode_changed(self, *_) -> None:
        """模式切换：同步参数并只显示该模式相关的控件。"""
        mode = self.cmb_mode.currentData()
        for w in (self.lbl_gallery, self.lst_bg):
            w.setVisible(mode == MODE_IMAGE)
        for w in (self.lbl_color, self.cmb_color):
            w.setVisible(mode == MODE_COLOR)
        self.row_strength.setVisible(mode == MODE_BLUR)
        self._set(mode=mode)()

    def _load_gallery(self) -> None:
        """把内置图库填进列表（缩略图）。列表为空时给一行提示。"""
        paths = list_backgrounds()
        if not paths:
            item = QListWidgetItem("图库为空\n先跑\nmake_backgrounds.py")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.lst_bg.addItem(item)
            return
        for path in paths:
            img = load_image(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = min(BG_THUMB[0] / w, BG_THUMB[1] / h)
            thumb = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
            qimg = QImage(thumb.data, thumb.shape[1], thumb.shape[0],
                          3 * thumb.shape[1], QImage.Format.Format_BGR888)
            item = QListWidgetItem(QPixmap.fromImage(qimg),
                                   os.path.splitext(os.path.basename(path))[0])
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.lst_bg.addItem(item)

    def _pick_gallery_bg(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self._set(bg_path=path)()

    def _pick_file_bg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择背景图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if path:
            self._set(mode=MODE_IMAGE, bg_path=path)()
            self.cmb_mode.setCurrentIndex(self.cmb_mode.findData(MODE_IMAGE))

    def _model_changed(self, *_) -> None:
        key = self.cmb_model.currentData()
        if key and self._on_model_change is not None:
            self._on_model_change(key)


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
