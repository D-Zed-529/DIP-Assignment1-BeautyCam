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
from core.infer import (
    DEFAULT_SEGMENTER, SEGMENTER_SPECS, TORCH_ONLY_SEGMENTERS, get_engine,
)
from core.pipeline import Pipeline
from demos.faceswap.effect import list_face_presets

# 低光增强引擎选择（CUDA 迁移后双深度档：快速 SCI / 质量 Retinexformer）
LOWLIGHT_CHOICES = [
    ("关闭", "off"),
    ("经典启发式（基线）", "heuristic"),
    ("SCI 深度模型（快速）", "sci"),
    ("Retinexformer（高质量·慢）", "retinex"),
]
LOWLIGHT_SCI_LEVELS = [("轻度 easy", "easy"), ("中度 medium", "medium"),
                       ("强力 difficult", "difficult")]

# 背景图库缩略图尺寸
BG_THUMB = (72, 41)
FACE_THUMB = (72, 72)

# 纯色背景预设：(显示名, 十六进制)。绿幕色便于后续做抠像演示
COLOR_PRESETS = [
    ("低饱和青灰", "#3C6E71"),
    ("纯白", "#FFFFFF"),
    ("浅灰", "#D8D8D6"),
    ("天蓝", "#BBD3E0"),
    ("深灰蓝", "#2E3A45"),
    ("绿幕", "#00B140"),
]

# 分割模型下拉：(显示名, 模型键)。torch 后端多出 RVM 视频抠图（默认，
# 发丝级 alpha + 时域一致）；mediapipe 后端只有两个 tflite。
SEGMENTER_LABELS = {
    "rvm": "RVM 视频抠图·推荐",
    "selfie_segmenter_binary": "二元·快（13ms）",
    "selfie_segmenter": "多分类·慢（155ms）",
}


def segmenter_choices() -> list[tuple[str, str]]:
    """按当前推理后端返回可用分割模型（torch-only 的 RVM 在 mediapipe
    后端下隐藏，避免选了报错）。"""
    try:
        backend = get_engine().backend_name
    except Exception:   # noqa: BLE001 —— 无模型环境（纯函数单测）退 mediapipe
        backend = "mediapipe:cpu"
    keys = [k for k in SEGMENTER_SPECS
            if backend.startswith("torch") or k not in TORCH_ONLY_SEGMENTERS]
    return [(SEGMENTER_LABELS.get(k, k), k) for k in keys]


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

        self.chk_finish = QCheckBox("收尾去噪与锐化")
        self.chk_finish.setChecked(p["finish"])
        self.chk_finish.setToolTip("默认关闭，避免实时预览出现过锐的边缘和失去皮肤纹理")
        self.chk_finish.toggled.connect(self._set("finish"))
        grid.addWidget(self.chk_finish, 6, 0, 1, 3)

    def _set(self, key: str):
        return lambda v: self.pipeline.set_params("beauty", **{key: v})


class AutoEnhancePanel(QGroupBox):
    """自适应画质（Phase 4）：分区自动曝光 + CLAHE + 白平衡 + 饱和度。

    全时段经典 DIP 校正（与低光增强互补：低光管极端暗光，本面板管
    逆光脸黑 / 轻度过曝 / 偏色 / 发灰等常态问题）。人脸区域来自
    FaceMesh 轮廓（美颜默认也在跑，零额外推理）。
    """

    def __init__(self, pipeline: Pipeline, parent=None):
        super().__init__("自适应画质", parent)
        self.pipeline = pipeline
        effect = pipeline.get_effect("autoenhance")
        p = effect.get_params()

        lay = QVBoxLayout(self)
        self.chk_enabled = QCheckBox("启用")
        self.chk_enabled.setChecked(effect.enabled)
        self.chk_enabled.setToolTip(
            "分区自动曝光（人脸优先）+ CLAHE 对比度 + 灰世界白平衡 + 饱和度，\n"
            "统计量时域平滑防闪。全时段生效，与低光增强可叠加。")
        self.chk_enabled.toggled.connect(
            lambda on: pipeline.set_enabled("autoenhance", on))
        lay.addWidget(self.chk_enabled)

        self.row_strength = SliderRow(
            "总强度", 0.0, 1.0, p["strength"], self._set("strength"))
        lay.addWidget(self.row_strength)
        self.row_face = SliderRow(
            "人脸曝光优先", 0.0, 1.0, p["face_exposure"],
            self._set("face_exposure"))
        self.row_face.setToolTip("人脸目标亮度从 115 插值到 150；0 = 全图统一曝光校正")
        lay.addWidget(self.row_face)
        self.row_contrast = SliderRow(
            "对比度 (CLAHE)", 0.0, 1.0, p["contrast"], self._set("contrast"))
        lay.addWidget(self.row_contrast)
        self.row_color = SliderRow(
            "白平衡", 0.0, 1.0, p["color"], self._set("color"))
        self.row_color.setToolTip("灰世界假设，在背景区估计通道增益（避开肤色污染）")
        lay.addWidget(self.row_color)
        self.row_sat = SliderRow(
            "饱和度", 0.0, 1.0, p["saturation"], self._set("saturation"))
        lay.addWidget(self.row_sat)
        self.row_smooth = SliderRow(
            "时域平滑", 0.0, 0.95, p["smooth"], self._set("smooth"))
        lay.addWidget(self.row_smooth)

    def _set(self, key: str):
        return lambda v: self.pipeline.set_params("autoenhance", **{key: v})


class LowLightPanel(QGroupBox):
    """低光增强（Phase 2）：经典启发式基线 / SCI 深度模型二选一。

    管线里同时挂着两个 effect（name=lowlight / lowlight_dnn），面板下拉
    谁就启用谁、另一个关闭（互斥）。客观对比数据见 scripts/eval_lowlight.py
    （合成暗图上 SCI 22.1dB vs 启发式 14.4dB PSNR）。
    """

    def __init__(self, pipeline: Pipeline, parent=None):
        super().__init__("低光增强", parent)
        self.pipeline = pipeline

        lay = QVBoxLayout(self)
        grid = QGridLayout()
        grid.addWidget(QLabel("引擎"), 0, 0)
        self.cmb_engine = QComboBox()
        for label, value in LOWLIGHT_CHOICES:
            self.cmb_engine.addItem(label, value)
        self.cmb_engine.setCurrentIndex(0)
        self.cmb_engine.currentIndexChanged.connect(self._engine_changed)
        grid.addWidget(self.cmb_engine, 0, 1)
        lay.addLayout(grid)

        self.lbl_level = QLabel("SCI 强度档")
        self.cmb_level = QComboBox()
        for label, value in LOWLIGHT_SCI_LEVELS:
            self.cmb_level.addItem(label, value)
        self.cmb_level.setCurrentIndex(1)   # medium 默认（评测最优档）
        self.cmb_level.currentIndexChanged.connect(
            lambda i: self.pipeline.set_params(
                "lowlight_dnn", level=self.cmb_level.itemData(i)))
        lay.addWidget(self.lbl_level)
        lay.addWidget(self.cmb_level)

        self.row_strength = SliderRow(
            "增强强度", 0.0, 1.0,
            pipeline.get_effect("lowlight").get_params()["strength"],
            self._set_strength)
        lay.addWidget(self.row_strength)

        self.chk_auto = QCheckBox("仅暗光时自动增强（灰度均值 < 60）")
        self.chk_auto.setChecked(True)
        self.chk_auto.toggled.connect(self._set_auto)
        lay.addWidget(self.chk_auto)

        self._engine_changed()

    # ------- 联动 -------

    def _engine_changed(self, *_) -> None:
        mode = self.cmb_engine.currentData()
        dnn_on = mode in ("sci", "retinex")
        if dnn_on:   # 切引擎前先落参数（低光链首帧创建会话时读到）
            self.pipeline.set_params("lowlight_dnn", engine=mode)
        self.pipeline.set_enabled("lowlight", mode == "heuristic")
        self.pipeline.set_enabled("lowlight_dnn", dnn_on)
        sci = mode == "sci"
        self.lbl_level.setVisible(sci)
        self.cmb_level.setVisible(sci)

    def _set_strength(self, v: float) -> None:
        self.pipeline.set_params("lowlight", strength=v)
        self.pipeline.set_params("lowlight_dnn", strength=v)

    def _set_auto(self, on: bool) -> None:
        self.pipeline.set_params("lowlight", auto=on)
        self.pipeline.set_params("lowlight_dnn", auto=on)


class HdrPanel(QGroupBox):
    """自动 HDR 连拍（Phase 1，拍照模式）：EV 预设 + 色调映射 + 连拍按钮。

    HDR 不进逐帧效果链（预览不受影响）；点「连拍融合」由 worker 在帧循环
    顶部进入连拍节奏（0.12s 间隔采集自然抖动）→ ECC 对齐 → Mertens 融合。
    成片与各 EV 原图、对照图一并存 photos/（验收要求）。
    """

    def __init__(self, on_capture: Callable[[str, object], None], parent=None):
        super().__init__("自动 HDR 拍照", parent)
        from core.effects.hdr import DEFAULT_EV_PRESET, EV_PRESETS

        lay = QVBoxLayout(self)
        grid = QGridLayout()
        grid.addWidget(QLabel("包围曝光"), 0, 0)
        self.cmb_evs = QComboBox()
        for name in EV_PRESETS:
            self.cmb_evs.addItem(name, name)
        self.cmb_evs.setCurrentIndex(
            max(0, self.cmb_evs.findData(DEFAULT_EV_PRESET)))
        grid.addWidget(self.cmb_evs, 0, 1)
        grid.addWidget(QLabel("色调映射"), 1, 0)
        self.cmb_tonemap = QComboBox()
        for label, value in (("关闭（Mertens 直出）", None),
                             ("Drago（柔和高光）", "drago"),
                             ("Reinhard（局部对比）", "reinhard")):
            self.cmb_tonemap.addItem(label, value)
        grid.addWidget(self.cmb_tonemap, 1, 1)
        lay.addLayout(grid)

        self.btn_burst = QPushButton("✨ HDR 连拍融合")
        self.btn_burst.setObjectName("hdrBtn")
        self.btn_burst.setEnabled(False)
        self.btn_burst.setToolTip(
            "连拍多张并以 gamma 曲线模拟包围曝光 → ECC 对齐 → Mertens 融合。\n"
            "macOS 不支持手动包围曝光，用 gamma 模拟（效果等价、可复现）。")
        self._on_capture = on_capture
        self.btn_burst.clicked.connect(self._capture)
        lay.addWidget(self.btn_burst)

    def set_running(self, running: bool) -> None:
        """相机启停联动（连拍需要活的采集源）。"""
        self.btn_burst.setEnabled(running)

    def _capture(self) -> None:
        self._on_capture(self.cmb_evs.currentData(),
                         self.cmb_tonemap.currentData())


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

        self.lbl_bg_status = QLabel("尚未选择背景图片")
        self.lbl_bg_status.setWordWrap(True)
        lay.addWidget(self.lbl_bg_status)

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
        for label, value in segmenter_choices():
            self.cmb_model.addItem(label, value)
        # 默认跟随引擎（torch 后端默认 rvm；查不到就退 DEFAULT_SEGMENTER）
        engine_default = getattr(get_engine(), "segmenter_model",
                                 DEFAULT_SEGMENTER)
        self.cmb_model.setCurrentIndex(
            max(0, self.cmb_model.findData(engine_default)))
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

    def _mode_changed(self, index: int | None = None) -> None:
        """切换模式时立即生效；首次选图片模式自动使用内置背景。"""
        mode = self.cmb_mode.currentData()
        for w in (self.lbl_gallery, self.lst_bg, self.lbl_bg_status):
            w.setVisible(mode == MODE_IMAGE)
        for w in (self.lbl_color, self.cmb_color):
            w.setVisible(mode == MODE_COLOR)
        self.row_strength.setVisible(mode == MODE_BLUR)
        self._set(mode=mode)()
        if mode == MODE_IMAGE:
            path = self.pipeline.get_effect("segment").get_params()["bg_path"]
            if not path or load_image(path) is None:
                paths = list_backgrounds()
                path = paths[0] if paths else ""
                self._set(bg_path=path)()
            self.lbl_bg_status.setText(
                f"当前背景：{os.path.basename(path)}" if path else
                "没有可用背景图，请点击「选择图片…」")
        # 构造函数末尾也会调用一次以初始化可见性；那次不改变总开关。
        if index is not None:
            self.chk_enabled.setChecked(True)

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
            self._set(mode=MODE_IMAGE, bg_path=path)()
            self.cmb_mode.setCurrentIndex(self.cmb_mode.findData(MODE_IMAGE))
            self.lbl_bg_status.setText(f"当前背景：{os.path.basename(path)}")
            self.chk_enabled.setChecked(True)

    def _pick_file_bg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择背景图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if path:
            if load_image(path) is None:
                self.lbl_bg_status.setText("图片无法读取，请重新选择")
                return
            self._set(mode=MODE_IMAGE, bg_path=path)()
            self.cmb_mode.setCurrentIndex(self.cmb_mode.findData(MODE_IMAGE))
            self.lbl_bg_status.setText(f"当前背景：{os.path.basename(path)}")
            self.chk_enabled.setChecked(True)

    def _model_changed(self, *_) -> None:
        key = self.cmb_model.currentData()
        if key and self._on_model_change is not None:
            self._on_model_change(key)


class BokehPanel(QGroupBox):
    """深度渐进虚化（P3-4，CUDA 迁移补上）：近清远糊，焦平面对齐人物。

    需要 torch 后端 + Depth Anything V2 权重；mediapipe 后端下效果自动
    透传（core/effects/bokeh.py 告警一次）。与"人像虚化"的均匀模糊互补：
    这里的模糊量随深度连续变化，观感对齐单反镜头。
    """

    def __init__(self, pipeline: Pipeline, parent=None):
        super().__init__("深度渐进虚化（单反感）", parent)
        self.pipeline = pipeline
        effect = pipeline.get_effect("bokeh")
        p = effect.get_params()

        lay = QVBoxLayout(self)
        self.chk_enabled = QCheckBox("启用")
        self.chk_enabled.setChecked(effect.enabled)
        self.chk_enabled.toggled.connect(
            lambda on: pipeline.set_enabled("bokeh", on))
        lay.addWidget(self.chk_enabled)

        self.row_strength = SliderRow(
            "虚化强度", 0.0, 1.0, p["strength"],
            lambda v: pipeline.set_params("bokeh", strength=v))
        lay.addWidget(self.row_strength)
        self.row_range = SliderRow(
            "焦外范围", 0.1, 0.8, p["range"],
            lambda v: pipeline.set_params("bokeh", range=v))
        lay.addWidget(self.row_range)

        self.chk_matte = QCheckBox("人像保持清晰（需分割模型）")
        self.chk_matte.setChecked(p["use_matte"])
        self.chk_matte.toggled.connect(
            lambda on: pipeline.set_params("bokeh", use_matte=on))
        lay.addWidget(self.chk_matte)


class FaceSwapPanel(QGroupBox):
    """实时换脸：本地选择源脸、授权确认和效果开关。"""

    def __init__(self, pipeline: Pipeline, parent=None):
        super().__init__("换脸（演示级）", parent)
        self.pipeline = pipeline
        self.effect = pipeline.get_effect("faceswap")
        p = self.effect.get_params()

        lay = QVBoxLayout(self)
        self.chk_consent = QCheckBox("我确认仅使用本人 / 已授权者 / 动漫形象")
        self.chk_consent.setChecked(bool(p["consent"]))
        self.chk_consent.toggled.connect(self._consent_changed)
        lay.addWidget(self.chk_consent)

        lay.addWidget(QLabel("内置脸库（均为原创虚构人物）"))
        self.lst_faces = QListWidget()
        self.lst_faces.setViewMode(QListWidget.ViewMode.IconMode)
        self.lst_faces.setIconSize(QPixmap(*FACE_THUMB).size())
        self.lst_faces.setGridSize(self.lst_faces.iconSize())
        self.lst_faces.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.lst_faces.setFixedHeight(2 * (FACE_THUMB[1] + 20))
        self.lst_faces.setMovement(QListWidget.Movement.Static)
        self.lst_faces.itemClicked.connect(self._pick_gallery_source)
        lay.addWidget(self.lst_faces)
        self._load_face_gallery()

        self.btn_source = QPushButton("自行上传源脸图片…")
        self.btn_source.clicked.connect(self._pick_source)
        lay.addWidget(self.btn_source)

        self.lbl_preview = QLabel("尚未选择源脸")
        self.lbl_preview.setFixedHeight(92)
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setStyleSheet(
            "background:#111827;border:1px solid #334155;border-radius:6px;")
        lay.addWidget(self.lbl_preview)

        self.lbl_status = QLabel("请选择一张正面清晰照片")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        self.chk_color = QCheckBox("肤色匹配（Reinhard）")
        self.chk_color.setChecked(bool(p["color_transfer"]))
        self.chk_color.toggled.connect(
            lambda on: self.pipeline.set_params("faceswap", color_transfer=on))
        lay.addWidget(self.chk_color)

        self.chk_enabled = QCheckBox("启用实时换脸")
        self.chk_enabled.setChecked(self.effect.enabled)
        self.chk_enabled.toggled.connect(self._enabled_changed)
        lay.addWidget(self.chk_enabled)

        path = str(p["source_path"])
        if path:
            self._show_source(path)
        self._sync_enabled_state()

    def _consent_changed(self, on: bool) -> None:
        self.pipeline.set_params("faceswap", consent=on)
        if not on:
            self.chk_enabled.setChecked(False)
        self._sync_enabled_state()

    def _enabled_changed(self, on: bool) -> None:
        allowed = bool(self.chk_consent.isChecked()
                       and self.effect.get_params()["source_path"])
        self.pipeline.set_enabled("faceswap", bool(on and allowed))
        if on and not allowed:
            self.chk_enabled.setChecked(False)

    def _sync_enabled_state(self) -> None:
        ready = bool(self.chk_consent.isChecked()
                     and self.effect.get_params()["source_path"])
        self.chk_enabled.setEnabled(ready)

    def _pick_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择已获授权的源脸图片", "",
            "图片 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not path:
            return
        if load_image(path) is None:
            self.lbl_status.setText("图片无法读取，请重新选择")
            return
        self._apply_source(path)

    def _load_face_gallery(self) -> None:
        paths = list_face_presets()
        if not paths:
            item = QListWidgetItem("内置脸库为空")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.lst_faces.addItem(item)
            return
        for path in paths:
            image = load_image(path)
            if image is None:
                continue
            h, w = image.shape[:2]
            scale = min(FACE_THUMB[0] / w, FACE_THUMB[1] / h)
            thumb = cv2.resize(
                image, (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA)
            qimg = QImage(thumb.data, thumb.shape[1], thumb.shape[0],
                          3 * thumb.shape[1], QImage.Format.Format_BGR888)
            stem = os.path.splitext(os.path.basename(path))[0]
            label = stem.split("_", 1)[-1]
            item = QListWidgetItem(QPixmap.fromImage(qimg), label)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(f"原创虚构人物：{label}")
            self.lst_faces.addItem(item)

    def _pick_gallery_source(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self._apply_source(path)

    def _apply_source(self, path: str) -> None:
        self.pipeline.set_params("faceswap", source_path=path)
        self._show_source(path)
        self.lbl_status.setText(f"源脸已选择：{os.path.basename(path)}")
        self._sync_enabled_state()
        if self.chk_consent.isChecked():
            self.chk_enabled.setChecked(True)

    def _show_source(self, path: str) -> None:
        image = load_image(path)
        if image is None:
            return
        h, w = image.shape[:2]
        qimg = QImage(image.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        pix = QPixmap.fromImage(qimg).scaled(
            270, 86, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.lbl_preview.setPixmap(pix)
        self.lbl_preview.setToolTip(path)

    def update_runtime_status(self, text: str) -> None:
        if text and self.lbl_status.text() != text:
            self.lbl_status.setText(text)


class CapturePanel(QGroupBox):
    """拍照：手动按钮 + V 手势 / 笑脸自动触发。"""

    def __init__(self, on_manual: Callable[[], None], parent=None):
        super().__init__("拍照", parent)
        v = QVBoxLayout(self)
        self.btn_capture = QPushButton("📸 拍照（空格）")
        self.btn_capture.setEnabled(False)
        self.btn_capture.clicked.connect(on_manual)
        self.chk_vsign = QCheckBox("V 手势自动拍照（持续 1s）")
        self.chk_vsign.setChecked(False)
        self.chk_vsign.setToolTip("开启后每帧运行手部检测，会降低预览帧率")
        self.chk_smile = QCheckBox("笑脸自动拍照（持续 0.5s）")
        self.chk_smile.setChecked(False)
        self.chk_smile.setToolTip("开启后每帧计算表情置信度，会降低预览帧率")
        # 流畅优先：预览链在 540p 处理（像素域开销 ~½²），拍照自动回到
        # 原始分辨率出片。全开效果时建议开启；追求逐像素画质可关闭。
        self.chk_smooth = QCheckBox("流畅优先（预览 540p 处理，拍照全分辨率）")
        self.chk_smooth.setChecked(True)
        for w in (self.btn_capture, self.chk_vsign, self.chk_smile,
                  self.chk_smooth):
            v.addWidget(w)

    def smooth_mode(self) -> bool:
        return self.chk_smooth.isChecked()

    def triggers(self) -> set[str]:
        from gui.workers import TRIGGER_SMILE, TRIGGER_V_SIGN
        out = set()
        if self.chk_vsign.isChecked():
            out.add(TRIGGER_V_SIGN)
        if self.chk_smile.isChecked():
            out.add(TRIGGER_SMILE)
        return out
