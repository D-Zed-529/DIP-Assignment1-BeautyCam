"""MediaPipe / ONNX 会话管理（单例）。

P0-1 冒烟结论（2026-09，macOS darwin 27 / Apple M4）：
  - mediapipe 1.0.1：已移除旧 `mp.solutions`；且 Tasks API 检测类图
    （FaceDetector / FaceLandmarker / HandLandmarker）在 Open() 阶段因
    DrishtiMetalHelper 硬崩溃（CPU/GPU 委托均崩）→ 不可用。
  - mediapipe 0.10.21：Tasks API + 显式 CPU 委托全部正常（分割器亦可用）。
  因此锁定 mediapipe==0.10.21，且所有会话显式指定 CPU 委托。

FaceDetector 已弃用：macOS Tasks 版存在 Metal 后处理缺陷；人脸框改由
FaceMesh 轮廓关键点导出（见 _landmarks_to_face_info），还省一次前向。

P3-0 分割模型冒烟结论（2026-09，1280×720，本机 CPU）：
  - `selfie_multiclass_256x256.tflite`（16.4MB，6 类）：**155 ms/帧**，且耗时与
    输入分辨率无关（256×144 与 1280×720 同为 ~140ms），不取任何输出仍 ~142ms
    —— 瓶颈纯在模型推理本身；照此直接接虚拟背景只能跑到 ~7fps，必然不达标。
  - `selfie_segmenter.tflite`（250KB，二元人/非人）：**13.2 ms/帧**，快 11.7 倍。
    虚拟背景只需前景/背景二分类，多分类那 5 类信息用不上却要多付 10 倍算力，
    故**默认二元模型**；多分类保留为可切换选项，用于"质量 vs 速度"对比实验。
  - **⚠️ 两个模型的类别编码与置信图极性恰好相反**（详见 SEGMENTER_SPECS）：
      二元      : 类别 0 = 人 / 255 = 背景；`conf[0]` = **人**的概率
      多分类    : 类别 0 = 背景 / 1..5 = 人；`conf[0]` = **背景**的概率
    搞反**不会抛异常**，只会静默产出整体反相的掩膜（人景对调）或全幅掩膜
    （背景替换什么都不换），是这块最容易埋雷的地方。本模块曾两次把这张表写反
    （"conf[0] 都是背景概率"这一说法看起来非常合理），因此单测用**与假设无关
    的探针法**独立复核（图像四角必为背景、画面中下部必为人），并加了运行时
    边框先验自检（_self_check_border）作为第二道保险。
  - 二元模型的置信度本身已接近二值（背景区 alpha 恒为 0.000），多分类的则有
    系统性偏置（背景区 alpha 恒为 ~0.084），直接当 alpha 用会给新背景叠上一层
    均匀的 8.4% 鬼影（表现为背景发灰）。用对比度拉伸即可归零 —— 详见
    core/effects/segment.py 的 matte_contrast 及其模块头实测数据。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from .context import FaceInfo, FrameContext, HandInfo

# 模型目录：仓库根/models（权重不入库，由 scripts/download_models.py 拉取）
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_FILES = {
    "face_landmarker": "face_landmarker.task",
    "hand_landmarker": "hand_landmarker.task",
    # Phase 3 自拍分割，两个模型都保留（切换见 InferenceEngine.set_segmenter_model）：
    #   binary     二元「人 / 非人」，250KB，13.2ms —— 虚拟背景默认用它
    #   multiclass 6 类（0背景/1头发/2躯体皮肤/3面部皮肤/4衣服/5其他），16.4MB，155ms
    "selfie_segmenter_binary": "selfie_segmenter.tflite",
    "selfie_segmenter": "selfie_multiclass_256x256.tflite",
}

DEFAULT_SEGMENTER = "selfie_segmenter_binary"

# 分割模型 -> 建议的推理间隔帧数（虚拟背景的 SegmentEffect 据此设置隔帧降载）。
# 二元模型 13ms 可每帧跑；多分类 155ms 必须隔帧，靠时域平滑复用中间帧。
SEGMENTER_INTERVAL = {
    "selfie_segmenter_binary": 1,
    "selfie_segmenter": 4,
}

# 分割模型规格表 —— 两个模型的「类别编码」与「置信图极性」**恰好相反**，
# 且搞反了不抛异常、只静默产出反转掩膜（人景对调）或全幅掩膜（什么都不换），
# 所以必须按模型显式声明，不能靠启发式猜。
#
# 用**与假设无关的探针法**实测确认（图像四角必为背景、画面中下部必为人；
# 见 tests/test_infer.py::TestSegmenterSemantics 的同一套探针）：
#   binary     类别 0 = 人 / 255 = 背景，conf[0] = **人**的概率
#   multiclass 类别 0 = 背景 / 1..5 = 人，conf[0] = **背景**的概率
#
# 注意：这张表曾经两次被写反（"conf[0] 都是背景概率"看起来非常合理），
# 因此配套的单测用探针法独立复核，而不是在表内自证。
SEGMENTER_SPECS = {
    "selfie_segmenter_binary": {"person_is_zero": True, "alpha_invert": False},
    "selfie_segmenter": {"person_is_zero": False, "alpha_invert": True},
}

# 边框先验自检阈值：画面边缘 alpha 均值超过它就告警（见 _self_check_border）
BORDER_FOREGROUND_WARN = 0.5

# FaceMesh 脸部轮廓关键点（一期沿用），用于生成人脸区域掩膜/人脸框
FACE_OVAL_IDS = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
]


def _landmarks_to_array(landmarks) -> np.ndarray:
    """MediaPipe 归一化关键点列表 -> (N, 3) float32 ndarray。"""
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)


def _landmarks_to_face_info(lm: np.ndarray, blendshapes=None) -> FaceInfo:
    """由 FaceMesh 关键点导出 FaceInfo：人脸框（轮廓 min/max，含下巴余量）。"""
    oval = lm[FACE_OVAL_IDS]
    x1, y1 = oval[:, 0].min(), oval[:, 1].min()
    x2, y2 = oval[:, 0].max(), oval[:, 1].max()
    # 向下扩 8% 高度包住下巴（一期 get_face_region_mask 的扩边思想）
    h_box = y2 - y1
    box = (float(x1), float(y1), float(x2), float(min(y2 + 0.08 * h_box, 1.0)))

    smile: Optional[float] = None
    if blendshapes:
        # ARKit 52 维 blendshape 无裸 smile，取左右嘴角微笑均值
        scores = [bs.score for bs in blendshapes
                  if bs.category_name in ("mouthSmileLeft", "mouthSmileRight")]
        if scores:
            smile = float(sum(scores) / len(scores))
    return FaceInfo(landmarks=lm, box=box, smile=smile)


def category_to_person_mask(cat: np.ndarray, person_is_zero: bool) -> np.ndarray:
    """分割类别掩膜 -> 人像 bool 掩膜。

    编码随模型而变（见 SEGMENTER_SPECS）：二元模型 0=人/255=背景，
    多分类模型 0=背景/1..5=人。搞反了会得到整体反相的掩膜，且不会报错。
    """
    return (cat == 0) if person_is_zero else (cat > 0)


def foreground_from_confidence(conf0: np.ndarray,
                              alpha_invert: bool) -> np.ndarray:
    """单张置信图 -> 前景概率 float32（0~1，未做任何锐化）。

    alpha_invert 由模型决定（见 SEGMENTER_SPECS）：多分类的 conf[0] 是背景概率
    需取反，二元的 conf[0] 直接就是前景概率。
    """
    a = conf0.astype(np.float32)
    return np.clip(1.0 - a if alpha_invert else a, 0.0, 1.0)


def border_foreground_ratio(alpha: np.ndarray, margin: int = 12) -> float:
    """画面四周边缘区域内 alpha 的均值。

    自拍场景下画面四边几乎总是背景，因此这个值应接近 0。它是**与掩膜语义无关**
    的独立参照（不依赖任何类别/置信图约定），用于运行时自检掩膜是否整体反相。
    """
    m = max(1, int(margin))
    band = np.concatenate([
        alpha[:m].ravel(), alpha[-m:].ravel(),
        alpha[:, :m].ravel(), alpha[:, -m:].ravel()])
    return float(band.mean())


class InferenceEngine:
    """每帧统一推理入口：FaceMesh / Hands / 自拍分割，结果填入 FrameContext。

    会话懒加载 + 单例（get_engine()）；创建开销大，帧循环里只调 process()。
    """

    def __init__(self, models_dir: Path | str = MODELS_DIR,
                 delegate: int | None = None,
                 segmenter_model: str = DEFAULT_SEGMENTER):
        self.models_dir = Path(models_dir)
        # None 表示用 mediapipe 默认；本机验证默认委托稳定，但保险起见
        # 全部显式 CPU（GPU/Metal 委托在本机崩溃，见模块头注释）
        self._delegate = (
            delegate if delegate is not None
            else mp_tasks.BaseOptions.Delegate.CPU
        )
        if segmenter_model not in SEGMENTER_SPECS:
            raise ValueError(f"未知分割模型：{segmenter_model}")
        self._segmenter_key = segmenter_model
        self._face_lm: Optional[vision.FaceLandmarker] = None
        self._hand_lm: Optional[vision.HandLandmarker] = None
        self._segmenter: Optional[vision.ImageSegmenter] = None
        # 边框先验自检只做一次（告警用，不自动翻转掩膜）
        self._border_checked = False
        self._ts = 0          # VIDEO 模式时间戳必须严格递增
        self._frame_id = 0
        self._lock = threading.Lock()   # 会话懒加载互斥

    # ------- 分割模型选择 -------

    @property
    def segmenter_model(self) -> str:
        return self._segmenter_key

    @property
    def segmenter_spec(self) -> dict:
        return SEGMENTER_SPECS[self._segmenter_key]

    @property
    def recommended_interval(self) -> int:
        """当前分割模型建议的推理间隔帧数（GUI 据此设 SegmentEffect 的 infer_interval）。"""
        return SEGMENTER_INTERVAL[self._segmenter_key]

    def set_segmenter_model(self, key: str) -> None:
        """切换分割模型（关闭旧会话，下次推理时重建）。"""
        if key not in SEGMENTER_SPECS:
            raise ValueError(f"未知分割模型：{key}")
        with self._lock:
            if key == self._segmenter_key:
                return
            if self._segmenter is not None:
                self._segmenter.close()
                self._segmenter = None
            self._segmenter_key = key
            self._border_checked = False

    # ------- 会话懒加载 -------

    def _base(self, key: str) -> mp_tasks.BaseOptions:
        path = self.models_dir / MODEL_FILES[key]
        if not path.exists():
            raise FileNotFoundError(
                f"模型缺失：{path}，请先运行 python scripts/download_models.py"
            )
        return mp_tasks.BaseOptions(
            model_asset_path=str(path), delegate=self._delegate)

    def _get_face_landmarker(self) -> vision.FaceLandmarker:
        if self._face_lm is None:
            with self._lock:
                if self._face_lm is None:
                    # blendshapes 提供微笑置信度（小头部网络，代价低），
                    # 笑脸判定比嘴部张合更稳（不受头姿影响）
                    self._face_lm = vision.FaceLandmarker.create_from_options(
                        vision.FaceLandmarkerOptions(
                            base_options=self._base("face_landmarker"),
                            running_mode=vision.RunningMode.VIDEO,
                            num_faces=5,
                            output_face_blendshapes=True,
                        ))
        return self._face_lm

    def _get_hand_landmarker(self) -> vision.HandLandmarker:
        if self._hand_lm is None:
            with self._lock:
                if self._hand_lm is None:
                    self._hand_lm = vision.HandLandmarker.create_from_options(
                        vision.HandLandmarkerOptions(
                            base_options=self._base("hand_landmarker"),
                            running_mode=vision.RunningMode.VIDEO,
                            num_hands=2,
                        ))
        return self._hand_lm

    def _get_segmenter(self) -> vision.ImageSegmenter:
        if self._segmenter is None:
            with self._lock:
                if self._segmenter is None:
                    self._segmenter = vision.ImageSegmenter.create_from_options(
                        vision.ImageSegmenterOptions(
                            base_options=self._base(self._segmenter_key),
                            running_mode=vision.RunningMode.VIDEO,
                            # 硬掩膜（类别）给 person_mask，置信图给软 alpha
                            output_category_mask=True,
                            output_confidence_masks=True,
                        ))
        return self._segmenter

    # ------- 每帧统一推理 -------

    def process(self, frame_bgr: np.ndarray, *,
                faces: bool = True, hands: bool = True,
                segmentation: bool = False) -> FrameContext:
        """跑一次全部所需推理，返回 FrameContext。frame 需为连续 BGR ndarray。"""
        h, w = frame_bgr.shape[:2]
        self._frame_id += 1
        self._ts += 33  # 约 30fps 的单调时间戳（VIDEO 模式要求严格递增即可）
        ctx = FrameContext(frame_id=self._frame_id, timestamp_ms=self._ts,
                           width=w, height=h)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if faces:
            r = self._get_face_landmarker().detect_for_video(mp_img, self._ts)
            for lms, bss in zip(r.face_landmarks, r.face_blendshapes):
                lm = _landmarks_to_array(lms)
                ctx.faces.append(_landmarks_to_face_info(lm, bss))

        if hands:
            r = self._get_hand_landmarker().detect_for_video(mp_img, self._ts)
            for lms, hd in zip(r.hand_landmarks, r.handedness):
                info = HandInfo(landmarks=_landmarks_to_array(lms))
                if hd:
                    info.handedness = hd[0].category_name
                ctx.hands.append(info)

        if segmentation:
            spec = self.segmenter_spec
            r = self._get_segmenter().segment_for_video(mp_img, self._ts)
            cat = np.squeeze(r.category_mask.numpy_view())   # (h, w)，0.10 为 2 维
            person = category_to_person_mask(cat, spec["person_is_zero"])
            ctx.person_mask = np.where(person, 255, 0).astype(np.uint8)
            # 软 alpha：由置信图导出的原始前景概率（锐化/精修留给 SegmentEffect）
            if r.confidence_masks:
                ctx.person_alpha = foreground_from_confidence(
                    r.confidence_masks[0].numpy_view(), spec["alpha_invert"])
                self._self_check_border(ctx.person_alpha)

        return ctx

    def _self_check_border(self, alpha: np.ndarray) -> None:
        """边框先验自检：四边几乎总是背景，若边缘 alpha 偏高则掩膜可能整体反相。

        只告警不自动翻转 —— 自动翻转会在"人物占满画面"时误判，反而制造故障。
        与掩膜语义无关的独立参照，能抓住"模型版本升级后约定变了"这类静默问题。
        """
        if self._border_checked:
            return
        self._border_checked = True
        ratio = border_foreground_ratio(alpha)
        if ratio > BORDER_FOREGROUND_WARN:
            print(f"[警告] 分割掩膜疑似整体反相：画面边缘 alpha 均值 {ratio:.2f} "
                  f"(> {BORDER_FOREGROUND_WARN})。请核对 "
                  f"core/infer.py 的 SEGMENTER_SPECS 与模型 {self._segmenter_key} 的约定。")

    def close(self) -> None:
        for s in (self._face_lm, self._hand_lm, self._segmenter):
            if s is not None:
                s.close()
        self._face_lm = self._hand_lm = self._segmenter = None


# ------- 模块级单例 -------

_engine: Optional[InferenceEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> InferenceEngine:
    """全局唯一 InferenceEngine（线程安全）。"""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = InferenceEngine()
    return _engine
