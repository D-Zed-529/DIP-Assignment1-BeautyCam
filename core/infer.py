"""MediaPipe / ONNX 会话管理（单例）。

P0-1 冒烟结论（2026-09，macOS darwin 27 / Apple M4）：
  - mediapipe 1.0.1：已移除旧 `mp.solutions`；且 Tasks API 检测类图
    （FaceDetector / FaceLandmarker / HandLandmarker）在 Open() 阶段因
    DrishtiMetalHelper 硬崩溃（CPU/GPU 委托均崩）→ 不可用。
  - mediapipe 0.10.21：Tasks API + 显式 CPU 委托全部正常（分割器亦可用）。
  因此锁定 mediapipe==0.10.21，且所有会话显式指定 CPU 委托。

FaceDetector 已弃用：macOS Tasks 版存在 Metal 后处理缺陷；人脸框改由
FaceMesh 轮廓关键点导出（见 _landmarks_to_face_info），还省一次前向。
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
    # Phase 3 起用的自拍分割（multiclass：0背景/1头发/2躯体皮肤/3面部皮肤/4衣服/5其他）
    "selfie_segmenter": "selfie_multiclass_256x256.tflite",
}

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


class InferenceEngine:
    """每帧统一推理入口：FaceMesh / Hands / 自拍分割，结果填入 FrameContext。

    会话懒加载 + 单例（get_engine()）；创建开销大，帧循环里只调 process()。
    """

    def __init__(self, models_dir: Path | str = MODELS_DIR,
                 delegate: int | None = None):
        self.models_dir = Path(models_dir)
        # None 表示用 mediapipe 默认；本机验证默认委托稳定，但保险起见
        # 全部显式 CPU（GPU/Metal 委托在本机崩溃，见模块头注释）
        self._delegate = (
            delegate if delegate is not None
            else mp_tasks.BaseOptions.Delegate.CPU
        )
        self._face_lm: Optional[vision.FaceLandmarker] = None
        self._hand_lm: Optional[vision.HandLandmarker] = None
        self._segmenter: Optional[vision.ImageSegmenter] = None
        self._ts = 0          # VIDEO 模式时间戳必须严格递增
        self._frame_id = 0
        self._lock = threading.Lock()   # 会话懒加载互斥

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
                            base_options=self._base("selfie_segmenter"),
                            running_mode=vision.RunningMode.VIDEO,
                            output_category_mask=True,
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
            r = self._get_segmenter().segment_for_video(mp_img, self._ts)
            cat = np.squeeze(r.category_mask.numpy_view())   # (h, w)，0.10 为 2 维
            # 前景 = 非背景类（头发/皮肤/衣服等全部算人）
            ctx.person_mask = np.where(cat == 0, 0, 255).astype(np.uint8)

        return ctx

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
