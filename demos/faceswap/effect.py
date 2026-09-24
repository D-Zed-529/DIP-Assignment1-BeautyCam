"""相机效果链中的实时换脸插件。

源脸图片只在首次启用或路径变化时读取、检测并建立 Delaunay 拓扑；目标脸
关键点直接复用本帧 ``FrameContext``，不会为换脸重复启动推理会话。
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from core.context import FrameContext
from core.effects.segment import load_image
from core.infer import FACE_OVAL_IDS, get_engine
from core.pipeline import Effect, NEED_FACES
from demos.faceswap.faceswap import delaunay_triangles, faceswap

MAX_SOURCE_SIDE = 1200  # 超大照片先等比缩小，降低首次建网与逐帧仿射开销
PREVIEW_MAX_SIDE = 960  # GUI 流畅预览上限；拍照原尺寸仍用完整三角网
# 预览保留半数面片控制点及轮廓/五官锚点，减少每帧约 900 次小块仿射。
PREVIEW_FEATURE_IDS = frozenset((
    33, 133, 159, 145, 362, 263, 386, 374,  # 眼睛
    1, 4, 6, 168, 197, 2, 98, 327,            # 鼻子
    0, 13, 14, 17, 61, 78, 81, 87, 91, 95,
    291, 308, 311, 317, 321, 324,             # 嘴唇
))
FACE_LIBRARY_DIR = Path(__file__).resolve().parents[2] / "assets" / "faces"
FACE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_face_presets() -> list[str]:
    """返回随项目发布的原创源脸素材，按文件名稳定排序。"""
    if not FACE_LIBRARY_DIR.is_dir():
        return []
    return [str(p) for p in sorted(FACE_LIBRARY_DIR.iterdir())
            if p.suffix.lower() in FACE_EXTENSIONS]


class FaceSwapEffect(Effect):
    """用一张已授权源脸图片替换相机中的第一张人脸。"""

    name = "faceswap"
    needs = frozenset({NEED_FACES})

    def __init__(self, enabled: bool = False, params: dict | None = None):
        super().__init__(enabled=enabled, params=params)
        self._cached_path = ""
        self._source: np.ndarray | None = None
        self._source_lm: np.ndarray | None = None
        self._triangles: list[tuple[int, int, int]] | None = None
        self._preview_triangles: list[tuple[int, int, int]] | None = None
        self._runtime_status = "请选择一张已获授权的源脸图片"

    @staticmethod
    def default_params() -> dict:
        return {
            "source_path": "",
            "consent": False,
            "color_transfer": True,
        }

    @property
    def runtime_status(self) -> str:
        return self._runtime_status

    def reset_temporal(self) -> None:
        """采集源切换时保留源脸缓存；它与相机帧尺寸无关。"""

    @staticmethod
    def _resize_source(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = min(1.0, MAX_SOURCE_SIDE / max(h, w))
        if scale >= 1.0:
            return image
        return cv2.resize(
            image, (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA)

    def _load_source(self, path: str) -> bool:
        self._cached_path = path
        self._source = self._source_lm = self._triangles = None
        self._preview_triangles = None
        image = load_image(path)
        if image is None:
            self._runtime_status = "源脸图片无法读取，请重新选择"
            return False
        image = self._resize_source(image)
        # 与实时帧共用 core.infer 的单例会话；此调用仅在换图后执行一次。
        ctx = get_engine().process(
            image, faces=True, hands=False, segmentation=False)
        if not ctx.faces:
            self._runtime_status = "源图未检测到人脸，请换一张正面清晰照片"
            return False
        lm = ctx.faces[0].landmarks
        pts = lm[:, :2] * np.array([image.shape[1], image.shape[0]])
        triangles = delaunay_triangles(pts, image.shape[:2])
        if not triangles:
            self._runtime_status = "源脸三角网生成失败，请重新选择照片"
            return False
        keep = sorted(set(range(0, len(pts), 2))
                      | set(FACE_OVAL_IDS) | PREVIEW_FEATURE_IDS)
        keep = [i for i in keep if i < len(pts)]
        preview = delaunay_triangles(pts[keep], image.shape[:2])
        self._preview_triangles = [tuple(keep[i] for i in tri)
                                   for tri in preview] if preview else triangles
        self._source, self._source_lm, self._triangles = image, lm, triangles
        self._runtime_status = f"换脸生效中：{os.path.basename(path)}"
        return True

    def process(self, frame: np.ndarray, ctx: FrameContext) -> np.ndarray:
        if not self.enabled:
            return frame
        p = self._p()
        path = str(p["source_path"])
        if not p["consent"]:
            self._runtime_status = "请先确认照片已获授权"
            return frame
        if not path:
            self._runtime_status = "请先选择源脸图片"
            return frame
        if path != self._cached_path and not self._load_source(path):
            return frame
        if self._source is None or self._source_lm is None or not self._triangles:
            return frame
        if not ctx.faces:
            self._runtime_status = "相机中未检测到人脸"
            return frame
        self._runtime_status = f"换脸生效中：{os.path.basename(path)}"
        return faceswap(
            self._source, self._source_lm, frame, ctx.faces[0].landmarks,
            color_transfer=bool(p["color_transfer"]),
            triangles=(self._preview_triangles
                       if self._preview_triangles is not None
                       and max(frame.shape[:2]) <= PREVIEW_MAX_SIDE
                       else self._triangles),
        )
