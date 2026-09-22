"""美颜效果：磨皮 / 美白 / 瘦脸 / 大眼（迁移自一期，全部参数化）。

相对一期的实现改进：
  - 大眼：一期为逐像素 Python 双重循环（~3 万次/帧），v2 用局部 cv2.remap
    向量化并加边缘羽化，消除圆形硬边。
  - 瘦脸：一期只是在下颌点位画 1px 黑点（实际无变形效果），v2 实现为
    下颌带状区域的 liquify 平移（朝脸中心，距离轮廓高斯衰减）。
  - 美白掩膜：一期用 FaceDetection 框 + FaceMesh 轮廓双限制；v2 弃用
    FaceDetector（macOS Tasks 版崩溃，见 core/infer.py），仅用 FaceMesh
    轮廓多边形掩膜，语义等价且少一次前向。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..context import FrameContext
from ..pipeline import Effect, NEED_FACES
from ..infer import FACE_OVAL_IDS

# ------- 调参常量（模块顶部集中，中文注释） -------
SMOOTH_DIAMETER = 9        #双边滤波邻域直径（一期口径）
SMOOTH_SIGMA = 60          #双边滤波颜色/空间 sigma（一期口径）
WHITEN_L_MAX = 30.0        # 美白滑杆上限（LAB L 增量；一期 15~18 推荐档）
SLIM_MAX_SHIFT_RATIO = 0.025   # 瘦脸强度=1.0 时的最大水平位移（占帧宽比例）
SLIM_BAND_SIGMA = 30.0     # 瘦脸影响域高斯衰减尺度（像素）
EYE_RADIUS_RATIO = 0.045   # 大眼作用半径（占帧宽比例，一期口径）
EYE_STRENGTH_MAX = 0.5     # 大眼滑杆上限（一期默认 0.18）

# 左/右眼关键点（外角、内角、上睑、下睑 —— 一期口径）
LEFT_EYE_IDS = (33, 133, 159, 145)
RIGHT_EYE_IDS = (362, 263, 386, 374)

# 下颌轮廓左右各 10 点（一期口径）
JAW_LEFT_IDS = list(range(234, 244))
JAW_RIGHT_IDS = list(range(454, 464))


def get_skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """YCrCb 经典肤色阈值掩膜（亚洲人群稳，一期口径）。"""
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
    skin = cv2.GaussianBlur(skin, (7, 7), 0)
    return skin


def face_oval_mask(frame_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """单张脸的轮廓多边形掩膜（高斯羽化）。landmarks: (468,3) 归一化。"""
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = landmarks[FACE_OVAL_IDS][:, :2] * np.array([w, h])
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    return cv2.GaussianBlur(mask, (11, 11), 0)


def whitening(frame_bgr: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """LAB 空间定向美白：仅 mask 区域提升 L 通道。"""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_white = np.clip(l_ch.astype(np.int16) + strength, 0, 255).astype(np.uint8)
    l_new = np.where(mask > 0, l_white, l_ch)
    return cv2.cvtColor(cv2.merge((l_new, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def enlarge_eyes(frame_bgr: np.ndarray, landmarks: np.ndarray,
                 strength: float = 0.18,
                 radius_ratio: float = EYE_RADIUS_RATIO) -> np.ndarray:
    """大眼：双眼局部 remap 放大（向中心收缩采样），边缘羽化混合。"""
    if strength <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()
    for eye_ids in (LEFT_EYE_IDS, RIGHT_EYE_IDS):
        cx = float(np.mean(landmarks[list(eye_ids), 0]) * w)
        cy = float(np.mean(landmarks[list(eye_ids), 1]) * h)
        r = int(radius_ratio * w)
        if r < 4:
            continue
        x1, x2 = max(int(cx - r), 0), min(int(cx + r), w)
        y1, y2 = max(int(cy - r), 0), min(int(cy + r), h)
        if x2 <= x1 or y2 <= y1:
            continue
        xs, ys = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
        dx, dy = xs - cx, ys - cy
        dist = np.sqrt(dx * dx + dy * dy)
        # 采样坐标向中心收缩 => 局部放大（一期公式）
        scale = 1.0 - strength * np.clip(1.0 - dist / r, 0.0, 1.0)
        src_x = np.clip(cx + dx * scale, 0, w - 1).astype(np.float32)
        src_y = np.clip(cy + dy * scale, 0, h - 1).astype(np.float32)
        roi = cv2.remap(frame_bgr, src_x, src_y, cv2.INTER_LINEAR)
        # 边缘权重：靠近圆心 1、圆边 0，避免硬边
        weight = np.clip(1.0 - dist / r, 0.0, 1.0)[..., None].astype(np.float32)
        blended = roi.astype(np.float32) * weight + \
            out[y1:y2, x1:x2].astype(np.float32) * (1.0 - weight)
        out[y1:y2, x1:x2] = np.rint(blended).astype(np.uint8)
    return out


def slim_face(frame_bgr: np.ndarray, landmarks: np.ndarray,
              strength: float = 0.35) -> np.ndarray:
    """瘦脸：下颌两侧带状区域像素朝脸中心水平平移（liquify，remap 实现）。"""
    if strength <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    lm_px = landmarks[:, :2] * np.array([w, h], dtype=np.float32)
    nose_x = float(lm_px[1, 0])   # 鼻尖 x 作为脸中心参考（比一期 w//2 稳）
    out = frame_bgr.copy()
    max_shift = strength * SLIM_MAX_SHIFT_RATIO * w
    for side_ids in (JAW_LEFT_IDS, JAW_RIGHT_IDS):
        pts = lm_px[side_ids]
        y0 = int(max(pts[:, 1].min() - 6, 0))
        y1 = int(min(pts[:, 1].max() + 6, h))
        x_lo = int(max(pts[:, 0].min() - 24, 0))
        x_hi = int(min(pts[:, 0].max() + 24, w))
        if x_hi - x_lo < 8 or y1 - y0 < 8:
            continue
        xs, ys = np.meshgrid(
            np.arange(x_lo, x_hi, dtype=np.float32),
            np.arange(y0, y1, dtype=np.float32))
        # 到最近轮廓点距离 -> 高斯衰减影响域（近似带状）
        d2 = (xs[..., None] - pts[:, 0]) ** 2 + (ys[..., None] - pts[:, 1]) ** 2
        dist = np.sqrt(d2.min(axis=-1))
        w_band = np.exp(-(dist / SLIM_BAND_SIGMA) ** 2)
        # 竖直窗：下颌中段位移最大，向上下衰减（脸颊上部不动）
        t = (ys - ys.min()) / max(ys.max() - ys.min(), 1.0)
        v_win = np.exp(-((t - 0.65) / 0.30) ** 2)
        shift = np.sign(nose_x - xs) * (max_shift * w_band * v_win)
        map_x = (xs + shift).astype(np.float32)
        # 两侧不重叠，均从原始帧采样写入 out，避免左右顺序依赖
        roi = cv2.remap(frame_bgr, map_x, ys.astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out[y0:y1, x_lo:x_hi] = roi
    return out


class BeautyEffect(Effect):
    """实时美颜链：磨皮 → 美白（肤色∩脸轮廓掩膜）→ 瘦脸/大眼 → 收尾锐化。"""

    name = "beauty"
    needs = frozenset({NEED_FACES})

    @staticmethod
    def default_params() -> dict:
        return {
            "smooth": 0.6,       # 磨皮混合比 0~1（一期 0.6）
            "whiten": 15.0,      # 美白强度（LAB L 增量）0~30（一期 15）
            "slim": 0.35,        # 瘦脸强度 0~1
            "eye_enabled": False,  # 大眼开关（一期默认关）
            "eye_strength": 0.18,   # 大眼强度 0~0.5（一期 0.18）
        }

    def process(self, frame: np.ndarray, ctx: FrameContext) -> np.ndarray:
        if not self.enabled:     # 防御直接调用；Pipeline 本身也会跳过
            return frame
        p = self._p()
        if not ctx.faces:
            # 无脸时只做全帧轻磨皮，避免肤色掩膜误伤背景
            if p["smooth"] > 0:
                return self._smooth(frame, p["smooth"])
            return frame

        # 1. 磨皮（保留纹理）
        if p["smooth"] > 0:
            frame = self._smooth(frame, p["smooth"])

        # 2. 美白：肤色掩膜 ∩ 人脸轮廓掩膜
        if p["whiten"] > 0:
            skin = get_skin_mask(frame)
            oval_total = np.zeros_like(skin)
            for f in ctx.faces:
                oval_total = cv2.bitwise_or(oval_total, face_oval_mask(frame, f.landmarks))
            skin = cv2.bitwise_and(skin, oval_total)
            if skin.any():
                frame = whitening(frame, skin, p["whiten"])

        # 3. 瘦脸 / 大眼（逐脸）
        for f in ctx.faces:
            frame = slim_face(frame, f.landmarks, p["slim"])
            if p["eye_enabled"] and p["eye_strength"] > 0:
                frame = enlarge_eyes(frame, f.landmarks, p["eye_strength"])

        # 4. 收尾：轻去噪 + 锐化防糊（一期口径）
        frame = cv2.medianBlur(frame, 3)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        frame = cv2.filter2D(frame, -1, kernel)
        return frame

    @staticmethod
    def _smooth(frame: np.ndarray, mix: float) -> np.ndarray:
        smooth = cv2.bilateralFilter(frame, SMOOTH_DIAMETER, SMOOTH_SIGMA, SMOOTH_SIGMA)
        return cv2.addWeighted(frame, 1.0 - mix, smooth, mix, 0)
