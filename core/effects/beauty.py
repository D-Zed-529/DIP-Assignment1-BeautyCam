"""美颜效果：磨皮 / 美白 / 瘦脸 / 大眼（迁移自一期，全部参数化）。

相对一期的实现改进：
  - 大眼：一期为逐像素 Python 双重循环（~3 万次/帧），v2 用局部 cv2.remap
    向量化并加边缘羽化，消除圆形硬边。
  - 瘦脸：一期只是在下颌点位画 1px 黑点（实际无变形效果），v2 实现为
    下颌带状区域的 liquify 内收（采样朝脸外侧、内容向中心移动）。
  - 美白：默认全身肤色（含脖子/手臂），可用 whiten_scope="face" 退回
    一期"仅脸部"口径；美白量随掩膜置信度渐变（软 alpha，无二值硬边）。
  - 美白掩膜：一期用 FaceDetection 框 + FaceMesh 轮廓双限制；v2 弃用
    FaceDetector（macOS Tasks 版崩溃，见 core/infer.py），仅用 FaceMesh
    轮廓多边形掩膜，语义等价且少一次前向。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..context import FrameContext
from ..mls import identity_maps, mls_similarity_maps
from ..pipeline import Effect, NEED_FACES
from ..infer import FACE_OVAL_IDS

# ------- 调参常量（模块顶部集中，中文注释） -------
SMOOTH_DIAMETER = 9        #双边滤波邻域直径（一期口径）
SMOOTH_SIGMA = 60          #双边滤波颜色/空间 sigma（一期口径）
WHITEN_L_MAX = 30.0        # 美白滑杆上限（LAB L 增量；一期 15~18 推荐档）
SLIM_MAX_SHIFT_RATIO = 0.055   # 瘦脸强度=1.0 时的最大水平位移（占帧宽比例）
SLIM_PROTECT_FADE_PX = 32.0     # 上半脸保护窗衰减带宽度（px）：眉线以上场→0
EYE_RADIUS_RATIO = 0.045   # 大眼作用半径（占帧宽比例，作半径上限兜底）
EYE_RADIUS_FROM_CORNERS = 0.85   # 大眼半径 = 眼角距 × 系数（自适应透视缩短）
EYE_MIN_CORNER_RATIO = 0.02      # 眼角距（归一化）低于此值视为透视塌缩，跳过该眼
EYE_STRENGTH_MAX = 0.5     # 大眼滑杆上限（一期默认 0.18）

# ------- 头姿门控（侧脸防伪影，实机反馈驱动） -------
# 2D 液化变形隐含"正脸假设"：头偏航后远端下颌链的投影塌缩进脸颊中部，
# 变形带会横穿脸面产生拉扯伪影。通行做法（MediaPipe 刚体变换头姿估计、
# MLLS 变形文献）是按头姿给变形强度加门控，超阈值直接关闭。
# 实机反馈：窄带+紧门控让正脸瘦脸几乎不可见 → 幅度与内侧衰减放宽，
# 门控全强度上限放宽到 20°（正脸小幅姿态波动不再吃掉强度）。
YAW_FULL_DEG = 20.0        # |yaw| ≤ 此值：变形全强度
YAW_ZERO_DEG = 40.0        # |yaw| ≥ 此值：变形完全关闭
FAR_SIDE_SKIP_DEG = 15.0   # |yaw| 超过此值：跳过"远端"下颌链（投影已塌缩）

# 左/右眼关键点（外角、内角、上睑、下睑 —— 一期口径）
LEFT_EYE_IDS = (33, 133, 159, 145)
RIGHT_EYE_IDS = (362, 263, 386, 374)

# 下颌链（取自 FACE_OVAL 轮廓序的下半段，下巴 152 两侧）。
# 注意：一期 range(234,244)/range(454,464) 并不在标准轮廓序上，会覆盖到
# 鼻翼/脸颊内测区域，导致瘦脸变形跑到鼻子上（实机反馈已验证）。
JAW_LEFT_IDS = [148, 176, 149, 150, 136, 172, 58, 132, 93]
JAW_RIGHT_IDS = [377, 400, 378, 379, 365, 397, 288, 361, 323, 454]


def estimate_yaw_deg(landmarks: np.ndarray) -> float:
    """由关键点不对称度近似头姿偏航角（度）。

    鼻尖(1) 到左右脸缘(234/454) 的水平距离比：正脸约 0，侧脸趋向 ±1，
    乘 90° 线性近似。仅用于变形门控阈值判断（15°/32° 档），不需要
    MediaPipe 变换矩阵的精确欧拉角；且自动兼容预览镜像。
    """
    nose = float(landmarks[1, 0])
    d_left = abs(float(landmarks[234, 0]) - nose)
    d_right = abs(float(landmarks[454, 0]) - nose)
    if d_left + d_right < 1e-6:
        return 0.0
    return (d_right - d_left) / (d_right + d_left) * 90.0


def pose_gate(yaw_deg: float,
              full: float = YAW_FULL_DEG,
              zero: float = YAW_ZERO_DEG) -> float:
    """头姿门控系数：正脸全强度 → 侧脸线性衰减到 0。"""
    a = abs(yaw_deg)
    if a <= full:
        return 1.0
    if a >= zero:
        return 0.0
    return 1.0 - (a - full) / (zero - full)


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
    """LAB 空间定向美白：美白量随掩膜置信度（0~255）渐变，避免二值硬边。

    只在掩膜包围盒内运算（盒外 alpha=0，输出严格不变，也省 LAB 全帧往返）。
    """
    ys_, xs_ = np.nonzero(mask)
    if len(ys_) == 0:
        return frame_bgr
    y1, y2 = int(ys_.min()), int(ys_.max()) + 1
    x1, x2 = int(xs_.min()), int(xs_.max()) + 1
    crop = frame_bgr[y1:y2, x1:x2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    alpha = mask[y1:y2, x1:x2].astype(np.float32) / 255.0
    l_new = np.clip(l_ch + strength * alpha, 0, 255).astype(np.uint8)
    out = frame_bgr.copy()
    out[y1:y2, x1:x2] = cv2.cvtColor(
        cv2.merge((l_new, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
    return out


def enlarge_eyes(frame_bgr: np.ndarray, landmarks: np.ndarray,
                 strength: float = 0.18,
                 radius_ratio: float = EYE_RADIUS_RATIO,
                 yaw_deg: float | None = None) -> np.ndarray:
    """大眼：双眼局部 remap 放大（向中心收缩采样），边缘羽化混合。

    侧脸处理：
      - 半径按该眼的眼角距自适应（远端眼透视缩短 → 半径同比缩小，
        圆变形不再溢出到鼻梁/颧骨）；
      - 头姿门控：yaw 超阈值时强度线性衰减到 0；
      - 眼角距塌缩过小时直接跳过该眼（关键点在极端侧脸下不可信）。
    """
    if strength <= 0:
        return frame_bgr
    if yaw_deg is None:
        yaw_deg = estimate_yaw_deg(landmarks)
    strength = strength * pose_gate(yaw_deg)
    if strength <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()
    # 眼角对（外角/内角）：左眼 33-133，右眼 362-263
    corner_pairs = {LEFT_EYE_IDS: (33, 133), RIGHT_EYE_IDS: (362, 263)}
    for eye_ids, (c_out, c_in) in corner_pairs.items():
        cx = float(np.mean(landmarks[list(eye_ids), 0]) * w)
        cy = float(np.mean(landmarks[list(eye_ids), 1]) * h)
        corner_dist = abs(float(landmarks[c_out, 0]) - float(landmarks[c_in, 0]))
        if corner_dist < EYE_MIN_CORNER_RATIO:
            continue          # 该眼透视塌缩（极端侧脸），跳过
        # 自适应半径：眼角距 × 系数，并以上限兜底
        r = int(min(EYE_RADIUS_FROM_CORNERS * corner_dist * w,
                    radius_ratio * w))
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
              strength: float = 0.40,
              yaw_deg: float | None = None) -> np.ndarray:
    """瘦脸 v4：MLS 刚体变形（Schaefer 2006），关键点驱动。

    v2/v3 的手调高斯位移场是对 MLS 的粗糙近似：方向硬编码水平、σ 靠
    拍脑袋。v4 直接实现教科书算法（core/mls.py，复数形式 + 粗网格加速）：
      - 动点：下颌链关键点，沿"指向脸中心"方向内移（跟随轮廓法向，
        不再只是水平方向）；
      - 锚点：眼角/眉/鼻尖/额顶/嘴角/下巴（P==Q），MLS 自动保证这些
        区域稳住——替代 v3 的竖直渐变窗与远端衰减等全部手工机制；
      - 位移沿链渐变（下巴端小、下颌角端弱、中段最大），幅度上限
        0.055×帧宽（1280 下默认强度 0.4 ≈ 28px）。

    侧脸防伪影（沿用）：
      - 头姿门控：|yaw| ≤20° 全强度，20°~40° 线性衰减，≥40° 关闭；
      - |yaw| > 15° 时跳过"远端"下颌链（2D 质心更近鼻尖的塌缩侧）。
    """
    if strength <= 0:
        return frame_bgr
    if yaw_deg is None:
        yaw_deg = estimate_yaw_deg(landmarks)
    strength = strength * pose_gate(yaw_deg)
    if strength <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    map_x, map_y = slim_face_maps(w, h, landmarks, strength=strength,
                                  yaw_deg=yaw_deg)
    return cv2.remap(frame_bgr, map_x, map_y,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def slim_face_maps(w: int, h: int, landmarks: np.ndarray,
                   strength: float = 0.40,
                   yaw_deg: float | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """瘦脸 v4 的位移场（slim_face 的纯函数核，供测试直接断言场量）。

    返回 (map_x, map_y)：眼线以上被锚点钉住（亚像素级），ROI 外严格
    恒等（羽化保证），下颌区沿轮廓法向内收。
    """
    if strength <= 0:
        return identity_maps(h, w)
    if yaw_deg is None:
        yaw_deg = estimate_yaw_deg(landmarks)
    strength = strength * pose_gate(yaw_deg)
    if strength <= 0:
        return identity_maps(h, w)
    lm_px = landmarks[:, :2] * np.array([w, h], dtype=np.float32)
    nose_x = float(lm_px[1, 0])   # 鼻尖 x 作为脸中心参考
    max_shift = strength * SLIM_MAX_SHIFT_RATIO * w
    chains = [JAW_LEFT_IDS, JAW_RIGHT_IDS]
    if abs(yaw_deg) > FAR_SIDE_SKIP_DEG:
        # 远端链 = 2D 质心更靠近鼻尖的那条（投影塌缩方）
        chains.sort(key=lambda ids: abs(float(lm_px[ids, 0].mean()) - nose_x))
        chains = chains[1:]     # 跳过最靠近鼻尖的一条

    # 脸中心参考：鼻尖与下巴中点上移一点（避开下巴尖的极值）
    center = np.array([(float(lm_px[1, 0]) + float(lm_px[152, 0])) / 2,
                       (float(lm_px[1, 1]) + float(lm_px[152, 1])) / 2])

    # 动点门槛：下颌链上高于下眼睑线的点（颞/耳侧，如 454/93）转锚点。
    # 这些点在真实人脸上贴着耳朵，内移会把耳朵区域拉歪。
    eyeline_y = float(np.median(lm_px[[145, 374], 1]))
    # 颞侧高度衰减：动点越接近眼线幅度越小（嘴线处全量、眼线处归零）。
    # 没有这一档时眼线下方最近的动点幅度大、梯度陡，场会漏进额侧。
    mouth_y = float(np.median(lm_px[[61, 291], 1]))
    span = max(mouth_y - eyeline_y, 1e-3)

    movers_P, movers_Q = [], []

    def _pin(p_i):
        movers_P.append(p_i)
        movers_Q.append(p_i.copy())

    for side_ids in chains:
        pts = lm_px[side_ids]
        for p_i in pts[pts[:, 1] < eyeline_y]:
            _pin(p_i)
        moving = pts[pts[:, 1] >= eyeline_y]
        n = len(moving)
        if n == 0:
            continue
        # 沿链位置渐变：下巴端(t=0) 0.4 → 中段(t=0.5) 1.0 → 耳端(t=1) 0.4
        t = np.linspace(0.0, 1.0, n)
        taper = 0.4 + 2.4 * t * (1.0 - t)   # 抛物线，峰=1.0
        # 高度因子（smoothstep）：动点 y 从嘴线升到眼线，幅度 1 → 0
        hf = np.clip((moving[:, 1] - eyeline_y) / span, 0.0, 1.0)
        hfac = hf * hf * (3.0 - 2.0 * hf)
        for p_i, tp, h_i in zip(moving, taper, hfac):
            direction = center - p_i
            norm = float(np.hypot(*direction))
            if norm < 1e-3 or h_i <= 0.0:
                _pin(p_i)   # 高度因子归零的点直接锚住，控制点更干净
                continue
            amount = max_shift * float(tp) * float(h_i)
            movers_P.append(p_i)
            movers_Q.append(p_i + direction / norm * amount)

    # 锚点：眼角/眉/鼻尖/额顶/嘴角/下巴（不动，稳定身份区域）。
    # 上轮廓弧（额侧/颞部）必须整圈钉住：MLS 是全局法，只锚额顶单点时
    # 下颌动点的位移会沿平面向上漏到额头（实测 2~5px 漂移）。
    anchor_ids = (33, 133, 362, 263,          # 眼角
                  105, 334,                    # 眉
                  1,                           # 鼻尖
                  10,                          # 额顶
                  338, 297, 332, 384, 385,     # 右上轮廓弧（颞/额侧）
                  387, 388, 127, 234,          # 颞部收口（眼线高度的轮廓）
                  109, 67, 103, 54, 21,        # 左上轮廓弧
                  61, 291,                     # 嘴角
                  152)                         # 下巴
    for idx in anchor_ids:
        _pin(lm_px[idx])

    map_x, map_y = mls_similarity_maps(h, w, np.array(movers_P), np.array(movers_Q))

    # 上半脸保护窗：眉线以上竖直余弦衰减到严格恒等。MLS 是全局光滑
    # 场，锚点之间会有 1~3px 亚视觉泄漏；该区域场量本就 <2px，乘窗无
    # 可感知差异，但保证强度拉满时发际线以上比特级不动。
    brow_y = float(np.min(lm_px[[105, 334], 1]))
    yy = np.arange(h, dtype=np.float32)[:, None]
    band = np.clip((yy - brow_y) / SLIM_PROTECT_FADE_PX, 0.0, 1.0)
    band = band * band * (3.0 - 2.0 * band)   # smoothstep
    id_x, id_y = identity_maps(h, w)
    map_x = id_x + (map_x - id_x) * band
    map_y = id_y + (map_y - id_y) * band
    return map_x, map_y


class BeautyEffect(Effect):
    """实时美颜链：磨皮 → 美白（肤色∩脸轮廓掩膜）→ 瘦脸/大眼 → 收尾锐化。"""

    name = "beauty"
    needs = frozenset({NEED_FACES})

    @staticmethod
    def default_params() -> dict:
        return {
            "smooth": 0.6,       # 磨皮混合比 0~1（一期 0.6）
            "whiten": 15.0,      # 美白强度（LAB L 增量）0~30（一期 15）
            "whiten_scope": "skin",  # 美白范围："skin" 全身肤色 / "face" 仅脸部
            "slim": 0.40,        # 瘦脸强度 0~1
            "eye_enabled": False,  # 大眼开关（一期默认关）
            "eye_strength": 0.18,   # 大眼强度 0~0.5（一期 0.18）
        }

    def process(self, frame: np.ndarray, ctx: FrameContext) -> np.ndarray:
        if not self.enabled:     # 防御直接调用；Pipeline 本身也会跳过
            return frame
        p = self._p()

        # 1. 磨皮（保留纹理）
        if p["smooth"] > 0:
            frame = self._smooth(frame, p["smooth"])

        # 2. 美白：默认全身肤色（脖子/手臂等皮肤一并提亮）；选"仅脸部"时
        #    再与 FaceMesh 轮廓掩膜求交（一期口径）
        if p["whiten"] > 0:
            skin = get_skin_mask(frame)
            if p["whiten_scope"] == "face" and ctx.faces:
                oval_total = np.zeros_like(skin)
                for f in ctx.faces:
                    oval_total = cv2.bitwise_or(
                        oval_total, face_oval_mask(frame, f.landmarks))
                skin = cv2.min(skin, oval_total)
            if skin.any():
                frame = whitening(frame, skin, p["whiten"])

        if not ctx.faces:
            return frame

        # 3. 瘦脸 / 大眼（逐脸，需关键点；侧脸按头姿门控防伪影）
        for f in ctx.faces:
            yaw = estimate_yaw_deg(f.landmarks)
            frame = slim_face(frame, f.landmarks, p["slim"], yaw_deg=yaw)
            if p["eye_enabled"] and p["eye_strength"] > 0:
                frame = enlarge_eyes(frame, f.landmarks, p["eye_strength"],
                                     yaw_deg=yaw)

        # 4. 收尾：轻去噪 + 锐化防糊（一期口径）
        frame = cv2.medianBlur(frame, 3)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        frame = cv2.filter2D(frame, -1, kernel)
        return frame

    @staticmethod
    def _smooth(frame: np.ndarray, mix: float) -> np.ndarray:
        smooth = cv2.bilateralFilter(frame, SMOOTH_DIAMETER, SMOOTH_SIGMA, SMOOTH_SIGMA)
        return cv2.addWeighted(frame, 1.0 - mix, smooth, mix, 0)
