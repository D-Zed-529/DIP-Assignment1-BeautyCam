"""美颜效果：磨皮 / 美白 / 瘦脸 / 大眼（迁移自一期，全部参数化）。

相对一期的实现改进：
  - 大眼：一期为逐像素 Python 双重循环（~3 万次/帧），v2 用局部 cv2.remap
    向量化并加边缘羽化，消除圆形硬边。
  - 瘦脸：一期只是在下颌点位画 1px 黑点（实际无变形效果）；v2~v4 经
    过手调高斯场与 MLS 全局变形两代；v5（当前）改为紧支撑 RBF 液化
    笔刷（core/liquify.py），下颌链沿轮廓法向内收 + 下巴尖上收，效果
    与性能详见 slim_face 注释与 docs/research-notes.md。
  - 美白：默认全身肤色（含脖子/手臂），可用 whiten_scope="face" 退回
    一期"仅脸部"口径；美白量随掩膜置信度渐变（软 alpha，无二值硬边）。
  - 美白掩膜：一期用 FaceDetection 框 + FaceMesh 轮廓双限制；v2 弃用
    FaceDetector（macOS Tasks 版崩溃，见 core/infer.py），仅用 FaceMesh
    轮廓多边形掩膜，语义等价且少一次前向。

调试：scripts/debug_slim.py 可输出识别标注图 / 位移场箭头图 / A/B
对照 / wipe 拼接与客观位移指标，瘦脸调参先跑它再看真图。
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..context import FrameContext
from ..liquify import rbf_liquify_maps
from ..mls import identity_maps, mls_similarity_maps
from ..pipeline import Effect, NEED_FACES
from ..infer import FACE_OVAL_IDS

# ------- 调参常量（模块顶部集中，中文注释） -------
SMOOTH_DIAMETER = 9        #双边滤波邻域直径（一期口径）
SMOOTH_SIGMA = 60          #双边滤波颜色/空间 sigma（一期口径）
SMOOTH_DOWNSCALE = 2       # 磨皮在 1/2 分辨率进行（皮肤低频，视觉等价）：
                           # 720p 双边 5.0ms → 半分辨率 ~1.6ms
WHITEN_L_MAX = 30.0        # 美白滑杆上限（LAB L 增量；一期 15~18 推荐档）
SLIM_MAX_SHIFT_RATIO = 0.055   # 瘦脸强度=1.0 时的最大位移命令（占帧宽比例）
SLIM_PROTECT_FADE_PX = 32.0     # 上半脸保护窗衰减带宽度（px）：眉线以上场→0
SLIM_BRUSH_FACE_RATIO = 0.30   # v5 液化笔刷半径 = 0.30 × 脸宽（盖住脸颊主体）
SLIM_MIN_BRUSH_PX = 24.0       # 笔刷半径下限（px）：极小脸时防核过尖
SLIM_TAPER_EDGE = 0.32         # 下颌链两端（下巴/耳侧）幅度比例（中段峰值 1.0）
SLIM_CHIN_LIFT_RATIO = 0.25    # 下巴尖上收幅度 = 0.25 × max_shift（瘦下巴）
SLIM_CHIN_WEIGHT = 2.5         # 下巴动点权值（抵消下方保护点的分母衰减）
SLIM_UNDER_CHIN_DROP = 0.5     # 下巴下方合成保护点距离 = 0.5 ×（下巴到嘴距）
SLIM_UNDER_CHIN_WEIGHT = 2.0   # 下方保护点权值：挡住笔刷向下挖脖子
SLIM_FAR_RATIO_ZERO = 0.52     # 远端塌缩比 ≤ 此值：远端链幅度归零（重度侧脸）
SLIM_FAR_RATIO_FULL = 0.90     # 远端塌缩比 ≥ 此值：远端链全幅度（正脸实测 0.94）
SLIM_BRUSH_HEIGHT_RATIO = 0.62  # 笔刷半径兜底：0.62×(额顶到下巴高)，侧头时脸宽塌缩防笔刷缩水
EYE_RADIUS_RATIO = 0.045   # 大眼作用半径（占帧宽比例，作半径上限兜底）
EYE_RADIUS_FROM_CORNERS = 0.85   # 大眼半径 = 眼角距 × 系数（自适应透视缩短）
EYE_MIN_CORNER_RATIO = 0.02      # 眼角距（归一化）低于此值视为透视塌缩，跳过该眼
EYE_STRENGTH_MAX = 0.5     # 大眼滑杆上限（一期默认 0.18）

# ------- 头姿门控（侧脸防伪影，实机反馈驱动） -------
# 2D 液化变形隐含"正脸假设"：头偏航后远端下颌链的投影塌缩进脸颊中部，
# 两链互相拉扯会产生伪影。防伪影主体已改为"远端链连续衰减"
# （见 slim_face_controls：按两链塌缩比的几何量连续降幅度，转头全程
# 无硬切），全局门控只作幅度随角度的平滑递减：
#   - 估计偏航 ≤55° 全强度（估计值对中低角度系统性偏低，约对应真实
#     转头 72° 内效果不打折）；
#   - 55°~95° 线性衰减；
#   - ≥95° 保留 POSE_GATE_MIN（一半）强度——实机反馈"打一半试试"，
#     极端角度不再关死，衰减后只作用于可见侧近端轮廓，实测无伪影。
YAW_FULL_DEG = 55.0        # |yaw| ≤ 此值：变形全强度
YAW_ZERO_DEG = 95.0        # |yaw| ≥ 此值：变形衰减到下限（不再归零）
POSE_GATE_MIN = 0.5        # 门控下限（"打一半"）：极端角度保留的强度比例

# 左/右眼关键点（外角、内角、上睑、下睑 —— 一期口径）
LEFT_EYE_IDS = (33, 133, 159, 145)
RIGHT_EYE_IDS = (362, 263, 386, 374)

# 下颌链（取自 FACE_OVAL 轮廓序的下半段，下巴 152 两侧）。
# 注意：一期 range(234,244)/range(454,464) 并不在标准轮廓序上，会覆盖到
# 鼻翼/脸颊内测区域，导致瘦脸变形跑到鼻子上（实机反馈已验证）。
JAW_LEFT_IDS = [148, 176, 149, 150, 136, 172, 58, 132, 93]
JAW_RIGHT_IDS = [377, 400, 378, 379, 365, 397, 288, 361, 323, 454]


# v5 液化保护点（id, 权值）：D=0 只进 Shepard 分母，把五官附近的场
# 平滑拉向 0（凸加权平均，只衰减不振荡）。权值越大保护越强。
# 真图实测（1280×1920，R≈154px）：无保护点时嘴角被拖 12.5px；加保护
# 后 s=0.4 降到 ~2px。下巴尖不是保护点——它是动点（瘦下巴，见
# slim_face_controls），其正下方的合成保护点负责挡住笔刷挖进脖子。
SLIM_GUARDS = ((61, 5.0), (291, 5.0),          # 嘴角（最强保护）
               (33, 2.0), (133, 2.0), (362, 2.0), (263, 2.0),   # 眼角
               (1, 1.5),                        # 鼻尖
               (234, 2.0), (454, 2.0),          # 颞部/耳前
               (127, 1.5), (356, 1.5))

# 下颌链上高于眼线的点（颞/耳侧，贴着耳朵）不作为动点：紧支撑核下
# 直接丢弃即可，影响半径到不了那里（v4 需转锚点，v5 无此必要）。


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
    """头姿门控系数：正脸全强度 → 随角度线性衰减 → 极端角度保留
    POSE_GATE_MIN（"打一半"，不归零）。"""
    a = abs(yaw_deg)
    if a <= full:
        return 1.0
    if a >= zero:
        return POSE_GATE_MIN
    return 1.0 - (1.0 - POSE_GATE_MIN) * (a - full) / (zero - full)


def get_skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """YCrCb 经典肤色阈值掩膜（亚洲人群稳，一期口径）。

    半分辨率计算再上采样：掩膜本就要 7px 高斯羽化（软边缘），统计性的
    阈值结果在 1/2 分辨率上几乎不变，720p 实测 ~1.3ms → ~0.6ms。
    """
    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (max(w // 2, 1), max(h // 2, 1)),
                       interpolation=cv2.INTER_AREA)
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
    skin = cv2.GaussianBlur(skin, (7, 7), 0)
    return cv2.resize(skin, (w, h), interpolation=cv2.INTER_LINEAR)


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
    x, y, bw, bh = cv2.boundingRect(mask)   # native 单遍扫描，比 np.nonzero 快
    if bw == 0 or bh == 0:
        return frame_bgr
    crop = frame_bgr[y:y + bh, x:x + bw]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    alpha = mask[y:y + bh, x:x + bw].astype(np.float32) / 255.0
    l_new = np.clip(l_ch + strength * alpha, 0, 255).astype(np.uint8)
    out = frame_bgr.copy()
    out[y:y + bh, x:x + bw] = cv2.cvtColor(
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
              yaw_deg: float | None = None,
              method: str = "v5") -> np.ndarray:
    """瘦脸 v5：紧支撑 RBF 液化（core/liquify.py），关键点驱动。

    v4（MLS 全局最小二乘）实测失败模式：全局场 + 大量锚点互相拉锯，
    动点位移被稀释、方向被掰歪——strength=1.0 时下巴被挤成尖锥、嘴
    角被斜向拖拽，默认强度下又弱到不可见。v5 改用液化笔刷语义：

      - 方向：沿下颌链轮廓法向（垂直于链切向、指向脸内），不再是
        "指向脸中心"——下巴附近法向自然转为水平轻夹、下颌角处斜向
        上收，符合真实"瘦下颌"运动；
      - Shepard 归一化叠加：场是各笔刷位移命令的凸加权平均，处处
        有界、同向笔刷间形成平滑平台（脸颊整体内收，而非贴线窄带），
        动点处位移 ≈ 命令值，强度所见即所得；
      - 无显式锚点：眼线以上链点不进动点 + 笔刷半径（0.30×脸宽）盖
        不到眼/嘴 + 沿链锥形衰减（下巴/耳端弱）+ 眉线以上保护窗。

    侧脸防伪影（全部连续、无硬切）：远端下颌链按"塌缩比"（两链到鼻尖
    线距离比）连续降幅度；全局门控仅兜底。method="v4" 保留 MLS 路径
    供 A/B 对照。

    性能：变形只在脸围盒 ROI 内 remap（场在 ROI 边界严格为 0，逐位
    等价于全帧 remap），720p 实测 5.7ms → ~2ms。
    """
    if strength <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    map_x, map_y, roi = _slim_maps_core(w, h, landmarks, strength,
                                        yaw_deg=yaw_deg, method=method)
    if roi is None:
        return frame_bgr
    if roi == (0, 0, w, h):    # v4 MLS 路径（全帧场，A/B 对照不进热路径）
        return cv2.remap(frame_bgr, map_x, map_y,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    x1, y1, x2, y2 = roi
    out = frame_bgr.copy()
    # 采样图换算到 ROI 子图坐标系（减 ROI 原点）；场在 ROI 边界为 0，
    # 采样不会越过子图边缘，与全帧 remap 逐位一致
    sub = frame_bgr[y1:y2, x1:x2]
    out[y1:y2, x1:x2] = cv2.remap(
        sub, map_x - np.float32(x1), map_y - np.float32(y1),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def slim_face_controls(w: int, h: int, landmarks: np.ndarray,
                       strength: float = 0.40,
                       yaw_deg: float | None = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float,
                                  list[int], list[int]]:
    """瘦脸 v5 控制点构建（纯函数，供 slim_face_maps 与 debug 可视化）。

    返回 (P, D, W, R, mover_ids, guard_ids)：P (K,2) 控制点像素坐标；
    D (K,2) 内容位移命令（动点非零、保护点为 0）；W (K,) 点权；
    R 笔刷半径；mover_ids/guard_ids 为对应 landmark id（debug 标注
    用，合成保护点为 -1）。动点 = 下颌链（眼线以上丢弃）+ 下巴尖
    （上收缩下巴）；保护点 = 嘴角/眼/鼻/颞部 + 下巴正下方合成点。
    strength 须为已过门控的值。
    """
    lm_px = landmarks[:, :2] * np.array([w, h], dtype=np.float32)
    nose_x = float(lm_px[1, 0])
    max_shift = strength * SLIM_MAX_SHIFT_RATIO * w
    # 笔刷半径的尺寸基准：转头时两颧间距(234↔454)按 cos 塌缩，用
    # "额顶到下巴高度"（不受偏航影响）按比例兜底，效果范围不随转头缩小。
    face_w = float(np.hypot(float(lm_px[454, 0] - lm_px[234, 0]),
                            float(lm_px[454, 1] - lm_px[234, 1])))
    face_h = float(np.hypot(float(lm_px[10, 0] - lm_px[152, 0]),
                            float(lm_px[10, 1] - lm_px[152, 1])))
    face_base = max(face_w, SLIM_BRUSH_HEIGHT_RATIO * face_h)
    brush_r = max(face_base * SLIM_BRUSH_FACE_RATIO, SLIM_MIN_BRUSH_PX)

    # 远端链连续衰减（替代旧版"超过 15° 整条跳过"的硬切）：以两链质心
    # 到鼻尖竖直线的距离比度量远端塌缩程度——基本正脸 ≈1，转头越大越
    # 小。比值 ≥SLIM_FAR_RATIO_FULL 全幅度、≤SLIM_FAR_RATIO_ZERO 归零，
    # 中间 smoothstep 连续过渡，转头过程幅度渐变无突变。
    chains = [JAW_LEFT_IDS, JAW_RIGHT_IDS]
    c_dist = [abs(float(lm_px[ids, 0].mean()) - nose_x) for ids in chains]
    near_i = 0 if c_dist[0] >= c_dist[1] else 1
    ratio = c_dist[1 - near_i] / max(c_dist[near_i], 1e-3)
    tt = float(np.clip((ratio - SLIM_FAR_RATIO_ZERO)
                       / (SLIM_FAR_RATIO_FULL - SLIM_FAR_RATIO_ZERO), 0.0, 1.0))
    far_factor = tt * tt * (3.0 - 2.0 * tt)
    side_factors = [1.0, 1.0]
    side_factors[1 - near_i] = far_factor

    # 脸中心参考：鼻尖与下巴中点上移一点（避开下巴尖的极值）
    center = np.array([(float(lm_px[1, 0]) + float(lm_px[152, 0])) / 2,
                       (float(lm_px[1, 1]) + float(lm_px[152, 1])) / 2])
    eyeline_y = float(np.median(lm_px[[145, 374], 1]))
    mouth_y = float(np.median(lm_px[[61, 291], 1]))
    span = max(mouth_y - eyeline_y, 1e-3)

    movers_P, movers_D, mover_ids, movers_W = [], [], [], []
    guard_pts, guard_ws, guard_ids_f = [], [], []
    for side_ids, f_i in zip(chains, side_factors):
        if f_i <= 0.0:
            continue    # 重度侧脸：远端链完全塌缩，整条退出（比值已连续过渡）
        keep = lm_px[side_ids, 1] >= eyeline_y
        kept_ids = [i for i, k in zip(side_ids, keep) if k]
        kp = lm_px[kept_ids]
        n = len(kp)
        if n == 0:
            continue
        # 沿链位置渐变（t=0 下巴端 → 1 耳端）：中段（下颌角/脸颊）峰值
        # 1.0，两端收敛到 SLIM_TAPER_EDGE——下巴端弱避免尖锥，耳端弱
        # 避免扯到耳朵。
        t = np.linspace(0.0, 1.0, n)
        taper = SLIM_TAPER_EDGE + (1.0 - SLIM_TAPER_EDGE) * 4.0 * t * (1.0 - t)
        # 高度因子（smoothstep）：动点 y 从眼线(0)降到嘴线(1)，幅度渐起
        hf = np.clip((kp[:, 1] - eyeline_y) / span, 0.0, 1.0)
        hfac = hf * hf * (3.0 - 2.0 * hf)
        # 轮廓法向：链切向（中心差分）旋转 90°，定向指向脸中心一侧
        if n >= 2:
            tang = np.gradient(kp, axis=0)
            normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
        else:
            normal = (center - kp)[None, :]
        nrm = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = np.divide(normal, np.maximum(nrm, 1e-9))
        inward = (center[None, :] - kp) * normal
        flip = (inward.sum(axis=1) < 0)
        normal[flip] *= -1.0
        for pid, p_i, u_i, tp, h_i in zip(kept_ids, kp, normal, taper, hfac):
            amount = max_shift * float(tp) * float(h_i) * float(f_i)
            if amount < 0.5 or not np.isfinite(amount):
                continue    # 位移过小的点不进求解（等价于锚）
            movers_P.append(p_i)
            movers_D.append(u_i * amount)
            mover_ids.append(pid)
            movers_W.append(1.0)

    # 下巴尖上收（"瘦下巴"）：下巴 152 也作为动点，方向指向脸中心
    #（≈竖直向上，收出 V 线）；权值高于 1，抵消正下方保护点的分母
    # 衰减。下方合成保护点（无 landmark）防止笔刷把脖子内容拽上来。
    if max_shift >= 0.5:
        chin_p = lm_px[152]
        chin_dir = center - chin_p
        cn = float(np.hypot(*chin_dir))
        if cn > 1e-3:
            movers_P.append(chin_p)
            movers_D.append(chin_dir / cn * max_shift * SLIM_CHIN_LIFT_RATIO)
            mover_ids.append(152)
            movers_W.append(SLIM_CHIN_WEIGHT)
        under_chin = np.array([float(chin_p[0]),
                               float(chin_p[1])
                               + SLIM_UNDER_CHIN_DROP * (float(chin_p[1])
                                                         - mouth_y)])
        guard_pts.append(under_chin)
        guard_ws.append(SLIM_UNDER_CHIN_WEIGHT)
        guard_ids_f.append(-1)      # 合成点（无 landmark id）

    # 保护点：D=0、权值>1，只进 Shepard 分母（五官附近场平滑衰减）
    for idx, gw in SLIM_GUARDS:
        guard_pts.append(lm_px[idx])
        guard_ws.append(gw)
        guard_ids_f.append(idx)

    P = np.array(movers_P + guard_pts, dtype=np.float64)
    D = np.array(movers_D + [np.zeros(2)] * len(guard_pts), dtype=np.float64)
    W = np.array(movers_W + guard_ws, dtype=np.float64)
    return P, D, W, brush_r, mover_ids, guard_ids_f


def _slim_maps_mls(w: int, h: int, lm_px: np.ndarray, strength: float,
                   yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """瘦脸 v4 的 MLS 位移场（保留作 A/B 对照，不再默认使用）。"""
    nose_x = float(lm_px[1, 0])
    max_shift = strength * SLIM_MAX_SHIFT_RATIO * w
    chains = [JAW_LEFT_IDS, JAW_RIGHT_IDS]
    if abs(yaw_deg) > FAR_SIDE_SKIP_DEG:
        chains.sort(key=lambda ids: abs(float(lm_px[ids, 0].mean()) - nose_x))
        chains = chains[1:]
    center = np.array([(float(lm_px[1, 0]) + float(lm_px[152, 0])) / 2,
                       (float(lm_px[1, 1]) + float(lm_px[152, 1])) / 2])
    eyeline_y = float(np.median(lm_px[[145, 374], 1]))
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
        t = np.linspace(0.0, 1.0, n)
        taper = 0.4 + 2.4 * t * (1.0 - t)
        hf = np.clip((moving[:, 1] - eyeline_y) / span, 0.0, 1.0)
        hfac = hf * hf * (3.0 - 2.0 * hf)
        for p_i, tp, h_i in zip(moving, taper, hfac):
            direction = center - p_i
            norm = float(np.hypot(*direction))
            if norm < 1e-3 or h_i <= 0.0:
                _pin(p_i)
                continue
            amount = max_shift * float(tp) * float(h_i)
            movers_P.append(p_i)
            movers_Q.append(p_i + direction / norm * amount)

    anchor_ids = (33, 133, 362, 263, 105, 334, 1, 10,
                  338, 297, 332, 384, 385, 387, 388, 127, 234,
                  109, 67, 103, 54, 21, 61, 291, 152)
    for idx in anchor_ids:
        _pin(lm_px[idx])
    return mls_similarity_maps(h, w, np.array(movers_P), np.array(movers_Q))


def _slim_maps_core(w: int, h: int, landmarks: np.ndarray,
                    strength: float = 0.40,
                    yaw_deg: float | None = None,
                    method: str = "v5"
                    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray],
                               Optional[tuple[int, int, int, int]]]:
    """瘦脸位移场核心（ROI 形式，热路径）。

    返回 (map_x, map_y, roi)：map 仅 ROI 尺寸（绝对采样坐标），roi=None
    表示无需变形。眉线以上保护窗在 ROI 行内应用（行函数，与全帧应用
    逐位一致）。v4 MLS 路径仍生成全帧场（roi=整帧）。
    """
    if strength <= 0:
        return None, None, None
    if yaw_deg is None:
        yaw_deg = estimate_yaw_deg(landmarks)
    strength = strength * pose_gate(yaw_deg)
    if strength <= 0:
        return None, None, None
    lm_px = landmarks[:, :2] * np.array([w, h], dtype=np.float32)

    if method == "v4":
        map_x, map_y = _slim_maps_mls(w, h, lm_px, strength, yaw_deg)
        roi = (0, 0, w, h)
    else:
        P, D, W, brush_r, _m, _g = slim_face_controls(w, h, landmarks,
                                                      strength, yaw_deg)
        map_x, map_y, roi = rbf_liquify_maps(h, w, P, D, brush_r,
                                             weights=W, return_roi=True)
    if roi is None:
        return None, None, None

    # 上半脸保护窗：眉线以上竖直余弦衰减到严格恒等（行函数，ROI 内
    # 应用与全帧应用等价——ROI 外的 map 本来就恒等）。
    x1, y1, x2, y2 = roi
    brow_y = float(np.min(lm_px[[105, 334], 1]))
    yy = np.arange(y1, y2, dtype=np.float32)[:, None]
    band = np.clip((yy - brow_y) / SLIM_PROTECT_FADE_PX, 0.0, 1.0)
    band = (band * band * (3.0 - 2.0 * band)).astype(np.float32)
    id_x = np.tile(np.arange(x1, x2, dtype=np.float32), (y2 - y1, 1))
    id_y = np.tile(np.arange(y1, y2, dtype=np.float32)[:, None], (1, x2 - x1))
    map_x = id_x + (map_x - id_x) * band
    map_y = id_y + (map_y - id_y) * band
    return map_x, map_y, roi


def slim_face_maps(w: int, h: int, landmarks: np.ndarray,
                   strength: float = 0.40,
                   yaw_deg: float | None = None,
                   method: str = "v5"
                   ) -> tuple[np.ndarray, np.ndarray]:
    """瘦脸位移场（slim_face 的纯函数核，供测试直接断言场量）。

    method="v5"（默认）：紧支撑 RBF 液化；"v4"：MLS（A/B 对照）。
    返回 (map_x, map_y)：眼线以上被保护窗压到严格恒等，ROI 外恒等，
    下颌区沿轮廓法向内收。
    """
    mx, my, roi = _slim_maps_core(w, h, landmarks, strength,
                                  yaw_deg=yaw_deg, method=method)
    fx, fy = identity_maps(h, w)
    if roi is not None:
        x1, y1, x2, y2 = roi
        fx[y1:y2, x1:x2] = mx
        fy[y1:y2, x1:x2] = my
    return fx, fy


class BeautyEffect(Effect):
    """实时美颜链：磨皮 → 美白（肤色∩脸轮廓掩膜）→ 瘦脸/大眼 → 收尾锐化。"""

    name = "beauty"
    needs = frozenset({NEED_FACES})
    supports_gpu = True          # 张量快路径见 _torch_impl.beauty_process_t

    @staticmethod
    def default_params() -> dict:
        return {
            "smooth": 0.6,       # 磨皮混合比 0~1（一期 0.6）
            "whiten": 15.0,      # 美白强度（LAB L 增量）0~30（一期 15）
            "whiten_scope": "skin",  # 美白范围："skin" 全身肤色 / "face" 仅脸部
            "slim": 0.40,        # 瘦脸强度 0~1
            "eye_enabled": False,  # 大眼开关（一期默认关）
            "eye_strength": 0.18,   # 大眼强度 0~0.5（一期 0.18）
            "finish": True,        # 收尾中值去噪与锐化；实时自然档可关闭
        }

    def process_gpu(self, frame_t, ctx: FrameContext):
        from ._torch_impl import beauty_process_t
        return beauty_process_t(frame_t, ctx, self._p())

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
        if p["finish"]:
            frame = cv2.medianBlur(frame, 3)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            frame = cv2.filter2D(frame, -1, kernel)
        return frame

    @staticmethod
    def _smooth(frame: np.ndarray, mix: float) -> np.ndarray:
        """磨皮：半分辨率双边滤波后上采样混合（720p 实测 5.0ms → 1.6ms）。

        皮肤纹理是低频信号，1/2 分辨率双边在视觉上与全分辨率几乎不可
        分（上采样本身就是低通）；混合比语义与一期口径一致。
        """
        if mix <= 0:
            return frame
        h, w = frame.shape[:2]
        f = SMOOTH_DOWNSCALE
        small = cv2.resize(frame, (max(w // f, 1), max(h // f, 1)),
                           interpolation=cv2.INTER_AREA)
        small = cv2.bilateralFilter(small, SMOOTH_DIAMETER,
                                    SMOOTH_SIGMA, SMOOTH_SIGMA)
        smooth = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(frame, 1.0 - mix, smooth, mix, 0)
