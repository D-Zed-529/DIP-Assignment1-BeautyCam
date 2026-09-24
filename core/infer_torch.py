"""PyTorch CUDA 推理后端 —— 替代 MediaPipe Tasks / ONNX Runtime 的完整推理链。

技术栈全面转向 PyTorch（2026-09 Windows / RTX 3060 部署，算力升级后以
更大更高质量的模型替换原有小模型）。四个推理线全部由 TorchScript /
torch 模型在 GPU 上完成：

  FaceMesh（VIDEO 模式语义）
    首帧/丢失 → BlazeFace 短距检测器（128×128 全帧，[-1,1] 归一化）
    → 锚框解码 + NMS → 每个检测框跑 landmarks（256×256 旋转 ROI 裁剪，
    [0,1] 归一化，输出 478×3 在 256 像素域）→ 坐标回投；
    跟踪帧 → 由上一帧 landmarks 导出旋转 ROI（眼线 33↔263 定向）；
    blendshapes 由 52 维 HUND 头部网络计算（输入 = 146 点子集的**像素
    坐标** (1,146,2)，输出原始 0~1 值无需 sigmoid）。
  Hands
    palm 检测（192×192，[0,1] 归一化——实测 [-1,1] 会大量误检）→
    腕→中指根定向 ROI → 224×224 裁剪 → [21×3 landmarks, presence,
    handedness, world]。
  自拍分割（三档可切）
    rvm        RobustVideoMatting resnet50 官方 TorchScript（**默认**，
               带循环时域状态的视频抠图，发丝级 alpha，替代 250KB 二元
               模型成为虚拟背景主力）
    binary     二元 selfie segmenter（256×256，sigmoid 直接出前景概率）
    multiclass 6 类分割（logits → softmax，极性约定见 SEGMENTER_SPECS）
  SCI 低光（torch 版）
    512×512 ONNX→TorchScript 前向，输出取 [1]（增强图），口径同
    infer.LowLightSession（Retinexformer 质量档见 core/retinexformer.py）。
  深度（新增，P3-4）
    Depth Anything V2 Small（HuggingFace transformers 本地权重），
    ctx.depth 为 (h,w) float32 相对深度（值越大越近），供渐进虚化使用。

标定说明（2026-09-23 实测，scripts/calibrate_torch.py，基准 =
mediapipe 0.10.21 CPU 委托同图输出）：
  face_detector   真脸框解码偏差 < 0.005（归一化坐标）
  face_landmarks  478 点平均偏差 0.0018 / 0.0009（x/y）
  face_blendshapes 逐 52 名值与 mediapipe 完全一致（相关 best idx 全对齐）
  selfie_segmenter alpha MAE 0.0038
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .context import FaceInfo, FrameContext, HandInfo
from .gpuops import device, upload_frame
from .infer import (
    BORDER_FOREGROUND_WARN,
    FACE_OVAL_IDS,
    LOWLIGHT_DEFAULT_LEVEL,
    LOWLIGHT_INPUT_SIZE,
    LOWLIGHT_LEVELS,
    SEGMENTER_INTERVAL,
    SEGMENTER_SPECS,
    border_foreground_ratio,
)

TORCH_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "torch"

# ------- MediaPipe 图常数（复刻用；标定记录见模块头） -------
FACE_DETECT_INPUT = 128        # BlazeFace 短距输入
FACE_LM_INPUT = 256            # FaceMesh landmarks 输入（.task 版是 256 不是 192！）
HAND_DETECT_INPUT = 192        # palm 检测输入
HAND_LM_INPUT = 224            # hand landmarks 输入
SEG_INPUT = 256                # 自拍分割输入

FACE_RECT_SCALE = 1.5          # 检测框 → ROI 缩放（face_landmarks_detector_graph.cc）
FACE_TRACK_SCALE = 1.5         # landmarks 跟踪 ROI 缩放（同上 square_long）
HAND_RECT_SCALE = 2.6          # palm 关键点 → ROI 缩放
HAND_TRACK_SCALE = 2.6         # hand landmarks 跟踪 ROI 缩放

FACE_DETECT_NORM = "pm1"       # BlazeFace 输入 [-1,1]（mediapipe 官方口径）
HAND_DETECT_NORM = "01"        # palm 检测 [0,1]（pm1 实测大量误检）
FACE_LM_NORM = "01"            # landmarks 输入 [0,1]

FACE_DETECT_THRESH = 0.5       # 检测分数阈值（face_landmarker 默认）
FACE_TRACK_THRESH = 0.5        # landmarks presence 绝对门限（高清脸）
# presence 低清退化门限（2026-09 实测标定）：landmarks 模型的 presence
# （= sigmoid(Identity_1)）随画面条件整体漂移 —— 同一样例脸在 960 宽裁剪
# 上 0.985、在 480 宽整幅上只有 0.0076，而幻影框两种情况都 ≈0.0000。
# 绝对 0.5 会把低清真脸整个拒掉（0 脸），因此加**相对门限**：
# 帧内最大 presence < 0.5 时，接受 presence ≥ 0.4×pmax 且 > 0.002 的候选
# （真脸/幻影比值实测 > 50 倍，相对判别稳定；0.002 地板防纯背景误收）。
PRESENCE_REL = 0.4
PRESENCE_FLOOR = 0.002
HAND_DETECT_THRESH = 0.5       # palm 检测阈值
HAND_TRACK_THRESH = 0.5        # hand landmarks presence 阈值
NMS_IOU = 0.3                  # 贪心 NMS 的 IoU 阈值（MediaPipe 同值）

# 眼线关键点（ROI 旋转定向用）与腕→中指根（Hands 用）
EYE_ROT_IDS = (33, 263)
HAND_ROT_IDS = (0, 9)          # wrist, middle_finger_mcp

# blendshapes 输出下标（输出序 = mediapipe kBlendshapeNames，原始 0~1 值）
BLENDSHAPE_MOUTH_SMILE_L = 44  # mouthSmileLeft
BLENDSHAPE_MOUTH_SMILE_R = 45  # mouthSmileRight

# HUND blendshapes 网络的 146 点输入子集（mediapipe face_blendshapes_graph.cc
# kLandmarksSubsetIdxs 原样，输入是**像素坐标** (x*w, y*h)）
BLENDSHAPE_LANDMARK_SUBSET = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53,
    54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91,
    93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148,
    149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168,
    172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263,
    267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297,
    300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336,
    338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
    384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415,
    454, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
]

MAX_FACES = 5                  # 与原 engine 的 num_faces 对齐
MAX_HANDS = 2

# RVM（RobustVideoMatting）配置：官方 TorchScript，带 4 层循环状态。
# 默认 mobilenetv3 fp32（实测 7.8ms/帧@720p/3060，质量远超二元分割器）；
# rvm_resnet50_fp16.ts 为高质量档（fp16 需 half 输入，~130ms，离线/拍照用）。
RVM_TS = "rvm.ts"
RVM_TS_HQ = "rvm_resnet50_fp16.ts"
RVM_DOWNSAMPLE = 0.375         # 720p 推荐下采样比（官方 README 口径）


# ------- SSD 锚框生成（SsdAnchorsCalculator 复刻） -------

def generate_anchors(num_layers: int, input_size: int, strides: list[int],
                     dev: torch.device) -> torch.Tensor:
    """blaze* 系列的固定形锚框（宽=高=scale，插值层附加锚）。

    每位置两锚，行序遍历 —— 顺序与 SsdAnchorsCalculator 一致。解码只依赖
    锚中心，返回 (N,2) 中心。
    """
    anchors = []
    for layer in range(num_layers):
        f = input_size // strides[layer]
        for y in range(f):
            for x in range(f):
                anchors.append(((x + 0.5) / f, (y + 0.5) / f))
                anchors.append(((x + 0.5) / f, (y + 0.5) / f))
    return torch.tensor(anchors, dtype=torch.float32, device=dev)


def _face_info_from_lm(lm: np.ndarray, smile: Optional[float]) -> FaceInfo:
    """由 FaceMesh 关键点导出 FaceInfo（与 infer._landmarks_to_face_info 同口径：
    轮廓 min/max + 下巴 8% 余量；smile 为 blendshapes 均值标量）。"""
    oval = lm[FACE_OVAL_IDS]
    x1, y1 = oval[:, 0].min(), oval[:, 1].min()
    x2, y2 = oval[:, 0].max(), oval[:, 1].max()
    h_box = y2 - y1
    box = (float(x1), float(y1), float(x2), float(min(y2 + 0.08 * h_box, 1.0)))
    return FaceInfo(landmarks=lm, box=box, smile=smile)


# ------- 检测头解码（TensorsToDetectionsCalculator 复刻） -------

def decode_detections(scores_raw, boxes_raw, anchors: np.ndarray,
                      input_size: int, score_thresh: float, nms_iou: float,
                      max_det: int, num_keypoints: int) -> list[dict]:
    """blaze 检测输出 → 检测列表（归一化坐标）。

    解码口径：cx = anchor_cx + b0/S；w = b2/S（blaze 系列无锚尺度乘子，
    与 TensorsToDetectionsCalculator x/y/w/h_scale = input_size 一致）。
    NMS 在 CPU numpy 上做（过阈值候选通常 <100 个，开销可忽略）。
    scores_raw/boxes_raw 已去 batch 维：(N,1) / (N,4+2K)（GPU 张量自动转 CPU）。
    """
    scores = torch.sigmoid(
        torch.as_tensor(scores_raw).cpu().reshape(-1)).numpy()
    b = np.asarray(boxes_raw.cpu()
                   if hasattr(boxes_raw, "cpu") else boxes_raw,
                   dtype=np.float32)
    if b.ndim > 2:
        b = b.reshape(b.shape[-3], -1)
    b = b.reshape(len(scores), -1)

    cx = anchors[:, 0] + b[:, 0] / input_size
    cy = anchors[:, 1] + b[:, 1] / input_size
    w = b[:, 2] / input_size
    h = b[:, 3] / input_size
    kps = None
    if num_keypoints > 0:
        k = b[:, 4:4 + num_keypoints * 2].reshape(len(scores), num_keypoints, 2)
        kps = anchors[:, None, :] + k / input_size

    cand = np.where(scores > score_thresh)[0]
    if len(cand) == 0:
        return []
    order = cand[np.argsort(-scores[cand])]
    keep: list[int] = []
    for i in order:
        ok = True
        for j in keep:
            xx1 = max(cx[i] - w[i] / 2, cx[j] - w[j] / 2)
            yy1 = max(cy[i] - h[i] / 2, cy[j] - h[j] / 2)
            xx2 = min(cx[i] + w[i] / 2, cx[j] + w[j] / 2)
            yy2 = min(cy[i] + h[i] / 2, cy[j] + h[j] / 2)
            inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            union = w[i] * h[i] + w[j] * h[j] - inter
            if union > 0 and inter / union > nms_iou:
                ok = False
                break
        if ok:
            keep.append(int(i))
        if len(keep) >= max_det:
            break
    out = []
    for i in keep:
        det = {"score": float(scores[i]),
               "box": (float(cx[i]), float(cy[i]), float(w[i]), float(h[i]))}
        if kps is not None:
            det["kps"] = kps[i]
        out.append(det)
    return out


# ------- 旋转 ROI 裁剪 / 回投（ImageToTensorCalculator 复刻） -------

def crop_affine(frame_t: torch.Tensor, center_xy, side_px: float,
                rotation: float, out_size: int) -> torch.Tensor:
    """frame_t: (1,3,H,W) float RGB [0,1]；中心/旋转定义 ROI。

    以 center（像素坐标）为中心、边长 side_px 的旋转正方形 →
    out_size×out_size。rotation 为 ROI 局部 +x 轴相对图像 +x 的转角
    （弧度，图像 y 向下的顺时针为正 —— 与 MediaPipe NORM_RECT 一致）。
    """
    h, w = frame_t.shape[-2:]
    cos, sin = math.cos(rotation), math.sin(rotation)
    a11 = cos * side_px / (w - 1)
    a12 = -sin * side_px / (w - 1)
    a21 = sin * side_px / (h - 1)
    a22 = cos * side_px / (h - 1)
    mat = torch.tensor([[a11, a12, 2 * center_xy[0] / (w - 1) - 1],
                        [a21, a22, 2 * center_xy[1] / (h - 1) - 1]],
                       dtype=torch.float32, device=frame_t.device)
    grid = F.affine_grid(mat.view(1, 2, 3), (1, 3, out_size, out_size),
                         align_corners=True)
    return F.grid_sample(frame_t, grid, align_corners=True,
                         padding_mode="border")


def project_landmarks(lm_crop: torch.Tensor, center_xy, side_px: float,
                      rotation: float, frame_wh) -> torch.Tensor:
    """裁剪域 landmarks (N,3)（0~1 归一化于裁剪框）→ 帧归一化坐标 (N,3)。"""
    w, h = frame_wh
    cos, sin = math.cos(rotation), math.sin(rotation)
    nx = lm_crop[:, 0] * 2 - 1
    ny = lm_crop[:, 1] * 2 - 1
    lx = nx * side_px / 2
    ly = ny * side_px / 2
    px = center_xy[0] + cos * lx - sin * ly
    py = center_xy[1] + sin * lx + cos * ly
    out = torch.empty_like(lm_crop)
    out[:, 0] = px / w
    out[:, 1] = py / h
    out[:, 2] = lm_crop[:, 2] * side_px / w   # z 与 x 同尺度口径
    return out


def _rect_from_detection(det: dict, scale: float, frame_wh):
    """检测框 → 旋转正方形 ROI（square_long × scale）。

    旋转取 blaze 关键点 0→1（右眼→左眼）连线角（对齐
    DetectionsToRectsCalculator 的 rotation vector 语义，target 0°）。
    """
    w, h = frame_wh
    cx, cy, bw, bh = det["box"]
    side = max(bw * w, bh * h) * scale
    kps = det.get("kps")
    rot = 0.0
    if kps is not None and len(kps) >= 2:
        rot = math.atan2((kps[1][1] - kps[0][1]) * h,
                         (kps[1][0] - kps[0][0]) * w)
    return (cx * w, cy * h), side, rot


def _rect_from_landmarks(lm, scale: float, rot_ids,
                         frame_wh):
    """landmarks → 旋转正方形 ROI（bbox 中心 + 关键点对定向）。

    lm 兼容 numpy (N,3) 或 torch (N,3)（帧归一化坐标）；返回 Python 标量，
    供 crop_affine 的仿射矩阵构造（**一次** .cpu() 后在 numpy 上算，避免
    逐标量 .item() 触发多次 GPU 同步——WDDM 下每次同步 ~0.5ms 起）。
    旋转 = 关键点对 p0→p1 的连线角（对齐 DetectionsToRectsCalculator
    的 rotation_vector + target_angle=0°；眼线 33→263 / 腕→中指根同理）。
    """
    w, h = frame_wh
    if isinstance(lm, torch.Tensor):
        lm = lm.detach().cpu().numpy()
    xs, ys = lm[:, 0] * w, lm[:, 1] * h
    bx1, bx2 = xs.min(), xs.max()
    by1, by2 = ys.min(), ys.max()
    center = ((bx1 + bx2) / 2, (by1 + by2) / 2)
    side = max(bx2 - bx1, by2 - by1) * scale
    p0, p1 = lm[rot_ids[0]], lm[rot_ids[1]]
    dx = float(p1[0] - p0[0]) * w
    dy = float(p1[1] - p0[1]) * h
    return center, side, math.atan2(dy, dx)


def _lm_iou(a, b) -> float:
    """两组 landmarks（帧归一化坐标，numpy）的 bbox IoU —— 去重用。"""
    a = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
    b = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else b
    ax1, ay1 = a[:, 0].min(), a[:, 1].min()
    ax2, ay2 = a[:, 0].max(), a[:, 1].max()
    bx1, by1 = b[:, 0].min(), b[:, 1].min()
    bx2, by2 = b[:, 0].max(), b[:, 1].max()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


# ------- TorchScript 加载 -------

def load_ts(name: str) -> torch.jit.ScriptModule:
    path = TORCH_MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"TorchScript 缺失：{path}，请先运行 python scripts/convert_models.py")
    m = torch.jit.load(str(path), map_location=device())
    m.eval()
    return m


def _pick_detector_outputs(outs):
    """按形状区分 scores（末维 1）与 boxes（末维 ≥4）。"""
    scores = boxes = None
    for t in outs:
        t = t[0] if t.dim() == 3 else t
        if t.shape[-1] == 1:
            scores = t
        elif t.shape[-1] >= 4:
            boxes = t
    if scores is None or boxes is None:
        raise RuntimeError(f"检测器输出形状异常：{[o.shape for o in outs]}")
    return scores, boxes


def _normalize(x: torch.Tensor, norm: str) -> torch.Tensor:
    return x * 2.0 - 1.0 if norm == "pm1" else x


# ------- 推理引擎 -------

class TorchInferenceEngine:
    """MediaPipe InferenceEngine 的 PyTorch CUDA 等价替换（接口对齐 + 扩展）。

    与 mediapipe 引擎的行为差异（有意为之）：
      - VIDEO 模式的内建跟踪改为显式 ROI 状态机（丢失/低置信度重检测），
        语义等价：每帧 landmarks 都可用；
      - 分割默认 RVM 视频抠图（带循环状态，发丝级 alpha），二元/多分类
        仍可切换用于"质量 vs 速度"对比；
      - 新增 depth（Depth Anything V2）与 mediapipe 引擎没有的能力。
    """

    def __init__(self, models_dir=None,
                 segmenter_model: Optional[str] = None,
                 parallel_inference: Optional[bool] = None):
        self.models_dir = Path(models_dir) if models_dir \
            else TORCH_MODELS_DIR.parent
        # 默认分割模型：RVM 可用则用 RVM（质量档），否则退二元
        if segmenter_model is None:
            segmenter_model = ("rvm" if (TORCH_MODELS_DIR / RVM_TS).exists()
                               else "selfie_segmenter_binary")
        if segmenter_model not in SEGMENTER_SPECS:
            raise ValueError(f"未知分割模型：{segmenter_model}")
        self._segmenter_key = segmenter_model
        self._lock = threading.Lock()
        # 同帧的各类模型相互独立；CUDA 上用常驻线程和 stream 重叠执行。
        # 单模型帧仍直跑，省去线程调度开销。
        self.parallel_inference = (device().type == "cuda" if
                                   parallel_inference is None else
                                   parallel_inference)
        self._inference_pool: Optional[ThreadPoolExecutor] = None
        self._inference_streams = None
        self._frame_id = 0
        self._ts = 0
        self._border_checked = False
        # TorchScript 懒加载（+ 对应的 CUDA Graph 回放器；None = 未建）
        self._face_det: Optional[torch.jit.ScriptModule] = None
        self._face_lm: Optional[torch.jit.ScriptModule] = None
        self._face_bs: Optional[torch.jit.ScriptModule] = None
        self._hand_det: Optional[torch.jit.ScriptModule] = None
        self._hand_lm: Optional[torch.jit.ScriptModule] = None
        self._segmenter: Optional[torch.jit.ScriptModule] = None
        self._depth_any = None                     # core.depthany.DepthSession
        # 跟踪状态
        self._face_tracks: list[dict] = []   # {"lm": (478,3) 帧归一化}
        self._hand_tracks: list[dict] = []   # {"lm", "handedness"}
        # RVM 循环状态：(r1..r4) + 最近推理的 (h, w, downsample_ratio)
        self._rvm_rec: list[Optional[torch.Tensor]] = [None] * 4
        self._rvm_shape: Optional[tuple[int, int]] = None
        self._rvm_ts = RVM_TS
        # 常驻锚框（CPU numpy，解码用）
        self._face_anchors = generate_anchors(
            4, FACE_DETECT_INPUT, [8, 16, 16, 16],
            torch.device("cpu")).numpy()
        self._hand_anchors = generate_anchors(
            4, HAND_DETECT_INPUT, [8, 16, 16, 16],
            torch.device("cpu")).numpy()

    # ------- 对齐 mediapipe 引擎的属性 -------

    @property
    def segmenter_model(self) -> str:
        return self._segmenter_key

    @property
    def segmenter_spec(self) -> dict:
        return SEGMENTER_SPECS[self._segmenter_key]

    @property
    def recommended_interval(self) -> int:
        return SEGMENTER_INTERVAL[self._segmenter_key]

    @property
    def backend_name(self) -> str:
        return f"torch:{device().type}"

    def set_segmenter_model(self, key: str) -> None:
        if key not in SEGMENTER_SPECS:
            raise ValueError(f"未知分割模型：{key}")
        with self._lock:
            if key != self._segmenter_key:
                self._segmenter = None
                self._segmenter_key = key
                self._border_checked = False
                self._rvm_rec = [None] * 4
                self._rvm_shape = None

    def reset_temporal(self) -> None:
        """清空跨帧状态（切换采集源时调用，RVM 循环状态与跟踪 ROI 都要断开）。"""
        self._face_tracks.clear()
        self._hand_tracks.clear()
        self._rvm_rec = [None] * 4
        self._rvm_shape = None
        if self._depth_any is not None:
            self._depth_any.reset()

    def close(self) -> None:
        if self._inference_pool is not None:
            self._inference_pool.shutdown(wait=True)
            self._inference_pool = None
            self._inference_streams = None
        self.reset_temporal()
        self._face_det = self._face_lm = self._face_bs = None
        self._hand_det = self._hand_lm = None
        self._segmenter = None
        self._depth_any = None

    # ------- 帧预处理 -------

    @staticmethod
    def _prep_rgb(frame_bgr: np.ndarray) -> torch.Tensor:
        t = upload_frame(frame_bgr).float().div_(255.0)
        return t.flip(1)   # BGR→RGB

    @staticmethod
    def _nhwc(x_nchw: torch.Tensor, size: int) -> torch.Tensor:
        """(1,3,H,W) → (1,size,size,3)（tflite 系模型都是 NHWC 输入）。"""
        y = F.interpolate(x_nchw, size=(size, size), mode="bilinear",
                          align_corners=False)
        return y.permute(0, 2, 3, 1).contiguous()

    # ------- FaceMesh -------

    # 跟踪健康（有存活轨迹）时，每 N 帧才跑一次全帧检测发现新人脸
    DETECT_REFRESH_INTERVAL = 10

    def _detect_faces(self, rgb_t: torch.Tensor) -> list[dict]:
        if self._face_det is None:
            self._face_det = load_ts("face_detector.ts")
        small = _normalize(self._nhwc(rgb_t, FACE_DETECT_INPUT),
                           FACE_DETECT_NORM)
        # 注意：tflite 逐算子转换图**不要**包 CUDA Graph——它是 GPU 计算
        # 受限（每 conv 带 permute/pad 拷贝，见 _torch_impl 模块头），图回放
        # 无收益；且捕获失败会毒化整个 CUDA 上下文（实测 2.14/3060/WDDM）。
        with torch.no_grad():
            outs = self._face_det(small)
        scores, boxes = _pick_detector_outputs(outs)
        return decode_detections(scores, boxes, self._face_anchors,
                                 FACE_DETECT_INPUT, FACE_DETECT_THRESH,
                                 NMS_IOU, MAX_FACES, 6)

    def _face_landmarks(self, rgb_t: torch.Tensor, center, side, rotation,
                        frame_wh):
        """ROI 裁剪 → landmarks（478×3，256 像素域 → /256 归一化）。

        presence = sigmoid(Identity_1)（标定实测：真脸 +4.2→0.98，
        幻影/背景 −18~−22→≈0，与 mediapipe 0.5 阈值同语义）；
        Identity_2（另一标量）不是 presence，不能覆盖。
        返回 (帧归一化 landmarks 的 numpy (478,3), presence)——张量在
        本函数内一次性下载（每帧每脸一次 D2H，后续 ROI/IoU 全走 numpy）。
        """
        if self._face_lm is None:
            self._face_lm = load_ts("face_landmarks.ts")
        crop = crop_affine(rgb_t, center, side, rotation, FACE_LM_INPUT)
        x = _normalize(self._nhwc(crop, FACE_LM_INPUT), FACE_LM_NORM)
        with torch.no_grad():
            outs = self._face_lm(x)
        lm_raw, presence_t = None, None
        for t in outs:
            if t.numel() == 478 * 3:
                lm_raw = t.reshape(478, 3)
            elif t.numel() == 1 and lm_raw is not None:
                presence_t = t.flatten()[0]
                break
        if lm_raw is None:
            return None, 0.0
        lm_crop = lm_raw / float(FACE_LM_INPUT)   # 256 像素域 → [0,1]
        lm = project_landmarks(lm_crop, center, side, rotation, frame_wh)
        # 关键点和 presence 一次性下载，避免先 float() 再 .cpu() 两次同步。
        if presence_t is None:
            return lm.detach().cpu().numpy(), 0.0
        packed = torch.cat((lm.flatten(), torch.sigmoid(presence_t).view(1)))
        values = packed.detach().cpu().numpy()
        return values[:-1].reshape(478, 3), float(values[-1])

    def _blendshapes(self, lm: np.ndarray, frame_wh) -> Optional[float]:
        """146 点子集（像素坐标）→ HUND 52 维 → smile 均值。lm 为 numpy。"""
        if self._face_bs is None:
            try:
                self._face_bs = load_ts("face_blendshapes.ts")
            except FileNotFoundError:
                return None
        w, h = frame_wh
        pts_np = (lm[BLENDSHAPE_LANDMARK_SUBSET][:, :2]
                  * np.array([w, h], dtype=np.float32))
        pts = torch.from_numpy(pts_np).to(device())[None]
        with torch.no_grad():
            out = self._face_bs(pts)
        v = out.reshape(-1) if not isinstance(out, (tuple, list)) \
            else out[0].reshape(-1)
        return float((v[BLENDSHAPE_MOUTH_SMILE_L]
                      + v[BLENDSHAPE_MOUTH_SMILE_R]) / 2)

    def _process_faces(self, rgb_t: torch.Tensor, frame_wh,
                       blendshapes: bool = True) -> list[FaceInfo]:
        """候选池 + 自适应 presence 门限 + 检测/跟踪去重。

        跟踪候选（上一帧 landmarks 的 ROI）与检测候选（throttle 刷新）
        先汇入同一池子，统一按 presence 门限筛（绝对 0.5 或相对 0.4×pmax，
        见模块头标定记录），再按 bbox IoU 去重（检测命中已跟踪脸时跳过，
        否则每轮刷新会复制轨迹）。轨迹存 numpy landmarks（每帧每脸仅
        一次 D2H，见 _face_landmarks）。
        """
        cands: list[dict] = []      # {"lm"(np), "smile", "presence"}
        # 跟踪既有脸（ROI 来自上一帧 landmarks）
        for tr in self._face_tracks:
            center, side, rot = _rect_from_landmarks(
                tr["lm"], FACE_TRACK_SCALE, EYE_ROT_IDS, frame_wh)
            lm, presence = self._face_landmarks(
                rgb_t, center, side, rot, frame_wh)
            if lm is not None:
                cands.append(dict(lm=lm, presence=presence, smile=None))
        # 检测补齐：无轨迹时每帧跑；跟踪健康时按间隔刷新发现新脸
        if (not cands or self._frame_id % self.DETECT_REFRESH_INTERVAL == 0):
            for det in self._detect_faces(rgb_t)[:MAX_FACES]:
                center, side, rot = _rect_from_detection(
                    det, FACE_RECT_SCALE, frame_wh)
                lm, presence = self._face_landmarks(
                    rgb_t, center, side, rot, frame_wh)
                if lm is not None:
                    cands.append(dict(lm=lm, presence=presence, smile=None))

        # 自适应 presence 门限（绝对值或相对帧内最大值）
        pmax = max((c["presence"] for c in cands), default=0.0)
        gated = [c for c in cands
                 if c["presence"] > FACE_TRACK_THRESH
                 or (pmax < FACE_TRACK_THRESH
                     and c["presence"] >= PRESENCE_REL * pmax
                     and c["presence"] > PRESENCE_FLOOR)]
        # 检测/跟踪去重（IoU > 0.3 视为同一张脸，保留先出现者=跟踪候选）
        accepted: list[dict] = []
        for c in gated:
            if any(_lm_iou(c["lm"], a["lm"]) > 0.3 for a in accepted):
                continue
            accepted.append(c)
        self._face_tracks = [{"lm": c["lm"].copy()} for c in accepted]
        out = []
        for c in accepted:
            smile = c["smile"]
            if smile is None and blendshapes:
                smile = self._blendshapes(c["lm"], frame_wh)
            out.append(_face_info_from_lm(c["lm"], smile))
        return out

    # ------- Hands -------

    # 无手轨迹时，每 N 帧才跑一次 palm 检测（192 输入的逐算子图较重）
    HAND_DETECT_REFRESH_INTERVAL = 5

    def _detect_hands(self, rgb_t: torch.Tensor) -> list[dict]:
        if self._hand_det is None:
            self._hand_det = load_ts("hand_detector.ts")
        small = _normalize(self._nhwc(rgb_t, HAND_DETECT_INPUT),
                           HAND_DETECT_NORM)
        with torch.no_grad():
            outs = self._hand_det(small)
        scores, boxes = _pick_detector_outputs(outs)
        return decode_detections(scores, boxes, self._hand_anchors,
                                 HAND_DETECT_INPUT, HAND_DETECT_THRESH,
                                 NMS_IOU, MAX_HANDS, 7)

    def _hand_landmarks(self, rgb_t, center, side, rotation, frame_wh):
        if self._hand_lm is None:
            self._hand_lm = load_ts("hand_landmarks.ts")
        crop = crop_affine(rgb_t, center, side, rotation, HAND_LM_INPUT)
        x = self._nhwc(crop, HAND_LM_INPUT)
        with torch.no_grad():
            outs = self._hand_lm(x)
        # 输出序（按图声明）：[21×3 landmarks, presence, handedness, world]
        lm_raw = None
        scalars: list[torch.Tensor] = []
        for t in outs:
            n = t.numel()
            if n == 21 * 3 and lm_raw is None:
                lm_raw = t.reshape(21, 3)
            elif n == 1:
                scalars.append(torch.sigmoid(t.flatten()[0]))
        if lm_raw is None:
            return None, 0.0, ""
        lm = project_landmarks(lm_raw, center, side, rotation, frame_wh)
        # 两个标量与 21 点坐标合并回传，只触发一次 GPU→CPU 同步。
        packed = torch.cat([lm.flatten(), *[s.view(1) for s in scalars]])
        values = packed.detach().cpu().numpy()
        lm = values[:21 * 3].reshape(21, 3)
        presence = float(values[21 * 3]) if len(scalars) > 0 else 0.0
        handedness = float(values[21 * 3 + 1]) if len(scalars) > 1 else 0.5
        # mediapipe 口径：handedness sigmoid > 0.5 → Right（图像未镜像时）
        hand = "Right" if handedness > 0.5 else "Left"
        return lm, presence, hand

    def _process_hands(self, rgb_t, frame_wh) -> list[HandInfo]:
        cands: list[dict] = []      # {"lm"(np), "handedness", "presence"}
        for tr in self._hand_tracks:
            if not self._valid_hand_landmarks(tr["lm"]):
                continue
            center, side, rot = _rect_from_landmarks(
                tr["lm"], HAND_TRACK_SCALE, HAND_ROT_IDS, frame_wh)
            lm, presence, hd = self._hand_landmarks(rgb_t, center, side, rot,
                                                    frame_wh)
            if lm is not None and self._valid_hand_landmarks(lm):
                cands.append(dict(lm=lm, presence=presence, hd=hd))
        # palm 检测：无手轨迹时每帧跑，有手时隔帧刷新（192 输入的逐算子图较重）
        if (not cands or self._frame_id
                % self.HAND_DETECT_REFRESH_INTERVAL == 0):
            for det in self._detect_hands(rgb_t)[:MAX_HANDS]:
                kps = det.get("kps")
                if kps is None:
                    continue
                # ROI：关键点跨度 + 腕(0)→中指根(2) 定向
                w, h = frame_wh
                xs = kps[:, 0] * w
                ys = kps[:, 1] * h
                side = max(xs.max() - xs.min(), ys.max() - ys.min()) \
                    * HAND_RECT_SCALE
                center = (float(xs.mean()), float(ys.mean()))
                rot = math.atan2(ys[2] - ys[0], xs[2] - xs[0]) - math.pi / 2
                lm, presence, hd = self._hand_landmarks(rgb_t, center, side,
                                                        rot, frame_wh)
                if lm is not None and self._valid_hand_landmarks(lm):
                    cands.append(dict(lm=lm, presence=presence, hd=hd))
        # 与人脸同款自适应 presence 门限（绝对 0.5 或相对 0.4×pmax）
        pmax = max((c["presence"] for c in cands), default=0.0)
        gated = [c for c in cands
                 if c["presence"] > HAND_TRACK_THRESH
                 or (pmax < HAND_TRACK_THRESH
                     and c["presence"] >= PRESENCE_REL * pmax
                     and c["presence"] > PRESENCE_FLOOR)]
        accepted: list[dict] = []
        for c in gated:
            if any(_lm_iou(c["lm"], a["lm"]) > 0.3 for a in accepted):
                continue
            accepted.append(c)
            # 旧轨迹和本帧检测会同时进入候选池。去重失败时也必须遵守
            # max_num_hands=2，否则轨迹逐帧膨胀，前向次数和延迟失控。
            if len(accepted) >= MAX_HANDS:
                break
        self._hand_tracks = [{"lm": c["lm"].copy(),
                              "handedness": c["hd"]} for c in accepted]
        return [HandInfo(landmarks=c["lm"], handedness=c["hd"])
                for c in accepted]

    @staticmethod
    def _valid_hand_landmarks(lm: np.ndarray) -> bool:
        """拒绝非有限或远离画面的假轨迹，避免下帧 ROI 变成巨大框。"""
        xy = np.asarray(lm)[:, :2]
        if not np.isfinite(xy).all():
            return False
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        span = hi - lo
        return bool(np.all(span > 0.002) and np.all(span < 2.0)
                    and np.all(hi > -0.25) and np.all(lo < 1.25))

    # ------- 分割 -------

    def _segment_rvm(self, rgb_t: torch.Tensor, frame_wh) -> torch.Tensor:
        """RVM 视频抠图：带循环状态，返回全分辨率 alpha (1,1,H,W) 张量。

        默认 rvm.ts = mobilenetv3 fp32；换成 *_fp16.ts 文件时自动转 half
        输入（fp16 导出的参数/输入都必须 half）。循环状态原样回灌。
        CUDA Graph：捕获时用具体形状的状态张量做静态输入（首帧状态为
        None，先直调一次拿到真实状态再建图）；分辨率变化时重建。
        """
        if self._segmenter is None:
            self._segmenter = load_ts(self._rvm_ts)
        h, w = frame_wh[1], frame_wh[0]
        if self._rvm_shape != (h, w):
            self._rvm_rec = [None] * 4      # 尺寸变化：循环状态作废
            self._rvm_shape = (h, w)
            self._g_rvm = None
        src = rgb_t.half() if "fp16" in self._rvm_ts else rgb_t

        # dsr 必须传 Python float：签名是 float，传 CUDA 张量会逐帧隐式
        # item()（D2H 同步）
        with torch.no_grad():
            outs = self._segmenter(src, *self._rvm_rec, RVM_DOWNSAMPLE)
        self._rvm_rec = list(outs[2:6])
        pha = outs[1]
        pha = pha.float()
        if pha.shape[-2:] != (h, w):
            pha = F.interpolate(pha, size=(h, w), mode="bilinear",
                                align_corners=False)
        return pha[:, 0].clamp_(0.0, 1.0)

    def _segment_tflite(self, rgb_t: torch.Tensor, frame_wh):
        if self._segmenter is None:
            name = ("selfie_multiclass.ts"
                    if self._segmenter_key == "selfie_segmenter"
                    else "selfie_segmenter.ts")
            self._segmenter = load_ts(name)
        x = self._nhwc(rgb_t, SEG_INPUT)
        with torch.no_grad():
            out = self._segmenter(x)
        t = out[0] if isinstance(out, (tuple, list)) else out
        if t.dim() == 4 and t.shape[-1] in (1, 6) and t.shape[1] not in (1, 6):
            t = t.permute(0, 3, 1, 2)          # NHWC → NCHW
        return t

    def _segment(self, rgb_t: torch.Tensor, frame_wh):
        """返回 (前景概率 (1,h,w) float 张量，硬类别图 (1,1,h,w) 张量)。"""
        if self._segmenter_key == "rvm":
            alpha = self._segment_rvm(rgb_t, frame_wh)
            cat = (alpha > 0.5).float()
            return alpha, cat
        spec = self.segmenter_spec
        t = self._segment_tflite(rgb_t, frame_wh)
        h, w = frame_wh[1], frame_wh[0]
        probs = t.softmax(1)
        # tflite 输出为 logits，softmax 后与 mediapipe 置信图一致（标定记录）
        conf0 = probs[:, 0]
        alpha = conf0 if not spec["alpha_invert"] else 1.0 - conf0
        cat = probs.argmax(1, keepdim=True).float()
        if alpha.shape[-2:] != (h, w):
            alpha = F.interpolate(alpha, size=(h, w), mode="bilinear",
                                  align_corners=False)
        if cat.shape[-2:] != (h, w):
            cat = F.interpolate(cat, size=(h, w), mode="nearest")
        return alpha, cat

    # ------- 统一入口 -------

    @staticmethod
    def _run_on_stream(stream, fn):
        """在指定 CUDA stream 中完成一次模型调用，并等到结果可被主链使用。"""
        with torch.inference_mode(), torch.cuda.stream(stream):
            result = fn()
        stream.synchronize()
        return result

    def _run_model_tasks(self, rgb_t, tasks):
        """并发执行同帧独立模型；输入与结果都跨 stream 显式同步。"""
        if not self.parallel_inference or len(tasks) < 2 or \
                rgb_t.device.type != "cuda":
            return {name: fn() for name, fn in tasks}
        if self._inference_pool is None:
            self._inference_pool = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="beautycam-infer")
            self._inference_streams = [torch.cuda.Stream() for _ in range(4)]
        # _prep_rgb 在调用方当前 stream 产出，子 stream 必须等它完成。
        source = torch.cuda.current_stream(rgb_t.device)
        streams = self._inference_streams[:len(tasks)]
        for stream in streams:
            stream.wait_stream(source)
        futures = [(name, self._inference_pool.submit(
            self._run_on_stream, stream, fn))
            for (name, fn), stream in zip(tasks, streams)]
        return {name: future.result() for name, future in futures}

    def process(self, frame_bgr: np.ndarray, *, faces: bool = True,
                hands: bool = True, segmentation: bool = False,
                depth: bool = False,
                blendshapes: bool = True) -> FrameContext:
        """统一推理入口。

        blendshapes：微笑置信度只服务手势/笑脸自动拍照；纯美颜链不需要，
        传 False 可省每脸一次 HUND 前向 + 同步（GUI 无触发器时即此形态）。
        分割 alpha / 深度以 GPU 张量存入 ctx（懒物化为 numpy，见
        FrameContext）——GPU 效果链全程不落 CPU。
        """
        h, w = frame_bgr.shape[:2]
        self._frame_id += 1
        self._ts += 33
        ctx = FrameContext(frame_id=self._frame_id, timestamp_ms=self._ts,
                           width=w, height=h)
        rgb_t = self._prep_rgb(frame_bgr)
        tasks = []
        if faces:
            tasks.append(("faces", lambda: self._process_faces(
                rgb_t, (w, h), blendshapes=blendshapes)))
        if hands:
            tasks.append(("hands", lambda: self._process_hands(rgb_t, (w, h))))
        if segmentation:
            tasks.append(("segmentation", lambda: self._segment(rgb_t, (w, h))))
        if depth:
            if self._depth_any is None:
                from .depthany import DepthSession
                self._depth_any = DepthSession()
            tasks.append(("depth", lambda: self._depth_any.estimate_t(
                rgb_t, (w, h))))
        results = self._run_model_tasks(rgb_t, tasks)
        if faces:
            ctx.faces.extend(results["faces"])
        if hands:
            ctx.hands.extend(results["hands"])
        if segmentation:
            spec = self.segmenter_spec
            alpha, cat = results["segmentation"]
            if alpha.dim() == 3:
                alpha = alpha.unsqueeze(1)
            ctx.person_alpha_t = alpha          # (1,1,h,w) [0,1]
            if self._segmenter_key == "selfie_segmenter":
                # 多分类：类别图语义（cat>0 = 人）≠ alpha 阈值化，显式给出
                cat_np = cat.detach().cpu().numpy()
                ctx.person_mask = np.where(cat_np > 0, 255, 0).astype(np.uint8)
            if not self._border_checked:
                self._border_checked = True
                ratio = border_foreground_ratio(ctx.person_alpha)
                if ratio > BORDER_FOREGROUND_WARN:
                    print(f"[警告] 分割掩膜疑似整体反相：画面边缘 alpha 均值 "
                          f"{ratio:.2f} (> {BORDER_FOREGROUND_WARN})。")
        if depth:
            ctx.depth_t = results["depth"]
        return ctx


# ------- SCI 低光（torch 版会话，接口对齐 LowLightSession） -----------

class LowLightSessionTorch:
    """SCI ONNX → TorchScript 会话（CUDA 前向），接口同 infer.LowLightSession。"""

    def __init__(self, level: str = LOWLIGHT_DEFAULT_LEVEL,
                 models_dir=None):
        if level not in LOWLIGHT_LEVELS:
            raise ValueError(f"未知 SCI 强度档：{level}")
        path = TORCH_MODELS_DIR / f"sci_{level}.ts"
        if not path.exists():
            raise FileNotFoundError(
                f"TorchScript 缺失：{path}，请先运行 python scripts/convert_models.py")
        self.level = level
        self.model = torch.jit.load(str(path), map_location=device()).eval()
        self.provider = f"PyTorch-{device().type.upper()}EP"
        # CUDA Graph：SCI 是逐算子小图，WDDM 下启动开销占大头
        self._graph = None

    def _forward(self, x):
        with torch.no_grad():
            return self.model(x)

    def _forward_graphed(self, x):
        if device().type == "cuda":
            if self._graph is None:
                self._graph = graph_call(self._forward, f"sci_{self.level}",
                                         [torch.zeros_like(x)])
            return self._graph(x)
        return self._forward(x)

    def enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        """整帧增强（512×512 推理域），返回 RGB float32 (512,512,3)。"""
        t = upload_frame(frame_bgr).float().div_(255.0).flip(1)  # RGB
        x = F.interpolate(t, size=(LOWLIGHT_INPUT_SIZE,) * 2, mode="area")
        out = self._forward_graphed(x)
        enhanced = out[1] if isinstance(out, (tuple, list)) else out
        if isinstance(enhanced, (tuple, list)):
            enhanced = enhanced[1]
        return enhanced[0].clamp(0, 1).flip(1).permute(1, 2, 0).cpu().numpy()

    def enhance_tensor(self, f_bgr01: torch.Tensor) -> torch.Tensor:
        """GPU 张量版：BGR float [0,1] (1,3,H,W) → 增强结果（同域同尺寸）。

        供效果链 GPU 路径零拷贝使用（core/effects/_torch_impl.py）。
        """
        x = F.interpolate(f_bgr01.flip(1), size=(LOWLIGHT_INPUT_SIZE,) * 2,
                          mode="area")
        out = self._forward_graphed(x)
        enhanced = out[1] if isinstance(out, (tuple, list)) else out
        enhanced = enhanced[0] if enhanced.dim() == 4 else enhanced
        up = F.interpolate(enhanced, size=f_bgr01.shape[-2:], mode="bilinear",
                           align_corners=False)
        return up.flip(1).clamp(0, 1)
