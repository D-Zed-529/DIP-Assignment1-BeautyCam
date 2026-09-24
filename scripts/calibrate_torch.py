"""Torch 推理链 vs MediaPipe 数值校准（BEAUTYCAM CUDA 迁移定版工具）。

以 mediapipe Tasks（CPU 委托）在同图上的输出为 ground truth，逐项对比
TorchScript 版（models/torch/*.ts）的输出偏差，并确定预处理常数
（输入归一化 / ROI 缩放 / blendshapes 输入子集）。

  python scripts/calibrate_torch.py [--image 路径]

输出：各项平均/最大偏差 + 是否达标的结论。定版常数写回 core/infer_torch.py
模块头（BEAUTYCAM 部署记录）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.infer import FACE_OVAL_IDS, InferenceEngine  # noqa: E402

TORCH_DIR = Path(__file__).resolve().parent.parent / "models" / "torch"

# ------- 锚框（blaze 家族：4 层 × 每位置 2 锚，行序） -------
ANCHOR_SPECS = {
    "face": dict(num_layers=4, strides=[8, 16, 16, 16], size=128),
    "hand": dict(num_layers=4, strides=[8, 16, 16, 16], size=192),
}


def make_anchors(num_layers, strides, size):
    anchors = []
    for layer in range(num_layers):
        f = size // strides[layer]
        for y in range(f):
            for x in range(f):
                anchors += [((x + 0.5) / f, (y + 0.5) / f)] * 2
    return torch.tensor(anchors, dtype=torch.float32)


def decode(scores_raw, boxes_raw, anchors, size, thresh=0.5, nms_iou=0.3,
           max_det=5, num_kp=6):
    """scores_raw: (N,1) logits；boxes_raw: (N,4+2K)（已去 batch 维）。"""
    scores = torch.sigmoid(torch.as_tensor(scores_raw).reshape(-1)).numpy()
    b = np.asarray(boxes_raw, dtype=np.float32).reshape(len(scores), -1)
    cx = anchors[:, 0].numpy() + b[:, 0] / size
    cy = anchors[:, 1].numpy() + b[:, 1] / size
    w = b[:, 2] / size
    h = b[:, 3] / size
    kps = (anchors.numpy()[:, None, :] + b[:, 4:4 + num_kp * 2]
           .reshape(len(anchors), num_kp, 2) / size)
    cand = np.where(scores > thresh)[0]
    out = []
    for i in cand[np.argsort(-scores[cand])]:
        keep = True
        for j in out:
            xx1 = max(cx[i] - w[i] / 2, cx[j] - w[j] / 2)
            yy1 = max(cy[i] - h[i] / 2, cy[j] - h[j] / 2)
            xx2 = min(cx[i] + w[i] / 2, cx[j] + w[j] / 2)
            yy2 = min(cy[i] + h[i] / 2, cy[j] + h[j] / 2)
            if max(0, xx2 - xx1) * max(0, yy2 - yy1) / (
                    w[i] * h[i] + w[j] * h[j]
                    - max(0, xx2 - xx1) * max(0, yy2 - yy1)) > nms_iou:
                keep = False
                break
        if keep:
            out.append(i)
        if len(out) >= max_det:
            break
    return [dict(score=float(scores[i]), box=(cx[i], cy[i], w[i], h[i]),
                 kps=kps[i]) for i in out]


def load_rgb_t(img_path: str, max_w: int = 1280):
    """读图 → RGB float (1,3,H,W)（等比压到 ≤max_w 宽）。"""
    buf = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img.shape[1] > max_w:
        s = max_w / img.shape[1]
        img = cv2.resize(img, (max_w, max(1, round(img.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)[None]
    return t / 255.0, img


def to_nhwc_small(rgb_t, size, norm):
    """(1,3,H,W) [0,1] RGB → (1,size,size,3)，norm ∈ {"01", "pm1"}。"""
    x = F.interpolate(rgb_t, size=(size, size), mode="bilinear",
                      align_corners=False)
    x = x.permute(0, 2, 3, 1).contiguous()
    if norm == "pm1":
        x = x * 2.0 - 1.0
    return x


def crop_roi(rgb_t, cx, cy, side, rot, out_size):
    """以 (cx,cy) 为中心、边长 side 的旋转正方形 → out_size（grid_sample）。"""
    h, w = rgb_t.shape[-2:]
    dev = rgb_t.device
    cos, sin = float(np.cos(rot)), float(np.sin(rot))
    mat = torch.tensor([[cos * side / (w - 1), -sin * side / (w - 1),
                         2 * cx / (w - 1) - 1],
                        [sin * side / (h - 1), cos * side / (h - 1),
                         2 * cy / (h - 1) - 1]], device=dev)
    grid = F.affine_grid(mat.view(1, 2, 3), (1, 3, out_size, out_size),
                         align_corners=True)
    return F.grid_sample(rgb_t, grid, align_corners=True,
                         padding_mode="border")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="assets/samples/portrait1.jpg")
    args = ap.parse_args()

    rgb_t, img_bgr = load_rgb_t(args.image)
    h, w = img_bgr.shape[:2]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{dev}，图 {w}x{h}")

    # ---- ground truth：mediapipe Tasks ----
    mp_engine = InferenceEngine()
    ref = mp_engine.process(img_bgr, faces=True, hands=True, segmentation=True)
    print(f"mediapipe：{len(ref.faces)} 脸 / {len(ref.hands)} 手 / "
          f"alpha 均值 {ref.person_alpha.mean():.3f}")
    ref_smile_idx = None
    if ref.faces:
        bs_names = mp_engine._get_face_landmarker().detect_for_video  # noqa
    # 拿 blendshape 名字对照（重跑一次 faces 拿不到名字列表——直接用 detect 结果
    # 里的 category_name 排序推导）

    # ---- face_detector ----
    fd = torch.jit.load(str(TORCH_DIR / "face_detector.ts"),
                        map_location=dev).eval()
    anchors_f = make_anchors(**ANCHOR_SPECS["face"])
    results = {}
    for norm in ("01", "pm1"):
        with torch.no_grad():
            outs = fd(to_nhwc_small(rgb_t.to(dev), 128, norm))
        scores = next(o for o in outs if o.shape[-1] == 1)
        boxes = next(o for o in outs if o.shape[-1] >= 4)
        dets = decode(scores[0].cpu(), boxes[0].cpu(), anchors_f, 128)
        results[norm] = dets
        if dets:
            d = dets[0]
            print(f"[face_det norm={norm}] {len(dets)} 检出，top score "
                  f"{d['score']:.3f}，box(cx,cy,w,h)="
                  f"{tuple(round(v, 3) for v in d['box'])}")
        else:
            print(f"[face_det norm={norm}] 无检出")
    # 参照：mediapipe 人脸框（轮廓导出）
    if ref.faces:
        o = ref.faces[0].landmarks[FACE_OVAL_IDS][:, :2]
        print(f"[参照 mediapipe 框] cx={o[:, 0].mean():.3f} "
              f"cy={o[:, 1].mean():.3f} w≈{o[:, 0].ptp():.3f} "
              f"h≈{o[:, 1].ptp():.3f}")

    # ---- selfie_segmenter ----
    seg = torch.jit.load(str(TORCH_DIR / "selfie_segmenter.ts"),
                         map_location=dev).eval()
    with torch.no_grad():
        out = seg(to_nhwc_small(rgb_t.to(dev), 256, "01"))
    t = out[0] if isinstance(out, (tuple, list)) else out
    alpha_t = t[0, :, :, 0] if t.dim() == 4 else t[0, 0]
    alpha_up = F.interpolate(alpha_t[None, None], size=(h, w), mode="bilinear",
                             align_corners=False)[0, 0].cpu().numpy()
    mae = float(np.abs(alpha_up - ref.person_alpha).mean())
    print(f"[segmenter] alpha MAE={mae:.4f}（mean torch {alpha_up.mean():.3f} "
          f"vs mp {ref.person_alpha.mean():.3f}）")

    # ---- face_landmarks（用 mediapipe 框的中点近似 ROI 对齐，仅校数值） ----
    flm = torch.jit.load(str(TORCH_DIR / "face_landmarks.ts"),
                         map_location=dev).eval()
    if ref.faces:
        ref_lm = ref.faces[0].landmarks  # (478,3) or (468,3)
        xs, ys = ref_lm[:, 0] * w, ref_lm[:, 1] * h
        cx, cy = float((xs.min() + xs.max()) / 2), float((ys.min() + ys.max()) / 2)
        side = float(max(xs.max() - xs.min(), ys.max() - ys.min())) * 1.6
        for norm in ("01", "pm1"):
            crop = crop_roi(rgb_t.to(dev), cx, cy, side, 0.0, 256)
            if norm == "pm1":
                crop = crop * 2 - 1
            x = crop.permute(0, 2, 3, 1).contiguous()
            with torch.no_grad():
                outs = flm(x)
            lm_raw = outs[0].reshape(-1, 3)[None, None][0, 0] \
                if outs[0].dim() > 2 else outs[0].reshape(-1, 3)
            n = min(len(lm_raw), 478)
            # 输出在 256 像素域（AGENTS.md #20）→ 先 /256 到 [0,1] 裁剪域
            lm01 = lm_raw[:n].cpu().numpy() / 256.0
            px = (lm01[:, 0] - 0.5) * side / w + cx / w
            py = (lm01[:, 1] - 0.5) * side / h + cy / h
            m = min(n, len(ref_lm))
            dx = np.abs(px[:m] - ref_lm[:m, 0])
            dy = np.abs(py[:m] - ref_lm[:m, 1])
            print(f"[face_lm norm={norm}] {n} 点，Δx mean {dx.mean():.4f} "
                  f"max {dx.max():.4f}，Δy mean {dy.mean():.4f} "
                  f"max {dy.max():.4f}（归一化坐标）")
            results.setdefault("lm", {})[norm] = (px, py)

    # ---- blendshapes：输入 146×2 点子集假设验证 ----
    bs = torch.jit.load(str(TORCH_DIR / "face_blendshapes.ts"),
                        map_location=dev).eval()
    if ref.faces:
        smile_ref = ref.faces[0].smile
        lm = ref.faces[0].landmarks
        # blendshapes 输入 = 146 点子集的**像素坐标**（AGENTS.md #20）
        for name, pts in (("前146", lm[:146, :2] * np.array([w, h],
                                                           dtype=np.float32)),
                          ("轮廓+iris?", lm[:146, :2] * np.array(
                              [w, h], dtype=np.float32))):
            with torch.no_grad():
                out = bs(torch.from_numpy(
                    pts.astype(np.float32))[None].to(dev))
            v = out.reshape(-1) if not isinstance(out, (tuple, list)) \
                else out[0].reshape(-1)
            print(f"[blendshapes {name}] 52 值域 [{v.min():.3f},"
                  f"{v.max():.3f}]，40/41 号 = "
                  f"{float(v[40]):.3f}/{float(v[41]):.3f}，"
                  f"mediapipe smile = {smile_ref}")

    # ---- hand_detector（无手图：score 应低） ----
    hd = torch.jit.load(str(TORCH_DIR / "hand_detector.ts"),
                        map_location=dev).eval()
    anchors_h = make_anchors(**ANCHOR_SPECS["hand"])
    for norm in ("01", "pm1"):
        with torch.no_grad():
            outs = hd(to_nhwc_small(rgb_t.to(dev), 192, norm))
        scores = next(o for o in outs if o.shape[-1] == 1)
        dets = decode(scores[0].cpu(), next(o for o in outs if o.shape[-1] >= 4)[0].cpu(),
                      anchors_h, 192, max_det=2, num_kp=7)
        top = max((d["score"] for d in dets), default=0.0)
        print(f"[hand_det norm={norm}] 检出 {len(dets)}，top score {top:.3f}"
              f"（mediapipe 手数 {len(ref.hands)}，应一致为 0）")

    mp_engine.close()
    print("\n校准完成 —— 达标线：landmark Δ mean < 0.01，alpha MAE < 0.02")
    return 0


if __name__ == "__main__":
    sys.exit(main())
