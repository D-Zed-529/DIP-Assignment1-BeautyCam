"""瘦脸 debug 可视化：识别标注 + 位移场 + A/B 对照 + 客观指标。

debug 模式（多图批跑）：
  .venv/bin/python scripts/debug_slim.py                    # 默认素材全套
  .venv/bin/python scripts/debug_slim.py img.jpg --algo v5  # 指定图/算法
  .venv/bin/python scripts/debug_slim.py --strength 1.0     # 指定强度

每张图每个算法输出 4 张图到 outputs/debug/：
  <stem>_lm_<algo>.jpg     人脸识别标注：FaceMesh 五官轮廓 + 下颌动点箭头
                           （黄）+ 锚点（洋红圈）+ 关键点 id
  <stem>_field_<algo>.jpg  位移场箭头图（粗网格，3× 夸张，颜色=幅度）
  <stem>_ab_<algo>.jpg     脸部裁剪 A/B 并排（左原右瘦）
  <stem>_wipe_<algo>.jpg   全幅左右 wipe 拼接（左原右瘦，中线对齐检查）
外加 stdout 指标表：下颌内收量、锚点位移（应≈0）、背景位移（应=0）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp  # noqa: E402

from core.effects.beauty import (  # noqa: E402
    JAW_LEFT_IDS, JAW_RIGHT_IDS, estimate_yaw_deg, pose_gate,
    slim_face_controls, slim_face_maps,
)
from core.infer import InferenceEngine  # noqa: E402

DEFAULT_INPUTS = ["assets/samples/portrait1.jpg"] + sorted(
    str(p) for p in Path("photos").glob("*.jpg"))

# 关键点 id → 文字标注（debug 读图定位用）
LABEL_IDS = {1: "nose", 152: "chin", 61: "mL", 291: "mR",
             234: "eL", 454: "eR", 93: "jawL", 377: "jawR"}

# 客观指标的采样点：(说明, landmark id, 期望)
METRIC_JAW = [(f"下颌中点#{i}", i) for i in (149, 378)]      # 期望: 内收明显
METRIC_ANCHOR = [(f"锚点#{i}", i) for i in (152, 61, 291, 145, 374, 105, 334, 1)]
METRIC_BG = [("背景左上", None), ("背景右下", None)]


def draw_landmarks_debug(img: np.ndarray, lm_px: np.ndarray,
                         mover_ids: list[int], guard_ids: list[int],
                         P: np.ndarray, D: np.ndarray, n_movers: int,
                         guard_w: np.ndarray | None = None) -> np.ndarray:
    """人脸识别 debug 标注：五官轮廓 + 下颌链 + 动点位移箭头 + 锚点。"""
    vis = img.copy()
    h, w = vis.shape[:2]
    overlay = vis.copy()
    fm = mp.solutions.face_mesh
    # 五官轮廓（细线，半透明叠加）：脸缘/眉/眼/鼻/嘴
    for a, b in fm.FACEMESH_CONTOURS:
        pa = tuple(np.round(lm_px[a]).astype(int))
        pb = tuple(np.round(lm_px[b]).astype(int))
        cv2.line(overlay, pa, pb, (180, 255, 180), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    # 下颌链（瘦脸作用链）加粗高亮
    for ids, color in ((JAW_LEFT_IDS, (0, 220, 0)), (JAW_RIGHT_IDS, (0, 220, 0))):
        pts = np.round(lm_px[ids]).astype(int)
        cv2.polylines(vis, [pts], False, color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(vis, tuple(p), 3, color, -1, cv2.LINE_AA)

    # 动点位移箭头（命令值，黄色；实际场量在 field 图看）
    for k in range(n_movers):
        p0 = tuple(np.round(P[k]).astype(int))
        p1 = tuple(np.round(P[k] + D[k] * 3.0).astype(int))   # 3× 夸张
        cv2.arrowedLine(vis, p0, p1, (0, 255, 255), 2, cv2.LINE_AA,
                        tipLength=0.25)

    # 保护点（洋红圆圈：嘴/眼/鼻/颞部/下巴下方合成点，权值越大圈越大）
    for k, pid in enumerate(guard_ids):
        c = tuple(np.round(P[n_movers + k]).astype(int))
        r = 5 + (3 * int(round(float(guard_w[k]))) if guard_w is not None else 0)
        cv2.circle(vis, c, r, (255, 0, 255), 2, cv2.LINE_AA)
        if pid >= 0:
            cv2.putText(vis, str(pid), (c[0] + 4, c[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 0, 255), 1)

    # 关键点 id 文字
    for pid, name in LABEL_IDS.items():
        c = np.round(lm_px[pid]).astype(int)
        cv2.putText(vis, f"{pid}:{name}", tuple(c + [6, -6]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"{pid}:{name}", tuple(c + [6, -6]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"movers={n_movers} guards={len(guard_ids)}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, f"movers={n_movers} guards={len(guard_ids)}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_field(vis: np.ndarray, map_x: np.ndarray, map_y: np.ndarray,
               step: int = 28, scale: float = 3.0) -> np.ndarray:
    """位移场箭头图：粗网格采样，箭长 = 位移×scale，颜色按幅度归一。"""
    out = vis.copy()
    h, w = out.shape[:2]
    mag = np.hypot(map_x - np.arange(w, dtype=np.float32)[None, :],
                   map_y - np.arange(h, dtype=np.float32)[:, None])
    mmax = max(float(mag.max()), 1e-3)
    m8 = np.clip(mag / mmax * 255, 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(m8, cv2.COLORMAP_TURBO)
    ys = np.arange(step // 2, h, step)
    xs = np.arange(step // 2, w, step)
    for y in ys:
        for x in xs:
            dx = float(map_x[y, x] - x)
            dy = float(map_y[y, x] - y)
            if np.hypot(dx, dy) < 0.3:
                continue
            p1 = (int(x + dx * scale), int(y + dy * scale))
            cv2.arrowedLine(out, (x, y), p1, tuple(int(v) for v in colors[y, x]),
                            1, cv2.LINE_AA, tipLength=0.3)
    cv2.putText(out, f"max_disp={mmax:.1f}px arrows x{scale}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, f"max_disp={mmax:.1f}px arrows x{scale}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def disp_at(map_x: np.ndarray, map_y: np.ndarray, x: float, y: float) -> np.ndarray:
    """双线性采样位移场在 (x, y) 处的内容位移。"""
    h, w = map_x.shape
    xi = int(np.clip(np.floor(x), 0, w - 2))
    yi = int(np.clip(np.floor(y), 0, h - 2))
    fx = float(min(max(x - xi, 0.0), 1.0))
    fy = float(min(max(y - yi, 0.0), 1.0))

    def bilinear(m: np.ndarray) -> float:
        return float(m[yi, xi] * (1 - fx) * (1 - fy) +
                     m[yi, xi + 1] * fx * (1 - fy) +
                     m[yi + 1, xi] * (1 - fx) * fy +
                     m[yi + 1, xi + 1] * fx * fy)

    return np.array([bilinear(map_x) - x, bilinear(map_y) - y])


def ab_compare(src: np.ndarray, out: np.ndarray, lm_px: np.ndarray,
               label: str) -> np.ndarray:
    """脸部裁剪 A/B 并排（左原右瘦）。"""
    x1, y1 = lm_px[:, 0].min(), lm_px[:, 1].min()
    x2, y2 = lm_px[:, 0].max(), lm_px[:, 1].max()
    mw, mh = (x2 - x1) * 0.35, (y2 - y1) * 0.35
    x1, y1 = int(max(x1 - mw, 0)), int(max(y1 - mh, 0))
    x2, y2 = int(min(x2 + mw, src.shape[1])), int(min(y2 + mh, src.shape[0]))
    a = src[y1:y2, x1:x2]
    b = out[y1:y2, x1:x2]
    target_w = 640
    scale = target_w / a.shape[1]
    a = cv2.resize(a, (target_w, int(a.shape[0] * scale)))
    b = cv2.resize(b, (target_w, int(b.shape[0] * scale)))
    divider = np.full((a.shape[0], 4, 3), 255, np.uint8)
    canvas = np.hstack([a, divider, b])
    cv2.putText(canvas, "ORIGIN", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, "ORIGIN", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (target_w + 16, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, label, (target_w + 16, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def wipe_compare(src: np.ndarray, out: np.ndarray) -> np.ndarray:
    """全幅 wipe：左半原图、右半瘦脸，中线拼接检查对齐与接缝。"""
    h, w = src.shape[:2]
    canvas = src.copy()
    canvas[:, w // 2:] = out[:, w // 2:]
    cv2.line(canvas, (w // 2, 0), (w // 2, h), (0, 255, 255), 1, cv2.LINE_AA)
    scale = 900 / w
    return cv2.resize(canvas, (900, int(h * scale)))


def report_metrics(name: str, map_x: np.ndarray, map_y: np.ndarray,
                   lm_px: np.ndarray, h: int, w: int) -> None:
    """客观指标：下颌内收（期望大）、锚点位移（期望≈0）、背景（=0）。"""
    print(f"  -- {name} 指标 --")
    center_x = float(lm_px[1, 0])
    for label, pid in METRIC_JAW:
        x, y = lm_px[pid]
        d = -disp_at(map_x, map_y, float(x), float(y))   # 内容位移=−采样偏移
        inward = (1 if x < center_x else -1) * d[0]
        print(f"  {label}: 内收 {inward:+.1f}px  (内容位移 dx={d[0]:+.1f} dy={d[1]:+.1f})")
    for label, pid in METRIC_ANCHOR:
        x, y = lm_px[pid]
        d = -disp_at(map_x, map_y, float(x), float(y))
        print(f"  {label}位移: |d|={np.hypot(*d):.2f}px (dx={d[0]:+.1f} dy={d[1]:+.1f})")
    for label, (x, y) in zip(("背景左上", "背景右下"),
                             ((6.0, 6.0), (float(w - 7), float(h - 7)))):
        d = -disp_at(map_x, map_y, x, y)
        print(f"  {label}: |d|={np.hypot(*d):.2f}px")


def process_image(path: str, engine: InferenceEngine, algos: list[str],
                  strength: float, out_dir: Path) -> None:
    src = cv2.imread(path)
    if src is None:
        print(f"[跳过] 读图失败: {path}")
        return
    src = np.ascontiguousarray(src)
    stem = Path(path).stem
    ctx = engine.process(src, hands=False)
    if not ctx.faces:
        print(f"[跳过] 未检出人脸: {path}")
        return
    h, w = src.shape[:2]
    yaws = [estimate_yaw_deg(f.landmarks) for f in ctx.faces]
    print(f"\n== {path}  {w}x{h}  脸数={len(ctx.faces)}  "
          f"yaw={','.join(f'{y:+.1f}°' for y in yaws)} ==")

    for algo in algos:
        # 与 BeautyEffect.process 一致：逐脸串行 remap（前一张脸的输出
        # 是下一张脸的输入）
        out = src.copy()
        for fi, face in enumerate(ctx.faces):
            lm = face.landmarks
            lm_px = lm[:, :2] * np.array([w, h])
            yaw = yaws[fi]
            gated = strength * pose_gate(yaw)   # 与 slim_face_maps 内部一致
            P, D, W, brush_r, mover_ids, guard_ids = slim_face_controls(
                w, h, lm, gated, yaw)
            map_x, map_y = slim_face_maps(w, h, lm, strength=strength,
                                          yaw_deg=yaw, method=algo)
            out = cv2.remap(out, map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
            suffix = f"_f{fi}" if len(ctx.faces) > 1 else ""
            label = f"SLIM s={strength:.2f} {algo}"
            tag = f"{strength:.2f}".replace(".", "")
            cv2.imwrite(str(out_dir / f"{stem}{suffix}_lm_{algo}.jpg"),
                        draw_landmarks_debug(src, lm_px, mover_ids,
                                             guard_ids, P, D,
                                             n_movers=len(mover_ids),
                                             guard_w=W[len(mover_ids):]),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(out_dir / f"{stem}{suffix}_field_{algo}.jpg"),
                        draw_field(src, map_x, map_y),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(out_dir / f"{stem}{suffix}_ab_{algo}_s{tag}.jpg"),
                        ab_compare(src, out, lm_px, label),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            report_metrics(f"{algo} f{fi}", map_x, map_y, lm_px, h, w)
            print(f"  输出: outputs/debug/{stem}{suffix}_*_({algo}).jpg  "
                  f"(movers={len(mover_ids)} R={brush_r:.0f}px)")
        if len(ctx.faces) > 1:
            cv2.imwrite(str(out_dir / f"{stem}_wipe_{algo}_s{tag}.jpg"),
                        wipe_compare(src, out),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> int:
    ap = argparse.ArgumentParser(description="瘦脸 debug 可视化")
    ap.add_argument("inputs", nargs="*", help="图片路径（默认全套素材）")
    ap.add_argument("--algo", default="v5", choices=["v4", "v5", "both"])
    ap.add_argument("--strength", type=float, default=0.40, help="瘦脸强度 0~1")
    ap.add_argument("--out", default="outputs/debug")
    args = ap.parse_args()

    inputs = args.inputs or DEFAULT_INPUTS
    algos = ["v4", "v5"] if args.algo == "both" else [args.algo]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = InferenceEngine()
    for p in inputs:
        process_image(p, engine, algos, args.strength, out_dir)
    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
