"""MLS 瘦脸 v4 真图 A/B 评测：标记位移 + 性能 + 对照图。

用法：.venv/bin/python scripts/eval_mls_slim.py [图像路径]
输出：outputs/mls_v4_*.jpg 与位移/耗时指标（stdout）。
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FrameContext, FaceInfo  # noqa: E402
from core.effects.beauty import (  # noqa: E402
    JAW_LEFT_IDS, JAW_RIGHT_IDS, slim_face,
)
from core.infer import InferenceEngine  # noqa: E402

# 标记点（MediaPipe id）：下颌链中段 / 太阳穴 / 嘴角 / 额顶
JAW_MID = [149, 378]          # 双侧下颌中点（瘦脸主目标）
TEMPLE = [127, 356]           # 太阳穴（保护域）
MOUTH = [61, 291]             # 嘴角（保护域）
FOREHEAD = [10]               # 额顶（保护域）

COLORS = {"下颌": (0, 255, 0), "太阳穴": (0, 165, 255),
          "嘴角": (0, 0, 255), "额顶": (255, 0, 0)}


def draw_marker(img, xy, color, r=7):
    x, y = int(round(xy[0])), int(round(xy[1]))
    cv2.circle(img, (x, y), r, color, 2, cv2.LINE_AA)


def centroid_around(img, xy, color, win=14):
    """以 xy 为中心的局部窗口内标记质心（避免多标记串扰）。"""
    x, y = int(xy[0]), int(xy[1])
    roi = img[max(y - win, 0):y + win, max(x - win, 0):x + win]
    b, g, r = roi[..., 0].astype(int), roi[..., 1].astype(int), roi[..., 2].astype(int)
    m = (g > 200) & (b < 100) & (r < 100)
    if color == COLORS["太阳穴"]:      # 橙 (B,G,R)=(0,165,255)
        m = (r > 200) & (g > 100) & (g < 180) & (b < 60)
    elif color == COLORS["嘴角"]:      # 红 (0,0,255)
        m = (r > 200) & (g < 100) & (b < 100)
    elif color == COLORS["额顶"]:      # 蓝 (255,0,0)
        m = (b > 200) & (g < 100) & (r < 100)
    if m.sum() < 5:
        return None
    ys, xs = np.nonzero(m)
    return (xs.mean() + max(x - win, 0), ys.mean() + max(y - win, 0))


def main(img_path: str):
    src = cv2.imread(img_path)
    assert src is not None, f"读图失败: {img_path}"
    h, w = src.shape[:2]
    print(f"输入 {w}x{h}: {img_path}")

    eng = InferenceEngine()
    ctx = eng.process(src, hands=False)
    assert ctx.faces and ctx.faces[0].landmarks is not None, "未检出人脸/关键点"
    face = ctx.faces[0]
    lm = face.landmarks            # (468,3) 归一化
    print(f"人脸框 {face.box}, 关键点 {lm.shape}")

    # ---------- 位移测量（带标记） ----------
    marked = src.copy()
    lm_px = lm[:, :2] * np.array([w, h])
    groups = {"下颌": JAW_MID, "太阳穴": TEMPLE,
              "嘴角": MOUTH, "额顶": FOREHEAD}
    for label, ids in groups.items():
        for i in ids:
            draw_marker(marked, lm_px[i], COLORS[label], r=6)

    out = slim_face(marked.copy(), lm, strength=0.40)

    print("\n== 标记位移（strength=0.40，正值为靠近脸中心/保护域应≈0）==")
    face_cx = lm_px[1, 0]
    for label, ids in groups.items():
        for i in ids:
            c0 = centroid_around(marked, lm_px[i], COLORS[label])
            c1 = centroid_around(out, lm_px[i], COLORS[label])
            if c0 is None or c1 is None:
                print(f"  {label}#{i}: 标记丢失")
                continue
            dx, dy = c1[0] - c0[0], c1[1] - c0[1]
            inward = (1 if lm_px[i, 0] < face_cx else -1) * dx  # >0 = 内收
            print(f"  {label}#{i}: dx={dx:+.1f} dy={dy:+.1f} 内收量={inward:+.1f}px")

    # ---------- 性能 ----------
    t0 = time.perf_counter()
    for _ in range(20):
        slim_face(src, lm, strength=0.40)
    dt = (time.perf_counter() - t0) / 20 * 1000
    print(f"\nslim_face 均耗: {dt:.1f}ms/帧 ({w}x{h})")

    # ---------- 对照图 ----------
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "mls_v4_origin.jpg"), src, [cv2.IMWRITE_JPEG_QUALITY, 92])
    slim_vis = slim_face(src.copy(), lm, strength=0.40)
    cv2.imwrite(str(out_dir / "mls_v4_slim40.jpg"), slim_vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    slim100 = slim_face(src.copy(), lm, strength=1.0)
    cv2.imwrite(str(out_dir / "mls_v4_slim100.jpg"), slim100, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\n对照图已存 {out_dir}/mls_v4_{{origin,slim40,slim100}}.jpg")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "assets/samples/portrait1.jpg")
