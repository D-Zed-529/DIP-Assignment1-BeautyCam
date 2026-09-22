"""程序化生成虚拟背景图库到 assets/backgrounds/（Phase 3）。

用法：python scripts/make_backgrounds.py [--force] [--width 1280] [--height 720]

为什么程序化生成而不是找现成照片：**版权与肖像权干净**。演示与答辩材料里
"背景素材来自哪里"是可以被追问的问题，合成图没有这个负担，且体积只有几 KB、
可复现（固定随机种子）。观感上覆盖了视频会议常见的几类场景：渐变、虚化散景、
简洁几何、绿幕抠像底色。

生成的图会被 git 跟踪（.gitignore 未排除 assets/），供 GUI 图库选择器直接读取。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "backgrounds"
SEED = 20260922          # 固定随机种子：保证每次生成结果一致（可复现）
JPEG_QUALITY = 88


def _bgr(hex_color: str) -> tuple[int, int, int]:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return (b, g, r)


def gradient(w: int, h: int, top: str, bottom: str,
             angle: float = 0.35) -> np.ndarray:
    """两色线性渐变。angle 控制渐变方向（0=纯纵向，越大越斜）。"""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (ys / max(h - 1, 1)) * (1.0 - angle) + (xs / max(w - 1, 1)) * angle
    t = np.clip(t, 0.0, 1.0)[..., None]
    c0 = np.array(_bgr(top), np.float32)
    c1 = np.array(_bgr(bottom), np.float32)
    return np.clip(c0 * (1.0 - t) + c1 * t, 0, 255).astype(np.uint8)


def vignette(img: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """四角压暗，让画面中心的人物更突出（视频会议背景的常见处理）。"""
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xs - w / 2) / (w / 2)) ** 2 + ((ys - h / 2) / (h / 2)) ** 2)
    mask = np.clip(1.0 - strength * np.clip(d / 1.42, 0, 1) ** 2, 0, 1)[..., None]
    return (img.astype(np.float32) * mask).astype(np.uint8)


def bokeh(w: int, h: int, base: str, blob_color: str,
          count: int = 26, radius: int = 90) -> np.ndarray:
    """虚化散景：随机大小光斑 + 大幅高斯模糊（模拟大光圈背景）。"""
    rng = np.random.default_rng(SEED)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = _bgr(base)
    for _ in range(count):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(radius // 3, radius))
        shade = 0.35 + 0.65 * float(rng.random())
        color = tuple(int(c * shade) for c in _bgr(blob_color))
        cv2.circle(img, (cx, cy), r, color, -1, lineType=cv2.LINE_AA)
    # 半径足够大时先降采样再模糊再升采样，快且效果等价
    small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 9)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def grid(w: int, h: int, base: str, line: str, step: int = 80) -> np.ndarray:
    """浅色网格（简洁的"办公室墙面"观感）。"""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = _bgr(base)
    color = _bgr(line)
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), color, 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), color, 1, cv2.LINE_AA)
    return img


def stripes(w: int, h: int, base: str, line: str, step: int = 46) -> np.ndarray:
    """斜向细条纹（比网格更"设计感"）。"""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = _bgr(base)
    color = _bgr(line)
    for i in range(-h, w + h, step):
        cv2.line(img, (i, 0), (i + h, h), color, 3, cv2.LINE_AA)
    return img


def solid(w: int, h: int, color: str) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = _bgr(color)
    return img


def build(w: int, h: int) -> dict[str, np.ndarray]:
    """返回 {文件名: 图像}。命名用中文语义便于 GUI 图库展示与人工挑选。"""
    return {
        "01_暖色渐变.jpg": vignette(gradient(w, h, "#F3D9C0", "#B07A5A")),
        "02_冷色渐变.jpg": vignette(gradient(w, h, "#CFE0E8", "#4E6E81")),
        "03_深色渐变.jpg": vignette(gradient(w, h, "#3A4048", "#12161A")),
        "04_虚化暖光.jpg": bokeh(w, h, "#2A2118", "#FFC98A"),
        "05_虚化冷光.jpg": bokeh(w, h, "#151C24", "#8FC7E8"),
        "06_浅灰网格.jpg": grid(w, h, "#EDEDEA", "#D6D6D2"),
        "07_蓝灰条纹.jpg": stripes(w, h, "#2E3A45", "#3C4C5A"),
        "08_纯白.jpg": solid(w, h, "#FFFFFF"),
        "09_浅灰.jpg": solid(w, h, "#D8D8D6"),
        "10_绿幕.jpg": solid(w, h, "#00B140"),   # 标准绿幕色，便于后续抠像演示
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="生成虚拟背景图库（程序化合成，无版权风险）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--width", type=int, default=1280, help="背景宽度")
    ap.add_argument("--height", type=int, default=720, help="背景高度")
    ap.add_argument("--force", action="store_true", help="已存在也重新生成")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for name, img in build(args.width, args.height).items():
        path = OUT_DIR / name
        if path.exists() and not args.force:
            skipped += 1
            continue
        # 注意：不能用 cv2.imwrite —— 它在 Windows 上遇到非 ASCII 路径会
        # **静默写到错误编码的文件名**（中文名会变成乱码文件），而 cv2.imread
        # 对中文路径又会静默返回 None。统一走 imencode + Path.write_bytes。
        ok, buf = cv2.imencode(".jpg", img,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            print(f"[失败] {name}", file=sys.stderr)
            return 1
        path.write_bytes(buf.tobytes())
        written += 1
        print(f"[生成] {name}  {img.shape[1]}x{img.shape[0]}  "
              f"{path.stat().st_size / 1024:.1f} KB")
    print(f"\n完成：新生成 {written} 张，跳过已存在 {skipped} 张 -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
