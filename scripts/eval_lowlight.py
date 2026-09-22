"""低光增强客观评测：PSNR / SSIM 对比启发式 vs SCI 深度档（P5-3）。

协议（合成成对数据，可控可复现）：
  取正常光照图作为 ground truth → gamma 压暗合成"暗图"（模拟暗光成像）
  → 各方法增强 → 与 ground truth 算 PSNR / SSIM。

  暗图合成：I_dark = (I/255)^2.2 × 0.45 × 255（非线性压暗 + 能量衰减，
  亮度约 45~60，与真实暗光的分布特征接近；完全线性的 0.45×I 会让
  直方图均衡类方法虚高，不代表真实场景）。

SSIM 为标准实现（高斯 11×11 σ=1.5 窗口，亮度/对比度/结构三项），
不依赖 skimage（研究依赖最小化）。

用法：
  python scripts/eval_lowlight.py                       # 评 assets/samples
  python scripts/eval_lowlight.py --input my_dir --markdown-out outputs/eval_lowlight.md
缺 onnxruntime / SCI 权重时自动跳过深度档，启发式照常评。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FrameContext                        # noqa: E402
from core.effects.lowlight import LowLightDnnEffect, LowLightEffect  # noqa: E402

DARKEN_GAMMA = 2.2     # 暗图合成 gamma
DARKEN_SCALE = 0.45    # 暗图合成能量比例


# ---------------- 指标（纯函数，可单测） ----------------

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """峰值信噪比（dB）。uint8 输入，三通道合并计算。"""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """结构相似性（Wang et al. 2004，高斯窗标准实现）。

    输入 uint8 BGR；灰度域计算（与主流评测口径一致），返回 [0,1]。
    """
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(g, g.transpose())
    x = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mu_x = cv2.filter2D(x, -1, window)[5:-5, 5:-5]
    mu_y = cv2.filter2D(y, -1, window)[5:-5, 5:-5]
    var_x = cv2.filter2D(x * x, -1, window)[5:-5, 5:-5] - mu_x ** 2
    var_y = cv2.filter2D(y * y, -1, window)[5:-5, 5:-5] - mu_y ** 2
    cov = cv2.filter2D(x * y, -1, window)[5:-5, 5:-5] - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(np.clip(num / den, 0, 1).mean())


def synthesize_dark(frame: np.ndarray) -> np.ndarray:
    """正常光照图 → 合成暗图（评测用成对数据的"暗端"）。"""
    f = frame.astype(np.float32) / 255.0
    dark = np.power(f, DARKEN_GAMMA) * DARKEN_SCALE * 255.0
    return np.clip(dark, 0, 255).astype(np.uint8)


# ---------------- 主流程 ----------------

def evaluate(effect, dark: np.ndarray) -> np.ndarray:
    out = effect.process(dark.copy(), FrameContext())
    return out if isinstance(out, np.ndarray) else dark


def main() -> int:
    ap = argparse.ArgumentParser(description="低光增强 PSNR/SSIM 评测")
    ap.add_argument("--input", default="assets/samples", help="图片目录/单图")
    ap.add_argument("--markdown-out", default="", help="结果另存 markdown")
    ap.add_argument("--save-compare", action="store_true",
                    help="另存 暗图/各方法/GT 对照拼接图")
    args = ap.parse_args()

    p = Path(args.input)
    paths = ([p] if p.is_file() else sorted(
        q for q in p.iterdir() if q.suffix.lower() in
        (".jpg", ".jpeg", ".png", ".bmp", ".webp"))) if p.exists() else []
    if not paths:
        print(f"无输入图片：{args.input}", file=sys.stderr)
        return 1

    methods: list[tuple[str, object]] = [
        ("启发式(线性增益+直方图均衡)", LowLightEffect(params={"auto": False})),
        ("不增强(暗图原样)", None),
    ]
    for level in ("easy", "medium", "difficult"):
        methods.append((f"SCI-{level}",
                        LowLightDnnEffect(params={"auto": False, "level": level,
                                                  "infer_interval": 1})))

    rows: list[tuple[str, float, float]] = []
    per_method_strips: dict[str, list[np.ndarray]] = {}
    for img_path in paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        dark = synthesize_dark(frame)
        strips = {"GT": [frame], "暗图": [dark]}
        for name, eff in methods:
            out = dark if eff is None else evaluate(eff, dark)
            rows.append((name, psnr(out, frame), ssim(out, frame)))
            strips[name] = [out]
        per_method_strips[img_path.stem] = [
            np.hstack([frame, dark] + [strips[n][0] for n, _ in methods])]

    # 汇总：按方法聚合均值
    names = [n for n, _ in methods]
    print(f"\n输入 {len(paths)} 张（暗图合成 γ={DARKEN_GAMMA}, ×{DARKEN_SCALE}）\n")
    print(f"{'方法':<28s}{'PSNR(dB)':>10s}{'SSIM':>8s}")
    agg_lines = [f"# 低光增强客观评测（{len(paths)} 张，γ={DARKEN_GAMMA} ×{DARKEN_SCALE}）",
                 "", "| 方法 | PSNR(dB) | SSIM |", "|---|---|---|"]
    for name in names:
        vals = [(p_, s) for n, p_, s in rows if n == name]
        pm = float(np.mean([v[0] for v in vals]))
        sm = float(np.mean([v[1] for v in vals]))
        print(f"{name:<28s}{pm:>10.2f}{sm:>8.4f}")
        agg_lines.append(f"| {name} | {pm:.2f} | {sm:.4f} |")

    if args.save_compare and per_method_strips:
        out_dir = Path("outputs/eval_lowlight")
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem, strip in per_method_strips.items():
            ok, buf = cv2.imencode(".jpg", strip[0], [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                (out_dir / f"{stem}_compare.jpg").write_bytes(buf.tobytes())
        print(f"\n对照图已存 {out_dir}/（GT/暗图/{'/'.join(names)}）")

    if args.markdown_out:
        Path(args.markdown_out).write_text("\n".join(agg_lines), encoding="utf-8")
        print(f"结果已保存：{args.markdown_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
