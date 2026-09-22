"""headless 批跑管线（评测数据生产线，不依赖 GUI）。

用法示例：
  # 目录下所有图片跑美颜链，输出到 outputs/
  python scripts/run_pipeline.py --input assets/samples --output outputs/

  # 视频前 100 帧，美颜关闭、低光开启，逐帧存图
  python scripts/run_pipeline.py --input demo.mp4 --no-beauty --lowlight \
      --max-frames 100

  # 禁用全部推理效果（测采集/IO 吞吐基线）
  python scripts/run_pipeline.py --input dir/ --raw
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.camera import ImageSequenceSource, VideoFileSource  # noqa: E402
from core.context import FrameContext                          # noqa: E402
from core.effects.beauty import BeautyEffect                    # noqa: E402
from core.effects.lowlight import LowLightEffect                # noqa: E402
from core.infer import get_engine                               # noqa: E402
from core.pipeline import Pipeline                              # noqa: E402

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def build_source(path: str):
    p = Path(path)
    if p.is_dir():
        return ImageSequenceSource(directory=str(p))
    if p.suffix.lower() in VIDEO_EXTS:
        return VideoFileSource(str(p))
    if p.is_file():
        return ImageSequenceSource(paths=[str(p)])
    raise SystemExit(f"输入不存在或类型不支持：{path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="BeautyCam headless 管线批跑",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", "-i", required=True,
                    help="图片目录 / 单张图片 / 视频文件")
    ap.add_argument("--output", "-o", default="outputs", help="输出目录")
    ap.add_argument("--beauty", dest="beauty", action="store_true",
                    default=True, help="启用美颜（默认开）")
    ap.add_argument("--no-beauty", dest="beauty", action="store_false",
                    help="关闭美颜")
    ap.add_argument("--smooth", type=float, default=0.6, help="磨皮 0~1")
    ap.add_argument("--whiten", type=float, default=15.0, help="美白 0~30")
    ap.add_argument("--slim", type=float, default=0.35, help="瘦脸 0~1")
    ap.add_argument("--eye", type=float, default=0.0, help="大眼强度 0~0.5（0=关）")
    ap.add_argument("--lowlight", action="store_true", help="启用启发式低光增强")
    ap.add_argument("--raw", action="store_true",
                    help="关闭全部效果（IO/采集基线）")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="最多处理帧数（0=不限，视频调试用）")
    ap.add_argument("--save-video", action="store_true",
                    help="视频输入时输出合成视频而非逐帧图片")
    args = ap.parse_args()

    source = build_source(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 组装管线（与 GUI 同一套效果类）
    beauty = BeautyEffect(enabled=args.beauty and not args.raw, params={
        "smooth": args.smooth, "whiten": args.whiten,
        "slim": args.slim, "eye_enabled": args.eye > 0,
        "eye_strength": args.eye,
    })
    lowlight = LowLightEffect(enabled=args.lowlight and not args.raw)
    pipeline = Pipeline([lowlight, beauty])

    if args.raw:
        engine = None
    else:
        engine = get_engine()

    if not source.open():
        print(f"无法打开输入：{source.name}", file=sys.stderr)
        return 1

    writer = None
    if args.save_video and isinstance(source, VideoFileSource):
        ret, probe = source.read()
        if not ret:
            print("视频读取失败", file=sys.stderr)
            return 1
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_dir / "result.mp4"), fourcc,
                                 source.fps, (probe.shape[1], probe.shape[0]))
        first = probe
    else:
        first = None

    n, t_infer, t_effect = 0, 0.0, 0.0
    t_all0 = time.perf_counter()
    pending = [first] if first is not None else []

    while True:
        if pending:
            frame = pending.pop(0)
        else:
            ret, frame = source.read()
            if not ret:
                break
        if args.max_frames and n >= args.max_frames:
            break

        if engine is not None:
            t0 = time.perf_counter()
            needs = pipeline.infer_needs()
            ctx = engine.process(
                frame,
                faces="faces" in needs, hands="hands" in needs,
                segmentation="segmentation" in needs)
            t_infer += time.perf_counter() - t0
            t0 = time.perf_counter()
            frame = pipeline.process(frame, ctx)
            t_effect += time.perf_counter() - t0
        else:
            frame = pipeline.process(frame, FrameContext(width=frame.shape[1],
                                                         height=frame.shape[0]))

        if writer is not None:
            writer.write(frame)
        else:
            stem = Path(getattr(source, "current_path", "") or f"frame_{n:05d}").stem
            cv2.imwrite(str(out_dir / f"{stem}_out.jpg"), frame)
        n += 1

    if writer is not None:
        writer.release()
    source.release()
    dt = time.perf_counter() - t_all0

    print(f"\n处理 {n} 帧，总耗时 {dt:.2f}s（{n / max(dt, 1e-9):.1f} fps）")
    if engine is not None:
        print(f"  推理累计 {t_infer:.2f}s，效果链累计 {t_effect:.2f}s，"
              f"均摊 {(t_infer + t_effect) / max(n, 1) * 1000:.1f} ms/帧")
    print(f"输出目录：{out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
