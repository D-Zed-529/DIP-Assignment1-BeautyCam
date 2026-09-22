"""headless 批跑管线（评测数据生产线，不依赖 GUI）。

用法示例：
  # 目录下所有图片跑美颜链，输出到 outputs/
  python scripts/run_pipeline.py --input assets/samples --output outputs/

  # 视频前 100 帧，美颜关闭、低光开启，逐帧存图
  python scripts/run_pipeline.py --input demo.mp4 --no-beauty --lowlight \
      --max-frames 100

  # 虚拟背景：换背景图（内置图库在 assets/backgrounds/）
  python scripts/run_pipeline.py --input assets/samples --segment \
      --seg-mode image --seg-bg assets/backgrounds/02_冷色渐变.jpg

  # 背景虚化 + 三档边缘处理对比图（答辩素材，一张图看全差异）
  python scripts/run_pipeline.py --input assets/samples --segment \
      --seg-mode blur --seg-compare

  # 禁用全部推理效果（测采集/IO 吞吐基线）
  python scripts/run_pipeline.py --input dir/ --raw

  # 自适应画质：分区自动曝光 + CLAHE + 白平衡（全时段经典 DIP 校正）
  python scripts/run_pipeline.py --input assets/samples --no-beauty \
      --autoenhance --ae-strength 1.0 --ae-smooth 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.camera import ImageSequenceSource, VideoFileSource  # noqa: E402
from core.context import FrameContext                          # noqa: E402
from core.effects.autoenhance import AutoEnhanceEffect         # noqa: E402
from core.effects.beauty import BeautyEffect                   # noqa: E402
from core.effects.lowlight import LowLightDnnEffect, LowLightEffect  # noqa: E402
from core.effects.segment import (                              # noqa: E402
    MODE_BLUR, MODE_COLOR, MODE_IMAGE, SegmentEffect,
)
from core.infer import (                                        # noqa: E402
    DEFAULT_SEGMENTER, SEGMENTER_INTERVAL, SEGMENTER_SPECS, get_engine,
)
from core.pipeline import (                                     # noqa: E402
    NEED_FACES, NEED_HANDS, NEED_SEGMENTATION, Pipeline,
)

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")

# 边缘处理对比档（--seg-compare 用）：从硬边到推荐档，用来展示
# "为什么不能直接把模型置信图当 alpha"，以及引导滤波带来的边缘改善。
SEG_COMPARE_TIERS: list[tuple[str, dict]] = [
    ("硬掩膜 (对比度1.0)", dict(matte_contrast=1.0, refine=False, feather=0.0)),
    ("朴素软掩膜 (对比度0)", dict(matte_contrast=0.0, refine=False)),
    ("软掩膜+精修", dict(matte_contrast=0.0, refine=True)),
    ("推荐档 (0.3+精修)", dict(matte_contrast=0.3, refine=True)),
]


def write_image(path: Path, img: np.ndarray) -> None:
    """写图片（支持中文路径）。

    不能用 cv2.imwrite：Windows 上遇到非 ASCII 路径会**静默写错文件名**
    （中文名变乱码文件）而不报错，输入文件名带中文时结果目录会一片混乱。
    """
    ext = path.suffix or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        path.write_bytes(buf.tobytes())


# 中文字体候选路径（Pillow 画标签用；找不到就退回 ASCII，不影响产出）
CJK_FONTS = (
    "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)
_LABEL_FONT = None
_LABEL_FONT_TRIED = False


def _label_font(size: int = 22):
    """懒加载中文字体；找不到可用字体返回 None。"""
    global _LABEL_FONT, _LABEL_FONT_TRIED
    if not _LABEL_FONT_TRIED:
        _LABEL_FONT_TRIED = True
        for path in CJK_FONTS:
            if os.path.exists(path):
                try:
                    _LABEL_FONT = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
    return _LABEL_FONT


def _label_panel(img: np.ndarray, text: str) -> np.ndarray:
    """在图上贴左上角标签。

    用 Pillow 而不是 cv2.putText —— OpenCV 的 Hershey 字体**没有中文字形**，
    中文标签会渲染成一串方块。找不到中文字体时退回 ASCII 编号，保证可用。
    """
    out = img.copy()
    font = _label_font()
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    box = draw.textbbox((0, 0), text, font=font) if font else (0, 0, len(text) * 14, 20)
    draw.rectangle((0, 0, box[2] + 16, box[3] + 10), fill=(0, 0, 0))
    if font:
        draw.text((8, 5), text, font=font, fill=(255, 255, 255))
    else:
        cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def build_comparison(pipeline: Pipeline, segment: SegmentEffect,
                     frame: np.ndarray, ctx: FrameContext) -> np.ndarray:
    """把边缘处理各档并排出成一张对比图（答辩素材）。

    推理只跑一次、ctx 复用：只切换 alpha 的处理档位重跑合成，成本极低。
    每档先 reset_temporal，避免继承上一档的时域平滑状态而互相污染。
    """
    original_params = segment.get_params()

    segment.set_enabled(False)
    base = pipeline.process(frame.copy(), ctx)   # 低光 + 美颜，未做背景替换
    segment.set_enabled(True)

    panels = [_label_panel(base, "原图（未替换）")]
    try:
        for label, params in SEG_COMPARE_TIERS:
            segment.set_params(**params)
            segment.reset_temporal()
            panels.append(_label_panel(segment.process(base.copy(), ctx), label))
    finally:
        segment.set_params(**original_params)
        segment.reset_temporal()
    return np.hstack(panels)


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
    ap.add_argument("--lowlight-dnn", action="store_true",
                    help="启用 SCI 深度低光增强（Phase 2 主力档，覆盖 --lowlight）")
    ap.add_argument("--lowlight-level", choices=("easy", "medium", "difficult"),
                    default="medium", help="SCI 强度档（仅 --lowlight-dnn）")
    ap.add_argument("--lowlight-strength", type=float, default=1.0,
                    help="低光增强强度 0~1（结果与暗图混合）")
    ap.add_argument("--lowlight-force", action="store_true",
                    help="关闭亮度自动触发（强制所有帧都增强；评测用）")
    ap.add_argument("--raw", action="store_true",
                    help="关闭全部效果（IO/采集基线）")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="最多处理帧数（0=不限，视频调试用）")
    ap.add_argument("--save-video", action="store_true",
                    help="视频输入时输出合成视频而非逐帧图片")
    # ---- 自适应画质优化（Phase 4） ----
    ae = ap.add_argument_group("自适应画质优化")
    ae.add_argument("--autoenhance", action="store_true",
                    help="启用自适应画质（分区自动曝光 + CLAHE + 白平衡，全时段经典 DIP）")
    ae.add_argument("--ae-strength", type=float, default=0.8,
                    help="总强度 0~1（结果与原图混合）")
    ae.add_argument("--ae-face-expo", type=float, default=0.7,
                    help="人脸曝光优先 0~1（人脸目标亮度 115→150 插值；0=全图统一）")
    ae.add_argument("--ae-contrast", type=float, default=0.3,
                    help="CLAHE 对比度 0~1（0=关）")
    ae.add_argument("--ae-color", type=float, default=0.5,
                    help="灰世界白平衡强度 0~1（0=关）")
    ae.add_argument("--ae-sat", type=float, default=0.0,
                    help="饱和度增强 0~1（0=关）")
    ae.add_argument("--ae-smooth", type=float, default=0.8,
                    help="统计量时域平滑 0~0.95")
    # ---- 人像虚化 / 背景替换（Phase 3） ----
    seg = ap.add_argument_group("人像虚化 / 背景替换")
    seg.add_argument("--segment", action="store_true", help="启用背景替换")
    seg.add_argument("--seg-mode", choices=(MODE_BLUR, MODE_IMAGE, MODE_COLOR),
                     default=MODE_BLUR, help="虚化 / 换图 / 纯色")
    seg.add_argument("--seg-strength", type=float, default=0.6,
                     help="虚化强度 0~1（仅 --seg-mode blur）")
    seg.add_argument("--seg-bg", default="", help="背景图路径（仅 --seg-mode image）")
    seg.add_argument("--seg-bg-color", default="#3C6E71", help="纯色背景 #RRGGBB")
    seg.add_argument("--seg-matte-contrast", type=float, default=0.3,
                     help="matte 对比度拉伸 0~1；0 = 朴素置信图基线")
    seg.add_argument("--seg-no-refine", dest="seg_refine", action="store_false",
                     default=True, help="关闭 guided filter 边缘精修")
    seg.add_argument("--seg-feather", type=float, default=3.0, help="羽化半径 px")
    seg.add_argument("--seg-smooth", type=float, default=0.7, help="时域平滑 0~1")
    seg.add_argument("--seg-interval", type=int, default=0,
                     help="分割推理间隔帧数；0=跟随所选模型（二元 1 / 多分类 4）")
    seg.add_argument("--seg-model", choices=tuple(SEGMENTER_SPECS),
                     default=DEFAULT_SEGMENTER, help="分割模型")
    seg.add_argument("--seg-compare", action="store_true",
                     help="另出边缘处理四档对比图（答辩素材）")
    seg.add_argument("--seg-dump-alpha", action="store_true",
                     help="另存 alpha 灰度图（人应是白的；用于排查掩膜反转）")
    args = ap.parse_args()

    source = build_source(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 组装管线（与 GUI 同一套效果类）；autoenhance 放链首：先校正曝光/色调
    autoenhance = AutoEnhanceEffect(
        enabled=args.autoenhance and not args.raw,
        params={"strength": args.ae_strength,
                "face_exposure": args.ae_face_expo,
                "contrast": args.ae_contrast, "color": args.ae_color,
                "saturation": args.ae_sat, "smooth": args.ae_smooth})
    beauty = BeautyEffect(enabled=args.beauty and not args.raw, params={
        "smooth": args.smooth, "whiten": args.whiten,
        "slim": args.slim, "eye_enabled": args.eye > 0,
        "eye_strength": args.eye,
    })
    if args.lowlight_dnn:
        lowlight = LowLightDnnEffect(
            enabled=not args.raw,
            params={"level": args.lowlight_level,
                    "strength": args.lowlight_strength,
                    "auto": not args.lowlight_force})
    else:
        lowlight = LowLightEffect(
            enabled=args.lowlight and not args.raw,
            params={"strength": args.lowlight_strength,
                    "auto": not args.lowlight_force})
    bg_path = os.path.abspath(args.seg_bg) if args.seg_bg else ""
    # 间隔默认跟随模型：二元 13ms 可每帧跑，多分类 155ms 必须隔帧
    seg_interval = args.seg_interval or SEGMENTER_INTERVAL[args.seg_model]
    segment = SegmentEffect(enabled=args.segment and not args.raw, params={
        "mode": args.seg_mode, "strength": args.seg_strength,
        "bg_path": bg_path, "bg_color": args.seg_bg_color,
        "matte_contrast": args.seg_matte_contrast, "refine": args.seg_refine,
        "feather": args.seg_feather, "smooth": args.seg_smooth,
        "infer_interval": seg_interval,
    })
    pipeline = Pipeline([autoenhance, lowlight, beauty, segment])

    if args.raw:
        engine = None
    else:
        engine = get_engine()
        engine.set_segmenter_model(args.seg_model)

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

        if isinstance(source, ImageSequenceSource):
            # 每张图互相独立：不继承上一张的掩膜/背景缓存（否则串帧）
            pipeline.reset_temporal()

        ctx = None
        if engine is not None:
            t0 = time.perf_counter()
            needs = pipeline.infer_needs_for(n)
            ctx = engine.process(
                frame,
                faces=NEED_FACES in needs, hands=NEED_HANDS in needs,
                segmentation=NEED_SEGMENTATION in needs)
            t_infer += time.perf_counter() - t0
            t0 = time.perf_counter()
            frame = pipeline.process(frame, ctx)
            t_effect += time.perf_counter() - t0
        else:
            frame = pipeline.process(frame, FrameContext(width=frame.shape[1],
                                                         height=frame.shape[0]))

        stem = Path(getattr(source, "current_path", "") or f"frame_{n:05d}").stem
        if writer is not None:
            writer.write(frame)
        else:
            write_image(out_dir / f"{stem}_out.jpg", frame)
            if ctx is not None and args.segment and not args.raw:
                if args.seg_dump_alpha:
                    write_image(out_dir / f"{stem}_alpha.png",
                                (np.clip(segment.debug_alpha, 0, 1) * 255
                                 ).astype(np.uint8))
                if args.seg_compare:
                    write_image(out_dir / f"{stem}_compare.jpg",
                                build_comparison(pipeline, segment, frame, ctx))
        n += 1

    if writer is not None:
        writer.release()
    source.release()
    dt = time.perf_counter() - t_all0

    print(f"\n处理 {n} 帧，总耗时 {dt:.2f}s（{n / max(dt, 1e-9):.1f} fps）")
    if engine is not None:
        print(f"  推理累计 {t_infer:.2f}s，效果链累计 {t_effect:.2f}s，"
              f"均摊 {(t_infer + t_effect) / max(n, 1) * 1000:.1f} ms/帧")
        print(f"  分割模型：{engine.segmenter_model}"
              f"（建议推理间隔 {engine.recommended_interval} 帧）")
    print(f"输出目录：{out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
