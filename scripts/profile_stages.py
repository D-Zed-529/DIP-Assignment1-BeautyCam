"""逐阶段性能剖析：一帧在「推理 → 各效果」链上的耗时分解（headless）。

与 bench.py 的区别：bench 按"功能单开"计时，本脚本把一帧拆到
阶段/函数级（engine 内部各模型、每个效果、上传/下载/同步开销），
用于定位组合管线慢在哪一环。口径统一：
  - 先 gpu_spin 拉时钟（AGENTS.md #24：笔记本降频会虚高 3~50 倍）；
  - GPU 异步记时时用 cuda 事件 / 显式 synchronize，避免把上一段的
    队列时间记到下一段。

用法：python scripts/profile_stages.py [--input assets/samples/portrait1.jpg]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.bench import gpu_spin, letterbox  # noqa: E402
from core.camera import CAP_HEIGHT, CAP_WIDTH  # noqa: E402
from core.effects.autoenhance import AutoEnhanceEffect  # noqa: E402
from core.effects.beauty import BeautyEffect  # noqa: E402
from core.effects.bokeh import BokehEffect  # noqa: E402
from core.effects.lowlight import LowLightDnnEffect, LowLightEffect  # noqa: E402
from core.effects.segment import SegmentEffect  # noqa: E402
from core.infer import get_engine  # noqa: E402
from core.pipeline import Pipeline  # noqa: E402

N = 8  # 每阶段固定调用次数


def t_ms(fn, n=N, warmup=2) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def sync_t_ms(fn, n=N, warmup=2) -> float:
    """GPU 涉及项：每次调用后显式 synchronize 再计时（含同步等待）。"""
    import torch
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(
        Path(__file__).resolve().parent.parent / "assets/samples/portrait1.jpg"))
    args = ap.parse_args()
    frame = cv2.imread(args.input)
    if frame is None:
        print("样本图读取失败", file=sys.stderr)
        return 1
    if frame.shape[:2] != (CAP_HEIGHT, CAP_WIDTH):
        frame = letterbox(frame, CAP_WIDTH, CAP_HEIGHT)
    h, w = frame.shape[:2]
    dark = (np.power(frame.astype(np.float32) / 255.0, 2.2)
            * 255.0 * 0.5).astype(np.uint8)

    results: list[tuple[str, float]] = []

    def rec(name, ms):
        results.append((name, ms))
        print(f"  {name:<44s} {ms:8.2f} ms", flush=True)

    print(f"== 传输/基础（{w}×{h}）==", flush=True)
    import torch
    from core.gpuops import device, download_frame, upload_frame
    gpu_spin()
    rec("H2D upload_frame(u8 720p)", sync_t_ms(lambda: upload_frame(frame)))
    t = upload_frame(frame)
    rec("D2H download_frame(u8 720p)", sync_t_ms(lambda: download_frame(t)))
    ft = t.float().div_(255.0)
    rec("D2H download float 720p", sync_t_ms(lambda: download_frame(ft)))
    alpha_like = torch.rand(1, 1, h, w, device=device())
    rec("D2H alpha float32 720p（engine→numpy 现状）",
        sync_t_ms(lambda: alpha_like[0].cpu().numpy()))

    # ---- 引擎分解 ----
    print("\n== 引擎推理分解 ==", flush=True)
    engine = get_engine()
    print(f"  后端：{engine.backend_name}，分割模型 {engine.segmenter_model}",
          flush=True)
    ctx_full = engine.process(frame, faces=True, hands=True, segmentation=True,
                              depth=True)
    # 每项单独（RVM/depth 有时域状态，重复同帧即稳态口径）
    rec("engine.process(faces)", sync_t_ms(
        lambda: engine.process(frame, faces=True, hands=False,
                               segmentation=False)))
    rec("engine.process(hands)", sync_t_ms(
        lambda: engine.process(frame, faces=False, hands=True,
                               segmentation=False)))
    rec("engine.process(segmentation RVM)", sync_t_ms(
        lambda: engine.process(frame, faces=False, hands=False,
                               segmentation=True)))
    if getattr(engine, "backend_name", "").startswith("torch"):
        rec("engine.process(depth)", sync_t_ms(
            lambda: engine.process(frame, faces=False, hands=False,
                                   segmentation=False, depth=True)))

    # ---- 效果分解（CPU 现状路径）----
    print("\n== 效果 CPU 路径分解（现状）==", flush=True)
    beauty = BeautyEffect()
    rec("BeautyEffect.process", t_ms(lambda: beauty.process(frame.copy(),
                                                             ctx_full)))
    seg = SegmentEffect(params={"smooth": 0.0})
    rec("SegmentEffect.process(blur)", t_ms(
        lambda: seg.process(frame.copy(), ctx_full)))
    seg.reset_temporal()
    bokeh = BokehEffect()
    rec("BokehEffect.process", t_ms(lambda: bokeh.process(frame.copy(),
                                                          ctx_full)))
    aex = AutoEnhanceEffect()
    rec("AutoEnhanceEffect.process", t_ms(lambda: aex.process(frame.copy(),
                                                              ctx_full)))
    low = LowLightEffect(params={"auto": False})
    rec("LowLightEffect.process(暗帧)", t_ms(lambda: low.process(dark.copy(),
                                                                 ctx_full)))
    dnn = LowLightDnnEffect(params={"auto": False})
    rec("LowLightDnnEffect.process(sci,暗帧)", t_ms(
        lambda: dnn.process(dark.copy(), ctx_full)))

    # ---- 组合（GUI 全开形态）----
    print("\n== 组合（engine + 效果链，逐帧端到端）==", flush=True)
    aex2 = AutoEnhanceEffect(enabled=True)
    low2 = LowLightDnnEffect(enabled=True, params={"auto": False})
    beauty2 = BeautyEffect(enabled=True)
    seg2 = SegmentEffect(enabled=True, params={"smooth": 0.0})
    bk2 = BokehEffect(enabled=True)
    pipe = Pipeline([aex2, low2, beauty2, seg2, bk2])
    base = dark.copy()

    def step_full(i=[0]):
        needs = pipe.infer_needs_for(i[0])
        i[0] += 1
        c = engine.process(base, faces="faces" in needs, hands=False,
                           segmentation="segmentation" in needs,
                           depth="depth" in needs)
        return pipe.process(base.copy(), c)

    gpu_spin()
    rec("全开端到端（画质+低光+美颜+虚化+深度虚化）",
        sync_t_ms(step_full, n=N))

    # 不含低光（亮场景常见形态）
    low2.set_enabled(False)
    state = {"i": 0}

    def step_bright():
        needs = pipe.infer_needs_for(state["i"])
        state["i"] += 1
        c = engine.process(frame, faces="faces" in needs, hands=False,
                           segmentation="segmentation" in needs,
                           depth="depth" in needs)
        return pipe.process(frame.copy(), c)

    rec("全开(无低光)端到端", sync_t_ms(step_bright, n=N))

    print("\n== 小结（按耗时降序）==", flush=True)
    for name, ms in sorted(results, key=lambda x: -x[1])[:12]:
        print(f"  {name:<44s} {ms:8.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
