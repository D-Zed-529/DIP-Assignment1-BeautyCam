"""性能基准：各功能单开/组合开的耗时分解与 FPS（Phase 5-2，headless）。

用法：
  python scripts/bench.py                      # 全量基准（美颜/低光/分割/组合 + 推理分解）
  python scripts/bench.py --input my.jpg       # 换样本图（默认 assets/samples/portrait1.jpg）
  python scripts/bench.py --markdown-out outputs/bench.md

产出（stdout + 可选 markdown 文件）：
  1. 各效果单开/组合开的均摊 ms/帧 与等价 FPS（720p 口径）
  2. 推理阶段分解（FaceMesh / Hands / 分割）
  3. 美颜内部阶段分解（磨皮 / 美白掩膜 / 美白 / 瘦脸 / 全链）
  4. 低光增强 ONNX 的 CPU vs CoreML 对比（缺权重/缺 onnxruntime 自动跳过）

⚠️ 内存纪律（首版教训：自适应加倍循环把 mediapipe 推理刷了上千次，
VIDEO 模式每次调用的内部分配不归还，直接把内存打爆）：
  - 每个基准项固定调用次数：推理类 OP_INFER_CALLS=4、纯算子 OP_CALLS=8，
    绝不做"跑到累计 N 秒为止"的开放式循环；
  - 每节结束 gc.collect() 并打印 RSS，异常增长当场可见。
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

try:                       # POSIX 专属；Windows 走 psapi 兜底
    import resource
except ImportError:        # pragma: no cover - Windows
    resource = None

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.camera import CAP_HEIGHT, CAP_WIDTH, mean_brightness  # noqa: E402
from core.context import FrameContext                        # noqa: E402
from core.effects.beauty import (                            # noqa: E402
    BeautyEffect, get_skin_mask, slim_face, whitening,
)
from core.effects.lowlight import LowLightEffect             # noqa: E402
from core.effects.segment import SegmentEffect               # noqa: E402
from core.infer import get_engine, preprocess_sci                            # noqa: E402
from core.pipeline import Pipeline                           # noqa: E402

DEFAULT_SAMPLE = (Path(__file__).resolve().parent.parent
                  / "assets" / "samples" / "portrait1.jpg")

OP_CALLS = 8        # 纯算子基准：每项固定调用次数
OP_INFER_CALLS = 4  # 推理类基准：每项固定调用次数（mediapipe 有按次增长的内存代价）


def letterbox(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """等比缩放并补黑边到 (w, h)——不畸变、不裁剪（face 检测对纵横比敏感）。"""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y + nh, x:x + nw] = resized
    return out


def rss_mb() -> float:
    """当前进程峰值驻留内存（MB）。macOS 的 ru_maxrss 单位是字节，
    Windows 用 GetProcessMemoryInfo 的 PeakWorkingSetSize（同为字节）。"""
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    import ctypes
    import ctypes.wintypes
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.wintypes.DWORD),
                    ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    # GetProcessMemoryInfo 在 psapi.dll（kernel32 里叫 K32GetProcessMemoryInfo，
    # 且需要精确的 ctypes 原型才不崩 —— psapi 更稳，失败就退 0 不阻塞基准）
    try:
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(pmc), pmc.cb)
        return float(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0)
    except OSError:
        return 0.0


def gpu_spin(n: int = 24) -> None:
    """GPU 空转拉时钟（笔记本 GPU 空闲即降频，短基准会虚高；
    有 CUDA 时跑几次大矩阵乘把频率顶上去）。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        a = torch.randn(2048, 2048, device="cuda")
        with torch.no_grad():
            for _ in range(n):
                a = a @ a * 0.001
        torch.cuda.synchronize()
        del a
    except Exception:   # noqa: BLE001 —— 无 CUDA 环境直接跳过
        pass


def bench(name: str, fn, calls: int = OP_CALLS, warmup: int = 1,
          spin: bool = False) -> float:
    """固定次数计均值：先预热（懒加载/内核编译不计入），再跑 calls 次取均值。

    spin=True 时预热前先空转 GPU 拉时钟（推理类基准用，笔记本降频口径防虚高）。
    """
    if spin:
        gpu_spin()
    for _ in range(warmup):
        fn()
    if spin:
        gpu_spin(8)
    t0 = time.perf_counter()
    for _ in range(calls):
        fn()
    ms = (time.perf_counter() - t0) / calls * 1000.0
    print(f"  {name:<30s} {ms:8.2f} ms/帧  ({1000.0 / ms if ms > 0 else 0:6.1f} fps 单开)",
          flush=True)
    return ms


def main() -> int:
    ap = argparse.ArgumentParser(description="BeautyCam 性能基准")
    ap.add_argument("--input", default=str(DEFAULT_SAMPLE), help="样本图（需含人脸）")
    ap.add_argument("--markdown-out", default="", help="结果另存 markdown 路径")
    args = ap.parse_args()

    frame = cv2.imread(args.input)
    if frame is None:
        print(f"样本图读取失败：{args.input}", file=sys.stderr)
        return 1
    if frame.shape[:2] != (CAP_HEIGHT, CAP_WIDTH):
        # 统一到相机 720p 口径（与 PLAN 验收线、实机预览一致，数据才可比）。
        # 必须等比缩放 + 补边（letterbox）：竖图直接硬 resize 到 16:9 会把脸
        # 横向压扁（纵横比 2:3 → 16:9），FaceMesh 检测不到畸变脸，基准就失真。
        frame = letterbox(frame, CAP_WIDTH, CAP_HEIGHT)
    h, w = frame.shape[:2]
    print(f"样本：{args.input}（{w}×{h}），同帧重复；RSS 起 {rss_mb():.0f}MB", flush=True)

    lines: list[str] = [f"# 性能基准（{w}×{h}，同帧重复）"]

    def section(title: str):
        print(f"\n== {title} ==", flush=True)
        lines.append(f"\n## {title}\n")
        lines.append("| 项目 | ms/帧 | 等效单开 fps |")
        lines.append("|---|---|---|")

    def record(name: str, ms: float):
        lines.append(f"| {name} | {ms:.2f} | {1000.0 / ms if ms > 0 else 0:.1f} |")

    # ---- 1. 基础图像算子 ----
    section("基础图像算子（优化对照用）")
    record("BGR→灰度", bench("BGR→灰度",
                             lambda: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    record("亮度检测 mean_brightness",
           bench("mean_brightness", lambda: mean_brightness(frame)))
    id_x, id_y = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
    record("全帧 remap（恒等）",
           bench("全帧恒等 remap", lambda: cv2.remap(
               frame, id_x, id_y, cv2.INTER_LINEAR)))

    # ---- 2. 推理分解 ----
    ctx = FrameContext(width=w, height=h)
    engine = None
    try:
        engine = get_engine()
        section(f"推理分解（{engine.backend_name}，同帧重复）")
        record("FaceMesh",
               bench("FaceMesh", lambda: engine.process(
                   frame, faces=True, hands=False, segmentation=False),
                   calls=OP_INFER_CALLS, warmup=3, spin=True))
        record("HandLandmarker",
               bench("Hands", lambda: engine.process(
                   frame, faces=False, hands=True, segmentation=False),
                   calls=OP_INFER_CALLS, warmup=3, spin=True))
        record(f"分割({engine.segmenter_model})",
               bench(f"分割 {engine.segmenter_model}", lambda: engine.process(
                   frame, faces=False, hands=False, segmentation=True),
                   calls=OP_INFER_CALLS, warmup=3, spin=True))
        if getattr(engine, "backend_name", "").startswith("torch"):
            try:
                record("深度(DepthAnythingV2)",
                       bench("深度估计", lambda: engine.process(
                           frame, faces=False, hands=False, segmentation=False,
                           depth=True), calls=OP_INFER_CALLS, warmup=3, spin=True))
            except Exception as e:   # noqa: BLE001 —— 缺 DA-v2 权重属预期
                print(f"  [跳过深度基准] {e}", flush=True)
        ctx = engine.process(frame, faces=True, hands=True, segmentation=True)
    except FileNotFoundError as e:
        print(f"\n[跳过推理分解] 模型缺失：{e}", flush=True)
        engine = None

    # ---- 3. 美颜内部阶段 ----
    section("美颜内部阶段分解")
    record("磨皮 bilateral(9,60,60)",
           bench("磨皮 bilateral", lambda: cv2.bilateralFilter(frame, 9, 60, 60)))
    skin = get_skin_mask(frame)
    record("肤色掩膜 get_skin_mask",
           bench("肤色掩膜", lambda: get_skin_mask(frame)))
    record("美白 whitening(LAB)",
           bench("美白", lambda: whitening(frame, skin, 15.0)))
    if ctx.faces:
        lm = ctx.faces[0].landmarks
        record("瘦脸 slim_face(全帧 remap)",
               bench("瘦脸全帧", lambda: slim_face(frame, lm, 0.4)))
        beauty = BeautyEffect()
        record("美颜全链(默认参数)",
               bench("美颜全链", lambda: beauty.process(frame.copy(), ctx)))
    del skin
    gc.collect()

    # ---- 4. 低光增强 ----
    section("低光增强")
    dark = (np.power(frame.astype(np.float32) / 255.0, 2.2)
            * 255.0 * 0.5).astype(np.uint8)   # 合成暗帧（gamma 压暗）
    low = LowLightEffect(params={"auto": False})
    record("启发式低光(暗帧)",
           bench("启发式低光", lambda: low.process(dark.copy(), ctx)))
    try:
        from core.effects.lowlight import LowLightDnnEffect  # noqa: F401
        dnn = LowLightDnnEffect(params={"auto": False})
        record("Zero-DCE++ ONNX(暗帧)",
               bench("低光 ONNX", lambda: dnn.process(dark.copy(), ctx),
                     calls=OP_INFER_CALLS))
    except Exception as e:   # noqa: BLE001 —— 缺权重/缺 onnxruntime 属预期
        print(f"  [跳过低光 ONNX] {e}", flush=True)
        lines.append(f"\n> 低光 ONNX 跳过：{e}")

    # ---- 4.5 自适应画质 ----
    section("自适应画质优化")
    from core.effects.autoenhance import AutoEnhanceEffect  # noqa: E402
    aex = AutoEnhanceEffect()
    record("自适应画质全链(默认参数)",
           bench("自适应画质", lambda: aex.process(frame.copy(), ctx)))
    aex.reset_temporal()
    del aex
    gc.collect()

    # ---- 5. 分割效果 ----
    section("人像虚化/背景替换效果")
    seg = SegmentEffect(params={"mode": "blur", "strength": 0.6, "smooth": 0.0})
    record("虚化效果(含精修)",
           bench("背景虚化", lambda: seg.process(frame.copy(), ctx)))
    seg.reset_temporal()
    del seg
    gc.collect()

    # ---- 6. 组合管线 ----
    if engine is not None:
        torch_backend = getattr(engine, "backend_name", "").startswith("torch")
        section("组合管线（整链均摊，含推理）")
        combos = [
            ("美颜", dict(beauty=True)),
            ("美颜+虚化", dict(beauty=True, segment=True)),
            ("美颜+虚化+低光(启发式)", dict(beauty=True, segment=True, lowlight=True)),
            ("自适应画质+美颜", dict(autoenhance=True, beauty=True)),
        ]
        if torch_backend:
            combos.append(("美颜+RVM虚化+深度渐进虚化",
                           dict(beauty=True, segment=True, bokeh=True)))
        for label, on in combos:
            from core.effects.bokeh import BokehEffect  # noqa: E402
            aex = AutoEnhanceEffect(enabled=on.get("autoenhance", False))
            beauty = BeautyEffect(enabled=on.get("beauty", False))
            segx = SegmentEffect(enabled=on.get("segment", False),
                                 params={"smooth": 0.0})
            bkx = BokehEffect(enabled=on.get("bokeh", False),
                              params={"strength": 0.7})
            lowx = LowLightEffect(enabled=on.get("lowlight", False),
                                  params={"auto": False})
            pipe = Pipeline([aex, lowx, beauty, segx, bkx])
            base = dark.copy() if on.get("lowlight") else frame.copy()
            state = {"i": 0}

            def step():
                needs = pipe.infer_needs_for(state["i"])
                state["i"] += 1
                c = engine.process(base, faces="faces" in needs, hands=False,
                                   segmentation="segmentation" in needs,
                                   depth="depth" in needs)
                return pipe.process(base.copy(), c)

            ms = bench(label, step, calls=OP_INFER_CALLS, warmup=2, spin=True)
            record(label, ms)
            for e in pipe.effects():
                e.reset_temporal()
            if hasattr(engine, "reset_temporal"):
                engine.reset_temporal()   # RVM 循环状态在组合间隔离
            del pipe, aex, beauty, segx, bkx, lowx, base
            gc.collect()

    # ---- 7. 低光深度模型对比（torch 后端：SCI / Retinexformer；ONNX CPU 基线） ----
    section("低光深度模型对比（512/512 推理域，暗帧）")
    try:
        from core.infer import get_lowlight_session
        for engine_key, label in (("sci", "SCI-medium (torch CUDA)"),
                                  ("retinex", "Retinexformer LOL-v1")):
            try:
                s = get_lowlight_session("medium", engine=engine_key)
                bench(label, lambda s=s: s.enhance(dark), calls=OP_INFER_CALLS, warmup=2, spin=True)
                lines.append(f"| {label}（{getattr(s, 'provider', '')}） |"
                             f" 见 stdout |")
                del s
                gc.collect()
            except Exception as e:   # noqa: BLE001 —— 缺权重属预期部署形态
                print(f"  [跳过] {label}: {e}", flush=True)
    except Exception as e:   # noqa: BLE001
        print(f"  [跳过低光对比] {e}", flush=True)
    try:
        import onnxruntime as ort
        from core.infer import LOWLIGHT_LEVELS, MODELS_DIR
        dnn_path = MODELS_DIR / LOWLIGHT_LEVELS["medium"]
        if dnn_path.exists():
            small = cv2.resize(dark, (512, 512))
            x = preprocess_sci(small)
            if "CPUExecutionProvider" in ort.get_available_providers():
                sess = ort.InferenceSession(str(dnn_path),
                                            providers=["CPUExecutionProvider"])
                iname = sess.get_inputs()[0].name
                bench("SCI-medium (ONNX CPU)",
                      lambda s=sess, n=iname: s.run(None, {n: x}),
                      calls=OP_INFER_CALLS)
                del sess
                gc.collect()
        else:
            print("\n[跳过] models/ SCI 权重不存在", flush=True)
    except ImportError:
        print("\n[跳过] onnxruntime 未安装", flush=True)

    print(f"\n完成。峰值 RSS {rss_mb():.0f}MB", flush=True)
    if args.markdown_out:
        Path(args.markdown_out).parent.mkdir(exist_ok=True)
        Path(args.markdown_out).write_text("\n".join(lines), encoding="utf-8")
        print(f"结果已保存：{args.markdown_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
