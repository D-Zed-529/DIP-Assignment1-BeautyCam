"""模型转换：MediaPipe tflite 与 SCI onnx → PyTorch TorchScript（存 models/torch/）。

技术栈全面转向 PyTorch CUDA（2026-09 Windows / RTX 3060 部署）：
  - MediaPipe 的 .task 包本质是 ZIP，内含若干 tflite 子模型；
  - tflite 走 **自研转换器 scripts/tflite_to_torch.py**（tflite2onnx 0.4.1
    在 face_landmarks / hand_detector 上布局传播 IndexError、selfie 二元
    模型的 HARD_SWISH 与自定义算子 Convolution2DTransposeBias 不支持、
    SUM 无映射——Windows 实测记录见该模块头），直接解析 flatbuffer 构建
    torch.nn.Module；
  - SCI 低光 ONNX 走 onnx2torch（转换前先把 shape inference 结果写回文件，
    绕开 onnx2torch 在 Windows 上 NamedTemporaryFile 句柄独占的坑）；
  - 统一 trace + freeze 成 TorchScript，运行期 torch.jit.load（免 pickle
    风险、免运行期依赖 tflite/onnx2torch）。

转换只做一次（模型文件不变就不用重跑）；运行期由 core/infer_torch.py
加载 TorchScript 并复刻 MediaPipe Tasks 的图逻辑（锚框解码 / NMS /
ROI 跟踪裁剪 / 坐标回投）。

用法：
  python scripts/convert_models.py            # 全部转换
  python scripts/convert_models.py --force    # 强制重转
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO / "models"
TORCH_DIR = MODELS_DIR / "torch"

sys.path.insert(0, str(REPO / "scripts"))

# (TorchScript 文件名, 来源 tflite, 是否需要从 .task 解包)
TFLITE_CONVERSIONS: list[tuple[str, str, bool]] = [
    ("face_detector.ts",       "face_detector.tflite",            True),
    ("face_landmarks.ts",      "face_landmarks_detector.tflite",  True),
    ("face_blendshapes.ts",    "face_blendshapes.tflite",         True),
    ("hand_detector.ts",       "hand_detector.tflite",            True),
    ("hand_landmarks.ts",      "hand_landmarks_detector.tflite",  True),
    ("selfie_segmenter.ts",    "selfie_segmenter.tflite",         False),
    ("selfie_multiclass.ts",   "selfie_multiclass_256x256.tflite", False),
]

# .task 包内需要解出的 tflite（名字 -> 所属 .task）
TASK_BUNDLES = {
    "face_detector.tflite": "face_landmarker.task",
    "face_landmarks_detector.tflite": "face_landmarker.task",
    "face_blendshapes.tflite": "face_landmarker.task",
    "hand_detector.tflite": "hand_landmarker.task",
    "hand_landmarks_detector.tflite": "hand_landmarker.task",
}

# SCI 低光三档（onnx2torch 路线）
ONNX_CONVERSIONS: list[tuple[str, str]] = [
    ("sci_easy.ts", "sci_easy_512x512.onnx"),
    ("sci_medium.ts", "sci_medium_512x512.onnx"),
    ("sci_difficult.ts", "sci_difficult_512x512.onnx"),
]


def extract_task_bundles() -> None:
    """从 .task（ZIP）解出 tflite 子模型到 models/（平铺，gitignore 覆盖）。"""
    for inner, task in TASK_BUNDLES.items():
        src = MODELS_DIR / task
        dst = MODELS_DIR / inner
        if dst.exists() or not src.exists():
            continue
        with zipfile.ZipFile(src) as z:
            data = z.read(inner)
        dst.write_bytes(data)
        print(f"[解包] {task} -> {inner}（{len(data) / 1e6:.2f} MB）")


def convert_tflite(ts_name: str, src_name: str) -> Path:
    """tflite →（自研转换器）→ torch.nn.Module → trace → freeze → .ts。"""
    import torch
    from tflite_to_torch import convert_tflite_module

    src = MODELS_DIR / src_name
    out = TORCH_DIR / ts_name
    if out.exists():
        print(f"[跳过] {ts_name} 已存在")
        return out
    if not src.exists():
        raise FileNotFoundError(f"缺少源模型：{src}（先跑 scripts/download_models.py）")

    module = convert_tflite_module(src)
    # 输入 shape 直接读 tflite 图声明（NHWC float [0,1]）
    shapes = module.input_shapes()
    assert len(shapes) == 1, f"{src_name} 预期单输入，实际 {len(shapes)}"
    x = torch.rand(*shapes[0])
    with torch.no_grad():
        module(x)                                   # 触发一次惰性路径
        scripted = torch.jit.trace(module, x)
        scripted = torch.jit.freeze(scripted.eval())
    torch.jit.save(scripted, str(out))
    print(f"[完成] {ts_name}（{out.stat().st_size / 1e6:.2f} MB，"
          f"输入 {shapes[0]}，输出 {module.output_shapes()}）")
    return out


def convert_onnx(ts_name: str, src_name: str) -> Path:
    """SCI onnx →（onnx2torch）→ trace → freeze → .ts。"""
    import onnx
    import torch
    import onnx2torch

    src = MODELS_DIR / src_name
    out = TORCH_DIR / ts_name
    if out.exists():
        print(f"[跳过] {ts_name} 已存在")
        return out
    if not src.exists():
        raise FileNotFoundError(f"缺少源模型：{src}（先跑 scripts/download_models.py）")

    # 传 ModelProto 而非路径：onnx2torch 对 path 输入会走 NamedTemporaryFile
    # 落盘做 shape inference，Windows 上句柄独占导致 PermissionError；
    # proto 输入则走内存 infer_shapes（54KB 小模型毫无压力）
    module = onnx2torch.convert(onnx.load(str(src))).float().eval()
    x = torch.rand(1, 3, 512, 512)
    with torch.no_grad():
        module(x)
        scripted = torch.jit.trace(module, x)
        scripted = torch.jit.freeze(scripted.eval())
    torch.jit.save(scripted, str(out))
    print(f"[完成] {ts_name}（{out.stat().st_size / 1e6:.2f} MB）")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="删除已有 TorchScript 重转")
    args = parser.parse_args()

    import torch  # noqa: F401 —— 提前失败提示

    TORCH_DIR.mkdir(parents=True, exist_ok=True)
    extract_task_bundles()
    if args.force:
        for ts, _, _ in TFLITE_CONVERSIONS:
            (TORCH_DIR / ts).unlink(missing_ok=True)
        for ts, _ in ONNX_CONVERSIONS:
            (TORCH_DIR / ts).unlink(missing_ok=True)

    failures: list[tuple[str, Exception]] = []
    for ts_name, src_name, _ in TFLITE_CONVERSIONS:
        try:
            convert_tflite(ts_name, src_name)
        except Exception as exc:  # noqa: BLE001 —— 单个失败不阻塞其余
            failures.append((ts_name, exc))
            print(f"[失败] {ts_name}: {exc}")
    for ts_name, src_name in ONNX_CONVERSIONS:
        try:
            convert_onnx(ts_name, src_name)
        except Exception as exc:  # noqa: BLE001
            failures.append((ts_name, exc))
            print(f"[失败] {ts_name}: {exc}")

    if failures:
        print(f"\n转换失败：{[n for n, _ in failures]}", file=sys.stderr)
        return 1
    print("\n全部 TorchScript 就绪。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
