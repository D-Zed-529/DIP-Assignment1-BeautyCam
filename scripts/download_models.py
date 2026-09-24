"""拉取模型权重到 models/（models/ 不入库）。

用法：python scripts/download_models.py [--force] [--only 子串]

P0-1 实测可用的下载源（mediapipe-models 官方桶部分路径已 404，以下为
2026-09 实测可用地址；如再失效，按文件名到 MediaPipe 官方文档找新版本）。
CUDA 迁移后的新增模型（BEAUTYCAM 部署）：
  - RVM（RobustVideoMatting resnet50 官方 TorchScript，GitHub release 直链）
  - Retinexformer LOL-v1 权重（官方 Google Drive 文件夹，经 gdown 拉取
    —— Drive 大文件需要确认令牌，urllib 拿不下）
  - Depth Anything V2 Small（HuggingFace transformers 格式，resolve 直链）
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# (文件名, 下载地址, 期望字节数——用于完整性校验，0 表示跳过)
MANIFEST: list[tuple[str, str, int]] = [
    ("face_landmarker.task",
     "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
     "face_landmarker/float16/1/face_landmarker.task",
     3_758_596),
    ("hand_landmarker.task",
     "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
     "hand_landmarker/float16/1/hand_landmarker.task",
     7_819_105),
    ("blaze_face_short_range.tflite",
     "https://storage.googleapis.com/mediapipe-models/face_detector/"
     "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
     229_746),
    ("selfie_multiclass_256x256.tflite",
     "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
     "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
     16_371_837),
    # Phase 3 分割模型：二元「人 vs 非人」，250KB（mediapipe 后端默认）
    ("selfie_segmenter.tflite",
     "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
     "selfie_segmenter/float16/latest/selfie_segmenter.tflite",
     249_537),
    # Phase 2 低光增强 SCI（CVPR 2022）ONNX 三档强度：54KB、固定 512×512
    # 输入。源仓库 Kazuhito00/SCI-ONNX-Sample（原作 Tao et al. 官方权重的
    # ONNX 化）。
    ("sci_easy_512x512.onnx",
     "https://github.com/Kazuhito00/SCI-ONNX-Sample/raw/main/model/"
     "sci_easy_512x512.onnx",
     54_613),
    ("sci_medium_512x512.onnx",
     "https://github.com/Kazuhito00/SCI-ONNX-Sample/raw/main/model/"
     "sci_medium_512x512.onnx",
     54_613),
    ("sci_difficult_512x512.onnx",
     "https://github.com/Kazuhito00/SCI-ONNX-Sample/raw/main/model/"
     "sci_difficult_512x512.onnx",
     54_613),
    # ---- CUDA 迁移新增（2026-09，算力升级后的模型替换） ----
    # RVM 人像抠图（torch 后端默认分割模型，发丝级 alpha + 视频时域一致）。
    # 默认 mobilenetv3 fp32（7.8ms/帧@720p/3060）；resnet50 fp16 为高质量
    # 档（~130ms，离线出图用：把 core/infer_torch.py 的 RVM_TS 指向它即可）。
    ("torch/rvm.ts",
     "https://github.com/PeterL1n/RobustVideoMatting/releases/download/"
     "v1.0.0/rvm_mobilenetv3_fp32.torchscript",
     0),
    ("torch/rvm_resnet50_fp16.ts",
     "https://github.com/PeterL1n/RobustVideoMatting/releases/download/"
     "v1.0.0/rvm_resnet50_fp16.torchscript",
     0),
    # Depth Anything V2 Small（transformers 格式，深度渐进虚化用）
    ("hf/depth-anything-v2-small-hf/config.json",
     "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/"
     "resolve/main/config.json", 0),
    ("hf/depth-anything-v2-small-hf/preprocessor_config.json",
     "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/"
     "resolve/main/preprocessor_config.json", 0),
    ("hf/depth-anything-v2-small-hf/model.safetensors",
     "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/"
     "resolve/main/model.safetensors", 0),
]

# Retinexformer（LOL-v1 质量档低光）：官方 Drive 文件夹（README "Download our
# models"），经 gdown --folder 拉取后取出 LOL_v1.pth。大文件 + 文件夹结构
# 决定了只能走 gdown（urllib 处理不了 Drive 的确认令牌）。
RETINEXFORMER_DRIVE = ("https://drive.google.com/drive/folders/"
                       "1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV")
RETINEXFORMER_WANT = ("LOL_v1.pth", "retinexformer_lol_v1.pth")

CHUNK = 1 << 20   # 1 MiB


def download(name: str, url: str, expect_size: int, force: bool) -> bool:
    dest = MODELS_DIR / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[跳过] {name} 已存在（{dest.stat().st_size} 字节）")
        return True
    print(f"[下载] {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, \
                open(dest, "wb") as f:
            sha = hashlib.sha256()
            got = 0
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                sha.update(chunk)
                got += len(chunk)
                print(f"\r  {name}: {got / 1e6:.1f} MB", end="", flush=True)
        print()
    except Exception as exc:  # noqa: BLE001 —— 单个模型失败不阻塞其余
        print(f"\n[失败] {name}: {exc}")
        dest.unlink(missing_ok=True)
        return False
    if expect_size and got != expect_size:
        print(f"[警告] {name} 大小 {got} != 期望 {expect_size}，请核对版本")
    print(f"[完成] {name}（sha256 {sha.hexdigest()[:16]}…）")
    return True


def download_retinexformer(force: bool) -> bool:
    """经 gdown 从官方 Drive 文件夹拉取 LOL_v1.pth。"""
    dest = MODELS_DIR / RETINEXFORMER_WANT[1]
    if dest.exists() and not force:
        print(f"[跳过] {dest.name} 已存在（{dest.stat().st_size} 字节）")
        return True
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("[失败] 需要 gdown（pip install gdown）才能从 Google Drive "
              "拉取 Retinexformer 权重")
        return False
    tmp_dir = MODELS_DIR / "_retinex_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[下载] Retinexformer LOL-v1（官方 Drive 文件夹 → {tmp_dir}）")
    ok = subprocess.call(
        [sys.executable, "-m", "gdown", "--folder",
         RETINEXFORMER_DRIVE, "-O", str(tmp_dir)])
    if ok != 0:
        print("[失败] gdown 拉取 Retinexformer 失败")
        return False
    found = next(tmp_dir.rglob(RETINEXFORMER_WANT[0]), None)
    if found is None:
        print(f"[失败] 拉取内容里没有 {RETINEXFORMER_WANT[0]}（Drive 结构"
              f"可能变化，请手动下载后放到 {dest}）")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(found.read_bytes())
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[完成] {dest.name}（{dest.stat().st_size / 1e6:.1f} MB）")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    parser.add_argument("--only", default="", help="只下载文件名含该子串的项")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    failures = [name for name, url, size in MANIFEST
                if (not args.only or args.only in name)
                and not download(name, url, size, args.force)]
    if not args.only or "retinex" in args.only:
        if not download_retinexformer(args.force):
            failures.append("retinexformer_lol_v1.pth")
    if failures:
        print(f"\n以下模型下载失败：{failures}", file=sys.stderr)
        return 1
    print("\n全部模型就绪。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
