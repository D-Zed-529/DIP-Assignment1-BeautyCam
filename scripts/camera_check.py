"""摄像头诊断：枚举索引 × 后端，报告授权状态与可用性。

用法：python scripts/camera_check.py [--scan 5]

排障顺序（macOS）：
  1. 本脚本若全部 isOpened=False 且 stderr 出现
     "not authorized to capture video" → 授权问题，见第 2 步；
  2. 系统设置 → 隐私与安全性 → 摄像头 → 勾选运行 Python 的宿主应用
     （Terminal / iTerm / VS Code / PyCharm…），改完重启该应用；
     若列表里没有该应用：先跑一次本脚本触发授权弹窗；
     弹窗被误拒过：`tccutil reset Camera` 后重跑；
  3. 确认没有其他 App 独占摄像头（FaceTime/Zoom/虚拟摄像头驱动）；
  4. 本机有 Continuity Camera 时索引可能偏移，用 --scan 找可用索引，
     再在 GUI 采集源里填对应编号。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def probe(index: int, backend: int, warmup: int = 6) -> dict:
    cap = cv2.VideoCapture(index, backend)
    info = {"index": index, "opened": cap.isOpened()}
    if info["opened"]:
        ret = None
        for _ in range(warmup):
            ret, frame = cap.read()
            if ret:
                info["size"] = (frame.shape[1], frame.shape[0])
                break
            time.sleep(0.1)
        info["read"] = bool(ret)
    cap.release()
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="摄像头诊断")
    ap.add_argument("--scan", type=int, default=3,
                    help="枚举的摄像头索引数量（默认 3）")
    args = ap.parse_args()

    print(f"cv2 {cv2.__version__} | 平台后端 AVFoundation 可用: "
          f"{hasattr(cv2, 'CAP_AVFOUNDATION')}")
    any_ok = False
    for idx in range(args.scan):
        info = probe(idx, cv2.CAP_AVFOUNDATION)
        if info["opened"]:
            any_ok = True
            size = info.get("size", "?")
            print(f"[{idx}] isOpened=True  首帧={'OK' if info['read'] else '空'} "
                  f"分辨率={size}")
        else:
            print(f"[{idx}] isOpened=False")
    if not any_ok:
        print("\n全部索引打开失败。若上方 stderr 有 'not authorized to capture "
              "video'：\n  系统设置 → 隐私与安全性 → 摄像头 → 勾选运行 Python "
              "的终端应用，重启该应用后重试。\n  曾误拒授权可先执行 "
              "`tccutil reset Camera`。")
        return 1
    print("\n有可用摄像头，GUI 里选对索引即可。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
