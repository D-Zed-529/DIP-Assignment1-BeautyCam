"""相机工作线程（QThread）。

线程纪律（AGENTS.md / PLAN §3.3）：本线程只做 采集 → 推理 → 效果链，
**永不触碰 GUI 控件**；帧/照片/状态一律经信号 emit 到主线程。
参数更新走 pipeline.set_params()（内部锁保护）。

一期两个 bug 的根治：
  - open_camera() 启动了两个 update_camera 线程 → v2 只有一个 worker 实例，
    start 前保证未运行。
  - v_sign_frames / smile_start_time 缺 global → 状态封装在
    AutoCaptureState 实例里，无全局变量。
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.camera import CameraSource
from core.context import FrameContext
from core.effects.hdr import (
    BURST_INTERVAL_S, DEFAULT_EV_PRESET, EV_PRESETS, build_hdr_comparison,
    hdr_pipeline,
)
from core.gestures import AutoCaptureState, any_smiling, is_v_sign
from core.infer import InferenceEngine
from core.pipeline import (
    NEED_FACES, NEED_HANDS, NEED_SEGMENTATION, Pipeline,
)

# 自动拍照的触发器名（与 GUI 复选框一一对应）
TRIGGER_MANUAL, TRIGGER_V_SIGN, TRIGGER_SMILE = "manual", "v_sign", "smile"

PHOTOS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "photos")

FPS_SMOOTHING = 0.9   # FPS 指数平滑系数

# HDR 连拍读帧的失败重试（连拍期间偶发空帧，超过则放弃本组）
HDR_MAX_EMPTY = 10

# 连拍/融合结束后丢弃的帧数：连拍间隔的 sleep 期间不 read()，
# AVFoundation 采集缓冲会积压旧帧，恢复后预览有"回放感"卡顿，丢几帧追上实时
HDR_POST_DISCARD = 3

# 预览预缩放尺寸：worker 线程把处理后的帧等比缩到该尺寸内再 emit。
# 原因：主线程每帧做 Qt SmoothTransformation 缩放（960×540）要 3~6ms
# 且卡 GUI；cv2.INTER_AREA 在 worker 侧 ~0.5ms，主线程只剩 QImage 包装
# （零拷贝构造 + setPixmap），界面响应明显更顺。
DISPLAY_SIZE = (960, 540)


class CameraWorker(QThread):
    """采集 + 推理 + 效果链的工作线程。"""

    frame_ready = Signal(object)        # np.ndarray（BGR，处理后）
    photo_saved = Signal(str)           # 照片文件路径
    fps_changed = Signal(float)
    face_count_changed = Signal(int)    # 每帧人脸数（状态栏显示）
    status_message = Signal(str)
    source_finished = Signal()          # 视频/图片序列读完后发出
    failed = Signal(str)

    def __init__(self, source: CameraSource, pipeline: Pipeline,
                 engine: InferenceEngine,
                 triggers: Optional[set[str]] = None,
                 display_size: tuple[int, int] = DISPLAY_SIZE,
                 parent=None):
        super().__init__(parent)
        self.source = source
        self.pipeline = pipeline
        self.engine = engine
        self.display_size = display_size
        # 启用的自动触发器集合（GUI 可随时改，集合赋值本身原子）
        self.triggers: set[str] = triggers if triggers is not None else set()
        self._stop_flag = False
        self._capture_request = False    # 手动拍照请求（跨线程标志）
        self._segmenter_request: Optional[str] = None   # 待切换的分割模型
        # HDR 连拍请求：(EV 预设名, tonemap 方式) 或 None
        self._hdr_request: Optional[tuple[str, Optional[str]]] = None
        self._auto_state = AutoCaptureState()
        self._fps = 0.0
        self._last_frame_time = time.time()
        os.makedirs(PHOTOS_DIR, exist_ok=True)

    # ------- 外部控制（GUI 线程调用） -------

    def request_capture(self) -> None:
        """请求手动拍照（下一帧生效）。"""
        self._capture_request = True

    def request_hdr_capture(self, ev_preset: str = DEFAULT_EV_PRESET,
                            tonemap: Optional[str] = None) -> None:
        """请求 HDR 连拍融合（本帧循环顶部生效）。"""
        if ev_preset not in EV_PRESETS:
            raise ValueError(f"未知 EV 预设：{ev_preset}")
        self._hdr_request = (ev_preset, tonemap)

    def request_segmenter_model(self, key: str) -> None:
        """请求切换分割模型（下一帧生效）。

        会话重建必须在工作线程做 —— GUI 线程直接重建会和正在跑的
        segment_for_video 撞上。沿用 _capture_request 同款跨线程标志模式。
        """
        self._segmenter_request = key

    def stop(self) -> None:
        self._stop_flag = True
        self.wait(5000)

    # ------- 主循环 -------

    def run(self) -> None:
        if not self.source.open():
            detail = getattr(self.source, "last_error", "")
            self.failed.emit(f"无法打开采集源：{self.source.name}\n{detail}".strip())
            return
        self.status_message.emit(f"已打开 {self.source.name}")
        try:
            self._loop()
        except Exception as exc:  # noqa: BLE001 —— 帧循环兜底，不让线程静默死掉
            self.failed.emit(f"相机线程异常：{exc}")
        finally:
            self.source.release()

    def _loop(self) -> None:
        empty_streak = 0          # 连续空帧计数（实时源偶发空帧用）
        frame_index = 0           # 隔帧降载的相位来源（见 pipeline.infer_needs_for）
        self.pipeline.reset_temporal()   # 新采集源：作废旧掩膜/旧背景缓存
        while not self._stop_flag:
            ret, frame = self.source.read()
            if not ret or frame is None:
                if self.source.continuous:
                    # 实时相机偶发空帧（对焦/曝光切换）：退避重试，连败才算故障
                    empty_streak += 1
                    if empty_streak > 150:   # ~15s 无帧视为设备故障
                        self.failed.emit(
                            f"{self.source.name} 持续无帧，设备可能被占用或已断开")
                        break
                    time.sleep(0.1)
                    continue
                self.source_finished.emit()
                break
            empty_streak = 0

            if self._segmenter_request is not None:
                # 分割模型切换（GUI 请求）：会话重建只在本线程做
                self.engine.set_segmenter_model(self._segmenter_request)
                self._segmenter_request = None

            if self._hdr_request is not None:
                # HDR 连拍（Phase 1，拍照模式）：连拍期间不走效果链
                #（保持 0.12s 节奏的一致间隔，采集自然抖动供 ECC 对齐）。
                # 融合/写盘放后台线程 —— 首版在帧循环里同步做，worker 阻塞
                # 0.5~1s（预览冻结）且恢复后还要慢慢消化积压，实测"拍完卡一阵"
                preset, tonemap = self._hdr_request
                self._hdr_request = None
                evs = EV_PRESETS[preset]
                self.status_message.emit(f"📸 HDR 连拍中（{len(evs)} 张）…")
                frames = self._hdr_burst(frame, len(evs))
                if frames is not None:
                    self.status_message.emit("HDR 融合中（后台）…")
                    threading.Thread(
                        target=self._merge_and_save_hdr,
                        args=(frames, evs, tonemap), daemon=True).start()
                # 连拍耗时不许计入帧间隔：否则恢复首帧瞬时 FPS 被砸到 ~1fps，
                # 0.9 指数平滑要十几秒才爬回来（FPS 徽章长时间假红）
                self._fps = 0.0
                self._last_frame_time = time.time()
                for _ in range(HDR_POST_DISCARD):
                    self.source.read()   # 排掉采集缓冲积压的旧帧
                continue

            try:
                ctx = self._infer(frame, frame_index)
                frame = self.pipeline.process(frame, ctx)
            except Exception as exc:  # noqa: BLE001 —— 单帧兜底：坏帧跳过而非终止演示
                self.status_message.emit(f"跳过一帧（处理异常：{exc}）")
                frame_index += 1
                continue
            frame_index += 1
            self.face_count_changed.emit(len(ctx.faces))

            # 自动触发判定（时间持续 + 冷却，见 core/gestures）
            now = time.time()
            trigger = self._auto_state.update(
                now,
                v_sign=(TRIGGER_V_SIGN in self.triggers
                        and is_v_sign(ctx.hands, ctx.width, ctx.height)),
                smiling=(TRIGGER_SMILE in self.triggers and any_smiling(ctx.faces)),
            )
            if self._capture_request:
                self._capture_request = False
                trigger = TRIGGER_MANUAL
            if trigger is not None:
                self._save_photo(frame, trigger)

            # 帧发出（worker 侧预缩放到显示尺寸；拷贝防 QImage 读到复用缓冲）
            self.frame_ready.emit(
                np.ascontiguousarray(self._scale_for_display(frame)))

            # FPS 统计（指数平滑）
            t = time.time()
            inst = 1.0 / max(t - self._last_frame_time, 1e-6)
            if self._fps:
                self._fps = FPS_SMOOTHING * self._fps + (1 - FPS_SMOOTHING) * inst
            else:
                self._fps = inst
            self._last_frame_time = t
            self.fps_changed.emit(self._fps)

    def _scale_for_display(self, frame: np.ndarray) -> np.ndarray:
        """等比缩放进显示框内（KeepAspectRatio 语义，INTER_AREA 下采样）。"""
        dw, dh = self.display_size
        h, w = frame.shape[:2]
        if (w, h) == (dw, dh):
            return frame
        scale = min(dw / w, dh / h)
        if scale >= 1.0:
            return frame          # 小帧原样（放大反而糊）
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    def _infer(self, frame: np.ndarray, frame_index: int) -> FrameContext:
        """统一推理：效果链的本帧需求 ∨ 手势触发需求。

        隔帧降载只作用于效果链需求 —— 手势/笑脸判定必须每帧，否则 V 手势的
        持续时间判定会因缺帧而不断被打断。
        """
        needs = self.pipeline.infer_needs_for(frame_index)
        if self.triggers:
            needs |= {NEED_FACES, NEED_HANDS}
        return self.engine.process(
            frame,
            faces=NEED_FACES in needs,
            hands=NEED_HANDS in needs,
            segmentation=NEED_SEGMENTATION in needs,
        )

    def _hdr_burst(self, first: np.ndarray, count: int
                   ) -> Optional[list[np.ndarray]]:
        """连拍 count 张（含已读到的 first），帧间隔由 BURST_INTERVAL_S 控制。

        每张都 emit 预览（连拍语义"所见即拍"，也避免 0.36s 冻屏观感）；
        期间持续刷新 FPS 基准，不污染恢复后的帧间隔统计。
        读帧失败重试（对焦/曝光切换的偶发空帧）；彻底失败返回 None。
        """
        frames = [first]
        empty = 0
        while len(frames) < count and empty < HDR_MAX_EMPTY:
            time.sleep(BURST_INTERVAL_S)
            ret, frame = self.source.read()
            if ret and frame is not None:
                frames.append(frame)
                empty = 0
                self.frame_ready.emit(
                    np.ascontiguousarray(self._scale_for_display(frame)))
                self._last_frame_time = time.time()
            else:
                empty += 1
        return frames if len(frames) == count else None

    def _merge_and_save_hdr(self, frames: list[np.ndarray],
                            evs: tuple[float, ...],
                            tonemap: Optional[str]) -> None:
        """HDR 融合 + 存档（**后台线程**，不阻塞帧循环）。

        frames 是连拍时已拷贝的独立数组，与帧循环无共享；只经 Qt 信号
        （跨线程自动 queued）回报结果，不碰任何 GUI 控件。
        存档：成片 / 各 EV 原图 / 融合原片 / 对照图（P1 验收要求）。
        """
        try:
            final, exposed, merged = hdr_pipeline(frames, evs, tonemap)
        except Exception as exc:   # noqa: BLE001 —— 拍照失败不能带崩任何线程
            self.status_message.emit(f"HDR 融合失败：{exc}")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = os.path.join(PHOTOS_DIR, f"hdr_{ts}")
        try:
            for i, img in enumerate(exposed):
                cv2.imwrite(f"{base}_ev{i}_{evs[i]:+g}.jpg", img,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(f"{base}_merged.jpg", merged,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(f"{base}_final.jpg", final,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(
                f"{base}_compare.jpg",
                build_hdr_comparison(exposed[len(evs) // 2], merged, final, evs),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
        except OSError as exc:
            self.status_message.emit(f"HDR 存档写出失败：{exc}")
            return
        self.photo_saved.emit(f"{base}_final.jpg")
        self.status_message.emit(f"📸 HDR 成片已保存（{len(evs)} 张融合 + 对照图）")

    def _save_photo(self, frame: np.ndarray, trigger: str) -> None:
        filename = os.path.join(
            PHOTOS_DIR,
            f"{trigger}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
        ok = cv2.imwrite(filename, frame,
                         [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            self.photo_saved.emit(filename)
            label = {"v_sign": "V手势", "smile": "笑脸", "manual": "手动"}[trigger]
            self.status_message.emit(f"📸 已保存（{label}）")
