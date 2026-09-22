"""HDR 纯函数单测（gamma 模拟曝光 / ECC 对齐 / Mertens 融合 / tonemap）。

末尾附带 worker 连拍编排的集成测试（假采集源 + 后台融合线程，
不依赖摄像头；需 PySide6，无则自动跳过）。
"""

from __future__ import annotations

import glob
import os
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np

from core.effects.hdr import (
    DEFAULT_EV_PRESET, EV_PRESETS, align_to_reference, build_hdr_comparison,
    exposure_lut, hdr_pipeline, merge_exposure_stack, simulate_exposure,
    tonemap_frame,
)


def textured_frame(w=320, h=240, seed=5) -> np.ndarray:
    """带大块结构 + 细噪声的合成帧（ECC 需要大尺度梯度；纯小尺度噪声
    在 6px 位移下自相关性差，真实照片总有场景结构，测试图对齐它）。"""
    rng = np.random.default_rng(seed)
    canvas = rng.integers(0, 40, (h, w, 3), dtype=np.uint8)
    for _ in range(12):
        cx, cy = int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))
        r = int(rng.integers(15, 60))
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        cv2.circle(canvas, (cx, cy), r, color, -1)
    return cv2.GaussianBlur(canvas, (3, 3), 0)


class TestExposureSimulation(unittest.TestCase):
    def test_lut_identity_at_zero_ev(self):
        self.assertTrue(np.array_equal(exposure_lut(0.0),
                                       np.arange(256, dtype=np.uint8)))

    def test_lut_direction(self):
        """ev>0 提亮（模拟过曝）、ev<0 压暗，且端点保持 0/255。"""
        up, down = exposure_lut(1.0), exposure_lut(-1.0)
        self.assertGreater(int(up[128]), 128)
        self.assertLess(int(down[128]), 128)
        self.assertEqual((int(up[0]), int(up[255])), (0, 255))
        self.assertEqual((int(down[0]), int(down[255])), (0, 255))

    def test_simulate_exposure_shape(self):
        f = textured_frame()
        out = simulate_exposure(f, 1.0)
        self.assertEqual(out.shape, f.shape)
        self.assertEqual(out.dtype, np.uint8)


class TestAlignAndMerge(unittest.TestCase):
    def test_align_identity_same_frame(self):
        f = textured_frame()
        out = align_to_reference(f, f)
        self.assertEqual(out.shape, f.shape)
        diff = float(np.abs(out.astype(int) - f.astype(int)).mean())
        self.assertLess(diff, 2.0)     # warp 边缘外推可能有极小差异

    def test_align_recovers_translation(self):
        """平移 6px 的帧经 ECC 对齐后应显著接近参考帧。"""
        ref = textured_frame()
        M = np.float32([[1, 0, 6], [0, 1, 0]])
        moved = cv2.warpAffine(ref, M, (ref.shape[1], ref.shape[0]))
        before = float(np.abs(moved.astype(int) - ref.astype(int)).mean())
        aligned = align_to_reference(ref, moved)
        after = float(np.abs(aligned.astype(int) - ref.astype(int)).mean())
        self.assertLess(after, before * 0.7)

    def test_merge_between_extremes(self):
        """暗帧 + 亮帧融合：亮度介于两档之间（取长补短，不偏到某一侧）。"""
        f = textured_frame()
        dark = simulate_exposure(f, -2.0)
        bright = simulate_exposure(f, 2.0)
        merged = merge_exposure_stack([dark, f, bright])
        m, lo, hi = (float(x.mean()) for x in (merged, dark.mean(), bright.mean()))
        self.assertGreater(m, lo)
        self.assertLess(m, hi)


class TestPipeline(unittest.TestCase):
    def test_pipeline_shapes_and_archive(self):
        f = textured_frame()
        frames = [f.copy() for _ in range(3)]
        final, exposed, merged = hdr_pipeline(
            frames, EV_PRESETS[DEFAULT_EV_PRESET], tonemap="drago")
        self.assertEqual(final.shape, f.shape)
        self.assertEqual(merged.shape, f.shape)
        self.assertEqual(len(exposed), 3)
        for a, b in zip(exposed, EV_PRESETS[DEFAULT_EV_PRESET]):
            # 每张曝光图与对应 EV 档的独立模拟一致
            self.assertTrue(np.array_equal(a, simulate_exposure(f, b)))

    def test_pipeline_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            hdr_pipeline([np.zeros((8, 8, 3), np.uint8)] * 2, (0.0, 1.0, 2.0))

    def test_tonemap_variants(self):
        f = textured_frame()
        self.assertIs(tonemap_frame(f, None), f)
        for method in ("drago", "reinhard"):
            out = tonemap_frame(f, method)
            self.assertEqual(out.shape, f.shape)
            self.assertEqual(out.dtype, np.uint8)
            # 不得退化为常量图（全黑/全白）—— tonemap 输出必须有动态范围
            self.assertGreater(float(out.std()), 5.0)
        with self.assertRaises(ValueError):
            tonemap_frame(f, "xxx")

    def test_tonemap_drago_survives_black_pixels(self):
        """回归（实拍 bug）：输入含纯黑像素（Mertens 欠曝黑区必然存在，
        实测 merged.min()==0）时 Drago 的 log(0) 产出 NaN → cast uint8
        整图全黑（720p 全黑 JPEG 恒 ~15KB，两组实拍 final 一字节不差）。
        修复 = 输入抬底 + NaN 消毒；断言输出保有动态范围。"""
        rng = np.random.default_rng(3)
        f = rng.integers(0, 255, (120, 160, 3), np.uint8)
        f[:20] = 0                      # 纯黑区
        f[:, :20] = 0
        for method in ("drago", "reinhard"):
            out = tonemap_frame(f, method)
            self.assertGreater(float(out.std()), 5.0,
                               f"{method} 输出退化为常量图")
            self.assertGreater(int(out.max()), 200)

    def test_comparison_strip_width(self):
        f = textured_frame()
        strip = build_hdr_comparison(f, f, f, (0.0,))
        self.assertEqual(strip.shape[1], f.shape[1] * 3)

    def test_ev_presets_wellformed(self):
        for name, evs in EV_PRESETS.items():
            self.assertEqual(len(evs), len(EV_PRESETS[name]))
            self.assertIn(0.0, evs)     # 必须含 0EV 基准帧


# ---------------- worker 连拍编排（回归：拍完 HDR 不得卡住帧循环） ----------------

try:
    from PySide6.QtCore import QCoreApplication, Qt
    import gui.workers as W
    HAS_PYSIDE = True
except Exception:   # noqa: BLE001 —— 无 GUI 依赖环境跳过
    HAS_PYSIDE = False


class _FakeSource:
    """图片序列假采集源（read 耗尽即结束，不依赖摄像头）。"""

    continuous = False
    name = "fake"

    def __init__(self, frames):
        self.frames = list(frames)

    def open(self) -> bool:
        return True

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self) -> None:
        pass


@unittest.skipUnless(HAS_PYSIDE, "PySide6 不可用")
class TestHdrWorkerFlow(unittest.TestCase):
    """连拍在帧循环顶部同步完成、融合/写盘在后台线程不阻塞帧循环。"""

    def setUp(self):
        _app = QCoreApplication.instance() or QCoreApplication([])
        self._photos_dir = tempfile.mkdtemp(prefix="hdr_test_")
        self._orig_photos = W.PHOTOS_DIR
        W.PHOTOS_DIR = self._photos_dir          # 不污染真实 photos/

    def tearDown(self):
        W.PHOTOS_DIR = self._orig_photos
        for f in glob.glob(os.path.join(self._photos_dir, "*")):
            os.unlink(f)
        os.rmdir(self._photos_dir)

    def test_burst_then_background_merge(self):
        from core.infer import InferenceEngine
        from core.pipeline import Pipeline

        frames = [textured_frame(320, 240, seed=i) for i in range(6)]
        worker = W.CameraWorker(
            _FakeSource(frames), Pipeline([]), InferenceEngine())
        saved: list[str] = []
        statuses: list[str] = []
        # 后台融合线程 emit 的信号默认走 QueuedConnection，需要事件循环派发；
        # 测试进程没有 app.exec()，改 DirectConnection 在 emit 线程直调槽
        # （生产 GUI 有事件循环，两种连接行为等价）
        worker.photo_saved.connect(saved.append, Qt.DirectConnection)
        worker.status_message.connect(statuses.append, Qt.DirectConnection)

        worker.request_hdr_capture(DEFAULT_EV_PRESET, None)
        t0 = time.perf_counter()
        worker._loop()          # 同步跑：3 张连拍 → 后台融合 → 排空 → 源耗尽退出
        loop_done = time.perf_counter() - t0

        # 帧循环本身只耗连拍时间（2×0.12s sleep），融合在后台：
        # 循环返回时融合线程可能仍在写盘，轮询等待完成（上限 5s）
        deadline = time.perf_counter() + 5.0
        while not saved and time.perf_counter() < deadline:
            time.sleep(0.05)
        self.assertTrue(saved, f"后台融合未产出照片；status={statuses}")
        self.assertTrue(os.path.exists(saved[0]))
        stem = os.path.basename(saved[0]).rsplit("_final", 1)[0]
        files = os.listdir(self._photos_dir)
        for suffix in ("_final.jpg", "_merged.jpg", "_compare.jpg"):
            self.assertIn(stem + suffix, files)
        self.assertEqual(len([f for f in files if "_ev" in f]), 3)
        # FPS 基准已被重置（连拍耗时不许污染恢复后的帧间隔统计）
        self.assertEqual(worker._fps, 0.0)
        # 帧循环总耗时 ≈ 连拍 sleep + 排空，不应被融合/写盘拖住
        self.assertLess(loop_done, 3.0)


if __name__ == "__main__":
    unittest.main()
