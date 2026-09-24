"""换脸几何纯函数单测（三角剖分 / 分块仿射 / Reinhard 迁移，不依赖模型）。"""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from demos.faceswap.faceswap import (
    delaunay_triangles, draw_wireframe, face_oval_poly, faceswap,
    reinhard_color_transfer, run, side_by_side, warp_face,
)


def grid_points(w=200, h=150, n=5) -> np.ndarray:
    """n×n 均匀网格点（确定性、无共线退化）。"""
    xs = np.linspace(10, w - 10, n)
    ys = np.linspace(10, h - 10, n)
    return np.array([[x, y] for y in ys for x in xs], np.float64)


class TestDelaunay(unittest.TestCase):
    def test_triangle_count_and_coverage(self):
        """n×n 网格剖分：三角数 = 2·(n-1)²，且所有三角顶点索引合法。"""
        w, h, n = 200, 150, 5
        pts = grid_points(w, h, n)
        tris = delaunay_triangles(pts, (h, w))
        self.assertEqual(len(tris), 2 * (n - 1) ** 2)
        for i, j, k in tris:
            self.assertTrue(all(0 <= t < len(pts) for t in (i, j, k)))
            self.assertNotEqual(len({i, j, k}), 1)

    def test_boundary_points_clamped(self):
        """贴边点不抛异常（Subdiv2D 要求点严格在矩形内）。"""
        pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float64)
        tris = delaunay_triangles(pts, (100, 100))
        self.assertEqual(len(tris), 2)


class TestWarp(unittest.TestCase):
    def test_translation_recovered(self):
        """源点整体平移到目标点：warp 后内容应跟随平移。"""
        h, w = 120, 160
        src = np.zeros((h, w, 3), np.uint8)
        src[40:60, 40:100] = (0, 0, 255)     # 红条
        pts = grid_points(w, h, 5)
        offset = np.array([10.0, 6.0])
        tris = delaunay_triangles(pts, (h, w))
        warped, mask = warp_face(src, pts, pts + offset, tris, (h, w))
        # 红条整体移动 10,6：中心从 (70,50) 到 (80,56)
        m = warped[:, :, 2] > 200
        self.assertGreater(int(m.sum()), 0)
        cy, cx = np.nonzero(m)[0].mean(), np.nonzero(m)[1].mean()
        self.assertAlmostEqual(cx, 80.0, delta=2.0)
        self.assertAlmostEqual(cy, 56.0, delta=2.0)
        self.assertGreater(cv2.countNonZero(mask), 0)


class TestColorTransfer(unittest.TestCase):
    def test_mean_moves_toward_reference(self):
        """色彩迁移后 patch 的 LAB 均值应靠近 reference。"""
        patch = np.full((40, 40, 3), (200, 120, 80), np.uint8)
        ref = np.full((40, 40, 3), (60, 180, 150), np.uint8)
        out = reinhard_color_transfer(patch, ref)
        lab_p = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).mean(axis=(0, 1))
        lab_o = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).mean(axis=(0, 1))
        lab_r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).mean(axis=(0, 1))
        for c in range(3):
            self.assertLess(abs(lab_o[c] - lab_r[c]), abs(lab_p[c] - lab_r[c]))

    def test_mask_ignores_black_pixels_outside_face(self):
        patch = np.zeros((40, 40, 3), np.uint8)
        patch[10:30, 10:30] = (200, 120, 80)
        ref = np.full((40, 40, 3), (60, 180, 150), np.uint8)
        mask = np.zeros((40, 40), np.uint8)
        mask[10:30, 10:30] = 255
        out = reinhard_color_transfer(patch, ref, mask)
        self.assertTrue(np.array_equal(out[0, 0], patch[0, 0]))
        self.assertLess(np.abs(out[20, 20].astype(int) - ref[20, 20]).mean(),
                        np.abs(patch[20, 20].astype(int) - ref[20, 20]).mean())


class TestFaceswapEndToEnd(unittest.TestCase):
    def _synth_mesh(self, cx, cy, r, n=468):
        """合成 468 点脸网（归一化坐标，与 FaceMesh 口径一致）。

        前 64 个点在"轮廓"半径上（cx,cy,r 均为归一化值），其余收进
        0.6 倍半径（内部点）。"""
        ts = np.linspace(0, 2 * np.pi, n, endpoint=False)
        lm = np.zeros((n, 3), np.float32)
        for i, t in enumerate(ts):
            rr = r if i < 64 else r * 0.6     # 轮廓点 + 内部点
            lm[i, :2] = cx + rr * np.cos(t), cy + rr * 0.9 * np.sin(t)
        return lm

    def test_output_shape_and_content_change(self):
        """端到端（合成关键点）：输出尺寸不变、克隆区域内像素确实改变。"""
        h, w = 200, 160
        dst = np.zeros((h, w, 3), np.uint8)
        dst[:] = (40, 80, 160)
        # src 必须有梯度：seamlessClone 是梯度域方法，纯色 src 的梯度
        # 恒为 0，克隆解会直接退化为 dst（泊松方程边界条件支配）
        src = np.zeros((h, w, 3), np.uint8)
        grad = np.linspace(30, 230, w, dtype=np.uint8)
        src[:] = grad[None, :, None]
        src_lm = self._synth_mesh(0.5, 0.5, 0.3)
        dst_lm = self._synth_mesh(0.5, 0.5, 0.34)
        out = faceswap(src, src_lm, dst, dst_lm, color_transfer=False)
        self.assertEqual(out.shape, dst.shape)
        oval = face_oval_poly(dst_lm, (h, w)).astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [oval], 255)
        region = mask > 0
        self.assertGreater(int(region.sum()), 100)
        # 泊松克隆保留 src 梯度、电平被 dst 边界色锚定：均值差不像直接
        # 贴图那么大（合成场景 dst 为纯色、锚定效应强），>2 即证明克隆
        # 区域内容确实来自 src（真实人脸纹理差异远大于此）
        self.assertGreater(float(np.abs(
            out[region].astype(int) - dst[region].astype(int)).mean()), 2.0)

    def test_wireframe_draw_runs(self):
        img = np.zeros((100, 100, 3), np.uint8)
        lm = self._synth_mesh(0.5, 0.5, 0.3)
        tris = delaunay_triangles(lm[:, :2] * np.array([100, 100]), (100, 100))
        out = draw_wireframe(img, lm, tris)
        self.assertGreater(int(out.sum()), 0)

    def test_different_image_sizes_save_all_stages(self):
        """源/目标图高度不同时，阶段并排图仍能生成。"""
        src = np.full((120, 90, 3), 100, np.uint8)
        dst = np.full((200, 160, 3), 160, np.uint8)
        self.assertEqual(side_by_side(src, dst).shape[0], dst.shape[0])
        src_lm = self._synth_mesh(0.5, 0.5, 0.3)
        dst_lm = self._synth_mesh(0.5, 0.5, 0.3)
        with tempfile.TemporaryDirectory() as tmp:
            src_path = Path(tmp) / "源图.png"
            dst_path = Path(tmp) / "目标图.png"
            src_path.write_bytes(cv2.imencode(".png", src)[1].tobytes())
            dst_path.write_bytes(cv2.imencode(".png", dst)[1].tobytes())
            with patch("demos.faceswap.faceswap.detect_landmarks",
                       side_effect=[src_lm, dst_lm]):
                result = run(str(src_path), str(dst_path), str(Path(tmp) / "out"))
            self.assertIsNotNone(result)
            for name in ("stage1_landmarks.jpg", "stage2_delaunay.jpg",
                         "stage3_warp.jpg", "stage4_clone.jpg",
                         "stage5_result.jpg", "compare.jpg"):
                self.assertTrue((Path(tmp) / "out" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
