"""实时换脸效果插件：授权门、源脸缓存与管线复用。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from core.context import FaceInfo, FrameContext
from core.pipeline import NEED_FACES
from demos.faceswap.effect import FaceSwapEffect, list_face_presets


def _face() -> FaceInfo:
    lm = np.zeros((468, 3), np.float32)
    lm[:, :2] = 0.5
    return FaceInfo(landmarks=lm, box=(0.2, 0.2, 0.8, 0.8))


class TestFaceSwapEffect(unittest.TestCase):
    def test_builtin_face_library_has_detectable_assets(self):
        paths = list_face_presets()
        self.assertGreaterEqual(len(paths), 4)
        self.assertTrue(all(path.lower().endswith(".png") for path in paths))

    def test_requires_consent_and_source(self):
        effect = FaceSwapEffect(enabled=True)
        frame = np.full((40, 60, 3), 80, np.uint8)
        ctx = FrameContext(width=60, height=40, faces=[_face()])
        self.assertTrue(np.array_equal(effect.process(frame, ctx), frame))
        self.assertIn("授权", effect.runtime_status)
        self.assertEqual(effect.needs, frozenset({NEED_FACES}))

    @patch("demos.faceswap.effect.faceswap")
    @patch("demos.faceswap.effect.delaunay_triangles", return_value=[(0, 1, 2)])
    @patch("demos.faceswap.effect.load_image")
    @patch("demos.faceswap.effect.get_engine")
    def test_source_detected_once_and_triangles_reused(
            self, get_engine, load_image, delaunay, swap):
        source = np.full((80, 60, 3), 120, np.uint8)
        load_image.return_value = source
        get_engine.return_value.process.return_value = SimpleNamespace(faces=[_face()])
        swap.side_effect = lambda src, src_lm, dst, dst_lm, **kw: dst + 1
        effect = FaceSwapEffect(enabled=True, params={
            "source_path": "源脸.jpg", "consent": True,
        })
        frame = np.zeros((40, 60, 3), np.uint8)
        ctx = FrameContext(width=60, height=40, faces=[_face()])

        out1 = effect.process(frame, ctx)
        out2 = effect.process(frame, ctx)
        full = np.zeros((720, 1280, 3), np.uint8)
        effect.process(full, ctx)

        self.assertEqual(int(out1.mean()), 1)
        self.assertEqual(int(out2.mean()), 1)
        load_image.assert_called_once_with("源脸.jpg")
        get_engine.return_value.process.assert_called_once()
        self.assertEqual(delaunay.call_count, 2)
        self.assertEqual(swap.call_count, 3)
        self.assertEqual(swap.call_args.kwargs["triangles"], [(0, 1, 2)])
        self.assertEqual(swap.call_args_list[0].kwargs["triangles"],
                         effect._preview_triangles)

    def test_no_target_face_keeps_frame(self):
        effect = FaceSwapEffect(enabled=True, params={
            "source_path": "源脸.jpg", "consent": True,
        })
        effect._cached_path = "源脸.jpg"
        effect._source = np.zeros((20, 20, 3), np.uint8)
        effect._source_lm = _face().landmarks
        effect._triangles = [(0, 1, 2)]
        frame = np.zeros((40, 60, 3), np.uint8)
        out = effect.process(frame, FrameContext(width=60, height=40))
        self.assertTrue(np.array_equal(out, frame))
        self.assertIn("未检测到人脸", effect.runtime_status)


if __name__ == "__main__":
    unittest.main()
