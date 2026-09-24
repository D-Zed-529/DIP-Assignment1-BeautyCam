"""换脸演示（Phase 4，演示级）——经典 DIP 流水线，主打过程可视化。

算法链（每阶段产物都存图，课堂讲解素材）：
  ① FaceMesh 468 点（源脸 / 目标脸，复用 core/infer.py 的引擎单例）
  ② Delaunay 三角剖分（Subdiv2D，源脸点集）
  ③ 分块仿射变形：逐三角 getAffineTransform + bbox 子图 warpAffine
  ④ 泊松融合：cv2.seamlessClone（NORMAL_CLONE，梯度域消接缝）
  ⑤ Reinhard 色彩迁移：LAB 均值/方差对齐目标脸（肤色一致化）

定位（PLAN §1）：演示级——"可辨识换脸"即合格，不追求以假乱真；
主打三角剖分/变形/融合各阶段中间产物的可视化。

⚠️ 伦理约束（P4-4）：仅可用于本人面部、明确授权者或动漫形象；演示
数据须获得被摄者同意。CLI 需 --consent 显式确认，README 同步声明。

用法：
  python -m demos.faceswap.faceswap --src a.jpg --dst b.jpg \
      --out outputs/faceswap --consent
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.effects.segment import load_image          # noqa: E402
from core.infer import FACE_OVAL_IDS, get_engine     # noqa: E402

# ------- 调参常量 -------
FEATHER_PX = 7          # 换脸蒙版羽化半径（吃掉克隆硬边）
CLONE_MARGIN = 0.02     # 脸框外扩比例（seamlessClone 蒙版比 oval 稍大更稳）


# ---------------- 纯函数（可单测） ----------------

def delaunay_triangles(points: np.ndarray, size: tuple[int, int]
                       ) -> list[tuple[int, int, int]]:
    """点集 Delaunay 三角剖分 → 索引三元组列表。

    Subdiv2D 要求点严格落在 (0,0,w,h) 内（边界点会抛错，先夹紧）；
    返回的三角顶点为**原始点集索引**（getTriangleList 给坐标，需映射回）。
    """
    h, w = size
    pts = np.asarray(points, np.float64)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, w - 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, h - 1.0)
    # 坐标 → 索引查找表（round 到 0.01px 精度即可；剖分返回的是原坐标）
    lookup = {(round(float(x), 2), round(float(y), 2)): i
              for i, (x, y) in enumerate(pts)}
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for x, y in pts:
        subdiv.insert((float(x), float(y)))
    tri_list = np.array(subdiv.getTriangleList(), np.float64).reshape(-1, 3, 2)
    tris: list[tuple[int, int, int]] = []
    for a, b, c in tri_list:
        key = [(round(float(p[0]), 2), round(float(p[1]), 2)) for p in (a, b, c)]
        if all(k in lookup for k in key):
            tris.append(tuple(lookup[k] for k in key))
    return tris


def warp_face(src: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray,
              tris: list[tuple[int, int, int]],
              canvas_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """分块仿射变形：源脸按三角网 warp 到目标点位置。

    返回 (warped, mask)：warped 为画布尺寸的变形结果，mask 为已覆盖
    区域（uint8 0/255）。逐三角只 warp 其目标 bbox 子图（900+ 个三角
    整图 warp 不可行）。
    """
    H, W = canvas_shape
    warped = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), np.uint8)
    for i, j, k in tris:
        src_tri = src_pts[[i, j, k]].astype(np.float32)
        dst_tri = dst_pts[[i, j, k]].astype(np.float32)
        M = cv2.getAffineTransform(src_tri, dst_tri)
        # 目标三角 bbox 子图上 warp + 三角掩膜（反走样由羽化蒙版兜底）。
        # ⚠️ warpAffine 的输出窗口原点恒为 (0,0)：把仿射平移分量减去
        # bbox 原点，让目标三角落到 patch 的 (bx,by) 相对坐标上。
        # bbox 可能越出画布（边缘三角）：patch/mask 按原 bbox 生成，写入
        # 画布前裁到相交区，否则切片尺寸不一致直接 IndexError。
        bx, by, bw, bh = cv2.boundingRect(dst_tri.astype(np.int32))
        if bw == 0 or bh == 0:
            continue
        x0, y0 = max(bx, 0), max(by, 0)
        x1, y1 = min(bx + bw, W), min(by + bh, H)
        if x1 <= x0 or y1 <= y0:
            continue
        M_local = M.copy()
        M_local[0, 2] -= bx
        M_local[1, 2] -= by
        patch = cv2.warpAffine(src, M_local, (bw, bh))
        tri_mask = np.zeros((bh, bw), np.uint8)
        cv2.fillConvexPoly(tri_mask,
                           (dst_tri - [bx, by]).astype(np.int32), 255)
        sub = (slice(y0 - by, y1 - by), slice(x0 - bx, x1 - bx))
        sub_mask = tri_mask[sub]
        warped[y0:y1, x0:x1][sub_mask > 0] = patch[sub][sub_mask > 0]
        mask[y0:y1, x0:x1][sub_mask > 0] = 255
    return warped, mask


def face_oval_poly(landmarks: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """FaceMesh 轮廓关键点 → 脸部椭圆多边形（像素坐标，顺序即轮廓序）。"""
    h, w = size
    pts = landmarks[FACE_OVAL_IDS][:, :2] * np.array([w, h])
    return pts.astype(np.float32)


def reinhard_color_transfer(patch: np.ndarray, reference: np.ndarray,
                            mask: np.ndarray | None = None
                            ) -> np.ndarray:
    """Reinhard 色彩迁移（LAB 统计对齐）：patch 的色彩分布向 reference 靠拢。

    (x − μ_p) · σ_r / σ_p + μ_r，逐通道。σ 比值夹在 [1/3, 3] 防极端
    方差（纯色补丁 σ≈0）把噪声放大。指定 mask 时只统计和修改脸部，
    避免三角网边缘的黑色空白把肤色均值拉低。
    """
    if mask is not None and mask.shape != patch.shape[:2]:
        raise ValueError("色彩迁移掩膜尺寸必须与图像一致")
    lab_p = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    valid = np.ones(patch.shape[:2], bool) if mask is None else mask > 0
    if not np.any(valid):
        return patch.copy()
    out = lab_p.copy()
    for c in range(3):
        mp, mr = float(lab_p[:, :, c][valid].mean()), float(lab_r[:, :, c][valid].mean())
        sp = max(float(lab_p[:, :, c][valid].std()), 1e-3)
        sr = max(float(lab_r[:, :, c][valid].std()), 1e-3)
        ratio = float(np.clip(sr / sp, 1.0 / 3.0, 3.0))
        channel = out[:, :, c]
        channel[valid] = (lab_p[:, :, c][valid] - mp) * ratio + mr
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8),
                        cv2.COLOR_LAB2BGR)


def faceswap(src: np.ndarray, src_lm: np.ndarray,
             dst: np.ndarray, dst_lm: np.ndarray,
             color_transfer: bool = True,
             triangles: list[tuple[int, int, int]] | None = None) -> np.ndarray:
    """端到端换脸（已有关键点）：剖分 → 分块仿射 → 色彩迁移 → 泊松融合。"""
    h, w = dst.shape[:2]
    src_pts = src_lm[:, :2] * np.array([src.shape[1], src.shape[0]])
    dst_pts = dst_lm[:, :2] * np.array([w, h])
    # 实时相机逐帧复用源脸的三角拓扑，避免每帧重复做 Delaunay 剖分。
    tris = (triangles if triangles is not None else
            delaunay_triangles(src_pts, (src.shape[0], src.shape[1])))
    warped, warp_mask = warp_face(src, src_pts, dst_pts, tris, (h, w))

    # 只在目标脸 oval 区域内取变形结果（剖分凸包略大于 oval，夹一夹）
    oval = face_oval_poly(dst_lm, (h, w))
    oval_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(oval_mask, [oval.astype(np.int32)], 255)
    region = cv2.bitwise_and(warp_mask, oval_mask)
    if color_transfer and cv2.countNonZero(region) > 32:
        ys, xs = np.nonzero(region)
        patch = warped[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        ref = dst[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        face_mask = region[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        colorized = reinhard_color_transfer(patch, ref, face_mask)
        warped[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = colorized

    # seamlessClone：梯度域融合，蒙版羽化 + 略外扩（融合区要有真像素梯度）
    m = cv2.erode(region, np.ones((3, 3), np.uint8))
    m = cv2.GaussianBlur(m, (FEATHER_PX * 2 + 1, FEATHER_PX * 2 + 1), 0)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return dst
    cy, cx = int(ys.mean()), int(xs.mean())
    try:
        return cv2.seamlessClone(warped, dst, m, (cx, cy), cv2.NORMAL_CLONE)
    except cv2.error:
        return dst


# ---------------- 过程可视化 ----------------

def draw_landmarks(img: np.ndarray, lm: np.ndarray, color=(0, 255, 255)
                   ) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    pts = (lm[:, :2] * np.array([w, h])).astype(np.int32)
    for p in pts:
        cv2.circle(out, tuple(p), 1, color, -1, cv2.LINE_AA)
    return out


def draw_wireframe(img: np.ndarray, lm: np.ndarray,
                   tris: list[tuple[int, int, int]],
                   color=(120, 220, 255)) -> np.ndarray:
    """三角剖分线框（课堂讲解主素材：能看到"分块仿射"的分块）。"""
    out = img.copy()
    h, w = out.shape[:2]
    pts = (lm[:, :2] * np.array([w, h])).astype(np.int32)
    for i, j, k in tris:
        tri = pts[[i, j, k]]
        cv2.polylines(out, [tri], True, color, 1, cv2.LINE_AA)
    return out


def detect_landmarks(img_bgr: np.ndarray) -> np.ndarray | None:
    """FaceMesh 单脸检测（复用全局引擎；返回 468×3 归一化关键点）。"""
    ctx = get_engine().process(img_bgr, faces=True, hands=False,
                               segmentation=False)
    return ctx.faces[0].landmarks if ctx.faces else None


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """保持宽高比，将不同尺寸的阶段图并排（以目标图高度为准）。"""
    target_h = right.shape[0]
    new_w = max(1, round(left.shape[1] * target_h / left.shape[0]))
    interp = cv2.INTER_AREA if target_h < left.shape[0] else cv2.INTER_LINEAR
    resized = cv2.resize(left, (new_w, target_h), interpolation=interp)
    return np.hstack([resized, right])


def run(src_path: str, dst_path: str, out_dir: str) -> str | None:
    """跑完整流程并存各阶段产物。返回结果图路径（检测不到脸返回 None）。"""
    src, dst = load_image(src_path), load_image(dst_path)
    if src is None or dst is None:
        print("源图/目标图读取失败", file=sys.stderr)
        return None
    src_lm, dst_lm = detect_landmarks(src), detect_landmarks(dst)
    if src_lm is None or dst_lm is None:
        print("源图或目标图未检测到人脸", file=sys.stderr)
        return None

    os.makedirs(out_dir, exist_ok=True)

    def save(name: str, img: np.ndarray):
        p = os.path.join(out_dir, name)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            Path(p).write_bytes(buf.tobytes())

    h, w = dst.shape[:2]
    src_pts = src_lm[:, :2] * np.array([src.shape[1], src.shape[0]])
    dst_pts = dst_lm[:, :2] * np.array([w, h])
    tris = delaunay_triangles(src_pts, (src.shape[0], src.shape[1]))

    save("stage1_landmarks.jpg", side_by_side(
        draw_landmarks(src, src_lm), draw_landmarks(dst, dst_lm)))
    save("stage2_delaunay.jpg", side_by_side(
        draw_wireframe(src, src_lm, tris), draw_wireframe(dst, dst_lm, tris)))
    warped, _ = warp_face(src, src_pts, dst_pts, tris, (h, w))
    save("stage3_warp.jpg", warped)

    no_color = faceswap(src, src_lm, dst, dst_lm, color_transfer=False)
    save("stage4_clone.jpg", no_color)
    result = faceswap(src, src_lm, dst, dst_lm, color_transfer=True)
    save("stage5_result.jpg", result)
    save("compare.jpg", np.hstack([dst, no_color, result]))
    print(f"完成。阶段产物与结果已存 {out_dir}")
    return os.path.join(out_dir, "stage5_result.jpg")


CONSENT_TEXT = (
    "换脸功能仅可用于：本人面部 / 明确授权者 / 动漫形象。\n"
    "使用他人照片须已获得其同意。继续请加 --consent 参数确认。")


def main() -> int:
    ap = argparse.ArgumentParser(description="演示级换脸（Delaunay+仿射+泊松）")
    ap.add_argument("--src", required=True, help="源脸图片（提供五官者）")
    ap.add_argument("--dst", required=True, help="目标图片（被换脸者）")
    ap.add_argument("--out", default="outputs/faceswap", help="输出目录")
    ap.add_argument("--consent", action="store_true",
                    help="确认已获得双方授权（伦理约束，见 README）")
    args = ap.parse_args()
    if not args.consent:
        print(CONSENT_TEXT, file=sys.stderr)
        return 2
    return 0 if run(args.src, args.dst, args.out) else 1


if __name__ == "__main__":
    sys.exit(main())
