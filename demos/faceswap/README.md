# 换脸演示（Phase 4，演示级）

经典 DIP 流水线：FaceMesh 关键点 → Delaunay 三角剖分 → 分块仿射变形 →
泊松融合（seamlessClone）→ Reinhard 色彩迁移。**主打过程可视化**，每个
阶段的中间产物都存图，用于课堂讲解；不追求以假乱真（"可辨识"即达标）。

## 用法

```bash
python -m demos.faceswap.faceswap --src 源脸.jpg --dst 目标.jpg \
    --out outputs/faceswap --consent
```

输出（out 目录）：

| 文件 | 阶段 |
|---|---|
| stage1_landmarks.jpg | ① FaceMesh 468 关键点标注（源/目标并排） |
| stage2_delaunay.jpg | ② 三角剖分线框（分块仿射的"分块"） |
| stage3_warp.jpg | ③ 分块仿射变形结果（源脸已摆到目标位置） |
| stage4_clone.jpg | ④ 泊松融合（无色彩迁移） |
| stage5_result.jpg | ⑤ + Reinhard 色彩迁移（最终结果） |
| compare.jpg | 目标原图 / ④ / ⑤ 三联对照 |

## ⚠️ 伦理约束（P4-4，与 UI 同口径）

本功能**仅可用于**：

1. 本人面部；
2. 明确授权被使用面部者；
3. 动漫 / 绘画形象。

使用他人照片须事先获得其书面同意；演示数据集的采集已获得被摄者同意。
CLI 以 `--consent` 参数强制显式确认（缺省拒绝运行），答辩演示将展示
该约束的存在。
