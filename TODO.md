# TODO — BeautyCam v2

> 计划与架构见 [docs/PLAN.md](docs/PLAN.md)，选型依据见 [docs/research-notes.md](docs/research-notes.md)。
> 人日为单人净工作量估算。完成一项勾一项；验收标准见 PLAN §4。

## Phase 0 — 重构底座（6–9 人日）

- [ ] **P0-1** 冒烟验证：用 `.venv`（mediapipe 1.0.1）跑一遍 `beautycam_v1.py`，确认 `mp.solutions.*` 旧 API 是否兼容；不兼容则记录差异，`core/infer.py` 改用新 Tasks API 封装（0.25d）
- [ ] **P0-2** 建目录骨架：`core/{effects}`、`gui/`、`demos/`、`scripts/`、`tests/`、`assets/`、`legacy/`（0.5d）
- [ ] **P0-3** `core/camera.py`：采集源抽象（实时相机 / 视频文件 / 图片序列），固定 1280×720、镜像、弱光检测口径统一（0.5d）
- [ ] **P0-4** `core/pipeline.py` + `core/context.py`：Effect 基类（`name/enabled/params/process`）、FrameContext（共享推理结果）、线程安全 `set_params()`（1d）
- [ ] **P0-5** `core/infer.py`：MediaPipe 单例会话；每帧统一跑 FaceMesh/Hands/FaceDetection 填充 ctx（0.5d）
- [ ] **P0-6** `core/effects/beauty.py`：迁移磨皮/美白/瘦脸/大眼，全部参数化（磨皮混合比、美白强度、瘦脸度、大眼开关+强度）（1d）
- [ ] **P0-7** `gui/main_window.py`：PySide6 主窗口——视频区 / 效果控制面板 / 拍照预览条；`theme.qss` 暗色主题（1.5d）
- [ ] **P0-8** `gui/workers.py`：QThread 相机工作线程，信号发帧（ndarray）；主线程绘制；杜绝跨线程碰控件（1d）
- [ ] **P0-9** 拍照功能迁移：手动拍照 / V 手势 / 笑脸触发（**顺带修复一期 `global` 缺失与双线程启动 bug**）、快门音效、`photos/` 预览条点击放大（1d）
- [ ] **P0-10** `scripts/run_pipeline.py`：headless CLI，对图片目录/视频批跑管线并保存输出（评测数据生产线）（0.5d）
- [ ] **P0-11** `tests/`：美颜纯函数、V 手势判定、笑脸判定的单测（静态图，不依赖摄像头）（0.5d）
- [ ] **P0-12** `requirements.txt` 增加 PySide6；`beautycam_v1.py` 移入 `legacy/`；更新 README（0.25d）

**Phase 0 验收**：仅美颜链实时预览 ≥30fps；手势/笑脸/手动拍照全部可用；headless CLI 可出图；单测通过。

## Phase 1 — 自动 HDR（2–4 人日）

- [ ] **P1-1** 确定包围曝光模拟参数：gamma 曲线组合（如 γ∈{0.5, 1.0, 2.0}）、连拍张数、连拍间隔（0.5d）
- [ ] **P1-2** `core/effects/hdr.py`：连拍采集 → `cv2.findTransformECC` 帧间对齐 → `cv2.createMergeMertens` 融合（1d）
- [ ] **P1-3** GUI「HDR 拍照」模式：倒计时连拍 → 融合进度提示 → 成片保存（成片 + 各曝光原图一并存档便于对比）（1d）
- [ ] **P1-4** 显示用色调映射选项（Drago / Reinhard），预览不刺眼（0.5d）
- [ ] **P1-5**（进阶，可选）单帧 HDR：HDRUNet 或 Deep-HdrReconstruction 导 ONNX，作"经典 vs 深度"对比线（+2d）

**Phase 1 验收**：逆光场景成片高光不过曝、暗部有细节；每组照片有单张 vs 融合对照。

## Phase 2 — 低光增强（2–3 人日）

- [ ] **P2-1** 模型定版：SCI vs Zero-DCE++，按 ONNX 权重可得性 + 本机实测速度二选一（0.5d）
- [ ] **P2-2** `scripts/download_models.py` + `core/infer.py` ONNX 会话管理：CoreML EP 优先、CPU 回退，**打印实际 provider**（0.5d）
- [ ] **P2-3** `core/effects/lowlight.py`：≤480p 推理 + 结果上采样，强度参数映射到模型输入缩放（1d）
- [ ] **P2-4** 性能与自动化：隔帧推理 + 掩膜/亮度时域平滑；亮度阈值自动开关（替换一期"均值<60"启发式或与其协同）（1d）

**Phase 2 验收**：暗光环境预览 ≥20fps；与旧启发式增强有可视对比图。

## Phase 3 — 人像虚化 / 背景替换（3–5 人日）

- [ ] **P3-1** `core/effects/segment.py`：MediaPipe Selfie Segmentation 人像掩膜 + EMA 时域平滑 + 边缘羽化（1d）
- [ ] **P3-2** 基础虚化：背景高斯模糊，模糊半径 = 强度滑杆（0.5d）
- [ ] **P3-3** 背景替换：内置 `assets/backgrounds/` 图库 + 用户自选图片，色彩轻微协调（1d）
- [ ] **P3-4**（进阶）Depth Anything V2 small ONNX：深度图渐进虚化（近清远糊）（+1.5d）
- [ ] **P3-5** GUI：模式选择（无 / 虚化 / 换背景）+ 强度滑杆 + 背景选择器（0.5d）

**Phase 3 验收**：掩膜边缘无明显抖动/镶边；虚化+美颜+低光三开 ≥15fps。

## Phase 4 — 换脸·演示级（3–5 人日）

- [ ] **P4-1** `demos/faceswap/` 离线版：源脸/目标脸 FaceMesh → Delaunay 三角剖分 → 分块仿射变形 → `cv2.seamlessClone` 泊松融合 → Reinhard 色彩迁移（2d）
- [ ] **P4-2** 过程可视化：三角剖分绘制动画、变形蒙版、融合前后中间产物展示（课堂讲解主素材）（1d）
- [ ] **P4-3**（可选）实时演示模式：摄像头人脸 + 模板脸（1d）
- [ ] **P4-4** 伦理约束落地：UI 提示文案 + README/文档声明（仅本人/授权/动漫形象）（0.25d）

**Phase 4 验收**：能完整展示算法各阶段中间产物；换脸效果达"可辨识"级别。

## Phase 5 — GPU 加速、评测与答辩（3–5 人日）

- [ ] **P5-1** CoreML EP 全模型启用与验证（逐会话确认 provider，防静默回退）；如引入 PyTorch 模型则验证 MPS（1d）
- [ ] **P5-2** 性能基准：各功能单开/组合开 FPS 表、单帧各阶段耗时分解、CPU vs CoreML 对比（1d）
- [ ] **P5-3** 客观评测：低光增强 PSNR/SSIM（LOL 子集或自摄成对数据）；HDR 成片主观评分收集（1d）
- [ ] **P5-4** 主观评测：Likert 问卷（自然度/边缘质量/整体满意度），延续一期双盲方法学（0.5d）
- [ ] **P5-5** 答辩材料：对比图集、架构图、演示脚本（含故障预案：备用视频/截图）、PPT（1.5d）

**Phase 5 验收**：性能对比表 + ≥2 组量化实验数据 + 完整演示走查一次通过。

## 里程碑

| 里程碑 | 内容 | 依赖 |
|--------|------|------|
| M1 | 新架构 + PySide6 GUI 可日常使用（P0 完成） | — |
| M2 | HDR + 低光上线，核心拍照功能齐（P1–P2） | M1 |
| M3 | 虚化/换背景上线，效果全量（P3） | M1 |
| M4 | 换脸演示 + 全部评测数据 + 答辩材料（P4–P5） | M2、M3 |
