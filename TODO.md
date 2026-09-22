# TODO — BeautyCam v2

> 计划与架构见 [docs/PLAN.md](docs/PLAN.md)，选型依据见 [docs/research-notes.md](docs/research-notes.md)。
> 人日为单人净工作量估算。完成一项勾一项；验收标准见 PLAN §4。

## Phase 0 — 重构底座（6–9 人日）

- [x] **P0-1** 冒烟验证：mediapipe **1.0.1 不可用**（移除 `mp.solutions` + Tasks 检测类图 Metal 崩溃），锁定 **0.10.21** + Tasks API + 显式 CPU 委托；FaceDetector 弃用（人脸框由 FaceMesh 导出）。结论固化在 `core/infer.py` 模块头与 README（0.25d）
- [x] **P0-2** 建目录骨架：`core/{effects}`、`gui/`、`demos/`、`scripts/`、`tests/`、`assets/`、`legacy/`（0.5d）
- [x] **P0-3** `core/camera.py`：采集源抽象（实时相机 / 视频文件 / 图片序列），固定 1280×720、镜像、弱光检测口径统一（0.5d）
- [x] **P0-4** `core/pipeline.py` + `core/context.py`：Effect 基类（`name/enabled/params/process`）、FrameContext（共享推理结果）、线程安全 `set_params()`（整字典替换快照）（1d）
- [x] **P0-5** `core/infer.py`：Tasks API 单例会话；每帧统一跑 FaceMesh(+blendshapes)/Hands/分割填充 ctx（0.5d）
- [x] **P0-6** `core/effects/beauty.py`：磨皮/美白/瘦脸/大眼全部参数化；瘦脸改真 liquify 变形、大眼 remap 向量化（1d）
- [x] **P0-7** `gui/main_window.py` + `panels.py`：PySide6 主窗口——视频区 / 效果控制面板 / 拍照预览条；`theme.qss` 暗色主题（1.5d）
- [x] **P0-8** `gui/workers.py`：QThread 相机工作线程，信号发帧；杜绝跨线程碰控件（1d）
- [x] **P0-9** 拍照功能迁移：手动 / V 手势（时间持续判定替代帧计数）/ 笑脸（blendshapes 置信度 + 回退口径）、快门音效、预览条点击放大；**一期 `global` 缺失与双线程 bug 已根治**（待摄像头实机联测确认）（1d）
- [x] **P0-10** `scripts/run_pipeline.py`：headless CLI（图片目录/视频批跑，真图实测通过）+ `scripts/download_models.py`（0.5d）
- [x] **P0-11** `tests/`：44 个单测全绿——美颜纯函数 / V 手势 / 笑脸 / 自动拍照状态机 / 管线顺序与并发参数 / 采集源 / 推理引擎（0.5d）
- [x] **P0-12** `requirements.txt` 更新（锁 mediapipe 0.10.21 + PySide6）；`beautycam_v1.py` 移入 `legacy/`；README 重写（0.25d）

**Phase 0 验收**：仅美颜链实测 ~55fps（M4/720p，含推理）≥30fps ✓；headless CLI 可出图 ✓；单测 44/44 ✓；手势/笑脸/手动拍照需用户实机运行确认（GUI + 摄像头无法无头验证）。

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

- [x] **P3-0** 分割模型冒烟基准（决策门）：`selfie_multiclass_256x256`（16.4MB）实测 **155ms/帧**且与输入分辨率无关 → 直接接上去只有 ~7fps，不达标；官方二元 `selfie_segmenter`（250KB）实测 **13.2ms/帧**，快 11.7 倍。**定版默认二元模型**，多分类保留可切换作为"质量 vs 速度"对比线（0.5d）
- [x] **P3-1** `core/effects/segment.py`：人像掩膜 + 运动自适应 EMA 时域平滑（静止重平滑防抖、运动减轻防拖影）+ guided filter 边缘精修（自实现 He et al. 2010，与 `cv2.ximgproc` 数值一致到 1e-5，**不依赖 contrib**，macOS 可跑）（1d）
- [x] **P3-2** 基础虚化：背景高斯模糊，强度滑杆映射 sigma；大 sigma 走"降采样→模糊→升采样"加速（0.5d）
- [x] **P3-3** 背景替换：内置 `assets/backgrounds/` 图库（10 张，`scripts/make_backgrounds.py` 程序化生成，无版权风险）+ 用户自选图片（0.5d）
- [ ] **P3-4**（进阶）Depth Anything V2 small ONNX：深度图渐进虚化（近清远糊）（+1.5d）
- [x] **P3-5** GUI：`SegmentPanel`（模式下拉 + 图库缩略图选择器 + 自选图片 + 纯色预设 + 强度/matte 对比度/羽化/时域平滑滑杆 + 边缘精修开关 + 分割模型下拉）（0.5d）

**Phase 3 验收**：掩膜边缘无明显抖动/镶边 ✓（实测掩膜边界与图像边界偏移 9.6px 的固有模糊经精修后最大梯度提升约 4 倍；绿幕合成无镶边）；虚化+美颜+低光三开 **13.8fps**（本机 Windows x86，基线——同机同推理但不开虚拟背景——仅 26.3fps；PLAN 的 ≥15fps 是 M4 Mac 上的口径，本机约慢一倍，差距来自机器而非本模块）。

**P3-0 两个关键结论**（详见 `core/infer.py` 模块头）：
1. **两个分割模型的类别编码与置信图极性恰好相反**（二元 `cat==0` 是人、`conf[0]` 是前景概率；多分类 `cat!=0` 是人、`conf[0]` 是背景概率）。搞反不抛异常、只静默产出反转掩膜，故按模型显式声明在 `SEGMENTER_SPECS`，并用**与假设无关的探针法**（图像四角必为背景）+ 运行时边框先验自检双重把关。
2. **多分类模型背景置信度有系统偏置**（背景区 alpha 恒为 ~0.084），直接当 alpha 会给新背景叠一层均匀 8.4% 鬼影（背景发灰）；对比度拉伸 `matte_contrast` 可归零。二元模型无此问题。

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
| M3 | 虚化/换背景上线，效果全量（P3） | M1 | ~~已完成~~ |
| M4 | 换脸演示 + 全部评测数据 + 答辩材料（P4–P5） | M2、M3 |
