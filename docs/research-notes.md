# 调研笔记：热门 / 前沿数字图像处理开源项目（2025–2026）

> 目的：为 BeautyCam 二期课堂项目选型。筛选标准：① 有开源仓库且活跃；② 与"相机类"应用契合；③ 笔记本电脑可跑（实时预览）或适合拍照后处理；④ 有讲解点（经典算法 ↔ 深度学习对比）。

## 一、热门开源项目（工程级，可直接用）

| 项目 | 领域 | 与本项目的关系 |
|------|------|----------------|
| [Kornia](https://github.com/kornia/kornia) | 可微计算机视觉库（PyTorch） | 把经典 DIP 算子写成可微分模块，讲"传统算法如何接入深度学习"的最佳素材 |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | 图像/视频超分 | 拍照保存后自动超分增强，星标极高的实用工具 |
| [GFPGAN](https://github.com/TencentARC/GFPGAN) / [CodeFormer](https://github.com/sczhou/CodeFormer) | 盲人脸修复 | 与美颜相机天然契合：拍照后人脸清晰化（GFPGAN 更快，CodeFormer 身份保持更好） |
| [IOPaint](https://github.com/Sanster/IOPaint) | 一体化图像修复 GUI（原 Lama Cleaner） | 集成 GFPGAN / Real-ESRGAN / 各家扩散修复模型的本地 Web 工具，交互设计可借鉴 |
| [rembg](https://github.com/danielgatis/rembg) | 背景移除（U2Net / ISNet / BiRefNet 后端） | 人像抠图、证件照换底色功能 |
| [SAM 2](https://github.com/facebookresearch/sam2) + [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | 可提示分割 + 单目深度 | 人像虚化（bokeh）、背景替换；SAM2 支持视频流追踪 |
| [Restormer](https://github.com/swz30/Restormer) | 高效 Transformer 复原（去雨/去噪/去模糊） | 论文级复原 baseline，适合复现对比 |
| [NAFNet](https://github.com/megvii-research/NAFNet) | 简单高效的复原网络（ECCV 2022） | 结构极简，适合课堂讲解"为什么它比复杂网络好" |
| OpenCV `cv2.createMergeMertens()` | 曝光融合（经典 HDR） | **零依赖、实时友好的自动 HDR 路线**（不需要曝光时间元数据） |

## 二、前沿论文（均有官方开源代码）

### 低光增强
- **Retinexformer**（ICCV 2023）：一阶段 Retinex Transformer，1400+ 引用；仓库同时是支持 15+ 基准、4000×6000 高分的工具箱，NTIRE 2024 低光挑战赛基线 → [code](https://github.com/caiyuanhao1998/Retinexformer)
- **Zero-DCE / Zero-DCE++**（CVPR 2020 / TPAMI 2021）：零参考曲线估计，无需成对训练数据，轻量实时 → [项目页](https://li-chongyi.github.io/Zero-DCE.html)、[论文合集仓库](https://github.com/Li-Chongyi/Lighting-the-Darkness-in-the-Deep-Learning)
- **SCI**（CVPR 2022）：自校准光照，参数量 KB 级，4K 实时 → [code](https://github.com/vis-opt-group/SCI)

### 自动 HDR
- **Deep-HdrReconstruction**（Eilertsen et al.）：单帧 LDR→HDR，学习逆相机管线（饱和区修复）→ [code](https://github.com/marcelsan/Deep-HdrReconstruction)
- **HDRUNet**（ICCV 2021）：单帧 HDR 重建 + 联合去噪去量化，高引 baseline
- **NTIRE 2025 Efficient Burst HDR and Restoration** 挑战赛（CVPR workshop）：手机端多帧/burst HDR 方向，代表工业界最新玩法
- 论文索引：[Awesome-High-Dynamic-Range-Imaging](https://github.com/rebeccaeexu/Awesome-High-Dynamic-Range-Imaging)、[Awesome-HDR](https://github.com/ytZhang99/Awesome-HDR)

### 高效复原 / 新架构
- **MambaIR**（ECCV 2024）/ **MambaIRv2**（2025）：状态空间模型（Mamba）做复原，线性复杂度 + 全局感受野 → [code](https://github.com/cguoah/MambaIR)；索引：Awesome-Mamba-in-Low-Level-Vision
- **ResShift**（NeurIPS 2023 Spotlight / TPAMI 2024）：残差移位扩散，15 步采样完成超分/去模糊 → [code](https://github.com/zsyOAOA/ResShift)
- **InvSR**：扩散反演任意步数超分（ResShift 作者后续）→ [code](https://github.com/zsyOAOA/InvSR)
- **DiffBIR**（ECCV 2024）：两阶段盲复原，先回归去退化再用 SD 生成先验恢复细节 → [code](https://github.com/XPixelGroup/DiffBIR)
- **SUPIR**（CVPR 2024）：SDXL 规模化的"野生"图像复原，观感天花板 → [code](https://github.com/Fanghua-Yu/SUPIR)
- 综述索引：[Awesome-diffusion-model-for-image-processing](https://github.com/lixinustc/Awesome-diffusion-model-for-image-processing)

## 三、课堂项目建议（推荐功能组合）

按"实时预览（CPU/核显可跑）"和"拍照后处理（可重可慢）"分两层：

1. **自动 HDR 拍照（招牌功能，强烈推荐）**
   - 经典线：连拍 3 张不同曝光（改变 `cap.set(CAP_PROP_EXPOSURE, ...)`）→ `cv2.createMergeMertens()` 融合，教学点：对比度/饱和度/显著性三种权重。
   - 深度线：单帧 HDR（HDRUNet / Deep-HdrReconstruction 预训练模型，可导 ONNX）。
   - 答辩亮点：Mertens vs 深度方法 的对比实验（PSNR/μ-visual + 用户评分）。
2. **低光增强预览**：用 SCI 或 Zero-DCE++（ONNX）替换现有"亮度<60 才提亮"的启发式增强，实时性好、有论文故事（零参考学习）。
3. **拍照后处理流水线**：保存时自动跑 Real-ESRGAN 超分 + GFPGAN 人脸修复（GUI 上加"AI 增强"开关）。
4. **人像虚化 / 背景替换**：rembg(BiRefNet) 或 MediaPipe SelfieSegmentation 抠像 + Depth Anything V2 估计深度做渐进虚化，模拟大光圈。
5. **加分讲点**：用 Kornia 把本项目某个传统算子（如双边滤波）重写成可微版本，串起"DIP 经典 ↔ 深度学习"的叙事。

> 扩散类（DiffBIR / SUPIR / ResShift）效果最惊艳但需要 GPU 且秒级延迟，建议只作为拍照后处理的可选档位或 PPT 对比图，不做实时预览。

## 四、选型速查

| 需求 | 实时预览首选 | 拍照后处理首选 | 讲解点 |
|------|--------------|----------------|--------|
| 自动 HDR | Mertens 曝光融合 | HDRUNet / Deep-HdrReconstruction | 经典融合 vs 学习式逆管线 |
| 低光增强 | SCI / Zero-DCE++（ONNX） | Retinexformer | 零参考 / Retinex 理论 |
| 超分 | — | Real-ESRGAN（或 ResShift） | 退化建模、扩散加速 |
| 人脸 | — | GFPGAN / CodeFormer | 生成先验 |
| 虚化/换背景 | MediaPipe 自拍分割 + Depth Anything | rembg(BiRefNet) | 分割 + 深度 |

## 五、实机反馈驱动的补充调研：侧脸防伪影（2026-09，Phase 0 联调）

**问题**：美颜的液化变形（瘦脸/大眼）在侧脸下产生拉扯伪影——2D 变形隐含正脸假设，头偏航后"远端"下颌链的 2D 投影塌缩进脸颊中部，变形带横穿脸面；大眼固定半径在透视缩短的远端眼上溢出到鼻梁。

**参照**：
- MediaPipe 官方头姿路线：FaceLandmarker 可输出 `facialTransformationMatrixes`（规范脸模型 → 运行时关键点的刚体变换），分解欧拉角得 yaw（[官方博客](https://developers.googleblog.com)、[Face Landmarker 文档](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)）。
- [大偏航下关键点退化](https://pubmed.ncbi.nlm.nih.gov/23681991/)（Perakis et al.）与 [MLLS 图像变形](https://ar5iv.labs.arxiv.org)（Schaefer et al. 2006 系）是变形质量的两条经典线；工程通行做法是**按头姿给变形强度加门控**，超阈值直接关闭。

**落地（`core/effects/beauty.py`）**：
- `estimate_yaw_deg()`：鼻尖(1)到左右脸缘(234/454)的水平距离比 → 线性近似偏航角（无需变换矩阵，自动兼容镜像，只作门控阈值用）；
- `pose_gate()`：|yaw| ≤15° 全强度，15°~32° 线性衰减，≥32° 关闭变形；
- 瘦脸：|yaw| >12° 时跳过远端下颌链（远端 = 2D 质心更近鼻尖的塌缩侧，不依赖 yaw 符号）；
- 大眼：半径 = 眼角距 × 0.85（自适应透视缩短，上限 0.045w），眼角距 <2%w 判为塌缩跳过该眼。
- 性能：瘦脸距离场改为 `cv2.distanceTransform`（17.7ms → 5.4ms），美白限定掩膜包围盒；全链 720p 实测 ~28fps。

### 瘦脸变形模型 v3（实机反馈"效果非常不明显"后的复盘，2026-09）

v2 的三个隐藏缺陷（教训都写进了代码注释）：
1. **alpha 内容混合制造重影**：`patch = 位移内容×a + 原内容×(1−a)` 在权重<1 处内容同时出现在原位和位移处，观感位移减半且发虚。位移场本身连续，直接单次 remap 即可，无需混合；
2. **左右链先后写入互相覆盖**：两场在下半脸重叠，后写入者以原始帧采样覆盖先写入者的位移。修法：两侧场先合成单一总位移场再一次 remap；
3. **链条顶端（耳侧）带宽上探太阳穴/发际**：拖动头发是"正脸观感差"来源之一。修法：竖直渐变窗（眼线以下渐起、嘴线以下全量）。

v3 参数：幅度上限 0.055×帧宽（1280 下默认强度 0.4 ≈ 28px），轮廓内侧 σ=58px（脸颊整体内收）、外侧 σ=22px（仅轮廓边滑动），门控放宽为 ≤20° 全强度 / 40° 关闭、远端链跳过阈值 15°。真图实测：轮廓内收 29px、40px 深处 7px、太阳穴/嘴角 <1.5px、11.5ms@2.5MP。
