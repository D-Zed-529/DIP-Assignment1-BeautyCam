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

### 瘦脸变形模型 v4：MLS 相似变形（2026-09-22，`core/mls.py`）

v3 的手调高斯位移场本质上是对 MLS（Moving Least Squares，Schaefer et al. 2006）的粗糙近似：方向硬编码水平、σ 手拍、靠竖直窗防泄漏。v4 直接实现教科书算法（复数形式 + 粗网格求解 + 双线性上采样，`mls_similarity_maps`），控制点跟随轮廓法向。

**变体选择（实测教训）**：刚体变体（rigid，只旋转）禁缩放，而"下颌整体向中心收"是收缩运动，会被模型本身对抗——真图跟随率仅 0.2~0.5，观感偏弱。相似变体（similarity，旋转+均匀缩放）控制点精确跟随，中段下颌跟随率 0.87~0.98。rigid 保留给平移/拖拽类布局。

**三个实现坑（都来自实测，代码注释同款）**：
1. **remap 语义**：采样图是逆映射，求解时必须 P/Q 对调（内容 P→Q ↔ map(q)=p），方向反了变形方向恰好相反；
2. **上采样不能用 `cv2.resize`**：其像素中心约定使采样坐标系统性偏移 ~0.5 个网格节点（grid_step=6 即 2.5px 内容平移，锚点钉扎失效、额头整体漂移）。改用 `cv2.remap` 按节点坐标精确采样；网格覆盖不到 ROI 边缘时补"虚拟节点"（位置前进、位移=0），不能 clip 冻结——否则绝对位置场在尾部变成线性负斜坡（ROI 边缘假漂移）；
3. **MLS 是全局法**：只锚额顶单点时下颌动点的位移沿平面漏到额头（实测 2~5px）。修法 = 上轮廓弧整圈锚点（338/297/332/384/385/387/388/127/234/109/67/103/54/21）+ 动点高度门槛（高于下眼睑线的链端转锚点）+ 嘴线→眼线 smoothstep 高度衰减 + 眉线以上竖直余弦保护窗（该区场量 <2px，乘窗无可感知差异，但保证强度拉满时发际线以上比特级不动）。

**真图验收（portrait1.jpg，1280×1920）**：默认强度 0.40 下颌中段内收 21~23px（v3 为 29px，同级可见），极限档 59~69px；太阳穴/嘴角/额顶位移 0.0px；49ms@2.5MP（720p 预览约 18ms）。56 个单测全绿，其中额头保护改为**位移场断言**（值级 allclose 在高对比边缘会被亚像素重采样放大成大值差，不是 MLS 能满足的口径）。

## 六、Phase 1–5 逐项算法选型落地调研（2026-09-22）

> 针对 [TODO.md](../TODO.md) 未完成项逐条核查「开源可直接用 / 论文算法轻量自实现」。检索式调研（Web 搜索），标注：✅ = 搜索结果直接确认存在（标题+内容匹配），△ = 仅检索到间接引用、路径未逐一打开核实。
> 环境前提：M4 / Python 3.12 / OpenCV 4.11（当前装的是 `opencv-python`，**无 ximgproc**）/ numpy 1.26（<2 锁定）/ ONNX Runtime 待引入。

### 6.1 Phase 1 — 自动 HDR

**P1-1 伪包围曝光参数**：gamma 伪曝光 + 融合有论文直接背书——Kinoshita et al.《A Pseudo Multi-Exposure Fusion Method Using Single Image》（arXiv 2018 ✅），思路即「单图 gamma 生成伪多曝光再融合」。落地配方：
- 公式 `I_ev = 255·(I/255)^(2^(−EV))`（EV>0 提亮），3 张 {−2, 0, +2} 起步，最多 5 张 {−3, −1.5, 0, +1.5, +3}（边际收益快速递减）；
- Mertens 融合权重用 OpenCV 默认 `(contrast, saturation, exposure) = (1, 1, 1)`（[官方文档](https://docs.opencv.org) ✅，权重 = 拉普拉斯对比度 / 饱和度 / 高斯中心 0.5 的 well-exposedness，拉普拉斯金字塔多尺度合成无缝）；
- 已知缺陷与缓解：暗部被 gamma 提亮时噪声放大（融合前轻量去噪，如双边滤波）；高光区无新信息可恢复（EV 范围控制在 ±3 内）。该论文无官方代码（MATLAB 系），但实现仅几行，自写即可。

**P1-2 帧间对齐——建议改方案**：TODO 原定 `findTransformECC`，调研后**建议换成 `cv2.createAlignMTB`**（[OpenCV HDR 文档](https://docs.opencv.org) ✅、[LearnOpenCV HDR 教程](https://learnopencv.com/hdr-imaging-with-opencv/) ✅）：
- MTB（median threshold bitmap）+ 金字塔逐级平移搜索，是 OpenCV 专为 HDR 多曝光序列设计的对齐器，对曝光差异不变，速度快于 ECC 1–2 个数量级（ECC 是迭代梯度优化，社区实测大图栈可到分钟级 ✅）；
- 局限：只估整帧平移（无旋转/仿射）。手持连拍间隔 <100ms 时以抖动平移为主，够用；若实测有旋转残差，备选 ORB/AKAZE 特征 + `estimateAffinePartial2D`（比 ECC 快且稳）；
- 同曝光视频流连拍（我们的场景）MTB 同样适用（中值阈值对同曝光也稳健）。
- 答辩讲点：Google HDR+（Hasinoff et al. 2016）「对齐→合并→完成（align-merge-finish）」移动端 burst HDR 流水线思想。

**P1-4 色调映射**：`createTonemapDrago / Reinhard / Mantiuk` 均为 OpenCV 内置（零新依赖）。Mertens 输出本身已是可显示的 LDR float [0,1]，色调映射仅作润色选项——推荐 Drago 为默认（自适应对数，观感均衡）、Mantiuk 作备选。

**P1-5 单帧深度 HDR（可选）**：首选 **HDRUNet**（ICCV 2021 workshop，NTIRE 2021 单帧赛道第 2 名；官方 PyTorch + 预训练权重 ✅ [github.com/chxy95/HDRUNet](https://github.com/chxy95/HDRUNet)）。纯卷积结构，`torch.onnx.export` 一行导出；无官方 ONNX、仓库许可证未见明确声明（落地前核查）。Deep-HdrReconstruction（marcelsan，HDRCNN，2017）较老，作论文引用即可；NTIRE 2025 efficient burst HDR 赛道跟踪报告即可不必落地。

### 6.2 Phase 2 — 低光增强

**P2-1 定版建议：Zero-DCE++ 为主线，SCI 作对比线**：
- **Zero-DCE++**（TPAMI 2021）：DCE-Net 从原版 ~79K 参数缩到 **~10K**（权重几十 KB），纯卷积，`torch.onnx.export` 一行导出；无官方 ONNX 但社区先例多（Luxonis model zoo ✅、HuggingFace Space ✅）。权重经官方项目页 [li-chongyi.github.io](https://li-chongyi.github.io/Zero-DCE.html) / 论文合集仓库获取 ✅。**强度滑杆可自然映射到增强曲线的应用迭代/插值系数**（官方代码即逐次应用 8 条曲线），与本项目参数化需求契合。
- **SCI**（CVPR 2022 oral）：官方仓库 [vis-opt-group/SCI](https://github.com/vis-opt-group/SCI) ✅，**权重直接在仓库内**（zoom/mid/full.pt 按场景），论文宣称 4K 实时；但自校准结构无官方 ONNX 导出脚本，工作量略高。作「轻量光照自校准 vs 零参考曲线」的答辩对比线最合适。
- 排除：PairLUT（CVPR 2024，LUT 路线，**未检索到官方代码** ✗）；扩散类（NeRCo/LightenDiffusion）过重，仅 PPT 素材。

**P2-4 视频实时化**：隔帧推理 + 亮度增益场 EMA/光流平滑属标准工程做法即可满足，无需引入额外模型；讲点引用 **StableLLVE**（[zkawfanx/StableLLVE](https://github.com/zkawfanx/StableLLVE) ✅ 官方代码+模型；注意论文实际是 CVPR 2021《Learning Temporal Consistency for Low Light Video Enhancement from Single Images》——单图增广 + 时序一致性损失训练，正切「视频稳定」主题，可只引用不落地）。
自动触发口径建议改为**欠曝像素占比**（如亮度 <40 的像素比例超阈值）替代一期「均值 <60」，对局部高光更鲁棒。

**P5-3 关联数据**：LOL-v1 485 训练 / 15 测试对（RetinexNet, BMVC 2018）；LOL-v2-real 689/100。**直接可下载**：[HF datasets okhater/lolv2-real](https://huggingface.co/datasets/okhater/lolv2-real) ✅，LOL-v1 有 Kaggle 镜像 ✅。

### 6.3 Phase 3 — 人像虚化 / 背景替换

**P3-1 掩膜**：已选型 `selfie_multiclass_256x256.tflite`（16MB，在下载清单）输出 **6 类**：0 背景 / 1 头发 / 2 身体皮肤 / 3 脸部皮肤 / 4 衣服 / 5 其他（官方文档 ✅）；人像掩膜 = 非背景类合并。若多类用不上可换官方更轻的 `selfie_segmenter`（144×256，约 250KB 量级）提速。**边缘精化首选引导滤波**：`cv2.ximgproc.guidedFilter(原帧, 掩膜, radius, eps)`，O(N) 毫秒级，以原帧为引导图把软边界吸附到真实边缘（He Kaiming ECCV 2010 / TPAMI 2012 ✅）——**注意当前环境是 `opencv-python` 无 ximgproc**，要么换同版本 `opencv-contrib-python`，要么 numpy 手写（约 30 行，本身也是好讲点）。对比线（可选）：**RVM**（[PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) ✅ 官方即供 ONNX，rnn 隐状态天然时序稳定，mobilenetv3 版轻量）作「时序模型 vs 逐帧+EMA 平滑」答辩素材；MODNet 官方 ONNX 偏静态图；BiRefNet/BEN2 质量高但重，仅作拍照后处理对比。

**P3-2/P3-3 虚化与换背景**：大半径背景模糊用「缩小→模糊→放大」金字塔技巧避免全帧大核卷积；圆盘散景核可用迭代 box blur 近似。讲点：Google 单摄人像模式（Wadhwa et al., SIGGRAPH Asia 2018《Synthetic Depth-of-Field with a Single-Camera Mobile Phone》）。背景色彩协调：Reinhard 2001 Lab 统计迁移（LearnOpenCV 有现成 numpy 实现 ✅）或 `skimage.exposure.match_histograms` 轻量替代。

**P3-4 深度渐进虚化（进阶）**：首选 **Depth Anything V2 small**——HF [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small) ✅ **现成 `model.onnx` / fp16 / quantized 直接下载**（无需自己导出）；ViT-S 约 25M 参数、**Apache-2.0 许可**（注意 Base/Large/Giant 是 CC-BY-NC-4.0，课程可用但商用不可 ✅）。备选 MiDaS v2.1 small（[HF Heliosoph/midas-small-onnx](https://huggingface.co/Heliosoph/midas-small-onnx) ✅ / 官方 [isl-org/MiDaS](https://github.com/isl-org/MiDaS) ✅，EfficientNet-Lite3 骨干 256×256）。实现：深度→弥散圆半径分段线性映射 + 深度分 K 带（4–6 带）分层模糊合成（经典做法，纯 OpenCV）；BokehMe（2021，学习+物理混合）作讲点；BokehDiff（扩散）过重排除。

### 6.4 Phase 4 — 换脸（演示级经典路线）

**P4-1 参考实现**（均已核实存在 ✅）：
- [LearnOpenCV Face Swap](https://learnopencv.com/face-swap-using-opencv-c-python/)（2016 经典教程）：dlib 68 点 → Delaunay → 分块仿射 → `seamlessClone`；关键技巧「在固定参考形状上剖分、按索引映射到两张脸」，避免每帧剖分差异导致的三角形错位；
- [AsadiAhmad/Face-Swap](https://github.com/AsadiAhmad/Face-Swap)（GitHub，现代经典实现）与 [PySource 8 步图解](https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python/) ✅。

**关键优化——剖分不用每帧算**：MediaPipe **`FACEMESH_TESSELATION` 是官方在 canonical face model 上定义的固定三角拓扑**（468 点静态三角网，约 900 个三角形；官方文档 ✅），可直接当剖分结果使用——免每帧 Delaunay、帧间拓扑严格稳定不翻折、源/目标脸共享同一索引表，等价于 LearnOpenCV 技巧的官方现成版。`cv2.Subdiv2D` 仅用于离线演示对比。

**变形算法选型**：分块仿射（三角形边界有接缝，需在三角形间做 blending）vs **MLS 变形**（全场平滑无接缝）。`core/mls.py` 已实现 rigid 变体（瘦脸适用）；换脸脸型不同需要缩放，**建议补 similar 变体**——与 rigid 只差一步（复数缩放因子 `a` 不做 `a/|a|` 归一化），Schaefer 2006 的保真度排序为 affine > similarity > rigid，similar 档平滑度/实现成本平衡最好。

**融合与色彩**：`seamlessClone` 用 `NORMAL_CLONE`（MIXED_CLONE 会把目标纹理渗进换入脸）；掩膜取内部关键点凸包并腐蚀 1–2px + 羽化，缓解边界鬼影/渗色（Pérez 2003 Poisson editing 为讲点）。色彩迁移 Reinhard 2001，LearnOpenCV 有现成 numpy 实现 ✅。

**P4-3 实时可行性**：FaceMesh（~10ms）+ MLS remap（几 ms）+ seamlessClone（人脸 ROI 内约 10–20ms），720p 单人脸实时可行（参考美颜链 28fps）；若 seamlessClone 拖帧率可退化为凸包掩膜 alpha 羽化混合。
**P4-2 可视化素材**：OpenCV Subdiv2D Delaunay 绘制 demo 与 LearnOpenCV Delaunay/三角剖分文章 ✅；LearnOpenCV face morph 动画可加作「形变插值」讲解素材。

### 6.5 Phase 5 — GPU 加速与评测

**P5-1 CoreML EP**：macOS arm64 的现代 `onnxruntime` wheel 自带 `CoreMLExecutionProvider`（老版本需单独 `onnxruntime-coreml` 包；官方文档 ✅）。落地三要点：① 会话前 `ort.get_available_providers()` 确认 EP 在列；② 会话后 `session.get_providers()` 打印实际 provider（防静默回退，与项目现有惯例一致）；③ `MLComputeUnits` 用 provider options 指定（`CPUAndNeuralEngine` / `CPUAndGPU` 等）。已知坑：ANE 首次编译慢（有缓存）、动态形状易回退 CPU——**ONNX 导出固定 480p 形状**。MediaPipe tflite 不走 ORT，维持 CPU 委托（P0-1 结论，无加速空间）。

**P5-2 性能基准**：`time.perf_counter` 分阶段打点取 P50/P95 + `cv2.getTickCount` 备用；py-spy 采样定位热点即可，无需引入框架。

**P5-3 客观指标**：PSNR/SSIM 用 `scikit-image.metrics`（numpy 1.26 兼容，锁 0.22+；**当前未安装，需补依赖**）。曝光融合/HDR 专用指标：
- **MEF-SSIM**（Ma et al., IEEE TIP 2015）：官方 MATLAB（kedema.org ✅）；Python 可靠来源为 **MEFB 基准仓库**（Zhang et al. 2020，含多曝光融合指标库 △ 路径未逐一核实，搜 "MEFB imagefusion" 即得）；
- **PCQI**（Chen et al. 2015）：官方 MATLAB，无成熟 PyPI 包，numpy 手写约 50 行（分块均值/方差/相关三分量 △）；
- 务实口径：若搬运成本高，退化为「分区直方图 + 梯度能量」统计（暗部细节/高光不过曝双向验证）+ 主观评分，课程答辩足够。

**P5-4 主观评测**：延续一期双盲 Likert 5 级（自然度/边缘质量/整体满意度），方法学挂 ITU-R BT.500（MOS/ACR 框架）。

### 6.6 与现行 TODO 的差异建议（行动项汇总）

| TODO | 调研后的建议变更 |
|------|------------------|
| P1-2 | `findTransformECC` → **`cv2.createAlignMTB`**（快 1–2 个量级、专为 HDR 设计；ECC 降为备选） |
| P2-1 | 定版 **Zero-DCE++**（主，易导 ONNX + 强度参数好映射），SCI 作对比线 |
| P3-1 | 掩膜边缘精化引入 **guidedFilter**（需 `opencv-contrib-python` 或 numpy 手写） |
| P3-4 | 深度模型直接用 **HF onnx-community/depth-anything-v2-small 现成 ONNX**（fp16） |
| P4-1 | 剖分直接用 **FACEMESH_TESSELATION 固定拓扑**（免每帧 Delaunay）；变形补 **MLS similar 变体**（core/mls.py 扩展） |
| P5-1 | ONNX 导出**固定 480p 形状**；`get_providers()` 打印写入会话管理 |
| 依赖 | 新增 `onnxruntime`、`scikit-image`（评测）；`opencv-python` → `opencv-contrib-python`（同 4.11 版本线，为 ximgproc） |
