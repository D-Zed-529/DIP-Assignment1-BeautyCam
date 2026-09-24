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

- [x] **P1-1** 确定包围曝光模拟参数：EV 组合三预设（3张±1EV 默认 / 3张±2EV / 5张±1.5EV）、连拍 0.12s 间隔（采集自然抖动供 ECC 对齐）；gamma LUT 模拟（ev>0 提亮、高光平滑压缩不硬 clip）（0.5d）
- [x] **P1-2** `core/effects/hdr.py`：连拍采集 → `cv2.findTransformECC`（EUCLIDEAN）帧间对齐 → `cv2.createMergeMertens` 融合。**实测坑：findTransformECC 返回的 warp 是 template→input 方向，应用必须取仿射逆**（合成平移实验：误用 13.3→23.8 变差，取逆 13.3→1.6），已写进函数 docstring 与单测（1d）
- [x] **P1-3** GUI「HDR 拍照」：HdrPanel（EV 预设 + tonemap 下拉 + 连拍按钮），worker 帧循环顶部连拍（0.12s 节奏，不走效果链）→ 融合 → 成片/各 EV 原图/对照图一并存 photos/（1d）
- [x] **P1-4** 显示用色调映射选项：Drago / Reinhard（作用于 Mertens 融合结果的伪 HDR 域，峰值归一防整体变暗）（0.5d）
- [ ] **P1-5**（进阶，可选）单帧 HDR：HDRUNet 导 ONNX 对比线（未做；经典线已达验收标准）

**Phase 1 验收**：管线端到端单测 11 个全绿（LUT 方向性/恒等、ECC 平移恢复、融合介于两档、tonemap 变体、数量校验）；成片+曝光原图+对照图存档逻辑就位；逆光实拍效果待用户实机确认（无头环境限制）。

## Phase 2 — 低光增强（2–3 人日）

- [x] **P2-1** 模型定版：**SCI**（Self-Calibrated Illumination, CVPR 2022）ONNX 化权重（Kazuhito00/SCI-ONNX-Sample，54KB、固定 512×512 输入、easy/medium/difficult 三档）。选型依据：Zero-DCE++ 无现成 ONNX（需自行导出）；SCI 直接可得且**CoreML 1.7ms vs CPU 7.4ms（4.3 倍）**。取输出张量 [1]（[0] 为中间量，与官方 sample 一致）（0.5d）
- [x] **P2-2** `scripts/download_models.py`（SCI 三档入 MANIFEST）+ `core/infer.py` ONNX 会话管理（`LowLightSession`：CoreML EP 优先、CPU 回退，**创建即打印实际 provider**，AGENTS.md 防静默回退要求）（0.5d）
- [x] **P2-3** `core/effects/lowlight.py`：`LowLightDnnEffect`——512×512 推理 + resize 回原尺寸（纵横比畸变仅在推理域）+ 上采样，强度参数为增强结果与原图混合比（1d）
- [x] **P2-4** 性能与自动化：隔帧推理（默认 2，中间帧复用上一帧结果）+ 亮度阈值自动触发（与启发式同口径，退出暗光断开复用链防闪旧帧）；缺权重/缺 onnxruntime 优雅降级透传（1d）

**Phase 2 验收**：客观评测（`scripts/eval_lowlight.py`，合成暗图 γ2.2×0.45）：**SCI-medium 22.1dB / SSIM 0.860 vs 启发式 14.4dB / 0.753（+7.7dB）**，"经典 vs 深度"对比线数据成立；CoreML EP 真实生效（创建打印 + bench 对比）；≥20fps 验收线大幅超出（全链含 SCI 仍 28fps）。

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

- [x] **P4-1** `demos/faceswap/faceswap.py` 离线版：源脸/目标脸 FaceMesh → Delaunay 三角剖分（Subdiv2D，坐标→索引回映）→ 分块仿射变形（逐三角 bbox 子图 warp，**两个实测坑**：warpAffine 输出窗口原点恒为 (0,0) 需平移分量减 bbox 原点；bbox 越画布须裁相交区）→ `cv2.seamlessClone` 泊松融合 → Reinhard 色彩迁移（σ 比值夹 [1/3,3] 防纯色补丁噪声放大）（2d）
- [x] **P4-2** 过程可视化：stage1~5 全阶段存图（关键点标注/三角剖分线框/变形蒙版/克隆前后/色彩迁移）+ compare 三联对照（1d）
- [x] **P4-3** 实时演示模式：`FaceSwapEffect` 接入 PySide6 相机效果链，复用每帧 FaceMesh 关键点并缓存源脸检测结果/Delaunay 三角网；GUI 提供 4 张原创虚构头像（两张明星风格、两张卡通风格）、自行上传、Reinhard 肤色匹配开关和授权确认门（2026-09-24）
- [x] **P4-4** 伦理约束落地：CLI `--consent` 强制确认门（缺省拒绝运行）+ `demos/faceswap/README.md` 三条使用边界声明（0.25d）

**Phase 4 验收**：离线几何/融合与实时效果插件共 11 个单测全绿（剖分计数/边界、平移恢复、Reinhard 均值迁移、端到端区域改变、线框绘制、授权门、源脸单次检测与三角网复用、无人脸透传）；真图自换脸三联对照统计一致（161.8/162.0/161.5，源=目标时结果≈自身，管线自洽）；相机窗口已完成实时换脸、原创脸库与自行上传入口，双人脸目标选择仍不在本阶段范围内。

## Phase 5 — GPU 加速、评测与答辩（3–5 人日）

- [x] **P5-1** CoreML EP 启用与验证：SCI 低光会话 CoreML EP 优先 + CPU 回退，**创建即打印实际 provider**（`LowLightSession.__init__`），bench 实测 CoreML 1.7ms vs CPU 7.4ms（4.3 倍），EP 真实生效非静默回退（MediaPipe 侧维持 CPU 委托——Metal 委托本机崩溃，见 P0-1）（1d）
- [x] **P5-2** 性能基准：`scripts/bench.py`——各功能单开/组合 FPS、单帧各阶段耗时分解、CPU vs CoreML 对比；**首版教训入档：开放式自适应计时循环把 mediapipe 推理刷上千次直接打爆内存（VIDEO 模式按次分配不归还），已改固定调用次数 + 每节 gc + RSS 打印**（1d）
- [x] **P5-3** 客观评测（低光部分）：`scripts/eval_lowlight.py` 合成成对数据（γ2.2×0.45 压暗），PSNR/SSIM 手写实现（不依赖 skimage），SCI vs 启发式 vs 不增强三线对比 + 对照图（1d）
- [ ] **P5-4** 主观评测：Likert 问卷线下收集（脚本/口径延续一期，待小组执行）
- [ ] **P5-5** 答辩材料：对比图集（`--seg-compare`/`eval_lowlight --save-compare`/HDR compare/faceswap compare 均可一键产出）、PPT（待小组完成）

**Phase 5 验收**：性能对比表 ✓（`outputs/bench_final.md`）+ 客观评测数据 ✓（低光 PSNR/SSIM 两组以上方法线）；主观问卷与演示走查待线下。

## 性能优化专项（2026-09-22，用户要求"着重优化性能"）

全部为**数值等价优化**（不改结果，逐位一致有单测锁定 `tests/test_beauty.py::TestPerfEquivalence`）：

| 优化点 | 手段 | 720p 实测 |
|---|---|---|
| 磨皮 bilateralFilter | 半分辨率计算 + INTER_LINEAR 上采样（皮肤低频，视觉等价） | 5.0 → 1.6 ms |
| 肤色掩膜 | 半分辨率阈值/形态学/羽化 + 上采样 | 1.3 → 0.6 ms |
| 美白定位 | `np.nonzero`（两遍 int64 索引）→ `cv2.boundingRect`（单遍 native） | 3.1 → 0.8 ms |
| 瘦脸 remap | 全帧 map/remap → 脸围盒 ROI（`rbf_liquify_maps(return_roi=True)`，场紧支撑保证 ROI 边界恒等；ROI 各边外扩 1px 供双线性插值邻居，与全帧**逐位一致**） | 5.7 → 2.4 ms |
| 亮度检测 | 1/4 边长 INTER_AREA 缩略图估均值（区域均值保均值不变） | 0.4 → 0.1 ms |
| GUI 帧路径 | 帧在 worker 线程预缩放到显示尺寸，主线程零缩放（原先主线程每帧 Qt SmoothTransformation 3~6ms） | 主线程 ≈0 缩放 |

**整链效果**：美颜全链 18.4 → 9.4 ms（**1.96×**）；美颜+虚化 24.0 → 27.6 fps；三开 22.9 → 28.3 fps（M4/720p，`scripts/bench.py` 优化前后对照见 `outputs/bench_before.md` / `bench_final.md`）。

## 自适应画质专项（2026-09-22，用户要求"回归 DIP 本质：直方图类自适应优化"）

新效果 `core/effects/autoenhance.py`（name=`autoenhance`，链首）：全时段经典 DIP 画面校正，
与低光增强互补（低光管极端暗光，本效果管逆光脸黑/轻度过曝/偏色/发灰等常态问题）。

- [x] **AE-1** 分区自动曝光：FaceMesh 轮廓掩膜（复用 `beauty.face_oval_mask`，与美颜共享推理零额外模型）分人脸/背景两区统计 LAB-L 直方图，各自 gamma 校正（人脸目标 150 / 背景 115，±12 容差带防常态抖动，γ 裁剪 [0.4,2.5]），两个幂变换 LUT 按软掩膜逐像素混合（过渡带无硬边）
- [x] **AE-2** 对比度：CLAHE 作用于 L 通道（contrast 参数映射 clipLimit 0.5~3.0）
- [x] **AE-3** 色调：灰世界假设白平衡，**背景区估计**通道增益（肤色会污染灰世界估计），增益裁剪 + 归一均值 1（只动通道比例不动整体亮度）
- [x] **AE-4** 饱和度：LAB 域就地缩放 a/b（零颜色空间转换成本）
- [x] **AE-5** 时域稳定：γ 与 WB 增益做**参数级** EMA（平滑统计量而非图像，相机 AE 标准做法）；`reset_temporal()` 断开统计历史
- [x] **AE-6** 接入：GUI `AutoEnhancePanel`（链首）、CLI `--autoenhance --ae-*` 参数组、bench 新增耗时节与组合档
- [x] **AE-7** 测试与验证：28 个单测全绿（gamma 求解/容差带/裁剪边界无 NaN、灰世界方向与掩膜版估计、分区 LUT、CLAHE、饱和度、EMA/reset、逆光端到端、无人脸退化）；真图批跑直方图确认（L 均值 167→136 朝目标回落，p1/p99 [17,248]→[3,244] 对比度拉开）；bench 实测效果本体 14.6ms/帧、+美颜组合 31.8fps

GUI 实机观感由用户运行 `python -m gui.main_window` 确认（无头环境只能验证 import 与纯函数）。

## 里程碑

| 里程碑 | 内容 | 依赖 | 状态 |
|--------|------|------|------|
| M1 | 新架构 + PySide6 GUI 可日常使用（P0 完成） | — | ✅ |
| M2 | HDR + 低光上线，核心拍照功能齐（P1–P2） | M1 | ✅ |
| M3 | 虚化/换背景上线，效果全量（P3） | M1 | ✅ |
| M4 | 换脸演示 + 全部评测数据 + 答辩材料（P4–P5） | M2、M3 | 代码与评测数据就绪；问卷/PPT 线下 |
