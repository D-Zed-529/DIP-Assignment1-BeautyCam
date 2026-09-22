# DIP-26 二期开发计划 — BeautyCam v2

> 状态：**已定稿**（2026-09-22）。任务跟踪见根目录 [TODO.md](../TODO.md)，选型调研见 [research-notes.md](research-notes.md)。

## 1. 目标与范围

在一期"实时美颜 + 手势/笑脸自动拍照"的基础上，把项目升级为一个**多效果实时相机系统**，用于 DIP 课程结题汇报：

| # | 功能 | 定位 |
|---|------|------|
| 1 | 自动 HDR 拍照 | 招牌功能。经典线（gamma 模拟包围曝光 + Mertens 融合）必做；单帧深度 HDR 为可选进阶 |
| 2 | 低光增强预览 | 深度线（SCI / Zero-DCE++，ONNX 推理）替换现有启发式增强 |
| 3 | 人像虚化 / 背景替换 | 基础版（自拍分割 + 模糊/换背景）必做；深度渐进虚化为进阶档 |
| 4 | 实时美颜 | 迁移一期实现，参数化（强度滑杆） |
| 5 | 换脸（演示级） | 经典 DIP 路线：Delaunay 三角剖分 + 分块仿射 + 泊松融合；主打**过程可视化**，不追求以假乱真 |
| 6 | GPU 加速 | ONNX Runtime CoreML EP（Apple Silicon）；作为优化项 + 答辩性能对比素材，不是独立功能 |
| 7 | 全新 GUI | **PySide6** 重写，暗色主题，控制面板 + 实时预览 + 拍照预览条 |

**明确不做**：深度换脸（inswapper 类）、扩散模型实时预览（DiffBIR/SUPIR 仅可作 PPT 对比图）、换语言/跨平台打包、多人协作后端。

**伦理约束**（写进 UI 与文档）：换脸功能仅用于本人面部、明确授权者或动漫形象；演示数据须获得被摄者同意。

## 2. 技术栈决策

| 层 | 选型 | 理由 |
|----|------|------|
| 语言/运行时 | Python 3.12 + `.venv` | 不变 |
| 处理核心 | OpenCV + MediaPipe | 不变（一期资产迁移复用） |
| 深度推理 | ONNX Runtime（CoreML EP 优先，CPU 回退） | 苹果芯片无 CUDA；CoreML 是唯一现实 GPU 路线；需验证无静默回退 |
| GUI | **PySide6**（LGPL） | 信号槽解决跨线程帧传递；QSS 暗色主题演示观感好；控件能力撑得起多效果面板。学习成本约 2–3 天 |
| 评测 | 本仓库 headless CLI + numpy 手写指标 | 与 GUI 解耦，可批量出实验数据 |

**关键绕坑决策**：macOS 上 OpenCV（AVFoundation 后端）对手动曝光控制支持很差，自动 HDR 的包围曝光**不用 `CAP_PROP_EXPOSURE` 实现**，改用连拍后 gamma 曲线模拟不同曝光再融合（效果等价、完全可控、可复现）。

## 3. 系统架构

### 3.1 目录结构

```
.
├── core/                  # 处理核心：禁止 import 任何 GUI（PySide6/tkinter）
│   ├── camera.py          #   采集源抽象：实时相机 / 视频文件 / 图片序列（测试与评测用）
│   ├── pipeline.py        #   效果链：有序 Effect 列表 + 线程安全参数更新
│   ├── context.py         #   FrameContext：每帧共享的推理结果（人脸/手部/分割）
│   ├── infer.py           #   MediaPipe / ONNX 会话管理（单例、CoreML EP、provider 记录）
│   └── effects/           #   每个效果一个模块
│       ├── beauty.py      #     磨皮/美白/瘦脸/大眼（迁移自一期）
│       ├── hdr.py         #     HDR 连拍融合（拍照模式，非逐帧效果）
│       ├── lowlight.py    #     SCI / Zero-DCE++ ONNX 低光增强
│       └── segment.py     #     自拍分割 + 虚化/背景替换（+ 进阶：深度虚化）
├── gui/                   # PySide6 界面壳
│   ├── main_window.py     #   主窗口：视频区 / 控制面板 / 拍照预览条
│   ├── workers.py         #   QThread 相机工作线程，信号发帧
│   ├── panels.py          #   各效果控制面板（开关 + 滑杆）
│   └── theme.qss          #   暗色主题
├── demos/                 # 演示级功能与课堂可视化
│   └── faceswap/          #   换脸：离线版 + 三角剖分过程动画 +（可选）实时版
├── scripts/
│   ├── download_models.py #   拉取 ONNX 权重到 models/
│   └── run_pipeline.py    #   headless CLI：图片/视频批跑管线，产出评测数据
├── tests/                 # 纯函数单测（不依赖摄像头/GUI）
├── models/                # 权重目录（gitignore，不入库）
├── assets/                # 背景图库等小体积资源（入库）
├── docs/                  # PLAN / 调研笔记
├── legacy/                # 一期单文件版本归档（Phase 0 完成后移入）
└── beautycam_v1.py       # 过渡期保留，GUI 对齐后移入 legacy/
```

### 3.2 效果链（Effect Chain）

```python
class Effect:
    name: str
    enabled: bool
    params: dict            # 如 {"strength": 0.6}，GUI 滑杆写入

    def process(self, frame: np.ndarray, ctx: FrameContext) -> np.ndarray: ...
```

- 每帧先由 `infer.py` 统一跑一次 FaceMesh/Hands/FaceDetection/分割，结果放进 `FrameContext`，各效果共享，**禁止效果内部各自重复推理**。
- 管线顺序（预览链）：`低光增强 → 美颜 → 虚化/背景替换 → 调试可视化叠加`。
- **HDR 例外**：它是拍照模式不是逐帧效果。预览时仅显示模式提示；按下快门后瞬间连拍 N 张 → 对齐 → Mertens 融合 → 出片。
- 参数更新走锁保护（GUI 线程写、工作线程读），避免边跑边改的竞态。

### 3.3 线程模型

```
QThread CameraWorker：cap.read() → 前推理(ctx) → pipeline.process(frame)
        │ 信号 emit ndarray（BGR）
        ▼
MainWindow（主线程）：ndarray → QImage 绘制；用户操作 → pipeline.set_params()
```

- 工作线程**永不触碰控件**（一期 `root.after()` hack 与双线程 bug 的根治）。
- 重推理（ONNX 模型）可在 worker 内按"隔帧 + 降分辨率 + 时域平滑"策略降载。

## 4. 阶段划分与验收标准

任务编号与详细清单见 [TODO.md](../TODO.md)。人日为单人净工作量估算，不含联调缓冲。

### Phase 0 — 重构底座（6–9 人日）
新建 core/gui 分层架构，PySide6 界面跑通实时预览，一期功能（美颜、手势、笑脸拍照）全部迁入并修复已知 bug（`v_sign_frames`/`smile_start_time` 缺 `global`、双线程启动）。
**验收**：仅美颜链预览 ≥30fps；headless CLI 能对静态图批跑；旧 `beautycam_v1.py` 移入 `legacy/`。

### Phase 1 — 自动 HDR（2–4 人日，进阶 +2）
gamma 包围曝光模拟 → ECC 对齐 → MergeMertens 融合 →（可选）色调映射显示；（进阶）HDRUNet ONNX 单帧 HDR 作对比线。
**验收**：逆光场景出片高光不过曝、暗部有细节；每张成片保留"单张 vs 融合"对照。

> **已完成（2026-09-22）**。EV 三预设（3张±1EV 默认）+ gamma LUT 模拟（高光平滑压缩非硬 clip）+ ECC EUCLIDEAN 对齐 + Mertens + Drago/Reinhard 显示选项；成片/各 EV 原图/对照图一并存档。**实施坑（已入 AGENTS.md #12）**：findTransformECC 返回的 warp 是 template→input 方向，应用须取仿射逆——误用方向不报错、只会更歪（合成平移实验 13.3→23.8，取逆 →1.6）。11 个纯函数单测锁定。HDRUNet 进阶线未做（经典线已达验收）。

### Phase 2 — 低光增强（2–3 人日）
模型选型定版（SCI vs Zero-DCE++，按 ONNX 可得性与实测速度）→ ONNX 会话管理（CoreML EP + provider 记录）→ ≤480p 推理 + 上采样 → 隔帧 + 时域平滑 → 亮度自动触发。
**验收**：暗光环境预览 ≥20fps；输出与旧启发式增强的对比图。

> **已完成（2026-09-22），定版 SCI**（Zero-DCE++ 无现成 ONNX 需自行导出，SCI 直接可得）。54KB、固定 512×512 输入（输出取张量 [1]）、三档强度；**CoreML EP 1.7ms vs CPU 7.4ms（4.3 倍，创建即打印实际 provider）**。客观评测（`scripts/eval_lowlight.py`，合成暗图）：**SCI-medium 22.1dB/0.860 vs 启发式 14.4dB/0.753**，深度线 +7.7dB 完胜，答辩对比素材成立。隔帧默认 2 + 复用上一帧结果；启发式保留为可切基线（GUI 双引擎下拉）。

### Phase 3 — 人像虚化 / 背景替换（3–5 人日）
MediaPipe Selfie Segmentation 掩膜（EMA 平滑 + 羽化）→ 基础虚化（强度滑杆）→ 背景替换（图库 + 自选）；进阶档：Depth Anything V2 小模型深度渐进虚化。
**验收**：边缘无明显抖动/镶边；虚化+美颜+低光三开 ≥15fps。

> **已完成（2026-09-22），实施中有两处偏离原计划，均以实测为依据：**
>
> 1. **分割模型换为二元 `selfie_segmenter.tflite`**（原计划默认用多分类）。P3-0 冒烟实测多分类 **155ms/帧**且耗时与输入分辨率无关（256×144 与 1280×720 同为 ~140–155ms，不取任何输出仍 ~142ms），直接接上去只有 ~7fps，验收必然不达标；二元模型仅 250KB、**13.2ms/帧**，快 11.7 倍。虚拟背景只需"人/非人"二分类，多分类那 5 类信息用不上却要多付 10 倍算力。多分类保留为可切换选项，正好做成"质量 vs 速度"对比线（延续本计划的"经典 vs 深度"对比套路）。
> 2. **边缘精修用自实现 guided filter，不依赖 `cv2.ximgproc`**。项目主目标的 macOS 上 opencv-python 不含 contrib；自实现（纯 `cv2.boxFilter`，He et al. ECCV 2010）与 ximgproc 版本实测数值一致到 1e-5，且顺便成了答辩可讲的算法内容。
>
> **另外两个必须记录在案的事实**（详见 `core/infer.py` 模块头与 `core/effects/segment.py`）：
> - 两个模型的**类别编码与置信图极性恰好相反**。搞反不抛异常，只静默产出反转掩膜（人景对调）或全幅掩膜（什么都不换），是这块最隐蔽的坑；已用 `SEGMENTER_SPECS` 显式声明 + 探针式单测 + 运行时边框先验自检三道防线覆盖。
> - 多分类模型的背景置信度有系统偏置（背景区 alpha 恒为 ~0.084），直接当 alpha 用会给新背景叠一层均匀 8.4% 鬼影；对比度拉伸（`matte_contrast`）可归零，二元模型无此问题。
>
> **性能实测**（本机 Windows x86 / 1280×720）：虚拟背景效果本身 15.2ms/帧（已从初版 23.3ms 优化：`cv2.blendLinear` 替代乘加、uint 域 matte 运算、半分辨率精修）；三开全帧 72.4ms ≈ **13.8fps**。基线——同机同推理（人脸+手势+美颜+低光）但不开虚拟背景——仅 **26.3fps**，而本计划记录的 M4 Mac 同配置约 55fps，可见本机约慢一倍，**13.8 vs 15fps 的差距主要来自机器而非本模块**。若需进一步提高，最有效的杠杆是把送入分割器的帧降采样（二元模型实测 480p 7.5ms vs 720p 13.4ms，代价是掩膜契约尺寸变化）。

### Phase 4 — 换脸·演示级（3–5 人日）
`demos/faceswap/`：FaceMesh 关键点 → Delaunay 三角剖分 → 分块仿射变形 → `seamlessClone` 泊松融合 → Reinhard 色彩迁移；三角剖分过程动画作为课堂讲解素材；（可选）实时换脸演示模式。
**验收**：能完整展示算法各阶段中间产物；效果达"可辨识换脸"级别即合格。

> **已完成（2026-09-22）**（离线版 + 过程可视化 + 伦理门；实时版 P4-3 为可选项未做）。stage1~5 全阶段存图 + 三联对照。两个实施坑入 AGENTS.md #13：warpAffine 输出窗口原点恒为 (0,0)（bbox 子图 warp 需平移分量减原点）、三角 bbox 越画布须裁相交区。泊松融合对纯色 src 会退化为 dst（梯度恒 0，数学正确）——测试要用带纹理的源图。伦理约束以 `--consent` 强制确认门落地（缺省拒绝运行）。

### Phase 5 — GPU 加速、评测与答辩（3–5 人日）
CoreML EP 全模型验证（防静默回退，记录实际 provider）→ 性能基准（单开/组合开 FPS、各阶段耗时分解、CPU vs CoreML）→ 客观评测（低光 PSNR/SSIM）→ 主观评测（Likert 问卷，延续一期方法学）→ PPT 与演示脚本。
**验收**：性能对比表 + 至少两组量化实验数据 + 完整演示走查。

> **代码与数据部分已完成（2026-09-22）**：① CoreML EP 启用并验证（SCI 4.3 倍，创建即打印 provider）；② `scripts/bench.py` 性能基准（各功能分解 + 后端对比）——**首版开放式计时循环把 mediapipe 刷爆内存（OOM），教训入 AGENTS.md #15**；③ `scripts/eval_lowlight.py` PSNR/SSIM（手写实现不依赖 skimage）。主观问卷（P5-4）与 PPT（P5-5）为小组线下工作。同日完成**性能优化专项**（用户要求）：磨皮/掩膜半分辨率、美白 boundingRect、瘦脸 ROI remap（逐位等价有单测锁定）、GUI worker 侧预缩放——美颜全链 18.4→9.4ms，组合链路详见 TODO.md 专项表。

### 时间线汇总

| 阶段 | 人日 | 累计 |
|------|------|------|
| P0 底座 | 6–9 | 6–9 |
| P1 HDR | 2–4 | 8–13 |
| P2 低光 | 2–3 | 10–16 |
| P3 虚化/换背景 | 3–5 | 13–21 |
| P4 换脸（演示级） | 3–5 | 16–26 |
| P5 GPU/评测/答辩 | 3–5 | **19–31** |

核心必做范围（P0–P3 + P5，即不含换脸）为 **16–26 人日**。3–4 人小组按每人每周 1–1.5 天投入、6–8 周周期（约 20–40 人日）可覆盖全量，留有缓冲。

**建议按模块认领分工**（避免互相踩）：① 架构 + GUI 壳；② HDR + 低光；③ 虚化 + 换脸；④ 评测 + 性能 + PPT。单人负责一个垂直模块的开发与自测。

## 5. 风险清单

| 风险 | 影响 | 对策 |
|------|------|------|
| macOS 手动曝光不可控 | HDR 拍不了真包围曝光 | 已定 gamma 模拟方案（见 §2） |
| CoreML EP 算子不支持静默回退 CPU | "GPU 加速"名存实亡 | 会话创建时记录并打印实际 provider；基准测试验证收益 |
| 多效果叠加帧率不达标 | 演示卡顿 | 模型 ≤480p 推理 + 隔帧 + 时域平滑 + 推理线程化；预案：降低预览分辨率档位 |
| mediapipe 1.0.1 API 与旧 `mp.solutions` 不兼容 | 迁移阻塞 | P0 第一项即冒烟验证，不兼容则按新 Tasks API 改写封装层 |
| 团队 PySide6 不熟 | Phase 0 延期 | P0 期间结对开发；界面壳先跑通再美化 |
| 换脸效果不佳 | 演示尴尬 | 定位演示级，主打过程可视化；离线版保底 |
| ONNX 权重体积大 | 仓库膨胀 | `models/` gitignore + 下载脚本；权重统一开源许可 |

## 6. 评测与答辩材料计划

延续一期 PPT 的评测方法学（Likert 主观评分 + 客观指标 + 性能数据）：

- **HDR**：3 组场景（逆光 / 室内 / 夜景）单张 vs 融合对照 + 主观评分；若做 HDRUNet 再加一列对比。
- **低光**：LOL 数据集子集或自摄成对数据，PSNR/SSIM；主观 MOS。
- **虚化/换背景**：边缘质量主观评分 + 该功能 FPS。
- **换脸**：算法各阶段中间产物展示 + 身份保持主观评分。
- **性能总表**：每功能单开/组合开的 FPS、各阶段耗时分解、CPU vs CoreML 对比（答辩核心 slide）。
