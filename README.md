# DIP-26 · BeautyCam 智能美颜与动作识别相机

数字图像处理（DIP）课程小组项目：基于 **Python + OpenCV + MediaPipe** 的桌面端实时美颜相机，支持手势 / 笑脸触发自动拍照。

**当前版本：v2（Phase 0/1/2/3/4 功能开发完成，Phase 5 评测数据就绪）** —— core/gui 分层架构 + PySide6 界面 + 效果链插件模式；一期 Tkinter 版归档于 `legacy/`。

## v2 已实现功能

| 模块 | 说明 |
|------|------|
| 分层架构 | `core/`（处理核心，禁 import GUI）/ `gui/`（PySide6 壳）/ `scripts/`（headless 评测 CLI）严格分离 |
| 采集源抽象 | `core/camera.py`：实时相机（1280×720 镜像）/ 视频文件 / 图片序列统一接口；弱光检测口径统一（灰度均值 < 60） |
| 统一推理 | `core/infer.py`：MediaPipe **Tasks API** 单例会话（FaceMesh468+blendshapes / Hands21 / 自拍分割），每帧一次推理结果放 `FrameContext` 共享，效果内部禁止重复推理 |
| 美颜（全参数化） | ① 双边滤波磨皮（混合比可调）② LAB 美白（默认**全身肤色**，含脖子/手臂；可切"仅脸部"=肤色∧轮廓掩膜；软 alpha 无硬边）③ 瘦脸（下颌链 liquify 内收，**真变形**，方向/作用域已修正）④ 大眼（remap 向量化 + 边缘羽化，替代一期逐像素循环）⑤ 收尾锐化 |
| **人像虚化 / 背景替换（Phase 3）** | `core/effects/segment.py`：三档模式（**背景虚化 / 换背景图 / 纯色**，对齐腾讯会议虚拟背景）。人像掩膜 = 二元自拍分割 → 运动自适应 EMA 时域平滑（静止防抖、运动防拖影）→ **guided filter 边缘精修**（自实现 He et al. 2010，边缘最大梯度提升约 4 倍）；内置 10 张程序化背景图库 + 用户自选图片 + 纯色预设。实测效果本体 15.2ms/帧 |
| 低光增强（Phase 2） | 双引擎可切：**SCI 深度模型**（CVPR 2022，ONNX 54KB、固定 512×512 推理 + 上采样、easy/medium/difficult 三档、隔帧复用降载、亮度自动触发；CoreML EP 实测 1.7ms/次 vs CPU 7.4ms，**4.3 倍**，合成暗图 PSNR 22.1dB 启发式 14.4dB）与一期启发式基线（线性增益+直方图均衡）共存，构成"经典 vs 深度"对比线 |
| **自适应画质优化** | `core/effects/autoenhance.py`：全时段经典 DIP 画面校正（与低光增强互补——本效果管逆光脸黑/轻度过曝/偏色/发灰等常态问题）。**FaceMesh 轮廓分区统计**直方图 → 人脸/背景各自 gamma 自动曝光（目标 150/115，容差带防抖，幂变换 LUT 按软掩膜混合）→ CLAHE 对比度 → **灰世界白平衡**（背景区估计，避开肤色污染）→ LAB 饱和度；统计量参数级 EMA 时域平滑防闪。人脸区域复用美颜的 FaceMesh 推理，零额外模型；实测效果本体 14.6ms/帧 |
| **自动 HDR 拍照（Phase 1）** | `core/effects/hdr.py`：连拍（0.12s 间隔采集自然抖动）→ gamma LUT 模拟包围曝光（macOS 不支持手动曝光的关键绕坑决策）→ findTransformECC 帧间对齐 → MergeMertens 融合 → 可选 Drago/Reinhard 色调映射；成片/各 EV 原图/对照图一并存档 |
| **换脸（演示级，Phase 4）** | `demos/faceswap/`：FaceMesh → Delaunay 三角剖分 → 分块仿射变形 → seamlessClone 泊松融合 → Reinhard 色彩迁移；**每阶段中间产物存图**（课堂讲解素材）；CLI `--consent` 强制伦理确认（仅本人/授权/动漫形象） |
| 手势拍照 | 剪刀手判定不变（食指+中指伸直、夹角 15°–65°），**改用墙钟时间持续 1s 判定**（一期帧计数在帧率波动时不稳） |
| 笑脸拍照 | FaceBlendshapes `mouthSmile` 置信度 > 0.45 持续 0.5s（比一期嘴部张合更抗头姿干扰），缺失时自动回退一期口径 |
| GUI | PySide6 暗色主题：视频区 + 效果面板（开关/滑杆）+ 采集源选择（摄像头/视频文件）+ 拍照预览条（点击放大）+ 状态栏 FPS |
| 线程模型 | QThread 工作线程只发信号不碰控件；参数走 `pipeline.set_params()`（锁保护）；**一期 global 缺失 / 双线程两个 bug 已根治** |
| headless CLI | `scripts/run_pipeline.py`：图片目录/视频批跑管线出评测数据；虚拟背景支持三档模式、模型切换、**边缘处理对比图一键产出**（答辩素材） |

照片按 `photos/{manual|v_sign|smile}_时间戳.jpg` 保存。

## 项目结构

```
.
├── core/                  # 处理核心（禁止 import GUI 库）
│   ├── camera.py          #   采集源抽象：相机 / 视频 / 图片序列
│   ├── context.py         #   FrameContext：每帧共享推理结果
│   ├── infer.py           #   MediaPipe Tasks 单例会话 + SCI 低光 ONNX 会话（CoreML EP）
│   ├── gestures.py        #   V 手势/笑脸判定 + 自动拍照状态机
│   ├── pipeline.py        #   Effect 基类 + 有序效果链（线程安全参数 + 隔帧降载）
│   └── effects/           #   beauty.py / lowlight.py（启发式+SCI）/ segment.py / hdr.py
├── gui/                   # PySide6 界面
│   ├── main_window.py     #   主窗口（python -m gui.main_window）
│   ├── panels.py          #   效果控制面板（美颜/低光/虚化/HDR/拍照）
│   ├── workers.py         #   QThread 相机工作线程（信号发帧 + HDR 连拍）
│   └── theme.qss          #   暗色主题（卡片化 + 徽章体系）
├── demos/faceswap/        # 换脸演示（Delaunay+泊松，过程可视化，--consent 伦理门）
├── scripts/
│   ├── download_models.py #   拉取 MediaPipe + SCI 模型到 models/
│   ├── make_backgrounds.py#   程序化生成虚拟背景图库（无版权风险）
│   ├── run_pipeline.py    #   headless 批跑 CLI
│   ├── bench.py           #   性能基准（各功能分解 + CPU vs CoreML 对比）
│   └── eval_lowlight.py   #   低光增强 PSNR/SSIM 客观评测
├── tests/                 # 纯函数单测（不依赖摄像头/GUI）
├── assets/samples/        # 测试样例图
├── assets/backgrounds/    # 虚拟背景图库（10 张，程序化生成）
├── legacy/                # 一期 Tkinter 单文件版（归档参考）
├── docs/                  # PLAN.md / research-notes.md
├── TODO.md                # 分阶段任务清单
└── models/                # 模型权重（gitignore，不入库）
```

## 环境与运行

Windows（含华为笔记本，建议 Python 3.11/3.12）：在 PyCharm 打开仓库根目录，
将解释器设为本项目的 `.venv\Scripts\python.exe`，运行模块
`gui.main_window`。首次使用先安装 `requirements.txt` 并运行
`python scripts/download_models.py`。摄像头保持 `0`，点击「开始」；
程序会优先尝试 DirectShow，失败时回退 OpenCV 自动后端。
在「人像虚化 / 背景替换」中选「背景虚化」即启用虚化；选「换成背景图」
会自动启用效果并选中第一张内置背景，也可点击图库缩略图或「选择图片…」。
选择图片后面板会显示当前文件名；若图片无法读取会提示重新选择。

实时换脸位于右侧「换脸（演示级）」面板：先选择一张正面清晰的源脸照片，
确认照片属于本人、已获授权者或动漫形象，再勾选「启用实时换脸」。程序会
复用相机已经得到的 FaceMesh 关键点，并缓存源脸的 Delaunay 三角网；可用
「肤色匹配（Reinhard）」开关对比色彩迁移前后的融合效果。该功能只在本机
处理图片，不会上传照片；取消授权确认会立即关闭换脸。面板内置 4 张原创
虚构人物头像（两张明星风格、两张卡通风格），也可点击「自行上传源脸图片…」
选择本机素材；项目不随附真实明星照片或现有版权角色。

PowerShell 首次安装示例：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_models.py
.\.venv\Scripts\python.exe -m gui.main_window
```

macOS（arm64）+ Python 3.12：

```bash
# 创建并激活虚拟环境（注意 PATH 中 python3 是 3.11，必须显式指定 3.12）
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate

# 安装依赖（mediapipe 必须锁 0.10.x，见下方"重要坑"）
pip install -r requirements.txt

# 拉取模型权重（models/ 不入库）
python scripts/download_models.py

# 启动 GUI（需摄像头权限 + 本地显示）
python -m gui.main_window

# 单测
python -m unittest discover tests

# 生成虚拟背景图库（已入库，一般无需重跑）
python scripts/make_backgrounds.py

# headless 批跑（评测数据生产线）
python scripts/run_pipeline.py --input assets/samples --output outputs/
python scripts/run_pipeline.py --input 某视频.mp4 --no-beauty --lowlight --max-frames 100

# SCI 深度低光增强（Phase 2 主力档）
python scripts/run_pipeline.py --input assets/samples --no-beauty \
    --lowlight-dnn --lowlight-level medium --lowlight-force

# 换脸演示（Phase 4；--consent 为伦理确认门，详见 demos/faceswap/README.md）
python -m demos.faceswap.faceswap --src 源脸.jpg --dst 目标.jpg \
    --out outputs/faceswap --consent

# 低光客观评测（PSNR/SSIM：SCI vs 启发式 vs 不增强）
python scripts/eval_lowlight.py --input assets/samples --save-compare

# 性能基准（各功能耗时分解 + SCI 的 CPU vs CoreML 对比）
python scripts/bench.py --markdown-out outputs/bench.md

# 虚拟背景：换背景图 / 背景虚化 / 纯色
python scripts/run_pipeline.py --input assets/samples --segment \
    --seg-mode image --seg-bg assets/backgrounds/02_冷色渐变.jpg
python scripts/run_pipeline.py --input assets/samples --segment --seg-mode blur

# 边缘处理四档对比图（答辩素材）

python scripts/run_pipeline.py --input assets/samples --segment --seg-mode color \
    --seg-bg-color "#00B140" --seg-compare
# 加 --seg-model selfie_segmenter 用多分类模型对比（155ms/帧，仅适合离线出图）
# 加 --seg-dump-alpha 另存 alpha 灰度图（人应是白的）
```

### 重要坑（P0-1 冒烟结论，2026-09 本机实测）

- **mediapipe 必须用 0.10.x**（锁 `0.10.21`）：
  - 1.0.x 已移除旧 `mp.solutions` API；
  - 且 1.0.x 的 Tasks 检测类图（FaceDetector/FaceLandmarker/HandLandmarker）在本机（macOS darwin 27）**Open() 阶段 Metal 服务硬崩溃**（CPU/GPU 委托均崩，SIGABRT 无法被 Python 捕获），仅 ImageSegmenter 在 CPU 委托下可用。
  - 0.10.21 的 Tasks API + 显式 CPU 委托全部正常（`core/infer.py` 已固定此配置）。
- **FaceDetector 已弃用**：除上述崩溃外，人脸框可由 FaceMesh 轮廓关键点导出（`_landmarks_to_face_info`），还省一次前向。
- 模型下载地址部分已 404，`scripts/download_models.py` 内是实测可用的 URL 清单。
- 一期"已知问题"（`v_sign_frames`/`smile_start_time` 缺 global、双线程启动）在 v2 架构下已根治，详情见 `legacy/beautycam_v1.py` 归档。

### 重要坑（P3-0 分割模型冒烟结论，2026-09 实测）

- **分割模型默认用二元 `selfie_segmenter.tflite`（250KB）而非多分类**：多分类 `selfie_multiclass_256x256.tflite`（16.4MB）实测 **155ms/帧**，且耗时与输入分辨率无关（降到 256×144 也一样，不取任何输出仍 ~142ms）——瓶颈纯在模型推理，直接接虚拟背景只有 ~7fps。二元模型 **13.2ms/帧**，快 11.7 倍，而虚拟背景只需要"人/非人"二分类。多分类保留为可切换选项，用于"质量 vs 速度"对比。
- **⚠️ 两个模型的类别编码与置信图极性恰好相反**：
  - 二元：类别 `0 = 人 / 255 = 背景`，`confidence_masks[0]` 是**人**的概率
  - 多分类：类别 `0 = 背景 / 1..5 = 人`，`confidence_masks[0]` 是**背景**的概率

  搞反**不会抛异常**，只会静默产出整体反相的掩膜（人景对调）或全幅掩膜（背景替换毫无效果），是这块最隐蔽的坑。约定显式声明在 `core/infer.py` 的 `SEGMENTER_SPECS`，并由三道防线兜底：探针式单测（用"图像四角必为背景"这种与掩膜语义无关的参照）、模型约定相反性断言、运行时边框先验自检告警。
- **多分类模型的背景置信度有系统偏置**（背景区 alpha 恒为 ~0.084，而非 0）。直接当 alpha 用会给新背景叠一层均匀的 8.4% 鬼影（背景发灰、颜色不实），靠 `matte_contrast` 对比度拉伸归零。二元模型无此问题。
- **不要依赖 `cv2.ximgproc`**：项目主目标的 macOS 上 opencv-python 不含 contrib。guided filter 已自实现（纯 `cv2.boxFilter`，与 ximgproc 版本数值一致到 1e-5）。
- **不要用 `cv2.imread`/`cv2.imwrite` 读写非 ASCII 路径**：Windows 上 `imread` 静默返回 `None`、`imwrite` 会**写到乱码文件名**且不报错。统一走 `np.fromfile`/`Path.write_bytes` + `cv2.imdecode`/`cv2.imencode`（见 `core/effects/segment.py::load_image`）。

## 性能基线（1280×720，2026-09 实测，Apple M4）

**美颜链性能优化（Phase 5）**：磨皮双边滤波降半分辨率（5.0→1.6ms）、肤色掩膜半分辨率（1.3→0.6ms）、美白 boundingRect 定位（3.1→0.8ms）、瘦脸 ROI remap（5.7→2.4ms，与全帧逐位一致有单测锁定）、亮度检测缩略图（0.4→0.1ms）——**美颜全链 18.4→9.4ms（1.96 倍）**，且全部不改数值结果（逐位等价）。

| 链路 | 优化前 | 优化后 |
|------|---------|--------|
| 美颜全链（含 FaceMesh） | 26.6 ms（37.6 fps） | **17.6 ms（56.8 fps）** |
| 美颜 + 虚化 | 41.6 ms（24.0 fps） | **36.3 ms（27.6 fps）** |
| 美颜 + 虚化 + 低光 | 43.7 ms（22.9 fps） | **35.4 ms（28.3 fps）** |

GUI 另有 worker 侧预缩放（帧在 worker 线程缩到显示尺寸，主线程零缩放，只做 QImage 包装）。

**推理分解**：FaceMesh 8.3ms / Hands 12.5ms（按需才跑）/ 分割二元 9.1ms / SCI 低光 CoreML **1.7ms** vs CPU 7.4ms（4.3 倍）。

分割模型选型对比：二元 13.2ms/帧（每帧可跑）vs 多分类 155ms/帧（必须隔帧，已实现通用 `inference_interval` 隔帧降载机制）。

低光增强客观评测（合成暗图，`scripts/eval_lowlight.py`）：**SCI-medium 22.1dB / SSIM 0.860** vs 启发式 14.4dB / 0.753 vs 不增强 6.7dB / 0.466。

## 二期计划收尾状态

v2 定位为**多效果实时相机系统**，详见 [docs/PLAN.md](docs/PLAN.md) 与 [TODO.md](TODO.md)：

- ~~**自动 HDR 拍照**（P1）~~：**已完成**（gamma 模拟包围曝光 + ECC 对齐 + Mertens + tonemap + 连拍存档）
- ~~**低光增强**（P2）~~：**已完成**（SCI ONNX 定版，CoreML EP，双引擎可切）
- ~~**人像虚化 / 背景替换**（P3）~~：**已完成**（三档模式 + 边缘精修 + 程序化背景图库）；进阶档深度渐进虚化（P3-4，Depth Anything V2）未做
- ~~**换脸（演示级）**（P4）~~：**已完成**（Delaunay + 泊松融合 + 过程可视化 + `--consent` 伦理门）；P4-3 实时版为可选项未做
- **GPU 加速与评测**（P5）：CoreML EP 已启用并验证（SCI 4.3 倍）、性能基准脚本（`bench.py`）与低光客观评测（`eval_lowlight.py`）就绪；主观问卷与 PPT 由小组线下完成

> 硬件部分（STM32F103 + LED 指示灯联动）的代码不在本仓库，PPT 中的相关内容为另一条交付线。
