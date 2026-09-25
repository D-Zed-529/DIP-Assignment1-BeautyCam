# DIP-26 · BeautyCam 智能美颜与动作识别相机

数字图像处理（DIP）课程小组项目：基于 **Python + OpenCV + PyTorch(CUDA)** 的桌面端实时美颜相机，支持手势 / 笑脸触发自动拍照。

**当前版本：v3（CUDA 迁移 + 模型升级完成）** —— 技术栈从 macOS/MediaPipe/CoreML 整体迁移到 **Windows + RTX GPU + PyTorch**，推理链全面 GPU 化，并按"算力富余换画质"的思路替换/新增了更大更好的模型：

| 组件 | v2（macOS / M4） | v3（Windows / RTX 3060） |
|------|------------------|--------------------------|
| 推理后端 | MediaPipe Tasks（CPU 委托） | **PyTorch TorchScript 全 GPU**（MediaPipe 自动回退） |
| 人像分割 | selfie_segmenter 二元（250KB） | **RVM RobustVideoMatting**（视频抠图，发丝级 alpha + 时域一致） |
| 低光增强 | SCI（54KB，快速档） | SCI 快速档 + **Retinexformer 质量档**（LOL-v1 SOTA 25.16dB） |
| 深度估计 | 无 | **Depth Anything V2 Small**（新增，P3-4 补上） |
| 效果 | 均匀背景虚化 | 均匀虚化 + **深度渐进虚化（单反感）** |

FaceMesh / Hands 保持 MediaPipe 原模型（已转 TorchScript 上 GPU，对 mediapipe 数值校准：landmark 平均偏差 0.002、分割 alpha MAE 0.004、blendshapes 逐名一致 —— 见 `scripts/calibrate_torch.py`）。

## v3 已实现功能

| 模块 | 说明 |
|------|------|
| 分层架构 | `core/`（处理核心，禁 import GUI）/ `gui/`（PySide6 壳）/ `scripts/`（headless 评测 CLI）严格分离 |
| **统一推理（双后端）** | `core/infer.py::get_engine()` 自动选择：CUDA + TorchScript 就绪 → `TorchInferenceEngine`（全 GPU，含跟踪状态机），否则回退 MediaPipe CPU。每帧一次推理结果放 `FrameContext` 共享 |
| 美颜（全参数化） | 双边滤波磨皮 / LAB 美白（人像区域或仅脸部；可选分割软掩膜精准圈人，避免背景同色误白）/ 瘦脸（下颌链 liquify 真变形）/ 大眼（remap 向量化）/ 收尾锐化 |
| **RVM 人像抠图** | 官方 TorchScript（mobilenetv3 fp32，~8ms/帧@720p），4 层循环时域状态，发丝级 alpha；隔帧降载、边缘精修（guided filter）全保留；resnet50 高质量档可切换（离线出图） |
| **深度渐进虚化（P3-4）** | Depth Anything V2 相对深度 → 焦平面对齐人物 → 模糊量随深度连续变化（sigma 金字塔 + 相邻档插值），观感对齐单反镜头 |
| 低光增强（双引擎） | **SCI 快速档**（默认 ONNX Runtime；torch CUDA 为实验路径）+ **Retinexformer 质量档**（ICCV 2023，LOL-v1 25.16dB，~67ms，隔帧推理摊薄）；启发式基线保留（"经典 vs 深度"对比线） |
| 自适应画质优化 | FaceMesh 分区统计直方图 → 人脸/背景细节保留 gamma 自动曝光（宽羽化、逆光抑制）→ CLAHE → 灰世界白平衡 → LAB 饱和度，统计量 EMA 时域平滑 |
| 自动 HDR 拍照 | 连拍 → gamma 模拟包围曝光 → ECC 对齐 → Mertens 融合 → 可选 Drago/Reinhard 色调映射 |
| 换脸（演示级） | 离线演示与相机实时预览；复用当前帧 FaceMesh，缓存源脸三角网；Delaunay 分块仿射 → 泊松融合 → Reinhard 色彩迁移；内置 4 张原创虚构头像，也可自行上传；CLI/GUI 均有授权确认门 |
| 手势 / 笑脸拍照 | V 手势（墙钟持续 1s）+ 笑脸（blendshapes mouthSmile 置信度），torch 后端下 blendshapes 由 HUND 头部网络计算（146 点子集·像素坐标输入） |
| GUI | PySide6 暗色主题：深度渐进虚化、实时换脸、低光引擎（SCI/Retinexformer）、分割模型（torch 后端默认 RVM）等面板 |
| headless CLI | `scripts/run_pipeline.py`：全部新参数（`--bokeh` / `--lowlight-engine` / `--seg-model rvm`） |

照片按 `photos/{manual|v_sign|smile}_时间戳.jpg` 保存。

## 项目结构

```
.
├── core/                   # 处理核心（禁止 import GUI 库）
│   ├── camera.py           #   采集源抽象：相机 / 视频 / 图片序列
│   ├── context.py          #   FrameContext（faces/hands/person_alpha/depth）
│   ├── infer.py            #   引擎工厂（torch/mediapipe 自动选择）+ ONNX 回退会话
│   ├── infer_torch.py      #   TorchInferenceEngine：FaceMesh/Hands/RVM/DA-v2 全 GPU
│   ├── gpuops.py           #   GPU 图像算子库（帧上传/滤波/引导滤波/CLAHE…）
│   ├── depthany.py         #   Depth Anything V2 Small 会话（transformers）
│   ├── retinexformer.py    #   Retinexformer 架构（vendor）+ LOL-v1 会话
│   ├── gestures.py / pipeline.py / mls.py / liquify.py
│   └── effects/            #   beauty / lowlight(双引擎) / segment / hdr / autoenhance / bokeh
├── gui/                    # PySide6 界面（main_window / panels / workers / theme.qss）
├── demos/faceswap/         # 离线/实时换脸（Delaunay+泊松，授权确认门）
├── assets/faces/          # 实时换脸内置原创脸库
├── scripts/
│   ├── download_models.py  #   拉取全部权重（mediapipe/RVM/Retinexformer/DA-v2）
│   ├── convert_models.py   #   tflite/onnx → TorchScript（一次性）
│   ├── tflite_to_torch.py  #   自研 TFLite→torch 转换器（tflite2onnx 在 Windows 不可用）
│   ├── calibrate_torch.py  #   torch vs mediapipe 数值校准（定版工具）
│   ├── run_pipeline.py     #   headless 批跑 CLI
│   ├── bench.py            #   性能基准（GPU 时钟拉频口径）
│   └── eval_lowlight.py    #   低光 PSNR/SSIM 客观评测
├── tests/                  # 236 个单测（不依赖摄像头/GUI；缺权重自动跳过）
├── models/torch/           # TorchScript 权重（gitignore，convert_models.py 产出）
├── models/hf/              # Depth Anything V2（transformers 格式，gitignore）
└── legacy/                 # 一期 Tkinter 单文件版（归档参考）
```

## 环境与运行（Windows + CUDA）

Windows 11 / RTX 3060 Laptop（6GB）+ Python 3.12（CUDA 12.6 轮子）。摄像头优先使用 DirectShow，打不开时回退 OpenCV 自动后端；右侧可扫描并选择设备。

```bash
# 1) 虚拟环境 + 依赖（torch 必须装 CUDA 轮子）
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2) 拉取模型权重（mediapipe 系 + RVM + Retinexformer(gdown) + Depth Anything V2）
python scripts/download_models.py

# 3) tflite/onnx → TorchScript 转换（一次性；产出 models/torch/*.ts）
python scripts/convert_models.py

# 4) （可选）torch vs mediapipe 数值校准（部署自检）
python scripts/calibrate_torch.py

# 5) 启动 GUI（需摄像头 + 本地显示）
python -m gui.main_window

# 单测（236 个；CUDA/权重缺失的项自动跳过）
python -m unittest discover tests
```

在 Windows 上也可以双击仓库根目录的 `start.bat` 启动；它会使用 `.venv` 中的 Python。运行 `start.bat --check` 可只检查 GUI 模块能否导入。

实时预览默认只运行轻度美颜；V 手势和笑脸自动拍照需要手动勾选，开启后会额外运行对应模型。拍照面板的“流畅优先”默认在 540p 处理预览，成片仍以原始分辨率重新处理。多种增强效果同时开启时，建议先从低强度调起；经典低光的直方图均衡可能放大暗部原有的色阶。
实时换脸位于右侧「换脸（演示级）」面板：选一张内置原创头像或自行上传源脸，确认使用权限后启用；可切换 Reinhard 肤色匹配。图片仅在本机处理，源脸关键点首次加载后缓存。
实时摄像头由独立线程持续采集，处理线程只领取最新帧；当效果链慢于相机帧率时会跳过过期帧，降低预览延迟。GPU 推理含多次小模型前向与 CPU 决策，任务管理器里的低平均利用率不等于显卡还有可直接转化为帧率的算力。
美颜、自适应画质和实时换脸共用本帧人脸关键点；背景处理与深度虚化需要人像掩膜时共用同一份分割结果。GUI 效果链把 CPU 换脸放在连续 CUDA 效果之后，组合开启时只需在末尾把帧下载到 CPU，避免中途回传再上传。
磨皮 GPU 路径利用额外显存展开邻域，减少 Windows 下小算子启动开销；540p 样图默认美颜整链约 21→14.5ms/帧，多效果（美颜＋自适应画质＋分割＋深度虚化）约 41→34ms/帧。此为预热后的离线样图处理耗时，不含摄像头取帧和界面显示。

headless 批跑与出图（评测数据生产线）：

```bash
python scripts/run_pipeline.py --input assets/samples --output outputs/

# RVM 虚拟背景三档（torch 后端默认 rvm；--seg-model 可切 binary/multiclass 对比）
python scripts/run_pipeline.py --input assets/samples --segment --seg-mode blur
python scripts/run_pipeline.py --input assets/samples --segment --seg-mode image --seg-bg assets/backgrounds/02_冷色渐变.jpg

# 深度渐进虚化（近清远糊，焦平面对齐人物）
python scripts/run_pipeline.py --input assets/samples --no-beauty --bokeh --bokeh-strength 0.9

# 低光双引擎：SCI 快速档 / Retinexformer 质量档
python scripts/run_pipeline.py --input assets/samples --no-beauty --lowlight-dnn --lowlight-force
python scripts/run_pipeline.py --input assets/samples --no-beauty --lowlight-dnn --lowlight-engine retinex --lowlight-force

# 低光客观评测 / 性能基准 / 换脸演示（同 v2）
python scripts/eval_lowlight.py --input assets/samples --save-compare
python scripts/bench.py --markdown-out outputs/bench.md
python -m demos.faceswap.faceswap --src 源脸.jpg --dst 目标.jpg --out outputs/faceswap --consent
```

## 重要坑（CUDA 迁移实测记录，2026-09）

- **tflite2onnx + onnx2torch 在 Windows 上不可用**（face_landmarks 布局传播 IndexError、HARD_SWISH / 自定义算子 Convolution2DTransposeBias 不支持、SUM 无映射；onnx2torch 的 shape inference 落盘临时文件句柄独占 PermissionError）→ 自研转换器 `scripts/tflite_to_torch.py` 直接解析 flatbuffer 构建 torch 模块（含 MediaPipe 自定义上采样算子，语义对照官方 transpose_conv_bias.cc 实现）。
- **face_landmarker.task 的 landmarks 模型输入是 256×256**（不是老的 192），输出 478×3 在 **256 像素域**（要 /256）；blendshapes 模型输入**不是图像**，是 146 点子集的**像素坐标** (1,146,2)，输出 52 维原始 0~1 值（mouthSmileLeft/Right = 下标 44/45）。
- **presence = sigmoid(Identity_1) 有低清退化**：同一张脸在 480 宽整幅上 presence 只有 0.0076（960 宽裁剪上 0.985），而幻影框恒 ≈0。绝对阈值 0.5 会把低清真脸整个拒掉 → 自适应门限（绝对 0.5 或相对帧内最大值 ×0.4，见 `core/infer_torch.py` 模块头标定记录）。
- **跟踪必须做检测/轨迹去重**（bbox IoU > 0.3 视为同一张脸），否则每轮检测刷新复制一份轨迹（实测 30 帧 1→4 递增）。
- **RVM 官方 TorchScript 的 fp16 版**（resnet50）输入/参数全 half，3060 上反而比 fp32 慢 16 倍（131ms vs 8ms）→ 默认 mobilenetv3 fp32。
- **笔记本 GPU 空闲即降频**，短基准虚高 3~50 倍：基准前必须空转拉时钟（`bench.py::gpu_spin`）；首次推理另有 ~15s 冷启动（模型加载）。
- **竖版图直接 resize 到 16:9 会把脸压扁到 FaceMesh 检测不到**：一律等比缩放 + 补边（AGENTS.md 坑 #14，对 torch 后端同样成立）。
- v2 的全部已知坑（mediapipe 版本、分割极性、cv2.imread 中文路径、ECC 方向、Drago NaN 等）见 AGENTS.md，多数只影响 mediapipe 回退路径。

## 性能基线（1280×720，2026-09 实测，RTX 3060 Laptop 6GB / torch:cuda）

实时预览在流畅优先模式下按 960×540 处理。此分辨率上，人脸 + RVM 的推理入口预热后顺序约 12ms/帧，CUDA stream 并发约 9ms/帧（本机三轮各 30 帧，范围 8.6～9.7ms）；单独美颜没有第二个模型可并发。效果链仍按顺序执行；9×9 磨皮滤波切块会增加调度开销，因此保持整图计算。

推理分解（GPU 时钟拉频口径，`scripts/bench.py`）：

| 推理项 | ms/帧 |
|---|---|
| FaceMesh（检测+关键点+blendshapes，跟踪态） | ~26 |
| Hands（palm+landmarks） | ~35 |
| 分割 RVM（mobilenetv3，含时域状态） | ~22 |
| 深度 Depth Anything V2 Small（392 推理域） | ~25 |
| 低光 SCI（512 域） | 默认 ONNX CPU 3.6；torch CUDA 实验路径 2.3 |
| 低光 Retinexformer（512 域，质量档） | ~67（隔帧推理摊薄到 ~34） |

组合链（整链均摊，含推理）：

| 链路 | ms/帧 | fps |
|---|---|---|
| 美颜 | ~60 | 16.7 |
| 美颜 + RVM 虚化 | ~96 | 10.4 |
| 美颜 + RVM 虚化 + 深度渐进虚化 | ~196 | 5.1（质量全开档） |

> 首帧另有 ~15s 冷启动（TorchScript/transformers 加载）；mediapipe 回退路径的基线见 git 历史（M4 口径）。

低光增强客观评测（合成暗图，`scripts/eval_lowlight.py`）：SCI-medium 22.1dB / SSIM 0.860 vs 启发式 14.4dB / 0.753；Retinexformer 为 LOL-v1 SOTA 口径（25.16dB，论文值）。

## 版本历史

- **v3（2026-09-23）**：Windows + CUDA 迁移；推理全 GPU（TorchScript）；RVM 分割、Retinexformer 低光、Depth Anything V2 深度虚化三项模型升级/新增；同帧独立模型 CUDA stream 并发；220 单测。
- **v3 合并（2026-09-24）**：接入队友的 Windows 摄像头回退、实时换脸与原创脸库；保留 CUDA 效果链并修复 GPU→CPU 插件交接；合并时 230 单测。
- **v3 修复（2026-09-24）**：自动拍照复选框运行中即时生效；修复 CUDA 大眼函数调用及局部坐标；无人像时纯色/图片背景仍执行替换；236 单测。
- **v2（2026-09）**：core/gui 分层 + PySide6 + 效果链插件；HDR / SCI 低光 / 虚拟背景 / 换脸 / 自适应画质；183 单测（macOS arm64 口径，详见 git 历史）。
- **v1**：Tkinter 单文件版（`legacy/`）。

> 硬件部分（STM32F103 + LED 指示灯联动）的代码不在本仓库，PPT 中的相关内容为另一条交付线。
