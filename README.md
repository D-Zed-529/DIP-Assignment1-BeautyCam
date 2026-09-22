# DIP-26 · BeautyCam 智能美颜与动作识别相机

数字图像处理（DIP）课程小组项目：基于 **Python + OpenCV + MediaPipe** 的桌面端实时美颜相机，支持手势 / 笑脸触发自动拍照。

**当前版本：v2（Phase 0 底座 + Phase 3 虚拟背景已完成）** —— core/gui 分层架构 + PySide6 界面 + 效果链插件模式；一期 Tkinter 版归档于 `legacy/`。

## v2 已实现功能

| 模块 | 说明 |
|------|------|
| 分层架构 | `core/`（处理核心，禁 import GUI）/ `gui/`（PySide6 壳）/ `scripts/`（headless 评测 CLI）严格分离 |
| 采集源抽象 | `core/camera.py`：实时相机（1280×720 镜像）/ 视频文件 / 图片序列统一接口；弱光检测口径统一（灰度均值 < 60） |
| 统一推理 | `core/infer.py`：MediaPipe **Tasks API** 单例会话（FaceMesh468+blendshapes / Hands21 / 自拍分割），每帧一次推理结果放 `FrameContext` 共享，效果内部禁止重复推理 |
| 美颜（全参数化） | ① 双边滤波磨皮（混合比可调）② LAB 美白（肤色∧FaceMesh 轮廓掩膜，强度可调）③ 瘦脸（下颌 liquify 位移，**真变形**，一期仅画点）④ 大眼（remap 向量化 + 边缘羽化，替代一期逐像素循环）⑤ 收尾锐化 |
| **人像虚化 / 背景替换（Phase 3）** | `core/effects/segment.py`：三档模式（**背景虚化 / 换背景图 / 纯色**，对齐腾讯会议虚拟背景）。人像掩膜 = 二元自拍分割 → 运动自适应 EMA 时域平滑（静止防抖、运动防拖影）→ **guided filter 边缘精修**（自实现 He et al. 2010，边缘最大梯度提升约 4 倍）；内置 10 张程序化背景图库 + 用户自选图片 + 纯色预设。实测效果本体 15.2ms/帧 |
| 低光增强 | 一期启发式迁移为效果模块（自动/手动、强度可调）；Phase 2 将替换为 SCI/Zero-DCE++ ONNX |
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
│   ├── infer.py           #   MediaPipe Tasks 单例会话（CPU 委托）
│   ├── gestures.py        #   V 手势/笑脸判定 + 自动拍照状态机
│   ├── pipeline.py        #   Effect 基类 + 有序效果链（线程安全参数 + 隔帧降载）
│   └── effects/           #   beauty.py / lowlight.py / segment.py（hdr 后续阶段）
├── gui/                   # PySide6 界面
│   ├── main_window.py     #   主窗口（python -m gui.main_window）
│   ├── panels.py          #   效果控制面板（开关+滑杆+图库选择器）
│   ├── workers.py         #   QThread 相机工作线程（信号发帧）
│   └── theme.qss          #   暗色主题
├── scripts/
│   ├── download_models.py #   拉取 MediaPipe 模型到 models/
│   ├── make_backgrounds.py#   程序化生成虚拟背景图库（无版权风险）
│   └── run_pipeline.py    #   headless 批跑 CLI
├── tests/                 # 纯函数单测（不依赖摄像头/GUI）
├── assets/samples/        # 测试样例图
├── assets/backgrounds/    # 虚拟背景图库（10 张，程序化生成）
├── legacy/                # 一期 Tkinter 单文件版（归档参考）
├── docs/                  # PLAN.md / research-notes.md
├── TODO.md                # 分阶段任务清单
└── models/                # 模型权重（gitignore，不入库）
```

## 环境与运行

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

## 性能基线（1280×720，2026-09 实测）

Apple M4（PLAN 的验收口径）：

| 链路 | 均摊耗时 | FPS |
|------|---------|-----|
| FaceMesh 推理 + 美颜（默认参数） | ~18 ms/帧 | ~55 |
| FaceMesh + Hands + 美颜（手势模式全开） | ~30 ms/帧 | ~34 |

Windows x86（本次 Phase 3 开发机）：

| 链路 | 均摊耗时 | FPS |
|------|---------|-----|
| 基线：人脸 + 手势 + 美颜 + 低光（**不开虚拟背景**） | 38.0 ms/帧 | 26.3 |
| 三开：低光 + 美颜 + 背景替换（其中背景替换 15.2ms） | 72.4 ms/帧 | 13.8 |

分割模型选型对比：二元 13.2ms/帧（每帧可跑）vs 多分类 155ms/帧（必须隔帧，已实现通用 `inference_interval` 隔帧降载机制）。

> 本机比 M4 慢约一倍（同配置基线 26.3 vs 55 fps），Phase 3 的 ≥15fps 验收线在 M4 上应有余量。若要进一步提高，最有效的杠杆是把送入分割器的帧降采样（二元模型 480p 实测 7.5ms vs 720p 13.4ms，代价是掩膜尺寸契约变化）。

## 二期计划（Phase 1–5 进行中）

v2 定位为**多效果实时相机系统**，详见 [docs/PLAN.md](docs/PLAN.md) 与 [TODO.md](TODO.md)：

- **自动 HDR 拍照**（P1）：gamma 模拟包围曝光 + Mertens 融合（经典线），可选单帧深度 HDR 对比线
- **低光增强**（P2）：SCI / Zero-DCE++（ONNX，CoreML EP）替换现有启发式增强
- ~~**人像虚化 / 背景替换**（P3）~~：**已完成**（三档模式 + 边缘精修 + 程序化背景图库）；进阶档深度渐进虚化（P3-4，Depth Anything V2）待做
- **换脸（演示级）**（P4）：Delaunay 剖分 + 泊松融合，主打过程可视化；仅限本人/授权/动漫形象
- **GPU 加速与评测**（P5）：ONNX Runtime CoreML EP + 性能对比基准 + 答辩材料

> 硬件部分（STM32F103 + LED 指示灯联动）的代码不在本仓库，PPT 中的相关内容为另一条交付线。
