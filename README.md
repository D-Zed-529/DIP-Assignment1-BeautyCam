# DIP-26 · BeautyCam 智能美颜与动作识别相机

数字图像处理（DIP）课程小组项目：基于 **Python + OpenCV + MediaPipe** 的桌面端实时美颜相机，支持手势 / 笑脸触发自动拍照。

**当前版本：v2（Phase 0 重构底座已完成）** —— core/gui 分层架构 + PySide6 界面 + 效果链插件模式；一期 Tkinter 版归档于 `legacy/`。

## v2 已实现功能（Phase 0）

| 模块 | 说明 |
|------|------|
| 分层架构 | `core/`（处理核心，禁 import GUI）/ `gui/`（PySide6 壳）/ `scripts/`（headless 评测 CLI）严格分离 |
| 采集源抽象 | `core/camera.py`：实时相机（1280×720 镜像）/ 视频文件 / 图片序列统一接口；弱光检测口径统一（灰度均值 < 60） |
| 统一推理 | `core/infer.py`：MediaPipe **Tasks API** 单例会话（FaceMesh468+blendshapes / Hands21 / 自拍分割），每帧一次推理结果放 `FrameContext` 共享，效果内部禁止重复推理 |
| 美颜（全参数化） | ① 双边滤波磨皮（混合比可调）② LAB 美白（肤色∧FaceMesh 轮廓掩膜，强度可调）③ 瘦脸（下颌 liquify 位移，**真变形**，一期仅画点）④ 大眼（remap 向量化 + 边缘羽化，替代一期逐像素循环）⑤ 收尾锐化 |
| 低光增强 | 一期启发式迁移为效果模块（自动/手动、强度可调）；Phase 2 将替换为 SCI/Zero-DCE++ ONNX |
| 手势拍照 | 剪刀手判定不变（食指+中指伸直、夹角 15°–65°），**改用墙钟时间持续 1s 判定**（一期帧计数在帧率波动时不稳） |
| 笑脸拍照 | FaceBlendshapes `mouthSmile` 置信度 > 0.45 持续 0.5s（比一期嘴部张合更抗头姿干扰），缺失时自动回退一期口径 |
| GUI | PySide6 暗色主题：视频区 + 效果面板（开关/滑杆）+ 采集源选择（摄像头/视频文件）+ 拍照预览条（点击放大）+ 状态栏 FPS |
| 线程模型 | QThread 工作线程只发信号不碰控件；参数走 `pipeline.set_params()`（锁保护）；**一期 global 缺失 / 双线程两个 bug 已根治** |
| headless CLI | `scripts/run_pipeline.py`：图片目录/视频批跑管线出评测数据 |

照片按 `photos/{manual|v_sign|smile}_时间戳.jpg` 保存。

## 项目结构

```
.
├── core/                  # 处理核心（禁止 import GUI 库）
│   ├── camera.py          #   采集源抽象：相机 / 视频 / 图片序列
│   ├── context.py         #   FrameContext：每帧共享推理结果
│   ├── infer.py           #   MediaPipe Tasks 单例会话（CPU 委托）
│   ├── gestures.py        #   V 手势/笑脸判定 + 自动拍照状态机
│   ├── pipeline.py        #   Effect 基类 + 有序效果链（线程安全参数）
│   └── effects/           #   beauty.py / lowlight.py（hdr/segment 后续阶段）
├── gui/                   # PySide6 界面
│   ├── main_window.py     #   主窗口（python -m gui.main_window）
│   ├── panels.py          #   效果控制面板（开关+滑杆）
│   ├── workers.py         #   QThread 相机工作线程（信号发帧）
│   └── theme.qss          #   暗色主题
├── scripts/
│   ├── download_models.py #   拉取 MediaPipe 模型到 models/
│   └── run_pipeline.py    #   headless 批跑 CLI
├── tests/                 # 纯函数单测（不依赖摄像头/GUI）
├── assets/samples/        # 测试样例图
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

# headless 批跑（评测数据生产线）
python scripts/run_pipeline.py --input assets/samples --output outputs/
python scripts/run_pipeline.py --input 某视频.mp4 --no-beauty --lowlight --max-frames 100
```

### 重要坑（P0-1 冒烟结论，2026-09 本机实测）

- **mediapipe 必须用 0.10.x**（锁 `0.10.21`）：
  - 1.0.x 已移除旧 `mp.solutions` API；
  - 且 1.0.x 的 Tasks 检测类图（FaceDetector/FaceLandmarker/HandLandmarker）在本机（macOS darwin 27）**Open() 阶段 Metal 服务硬崩溃**（CPU/GPU 委托均崩，SIGABRT 无法被 Python 捕获），仅 ImageSegmenter 在 CPU 委托下可用。
  - 0.10.21 的 Tasks API + 显式 CPU 委托全部正常（`core/infer.py` 已固定此配置）。
- **FaceDetector 已弃用**：除上述崩溃外，人脸框可由 FaceMesh 轮廓关键点导出（`_landmarks_to_face_info`），还省一次前向。
- 模型下载地址部分已 404，`scripts/download_models.py` 内是实测可用的 URL 清单。
- 一期"已知问题"（`v_sign_frames`/`smile_start_time` 缺 global、双线程启动）在 v2 架构下已根治，详情见 `legacy/beautycam_v1.py` 归档。

## 性能基线（Apple M4，1280×720，2026-09 实测）

| 链路 | 均摊耗时 | FPS |
|------|---------|-----|
| FaceMesh 推理 + 美颜（默认参数） | ~18 ms/帧 | ~55 |
| FaceMesh + Hands + 美颜（手势模式全开） | ~30 ms/帧 | ~34 |

Phase 0 验收线（美颜链 ≥30fps）达标；多效果叠加的进一步优化（隔帧推理/降分辨率）见 PLAN。

## 二期计划（Phase 1–5 进行中）

v2 定位为**多效果实时相机系统**，详见 [docs/PLAN.md](docs/PLAN.md) 与 [TODO.md](TODO.md)：

- **自动 HDR 拍照**（P1）：gamma 模拟包围曝光 + Mertens 融合（经典线），可选单帧深度 HDR 对比线
- **低光增强**（P2）：SCI / Zero-DCE++（ONNX，CoreML EP）替换现有启发式增强
- **人像虚化 / 背景替换**（P3）：自拍分割起步，进阶深度渐进虚化
- **换脸（演示级）**（P4）：Delaunay 剖分 + 泊松融合，主打过程可视化；仅限本人/授权/动漫形象
- **GPU 加速与评测**（P5）：ONNX Runtime CoreML EP + 性能对比基准 + 答辩材料

> 硬件部分（STM32F103 + LED 指示灯联动）的代码不在本仓库，PPT 中的相关内容为另一条交付线。
