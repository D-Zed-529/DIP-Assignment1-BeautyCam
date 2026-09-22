# AGENTS.md — AI 协作规范

本仓库是数字图像处理课程项目 BeautyCam（macOS arm64 本地开发，非服务器部署）。

## 语言与沟通

- 与用户交流、代码注释、文档一律使用**中文**。

## 环境与命令

- Python 版本：**3.12**（`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12`）。注意：PATH 中默认的 `python3` 来自 PlatformIO 的 penv（3.11），创建虚拟环境时务必显式指定 3.12 的解释器。
- 虚拟环境：`.venv/`（已 gitignore）。激活：`source .venv/bin/activate`。
- 安装依赖：`pip install -r requirements.txt`。改依赖时同步更新 `requirements.txt`（`pip freeze > requirements.txt` 或手动维护顶层依赖）。
- 运行程序：`python -m gui.main_window`（需要摄像头权限 + 本地 GUI，无法在无头环境验证 UI，只能验证 import 与纯函数逻辑）。首次运行前先 `python scripts/download_models.py` 拉取模型。

## 代码约定

- **v2 架构（Phase 0 已落地，见 `docs/PLAN.md`）**：`core/`（处理核心）与 `gui/`（PySide6 界面）严格分层——**`core/` 下任何模块禁止 import GUI 库**（PySide6/tkinter）；换脸等演示放 `demos/`；评测批跑走 `scripts/run_pipeline.py`（headless，不依赖 GUI）。
- 效果插件模式：每个效果是 `core/effects/` 下的一个模块，实现 Effect 基类（`name/needs/enabled/params/process(frame, ctx)`）；每帧推理结果统一放在 `FrameContext` 里共享，**效果内部禁止重复起 MediaPipe/ONNX 会话**。需要重推理的效果另外实现 `inference_interval(need)` 声明隔帧间隔（由 `Pipeline.infer_needs_for(frame_index)` 聚合）、`reset_temporal()` 清理跨帧状态。
- 模块名小写下划线（如 `beauty.py`、`lowlight.py`）。
- OpenCV 处理链以 BGR 帧为主，送入 MediaPipe 前转 RGB，注意别搞反。
- MediaPipe / ONNX 会话统一在 `core/infer.py` 管理单例、创建开销大，不要在帧循环里重复创建；ONNX 会话创建后必须记录并打印实际 execution provider（防 CoreML 静默回退 CPU）。
- 阈值 / 强度等调参常量集中在模块顶部定义并附中文注释（现有风格：`V_SIGN_HOLD`、`SMILE_HOLD` 等）。
- GUI 线程模型（PySide6）：相机与推理在 `gui/workers.py` 的 QThread 中，**工作线程永不触碰控件**，帧经信号 emit 到主线程绘制；参数更新走 `pipeline.set_params()`（内部加锁）。
- 一期 `legacy/beautycam_v1.py`（Tkinter）只作参考不再维护，勿在 legacy 上开发。

## 已知坑（改代码前必读）

1. **mediapipe 必须锁 0.10.x（当前 0.10.21）**：1.0.x 移除 `mp.solutions`，且其 Tasks 检测类图在本机（macOS darwin 27）Metal 服务上硬崩溃（SIGABRT，Python 捕不住）；0.10.21 用 Tasks API + 显式 CPU 委托全部正常。此配置固定在 `core/infer.py`，不要改动委托设置。P0-1 冒烟结论详见 `core/infer.py` 模块头注释与 README。
2. （历史）一期 `legacy/beautycam_v1.py` 的 `v_sign_frames`/`smile_start_time` 缺 `global`、`open_camera()` 双线程两个 bug，v2 架构（状态封装在 `AutoCaptureState`、单 worker 实例）已根治；改拍照逻辑请改 `core/gestures.py` 与 `gui/workers.py`，不要回头修 legacy。
3. `photos/`、`outputs/` 是运行时输出目录，入库时忽略。
4. `课程答辩PPT`（141MB）被 .gitignore 排除，超过 GitHub 100MB 限制，不要 `git add -f`。
5. `models/`（MediaPipe `.task`/`.tflite` 与 ONNX 权重）不入库，由 `scripts/download_models.py` 拉取；也不要把权重文件放在 `models/` 以外的地方。部分官方下载地址已 404，脚本内是实测可用清单。
6. numpy 与 mediapipe/opencv 版本联动（mediapipe 0.10.x 需要 numpy<2，opencv 需随之锁 4.11），升级任何一个前先跑 `python -m unittest discover tests`。
7. **⚠️ 两个分割模型的类别编码与置信图极性恰好相反**（二元 `cat==0` 是人、`conf[0]` 是前景概率；多分类 `cat!=0` 是人、`conf[0]` 是背景概率）。搞反**不抛异常**，只静默产出反转掩膜（人景对调）或全幅掩膜（背景替换毫无效果）。约定声明在 `core/infer.py` 的 `SEGMENTER_SPECS`，改这段务必跑 `tests/test_infer.py::TestSegmenterSemantics`（用"图像四角必为背景"这种与掩膜语义无关的探针，形状/dtype 断言抓不住反转）。
8. **分割默认用二元模型（13.2ms/帧），不要图省事改用多分类（155ms/帧）**。多分类仅用于离线出"质量 vs 速度"对比图；若确要用，`infer_interval` 必须跟着调到 4（GUI 模型下拉会自动同步）。
9. **不要依赖 `cv2.ximgproc`**（macOS 的 opencv-python 不含 contrib）。guided filter 已自实现于 `core/effects/segment.py`，与 ximgproc 数值一致到 1e-5。
10. **不要用 `cv2.imread`/`cv2.imwrite` 处理非 ASCII 路径**：Windows 上 `imread` 静默返回 `None`、`imwrite` 静默写到乱码文件名，都不报错。统一走 `core/effects/segment.py::load_image` 与 `scripts/run_pipeline.py::write_image`。
11. 带跨帧状态的 Effect（目前只有 `SegmentEffect`）必须实现 `reset_temporal()`，并在 `process` 里对"帧尺寸变化"做处理 —— 切换采集源（相机 720p → 视频文件 480p）时同一实例会复用，残留的旧尺寸数组会让 `cv2` 直接抛异常。`Pipeline.reset_temporal()` 已在 worker 启动与逐张图片批跑前调用。

## Git 约定

- 分支：`main` + 按阶段建功能分支（如 `feat/phase0-skeleton`、`feat/hdr`），完成自测后合回 `main`。
- 提交信息用中文、一行式，说清「改了什么 + 为什么」。
- 不提交 `.venv/`、`photos/`、`models/`、`*.pptx`、`.DS_Store`、`__pycache__/`。

## 验证方式

- 单测：`python -m unittest discover tests`（109 个用例：美颜纯函数 / V 手势 / 笑脸 / 状态机 / 管线 / 隔帧节流 / 采集源 / 推理引擎 / 分割模型语义 / 背景替换纯函数；不依赖摄像头与 GUI）。
- headless 批跑：`python scripts/run_pipeline.py --input assets/samples --output outputs/`（真图验证整条推理+效果链）。
- 虚拟背景出图自检：加 `--segment --seg-dump-alpha` 看 alpha 灰度图**人是不是白的**（这是识别掩膜反转最快的手段），加 `--seg-compare` 出边缘处理四档对比图。
- 涉及摄像头/GUI 的改动，请用户实机运行 `python -m gui.main_window` 确认。
