# AGENTS.md — AI 协作规范

本仓库是数字图像处理课程项目 BeautyCam（macOS arm64 本地开发，非服务器部署）。

## 语言与沟通

- 与用户交流、代码注释、文档一律使用**中文**。

## 环境与命令

- Python 版本：**3.12**（`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12`）。注意：PATH 中默认的 `python3` 来自 PlatformIO 的 penv（3.11），创建虚拟环境时务必显式指定 3.12 的解释器。
- 虚拟环境：`.venv/`（已 gitignore）。激活：`source .venv/bin/activate`。
- 安装依赖：`pip install -r requirements.txt`。改依赖时同步更新 `requirements.txt`（`pip freeze > requirements.txt` 或手动维护顶层依赖）。
- 运行程序：`python beautycam_v1.py`（需要摄像头权限 + 本地 GUI，无法在无头环境验证 UI，只能验证 import 与纯函数逻辑）。

## 代码约定

- **v2 架构（正在重构，见 `docs/PLAN.md`）**：`core/`（处理核心）与 `gui/`（PySide6 界面）严格分层——**`core/` 下任何模块禁止 import GUI 库**（PySide6/tkinter）；换脸等演示放 `demos/`；评测批跑走 `scripts/run_pipeline.py`（headless，不依赖 GUI）。
- 效果插件模式：每个效果是 `core/effects/` 下的一个模块，实现 Effect 基类（`name/enabled/params/process(frame, ctx)`）；每帧推理结果统一放在 `FrameContext` 里共享，**效果内部禁止重复起 MediaPipe/ONNX 会话**。
- 模块名小写下划线（如 `beauty.py`、`lowlight.py`）。
- OpenCV 处理链以 BGR 帧为主，送入 MediaPipe 前转 RGB，注意别搞反。
- MediaPipe / ONNX 会话统一在 `core/infer.py` 管理单例、创建开销大，不要在帧循环里重复创建；ONNX 会话创建后必须记录并打印实际 execution provider（防 CoreML 静默回退 CPU）。
- 阈值 / 强度等调参常量集中在模块顶部定义并附中文注释（现有风格：`V_SIGN_HOLD`、`SMILE_HOLD` 等）。
- GUI 线程模型（PySide6）：相机与推理在 `gui/workers.py` 的 QThread 中，**工作线程永不触碰控件**，帧经信号 emit 到主线程绘制；参数更新走 `pipeline.set_params()`（内部加锁）。
- 旧版 `beautycam_v1.py`（Tkinter）Phase 0 完成后移入 `legacy/`，只作参考不再维护。

## 已知坑（改代码前必读）

1. `update_camera()` 里 `v_sign_frames`、`smile_start_time` 缺 `global` 声明（详见 README「已知问题」），修任何拍照触发逻辑前先确认这个问题是否已被修复。
2. `open_camera()` 启动了两次 `update_camera` 线程。
3. `photos/` 是运行时输出目录，入库时忽略。
4. `课程答辩PPT`（141MB）被 .gitignore 排除，超过 GitHub 100MB 限制，不要 `git add -f`。
5. `models/`（ONNX 权重）不入库，由 `scripts/download_models.py` 拉取；也不要把权重文件放在 `models/` 以外的地方。

## Git 约定

- 分支：`main` + 按阶段建功能分支（如 `feat/phase0-skeleton`、`feat/hdr`），完成自测后合回 `main`。
- 提交信息用中文、一行式，说清「改了什么 + 为什么」。
- 不提交 `.venv/`、`photos/`、`models/`、`*.pptx`、`.DS_Store`、`__pycache__/`。

## 验证方式

- 无自动化测试。最低验证：`python -c "import beautycam_v1"` 会直接起 GUI（不适用），改为逐函数验证纯图像处理函数，或写 `tests/` 下的临时脚本用静态图跑 `beautify()`、`is_v_sign()` 等。
- 涉及摄像头的改动，请用户实机运行确认。
