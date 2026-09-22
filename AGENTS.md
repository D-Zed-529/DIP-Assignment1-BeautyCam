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

- 单文件脚本风格（`beautycam_v1.py`），若拆分模块，保持模块名小写下划线（如 `beauty.py`、`gestures.py`）。
- OpenCV 处理链以 BGR 帧为主，送入 MediaPipe 前转 RGB，注意别搞反。
- MediaPipe solution 实例（`face_mesh` / `hands` / `face_detect`）是全局单例、创建开销大，不要在帧循环里重复创建。
- 阈值 / 强度等调参常量集中在文件顶部定义并附中文注释（现有风格：`V_SIGN_HOLD`、`SMILE_HOLD` 等）。
- GUI 是 Tkinter 主线程 + 相机子线程：子线程不能直接碰 Tk 控件，必须走 `root.after(0, ...)` 回调（参考 `update_camera` 末尾的用法）。

## 已知坑（改代码前必读）

1. `update_camera()` 里 `v_sign_frames`、`smile_start_time` 缺 `global` 声明（详见 README「已知问题」），修任何拍照触发逻辑前先确认这个问题是否已被修复。
2. `open_camera()` 启动了两次 `update_camera` 线程。
3. `photos/` 是运行时输出目录，入库时忽略。
4. `课程答辩PPT`（141MB）被 .gitignore 排除，超过 GitHub 100MB 限制，不要 `git add -f`。

## Git 约定

- 分支：`main`。提交信息用中文、一行式，说清「改了什么 + 为什么」。
- 不提交 `.venv/`、`photos/`、`*.pptx`、`.DS_Store`。

## 验证方式

- 无自动化测试。最低验证：`python -c "import beautycam_v1"` 会直接起 GUI（不适用），改为逐函数验证纯图像处理函数，或写 `tests/` 下的临时脚本用静态图跑 `beautify()`、`is_v_sign()` 等。
- 涉及摄像头的改动，请用户实机运行确认。
