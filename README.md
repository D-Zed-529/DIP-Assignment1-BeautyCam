# DIP-26 · BeautyCam 智能美颜与动作识别相机

数字图像处理（DIP）课程小组项目：基于 **Python + OpenCV + MediaPipe** 的桌面端实时美颜相机，支持手势 / 笑脸触发自动拍照。

## 已实现功能

| 模块 | 说明 |
|------|------|
| 视频采集 | OpenCV 打开摄像头，1280×720，镜像显示，弱光自动增强（亮度阈值 60，alpha 提亮 + YCrCb 直方图均衡化） |
| 人脸 / 手部检测 | MediaPipe FaceMesh（468 点，最多 5 张脸）、Hands（21 点）、FaceDetection，实例全局复用 |
| 美颜（可开关） | ① 双边滤波磨皮（与原图 6:4 融合）② LAB 空间美白（L+15，肤色掩膜 ∧ 人脸框 ∧ FaceMesh 轮廓三层限定）③ 瘦脸（下颌关键点向脸中心收缩，强度 0.04）④ 大眼（局部液化放大，默认关闭）⑤ 中值去噪 + 锐化防糊 |
| 手势拍照 | 剪刀手判定：食指 + 中指伸直、其余弯曲、两指夹角 15°–65°；持续约 10 帧自动拍照，2 秒冷却 |
| 笑脸拍照 | 嘴部关键点 13/14 垂直距离 > 0.012 且持续 0.5 秒自动拍照 |
| GUI | Tkinter：双缓冲视频标签（防闪烁）、最近 4 张照片预览（点击放大）、打开相机 / 拍照 / 相册 / 退出 / 美颜 / 大眼按钮、快门提示音 |

照片按 `photos/auto_时间戳.jpg` / `photos/manual_时间戳.jpg` 保存。

## 项目结构

```
.
├── beautycam_v1.py   # 一期主程序（采集 → 检测 → 美颜 → 触发拍照 → GUI，重构后移入 legacy/）
├── 课程答辩PPT           # 一期答辩 PPT（141MB，不入库，见 .gitignore）
├── TODO.md            # 二期详细任务清单（分阶段 + 人日估算）
├── docs/
│   ├── PLAN.md        # 二期开发计划（架构 / 技术栈 / 阶段验收 / 风险）
│   └── research-notes.md  # 选型调研
├── AGENTS.md          # AI 协作规范
└── photos/            # 运行时照片输出（不入库）
```

## 环境与运行

macOS（arm64）+ Python 3.12：

```bash
# 创建并激活虚拟环境
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行
python beautycam_v1.py
```

依赖：`opencv-python`、`mediapipe`、`numpy`、`pillow`（见 `requirements.txt`）。

> 硬件部分（STM32F103 + LED 指示灯联动）的代码不在本仓库，PPT 中的相关内容为另一条交付线。

## 已知问题

- `update_camera()` 中 `v_sign_frames`、`smile_start_time` 存在赋值但未列入 `global` 声明，检测到手势 / 笑脸时会触发 `UnboundLocalError`（被 try/except 吞掉），表现为自动拍照可能失效。
- `open_camera()` 中相机更新线程被启动了两次。

## 二期计划（已定稿）

v2 定位为**多效果实时相机系统**，详见 [docs/PLAN.md](docs/PLAN.md) 与 [TODO.md](TODO.md)：

- **自动 HDR 拍照**：gamma 模拟包围曝光 + Mertens 融合（经典线），可选单帧深度 HDR 对比线
- **低光增强**：SCI / Zero-DCE++（ONNX，CoreML EP）替换现有启发式增强
- **人像虚化 / 背景替换**：自拍分割起步，进阶深度渐进虚化
- **实时美颜**：一期功能迁移并参数化
- **换脸（演示级）**：Delaunay 剖分 + 泊松融合，主打过程可视化；仅限本人/授权/动漫形象
- **GPU 加速**：ONNX Runtime CoreML EP + 性能对比基准
- **GUI 全面重写**：Tkinter → PySide6（暗色主题、效果控制面板、信号槽线程模型）

技术栈决策与架构设计（core/gui 分层、效果链、QThread 线程模型）见 PLAN §2–§3。
