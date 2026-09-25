# AGENTS.md — AI 协作规范

本仓库是数字图像处理课程项目 BeautyCam（**Windows 11 + RTX 3060 Laptop + CUDA 部署，
2026-09 从 macOS arm64 迁移**；推理链全 PyTorch，mediapipe 保留为回退后端）。

## 语言与沟通

- 与用户交流、代码注释、文档一律使用**中文**。

## 环境与命令

- Python 版本：**3.12**（`.venv/Scripts/python.exe`；系统 `python` 可能是 3.14，注意区分）。
- 虚拟环境：`.venv/`（已 gitignore）。激活：`.venv\Scripts\activate`。
- torch 必须 CUDA 轮子：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`。
- 安装依赖：`pip install -r requirements.txt`。改依赖时同步更新 `requirements.txt`。
- **首次部署三步**：`python scripts/download_models.py`（拉权重，含 gdown 下载 Retinexformer）→ `python scripts/convert_models.py`（tflite/onnx → models/torch/*.ts，一次性）→ `python -m unittest discover tests`。
- 运行程序：`python -m gui.main_window`（需摄像头 + 本地 GUI）。数值部署自检：`python scripts/calibrate_torch.py`（torch vs mediapipe 偏差报告）。
- 引擎后端选择：`core/infer.py::get_engine()` 自动（CUDA+TorchScript 就绪 → torch，否则 mediapipe）；`get_engine(prefer="mediapipe")` 可强制基准对照。

## 代码约定

- **v3 架构**：`core/`（处理核心）与 `gui/`（PySide6 界面）严格分层——**`core/` 下任何模块禁止 import GUI 库**；换脸等演示放 `demos/`；评测批跑走 `scripts/run_pipeline.py`。
- **推理引擎双后端**（接口对齐，`process(frame, faces=, hands=, segmentation=, depth=)`）：
  `TorchInferenceEngine`（`core/infer_torch.py`，全 GPU）与 `InferenceEngine`（mediapipe CPU）。
  **效果内部禁止重复起推理会话**；每帧结果统一进 `FrameContext`（含 `depth`）。
- 效果插件模式：Effect 基类（`name/needs/enabled/params/process`）；`needs` 可为 property（BokehEffect 按 use_matte 动态声明 NEED_SEGMENTATION）；跨帧状态效果必须实现 `reset_temporal()`；带循环状态的**引擎**也要实现 `reset_temporal()`（RVM）。
- 会话/模型懒加载 + 单例（`get_engine` / `get_lowlight_session(level, engine)` / `get_depth_session`）；torch 模型创建后打印实际后端（`backend_name` / `provider`），防静默回退。
- 模块名小写下划线；OpenCV 处理链以 BGR 帧为主，送模型前转 RGB。
- 阈值/强度调参常量集中模块顶部并附中文注释（现有风格：`V_SIGN_HOLD`、`PRESENCE_REL` 等）。
- GUI 线程模型（PySide6）：推理/效果在 `gui/workers.py` 的 QThread；**工作线程永不触碰控件**；参数走 `pipeline.set_params()`；分割模型切换走 `worker.request_segmenter_model`（会话重建只在工作线程做）。
- 一期 `legacy/beautycam_v1.py` 只作参考不再维护。

## 已知坑（改代码前必读）

1. **mediapipe 回退后端锁 0.10.x（当前 0.10.21）+ 显式 CPU 委托**（torch 后端不受此限；macOS Metal 崩溃历史见 git）。此配置固定在 `core/infer.py`。
2. （历史）一期 legacy 的 global 缺失 / 双线程 bug 已在 v2 根治；改拍照逻辑改 `core/gestures.py` 与 `gui/workers.py`。
3. `photos/`、`outputs/` 是运行时输出目录，入库时忽略。
4. `课程答辩PPT`（141MB）被 .gitignore 排除，不要 `git add -f`。
5. `models/` 不入库（含 `models/torch/`、`models/hf/`），由 `scripts/download_models.py` + `scripts/convert_models.py` 产出；Retinexformer 权重走 gdown（Google Drive 文件夹）。
6. numpy 与 mediapipe/opencv 版本联动（numpy<2，opencv 4.11），升级前先跑全量单测。
7. **分割模型类别编码与置信图极性按模型显式声明**（`SEGMENTER_SPECS`：二元 `cat==0` 是人；多分类 `cat!=0` 是人、`conf[0]` 是背景；RVM 直接出前景 alpha）。搞反**不抛异常**只静默反相。改这张表必跑 `tests/test_infer.py::TestSegmenterSemantics` 与 `tests/test_infer_torch.py::TestRvmSegmentation`（探针法：图像四角必为背景）。
8. 分割默认模型：torch 后端 = **RVM**（每帧），mediapipe 后端 = 二元；多分类仅离线对比（155ms/帧，CPU 口径）。GUI 下拉按后端过滤 torch-only 项。
9. 不要依赖 `cv2.ximgproc`（guided filter 已自实现于 `core/effects/segment.py` / `core/gpuops.py`）。
10. **不要用 `cv2.imread`/`cv2.imwrite` 处理非 ASCII 路径**：统一走 `np.fromfile`+`imdecode` / `imencode`+`write_bytes`（内置背景图就是中文文件名）。
11. 带跨帧状态的 Effect（Segment / LowLightDnn / Bokeh 的深度会话归一化）必须实现 `reset_temporal()` 并处理帧尺寸变化；**引擎侧 RVM 循环状态同理**（`TorchInferenceEngine.reset_temporal`，worker 启动与逐张批跑前调用）。
12. `findTransformECC` 的 warp 方向要取逆（`core/effects/hdr.py::align_to_reference`）。
13. 分块仿射 warp 两个坑见 `demos/faceswap/faceswap.py::warp_face`。
14. **竖版图直接 resize 到 16:9 会把脸压扁到 FaceMesh 检测不到**（torch 后端同样成立）："样本图喂检测器"一律等比缩放 + 补边（`scripts/bench.py::letterbox`、`tests/test_infer_torch.py::_sample_720p`）。
15. 基准/压力脚本禁止开放式计时循环（mediapipe VIDEO 按次分配不归还会 OOM）：固定调用次数 + gc + RSS 打印。
16. SCI 低光固定 512×512 输入、输出取张量 [1]；torch 版会话（`LowLightSessionTorch`）与 ONNX 版口径一致（推理域 RGB float）。
17. `createTonemapDrago` 对 0 值像素产出 NaN（抬底 + nan_to_num 双保险，`core/effects/hdr.py`）。
18. OpenCV 8bit LAB 与 sRGB 灰度非线性关系；掩膜区域统计口径见 `core/effects/autoenhance.py`。
19. **tflite→torch 不能用 tflite2onnx/onnx2torch**（Windows 实测：布局传播 IndexError、HARD_SWISH/自定义算子不支持、临时文件句柄独占）——自研转换器 `scripts/tflite_to_torch.py`（flatbuffer 直解；tflite 2.18 绑定的 API 坑：空 buffer `DataAsNumpy()` 返回 int、union 选项要 `Table.Init`、`Padding.SAME==0`、conv 权重 OHWI、自定义算子按 CustomCode 判定）。
20. **face_landmarker.task 模型口径**：landmarks 输入 256×256（非 192）、输出 478×3 在 **256 像素域**；blendshapes 输入 = 146 点子集（`BLENDSHAPE_LANDMARK_SUBSET`）的**像素坐标**，输出 52 维原始 0~1（mouthSmileL/R = 44/45），序 = mediapipe kBlendshapeNames。
21. **presence（=sigmoid(Identity_1)）有低清退化**（同一脸 480 宽 0.0076 vs 960 宽 0.985，幻影恒 ≈0）→ 检测/跟踪统一走自适应门限（绝对 0.5 或相对 0.4×pmax，`core/infer_torch.py::_process_faces`），改阈值先跑 `tests/test_infer_torch.py`。
22. **跟踪必须做检测/轨迹 IoU 去重**（>0.3 视为同一脸），否则检测刷新每 10 帧复制一份轨迹（曾实测 30 帧 1→4 递增）。
23. **RVM 官方 TorchScript**：fp16 版（resnet50）输入要 half 且在 3060 上比 fp32 慢 16 倍 → 默认 `models/torch/rvm.ts`（mobilenetv3 fp32）；换 fp16 文件时 `_segment_rvm` 会自动转 half。
24. **笔记本 GPU 空闲降频，短基准虚高 3~50 倍**：推理类基准必须先 `gpu_spin()` 拉时钟（`scripts/bench.py`）；首帧另有 ~15s 冷启动，评测一律先预热再计时。
25. mediapipe 引擎**不支持 depth**（请求时告警一次、效果透传）；`BokehEffect` 依赖 torch 后端 + DA-v2 权重。
26. **onnx2torch 转出的 SCI 与 ONNX 输出 MAE 0.15（数值不忠实）**：SCI 快速档**默认走 ONNX Runtime 会话**（v2 验证基线 22.1dB，CPU EP 3.6ms），torch 版只作 `get_lowlight_session(engine="torch")` 的实验选项；`eval_lowlight.py` 的 PSNR 是回归哨兵（掉了先查这条）。
27. **tflite 转换图禁止运行时 `.shape` 进 trace**（2026-09 性能重构核心坑）：trace 会把 `x.shape[i]` 记录成 `aten::size + NumToTensor + int()`，每次前向执行 = 一次 D2H 同步——face_landmarks 实测 141 次/帧 ≈ 19ms 纯同步开销（GPU 真实计算只有 ~2ms）。`tflite_to_torch.py` 已改为查**静态形状表**（`TfliteModule._tshape`，tflite 图声明），转换后 face_landmarks 21→5.2ms、face_detector 16.5→1.8ms。新增算子处理器时形状一律走 `self._sz(ti, x, d)`，注意：**x 若已被 permute 成 NCHW，其 shape[2]/[3] 对应 NHWC 声明的 dim 1/2**（曾写反过）。改动后必须重跑 calibrate + 全量单测。
28. **CUDA Graph 在本机（torch 2.14 / 3060 / WDDM）的边界**：tflite 逐算子图捕获会 `cudaErrorStreamCaptureInvalidated` 且**失败会毒化整个 CUDA 上下文**（同进程后续一切 CUDA 调用全挂），而它们本就是 GPU 计算受限、图回放无收益——**不要给 tflite 转换图包图**；DA-v2（11→8ms）与 Retinexformer（纯算力受限，图无益但无害）可捕获，见 `core/cudagraph.py`。另：RVM 的 `downsample_ratio` 形参是 float，**传 CUDA 张量会逐帧隐式 item() 同步**，必须传 Python float。
29. **效果链 GPU 融合**（`Pipeline(use_gpu=True)`，GUI/run_pipeline 在 torch 后端自动启用）：连续 `supports_gpu` 效果共享一次帧上传/下载、以 (1,3,H,W) uint8 BGR 张量直通（`Effect.process_gpu`，实现见 `core/effects/_torch_impl.py`）；抛 NotImplementedError 自动回退 CPU（SCI 的 ONNX 会话即此）。**FrameContext 的 person_alpha/person_mask/depth 支持张量/ numpy 双形态懒互转**——GPU 链读 `*_t`，CPU 消费方读 numpy 属性，整链不落回 CPU。CPU 路径仍是逐位基准（测试默认 `use_gpu=False`）。
30. **GPU 版图像算子的性能口径**（`gpuops.py` / `_torch_impl.py`）：小 kernel 链在 WDDM 下按"kernel 数量"计价。磨皮双边滤波 d=9 已改为邻域展开向量化，240×135 输入实测 10.3→1.45ms，额外消耗显存、展开张量超过 256MB 时回退逐位叠加；3×3 中值**不能**用 `stack+median`（36ms），用行列两段 `_med3` 可分离近似（~1.5ms，视觉等价）；LAB pow 往返 ~2.6ms。新增 GPU 算子先跑 `scripts/profile_stages.py` 看单段耗时再上。
31. **基准必须充分预热**：GPU 时钟爬升要 ~20 帧，冷态首测可虚低 2~3 倍（540p 全开曾测出 104ms，稳态实为 40ms）；单段微基准用 `tm()` 风格（预热 6 次 + 每次 synchronize）。`scripts/profile_stages.py` 是逐阶段分解工具（engine 内部各模型 / 各效果 GPU/CPU / 组合端到端）。
32. **深度虚化隔帧与批跑**：`BokehEffect.inference_interval(NEED_DEPTH)=3`（深度是低频信号），中间帧复用效果内缓存（`_depth_cache_t`）；`run_pipeline.py` 图片序列逐张 `reset_temporal`，帧序必须恒 0，否则独立图片会因隔帧拿不到深度。
33. **流畅优先模式**（GUI"拍照"面板复选，默认开）：worker 把预览链降到 540p 处理（显示本就是 960×540），**拍照时用最近一帧原始分辨率重跑一次管线**（一次性 ~0.5s 出全分辨率照片）；RVM/分割的时域状态在尺寸切换时自动作废重建，下一帧预览不受影响。
34. **GPU 颜色空间往返必须保色**：`gpuops.rgb_to_lab` 的 XYZ 须除以 D65 白点，`lab_to_rgb` 才能乘回；`rgb_to_ycrcb` 中色差乘 0.713/0.564，逆变换须除回。漏掉会使灰阶染色或低光画面去饱和，并在多效果叠加时放大色彩断层；回归测试见 `tests/test_gpuops_color.py`。
35. **GUI 自动拍照默认关闭**：同时开启 V 手势和笑脸会使每帧额外跑手部检测与表情网络，540p 样图整链稳态约 70.8ms/帧；仅美颜约 22.8ms/帧。worker 必须按各触发器分别声明 `NEED_HANDS` / `NEED_FACES` 和 blendshapes，不可用 `bool(triggers)` 一次性全开。
36. **实时采集与推理并行**：`LiveCamera` 独立线程持续读取设备，`read()` 只返回最新未消费帧，处理慢时丢过期帧以降低预览延迟；释放设备前先停止采集线程。Torch 人脸/手部关键点及标量结果打包一次 D2H，worker 整帧走 `torch.inference_mode()`。并发采集改善新鲜度，不能把计算受限的效果链 fps 直接翻倍。
37. **同帧独立模型并发**：`TorchInferenceEngine.process` 在 CUDA 上用常驻线程池与独立 stream 并发执行人脸、手、分割、深度中实际请求的模型；单模型帧直跑。上传 RGB 后子 stream 等待源 stream，返回前同步各 stream；RVM/跟踪/深度时域状态仍按帧顺序更新，勿将连续帧并发。540p 人脸+RVM 实测整入口约 12→9ms；9×9 磨皮滤波切 2/4 块虽无误差，却因额外调度从 1.4ms 增至 2.1/2.8ms，不启用切块。
37. **用显存换速度的边界**：`gpuops.bilateral_blur` 仅对较小的 CUDA 张量使用 `F.unfold` 全邻域向量化，计算前按展开张量估算显存；超过 256MB 或 CPU 路径保留逐位叠加。与参考最大误差 <1e-5 的回归测试见 `tests/test_gpuops_bilateral.py`。在 540p 预览口径，默认美颜样图整链约 21→14.5ms/帧。
38. **全开模式的 WDDM 长尾**：本机人脸+手+RVM+深度并发时，模型本身稳态约 20ms，但独立 CUDA stream 偶发 1.5~5s 停顿。GUI worker 对三路及以上推理需求改为同一 stream 串行提交；完整 Retinexformer+换脸预览样图 540p 在预热后约 104ms/帧、30 帧无秒级尖峰。Depth Anything 的 CUDA Graph 必须在提交并发任务前捕获，否则可能污染 CUDA 上下文。手部跟踪结果必须按 `MAX_HANDS` 封顶并拒绝非有限/越界关键点，否则假轨迹增殖让每帧前向次数失控。此处长尾只在当前 RTX 3060 Laptop + WDDM 组合验证，其他硬件要重新测。

## Git 约定

- 分支：`main` + 按阶段建功能分支，完成自测后合回 `main`。
- 提交信息用中文、一行式，说清「改了什么 + 为什么」。
- 不提交 `.venv/`、`photos/`、`outputs/`、`models/`、`*.pptx`、`__pycache__/`。

## 验证方式

- 单测：`python -m unittest discover tests`（251 个用例；不依赖摄像头与 GUI；CUDA / 各权重缺失的用例自动跳过；torch 引擎的跟踪稳定性、幻影拒检、RVM 探针语义、并发一致性、GPU 融合链与 ctx 双形态均有专测）。
- 数值校准：`python scripts/calibrate_torch.py`（torch vs mediapipe：landmark Δ、alpha MAE、blendshapes 对照；定版记录在 `core/infer_torch.py` 模块头。注意 face_lm 输出在 256 像素域、blendshapes 输入是像素坐标——脚本已按此口径比较）。
- 性能基准：`python scripts/bench.py --markdown-out outputs/bench.md`（GPU 时钟拉频口径）；逐阶段分解用 `python scripts/profile_stages.py`（推理/效果/组合分层计时，冷态数字无效、看预热后稳态）。
- 低光客观评测：`python scripts/eval_lowlight.py --save-compare`。
- headless 批跑：`python scripts/run_pipeline.py --input assets/samples --output outputs/`（真图验证整条推理+效果链；`--bokeh` / `--lowlight-engine retinex` / `--seg-model rvm` 为 v3 新参数）。
- 虚拟背景自检：`--seg-dump-alpha` 看 alpha 人是不是白的；`--seg-compare` 出边缘对比图。
- 涉及摄像头/GUI 的改动，请用户实机运行 `python -m gui.main_window` 确认。
