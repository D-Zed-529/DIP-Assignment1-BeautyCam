"""人像虚化 / 背景替换（Phase 3）—— 腾讯会议式虚拟背景。

三档模式共用同一套人像掩膜：`blur` 背景虚化 / `image` 换背景图 / `color` 纯色。

## 为什么不能直接把模型的置信图当 alpha 用

P3-0 在 720p 样本上实测（多个模型的背景区 alpha 统计）：

    模型              背景区 alpha      合成后新背景与原背景的残留偏差
    selfie_multiclass ~0.084（恒定）  8.2 / 255   ← 新背景均匀发灰
    selfie_segmenter  0.000          0.0         ← 已经干净

多分类模型的背景置信度有系统性偏置（背景区 conf[0]≈0.916 而非 ~1.0），
于是 `1 - conf[0]` 在背景区留下一个**均匀的 8.4% 鬼影地板**：新背景只以约
92% 的不透明度叠上去，整片背景发灰、颜色不实。它不是"边缘发虚"那种糊，
而是全局的透明度偏差 —— 所以肉眼扫一眼容易漏掉，但和纯色背景一对比就明显。

**对比度拉伸**（`matte_contrast`）正是治这个的：把 [0.24, 0.76] 线性拉满到
[0,1]，0.084 直接被压到 0，偏差 8.2 → 0.0。二元模型本来就没有这个地板
（背景区就是 0），所以它几乎不需要拉伸 —— 但保留该旋钮，多分类才有得救。

`matte_contrast=0` 即"朴素置信图"基线，GUI/CLI 里切一下就能看到上面这组对比。

第二步是 **guided filter 边缘精修**（`refine`，He et al. ECCV 2010）：以当前帧
灰度为引导，把掩膜过渡带吸附到图像真实边缘上。256×256 的模型输出上采样到 720p
后边缘本就有约 10px 的模糊（实测掩膜边界与图像边界的平均偏移 9.6px），精修把
最大梯度提升约 4 倍，边缘明显更实；它同时还修正"alpha 来自美颜之前、而美颜
瘦脸已挪动下颌"带来的潜在错位。

## 性能

分割推理在 core/infer.py（二元模型 13.2ms/帧，可每帧跑）；本模块的帧操作实测
约 20ms（拉伸 ~2 + EMA ~2 + 精修 ~6 + 背景模糊 ~5 + 混合 ~5），见 PLAN §4 验收线。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..context import FrameContext
from ..pipeline import Effect, NEED_SEGMENTATION

# ------- 模式（GUI 下拉与此一一对应） -------
MODE_BLUR, MODE_IMAGE, MODE_COLOR = "blur", "image", "color"
MODES = (MODE_BLUR, MODE_IMAGE, MODE_COLOR)

# 背景图库目录与可识别的图片扩展名
BACKGROUNDS_DIR = (Path(__file__).resolve().parent.parent.parent
                   / "assets" / "backgrounds")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ------- 调参常量 -------
# 运动量在缩略图上估计（帧间平均绝对差）。参考尺度经实测标定：静止场景
# ~0.002，人物正常说话/晃动 ~0.01~0.03，快速挥手/入画出画 >0.05。
MOTION_THUMB_W, MOTION_THUMB_H = 80, 45
MOTION_REF = 0.03        # 运动量参考尺度，达到它即认为"大运动"（不再信历史）
MOTION_RELIEF = 0.9      # 大运动时时域平滑权重最多削减到这个比例（防拖影）
REFINE_EPS = 1e-4        # guided filter 正则项（值小则更贴近引导图边缘）
BLUR_SIGMA_MAX = 20.0    # 虚化强度=1.0 时的高斯 sigma
BLUR_DOWNSCALE_AT = 4.0  # sigma 超过此值就改用"降采样→模糊→升采样"加速
DEFAULT_BG_COLOR = "#3C6E71"   # 纯色背景默认值（低饱和青灰，演示观感中性）


# ------- 纯函数（可单测，不依赖状态） -------

def validate_mode(mode: str) -> str:
    """校验背景模式取值（GUI 下拉/CLI 之外的手写参数容易拼错）。"""
    if mode not in MODES:
        raise ValueError(f"未知背景模式：{mode!r}，应为 {MODES} 之一")
    return mode


def parse_hex_color(text: str,
                    default: tuple[int, int, int] = (113, 110, 60)
                    ) -> tuple[int, int, int]:
    """'#RRGGBB' -> BGR 三元组；非法输入回退 default（注意 default 是 BGR）。"""
    s = str(text).strip().lstrip("#")
    if len(s) != 6:
        return default
    try:
        r, g, b = (int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return default
    return (b, g, r)


def list_backgrounds() -> list[str]:
    """内置背景图库（assets/backgrounds）里的图片绝对路径，按文件名排序。

    由 scripts/make_backgrounds.py 程序化生成（无版权风险）。过滤掉非图片文件，
    因此目录里放 README.md / .gitkeep 也不会污染 GUI 图库列表。
    """
    if not BACKGROUNDS_DIR.is_dir():
        return []
    return sorted(
        str(p) for p in BACKGROUNDS_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTS)


def load_image(path: str) -> Optional[np.ndarray]:
    """读取背景图（BGR）。读不到返回 None，**绝不抛异常**。

    不能用 cv2.imread：它在 Windows 上遇到非 ASCII 路径（本项目内置背景图就是
    中文名）会**静默返回 None**，表现为"图库里的图全都选不中"。统一走
    np.fromfile + cv2.imdecode，中文路径正常。
    """
    if not path:
        return None
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def cover_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """等比缩放并居中裁剪到 (w, h)，等价 CSS `background-size: cover`。"""
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return np.zeros((h, w, 3), np.uint8)
    scale = max(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    x, y = (nw - w) // 2, (nh - h) // 2
    return np.ascontiguousarray(resized[y:y + h, x:x + w])


def sharpen_matte(alpha: np.ndarray, contrast: float) -> np.ndarray:
    """matte 对比度拉伸：把 [(0.5-w/2), (0.5+w/2)] 线性映射到 [0, 1]。

    contrast=0 -> 恒等（朴素置信图基线，供对比实验）；越大过渡带越窄。
    见模块头的实测数据：多分类模型的置信图不做这一步会得到 0% 背景纯度。
    """
    c = float(np.clip(contrast, 0.0, 1.0))
    if c <= 0.0:
        return alpha
    width = 1.0 - 0.8 * c          # c=0 -> 1.0（不拉伸）；c=1 -> 0.2（最激进）
    lo = 0.5 - 0.5 * width
    # 全分辨率 float32 上逐元素造临时数组很贵，全部就地运算（720p 实测 ~1.8ms）
    out = alpha - lo
    out *= 1.0 / width
    np.clip(out, 0.0, 1.0, out=out)
    return out


def ema_matte(new: np.ndarray, prev: Optional[np.ndarray], smooth: float,
              motion_relief: float = MOTION_RELIEF) -> np.ndarray:
    """运动自适应时域 EMA：静止时重平滑（防抖动），运动时减轻（防拖影）。

    new/prev 为前景概率 float32 (h, w)。smooth=0 表示不做时域平滑。
    """
    if prev is None or prev.shape != new.shape or smooth <= 0.0:
        return new
    # 运动量在缩略图上估计：720p 上逐元素求绝对值差要 2~3ms，缩略图只要 ~0.3ms，
    # 而整体运动量这种统计特征不需要全分辨率精度
    motion = min(_motion_level(new, prev) / MOTION_REF, 1.0)
    w = float(np.clip(smooth, 0.0, 1.0)) * (1.0 - motion_relief * motion)
    return cv2.addWeighted(prev, w, new, 1.0 - w, 0.0)   # cv2 的 SIMD 版 EMA


def _motion_level(new: np.ndarray, prev: np.ndarray) -> float:
    """缩略图上的平均绝对差（0~1），作为帧间运动量的代理指标。"""
    tiny = (MOTION_THUMB_W, MOTION_THUMB_H)
    a = cv2.resize(new, tiny, interpolation=cv2.INTER_AREA)
    b = cv2.resize(prev, tiny, interpolation=cv2.INTER_AREA)
    return float(cv2.mean(cv2.absdiff(a, b))[0])


def box_filter(x: np.ndarray, radius: int) -> np.ndarray:
    """半径为 radius 的均值滤波（guided filter 的基本算子，O(1)/像素）。"""
    k = 2 * radius + 1
    return cv2.boxFilter(x, -1, (k, k), normalize=True,
                         borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int,
                  eps: float = REFINE_EPS) -> np.ndarray:
    """引导滤波（He et al., Guided Image Filtering, ECCV 2010），边缘保持平滑。

    以 guide 的边缘为依据修正 src：把低分辨率的粗糙掩膜吸附到图像真实边缘上。
    自实现（仅用 cv2.boxFilter）而非用 cv2.ximgproc.guidedFilter —— opencv-python
    在 macOS 上不含 contrib 模块；本实现与 ximgproc 版本数值一致到 1e-5。
    """
    mean_g = box_filter(guide, radius)
    mean_s = box_filter(src, radius)
    cov = box_filter(guide * src, radius) - mean_g * mean_s
    var = box_filter(guide * guide, radius) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_s - a * mean_g
    return box_filter(a, radius) * guide + box_filter(b, radius)


def refine_matte(alpha: np.ndarray, frame: np.ndarray,
                 radius: int) -> np.ndarray:
    """以当前帧灰度为引导图精修掩膜（半分辨率计算后上采样，约 7ms/720p）。"""
    h, w = alpha.shape[:2]
    half = (max(w // 2, 1), max(h // 2, 1))
    # 先把帧缩到半分辨率再转灰度：全分辨率做 BGR2GRAY + /255 要 ~2ms，
    # 半分辨率只要 ~0.4ms，而引导图本来就要降采样
    small = cv2.resize(frame, half, interpolation=cv2.INTER_AREA)
    guide = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    guide *= 1.0 / 255.0
    a_small = cv2.resize(alpha, half, interpolation=cv2.INTER_AREA)
    out = guided_filter(guide, a_small, max(1, int(radius)))
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def soften_matte(alpha: np.ndarray, radius: float) -> np.ndarray:
    """羽化：不做精修时的退路（对掩膜做小半径高斯模糊）。"""
    r = float(radius)
    if r <= 0:
        return alpha
    k = int(r) * 2 + 1
    return cv2.GaussianBlur(alpha, (k, k), 0)


def shift_edges(alpha: np.ndarray, px: int) -> np.ndarray:
    """掩膜边界收缩（px>0，吃掉背景镶边）/ 膨胀（px<0）。"""
    p = int(px)
    if p == 0:
        return alpha
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                 (2 * abs(p) + 1, 2 * abs(p) + 1))
    return cv2.erode(alpha, k) if p > 0 else cv2.dilate(alpha, k)


def blend_over(frame: np.ndarray, bg: np.ndarray,
               alpha: np.ndarray) -> np.ndarray:
    """out = frame * alpha + bg * (1 - alpha)。

    用 cv2.blendLinear（在 imgproc 里，非 contrib，各平台都有）：720p 实测
    0.79ms，比 multiply+add 快 4 倍、比 numpy float32 快 40 倍。权重必须是
    单通道 float32；a≡0 / a≡1 时逐像素精确（合成边界情形单测靠这一点）。
    """
    a = alpha if alpha.dtype == np.float32 else alpha.astype(np.float32) / 255.0
    return cv2.blendLinear(frame, bg, a, 1.0 - a)


# ------- 效果 -------

class SegmentEffect(Effect):
    """人像掩膜 + 背景替换（模糊 / 图片 / 纯色）。

    本效果是第一个带跨帧状态的 Effect：持有上一帧的 alpha 与背景图缓存，
    供时域平滑与隔帧复用。`process` 只在工作线程被调用，因此状态无需加锁；
    GUI 线程只经 `set_params`/`set_enabled` 交互（基类已加锁）。
    """

    name = "segment"
    needs = frozenset({NEED_SEGMENTATION})

    @staticmethod
    def default_params() -> dict:
        return {
            "mode": MODE_BLUR,        # blur 虚化 / image 换图 / color 纯色
            "strength": 0.6,          # 虚化强度 0~1（映射到高斯 sigma）
            "bg_path": "",            # 背景图路径（mode=image）
            "bg_color": DEFAULT_BG_COLOR,   # 纯色背景（mode=color）
            "matte_contrast": 0.3,    # matte 对比度拉伸 0~1；0 = 朴素基线（对照档）
            "refine": True,           # guided filter 边缘精修
            "feather": 3.0,           # 羽化 / 精修半径（px）
            "smooth": 0.7,            # 时域平滑系数 0~1
            "edge_shift": 0,          # 掩膜收缩(+)/膨胀(-) px，用于吃背景镶边
            "infer_interval": 1,      # 分割推理间隔帧数（选多分类模型时置 4）
            "min_person_ratio": 0.02,  # 人像占比低于此值则原样输出，防"人消失"
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        validate_mode(self._params["mode"])
        self._prev_alpha: Optional[np.ndarray] = None
        self._bg_key: Optional[tuple] = None
        self._bg_cache: Optional[np.ndarray] = None
        self._last_alpha: Optional[np.ndarray] = None

    def set_params(self, **kwargs) -> None:
        """额外校验 mode 取值：非法模式若被静默接受，会悄悄地按纯色处理。"""
        if "mode" in kwargs:
            validate_mode(kwargs["mode"])
        super().set_params(**kwargs)

    @property
    def debug_alpha(self) -> Optional[np.ndarray]:
        """最近一帧实际用于合成的 alpha（全分辨率 float32），只读。

        供 headless 的 --seg-dump-alpha 与 GUI 排障使用：把它存成灰度图看一眼
        "人是不是白的"，是识别掩膜整体反转最直接的手段。不进热路径。
        """
        return self._last_alpha

    # ------- 推理降载声明 -------

    def inference_interval(self, need: str) -> int:
        """分割推理的隔帧间隔（多分类模型 155ms/帧，必须隔帧跑）。"""
        if need == NEED_SEGMENTATION:
            return max(1, int(self._p()["infer_interval"]))
        return 1

    # ------- 开关（启用时清掉陈旧状态，避免禁用再启用后闪一帧旧掩膜） -------

    def set_enabled(self, on: bool) -> None:
        super().set_enabled(on)
        if on:
            self.reset_temporal()

    def reset_temporal(self) -> None:
        """清空跨帧状态（尺寸变化、切换采集源、重新启用时调用）。"""
        self._prev_alpha = None
        self._bg_key = None
        self._bg_cache = None
        self._last_alpha = None

    # ------- 帧处理 -------

    def process(self, frame: np.ndarray, ctx: FrameContext) -> np.ndarray:
        if not self.enabled:      # 防御直接调用；Pipeline 本身也会跳过
            return frame
        p = self._p()
        h, w = frame.shape[:2]

        alpha = self._take_alpha(ctx, frame, h, w, p)
        if alpha is None:
            return frame          # 还没有任何掩膜（未启用分割 / 模型未就绪）

        # 人离开画面时整帧都会变成背景（"人消失了"）——占比过低就原样输出
        if float(alpha.mean()) < float(p["min_person_ratio"]):
            return frame

        if p["edge_shift"]:
            alpha = shift_edges(alpha, int(p["edge_shift"]))
        self._last_alpha = alpha

        return blend_over(frame, self._background(frame, p), alpha)

    # ------- alpha 状态机 -------

    def _take_alpha(self, ctx: FrameContext, frame: np.ndarray, h: int, w: int,
                    p: dict) -> Optional[np.ndarray]:
        """取本帧的前景概率。

        有新的推理结果就推进时域状态；隔帧推理的中间帧（ctx 为空）复用上一帧
        的 alpha —— "本帧没有新数据"与"从来没有过 alpha"由 _prev_alpha 是否为
        None 区分，engine 侧因此可以完全无状态。
        """
        raw = ctx.person_alpha
        if raw is None and ctx.person_mask is not None:
            raw = ctx.person_mask.astype(np.float32) / 255.0

        if raw is not None:
            raw = np.asarray(raw, dtype=np.float32)
            if raw.shape[:2] != (h, w):
                # 采集源尺寸变化（如切换视频文件）：重采样并断开时域历史
                raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
                self._prev_alpha = None
            raw = sharpen_matte(raw, p["matte_contrast"])
            raw = ema_matte(raw, self._prev_alpha, float(p["smooth"]))
            self._prev_alpha = raw
        elif self._prev_alpha is not None and self._prev_alpha.shape[:2] != (h, w):
            self._prev_alpha = None

        alpha = self._prev_alpha
        if alpha is None:
            return None

        if p["refine"]:
            alpha = refine_matte(alpha, frame, int(round(float(p["feather"]))))
        else:
            alpha = soften_matte(alpha, p["feather"])
        return np.clip(alpha, 0.0, 1.0)

    # ------- 背景生成（单条缓存，键含尺寸与参数） -------

    def _cached(self, key: tuple, factory) -> Optional[np.ndarray]:
        if key != self._bg_key:
            self._bg_key, self._bg_cache = key, factory()
        return self._bg_cache

    def _background(self, frame: np.ndarray, p: dict) -> np.ndarray:
        h, w = frame.shape[:2]
        mode = p["mode"]
        if mode == MODE_BLUR:
            return self._blurred(frame, float(p["strength"]))
        if mode == MODE_IMAGE:
            img = self._cached((str(p["bg_path"]), w, h),
                               lambda: self._load_image(str(p["bg_path"]), w, h))
            if img is not None:
                return img
            # 背景图缺失/读取失败 -> 退回纯色，避免整帧变黑
        return self._cached(("color", str(p["bg_color"]), w, h),
                            lambda: self._solid(str(p["bg_color"]), w, h))

    @staticmethod
    def _blurred(frame: np.ndarray, strength: float) -> np.ndarray:
        sigma = BLUR_SIGMA_MAX * float(np.clip(strength, 0.0, 1.0))
        if sigma < 0.5:
            return frame.copy()
        # 高斯核宽度随 sigma 增长，大半径直接算很慢：先降采样再模糊再升采样。
        # 模糊本身是低通，视觉等价但快得多。
        if sigma > BLUR_DOWNSCALE_AT:
            f = max(1, int(sigma / BLUR_DOWNSCALE_AT))
            small = cv2.resize(frame, None, fx=1.0 / f, fy=1.0 / f,
                               interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), sigma / f)
            return cv2.resize(small, (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(frame, (0, 0), sigma)

    @staticmethod
    def _load_image(path: str, w: int, h: int) -> Optional[np.ndarray]:
        img = load_image(path)
        return None if img is None else cover_resize(img, w, h)

    @staticmethod
    def _solid(color: str, w: int, h: int) -> np.ndarray:
        out = np.empty((h, w, 3), np.uint8)
        out[:] = parse_hex_color(color)
        return out
