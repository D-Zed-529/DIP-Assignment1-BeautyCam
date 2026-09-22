"""效果插件：每个效果一个模块一个类（见 core/pipeline.Effect）。"""

from .beauty import BeautyEffect
from .lowlight import LowLightEffect

__all__ = ["BeautyEffect", "LowLightEffect"]
