"""通用 TFLite → PyTorch 转换器（覆盖本项目 7 个 MediaPipe/自训 tflite 模型）。

为什么不用 tflite2onnx + onnx2torch 直转（2026-09 Windows 实测记录）：
  - tflite2onnx 0.4.1 在 face_landmarks / hand_detector 上布局传播
    IndexError；selfie 二元模型含 HARD_SWISH 与自定义算子
    Convolution2DTransposeBias 直接不支持；SUM 无映射；
  - onnx2torch 的 shape inference 在 Windows 上用 NamedTemporaryFile
    （句柄独占导致 PermissionError，需先做内存内 shape inference 绕过）。
  自写转换器一次实现覆盖全部算子（本项目模型共 22 种标准算子 + 1 个
  MediaPipe 自定义算子），且可对每个模型做「tflite 数值参照」校验。

设计：
  - 直接解析 flatbuffer（tflite 包），按算子流构建 torch.nn.Module 的
    forward（数据无关控制流，可 jit.trace）；
  - 全程 NHWC 执行（conv 处 permute 进出），权重一次性转 fp32；
  - fp16 图的 DEQUANTIZE 按恒等跳过（const 统一物化为 fp32）；
  - 自定义算子 Convolution2DTransposeBias（MediaPipe 分割上采样）：
    语义 = stride 2、kernel 3、SAME 风格填充的 conv_transpose2d + bias，
    由输入/输出静态形状推导填充，端到端对 mediapipe 输出校验。

用法（被 scripts/convert_models.py 调用）：
  from tflite_to_torch import convert_tflite_module
  module = convert_tflite_module("models/face_detector.tflite")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tflite

logger = logging.getLogger("tflite_to_torch")

_PAD_SAME = tflite.Padding.SAME       # 注意：tflite 2.18 枚举 SAME=0 / VALID=1
_PAD_VALID = tflite.Padding.VALID
_ACT_NONE, _ACT_RELU, _ACT_RELU_N1, _ACT_RELU6, _ACT_TANH = 0, 1, 2, 3, 4

_DTYPE_MAP = {
    tflite.TensorType.FLOAT32: np.float32,
    tflite.TensorType.FLOAT16: np.float16,
    tflite.TensorType.INT32: np.int32,
    tflite.TensorType.INT64: np.int64,
    tflite.TensorType.UINT8: np.uint8,
    tflite.TensorType.INT8: np.int8,
    tflite.TensorType.BOOL: np.bool_,
}


def _op_name(model, opcode_index: int) -> str:
    code = model.OperatorCodes(opcode_index).BuiltinCode()
    if code == 127:
        return f"CUSTOM:{model.OperatorCodes(opcode_index).CustomCode()}"
    for k in dir(tflite.BuiltinOperator):
        if k.isupper() and getattr(tflite.BuiltinOperator, k) == code:
            return k
    return f"BUILTIN_{code}"


def _opts(op, cls):
    """读取内建选项结构体（union 字段 → Table.Init 模式）。

    tflite 2.18 绑定里 Operator.BuiltinOptions() 返回 flatbuffers Table，
    需按具体选项类 Init 重建（无 BuiltinOptionsAsNumpy 便捷方法）。
    """
    if op.BuiltinOptionsType() == 0:
        return None
    tab = op.BuiltinOptions()
    if tab is None:
        return None
    inst = cls()
    inst.Init(tab.Bytes, tab.Pos)
    return inst


def _pad_same(in_size: int, k: int, stride: int, dilation: int) -> tuple[int, int]:
    """TF SAME 填充（上/下不对称分布，多的给下/右）。"""
    k_eff = (k - 1) * dilation + 1
    out = (in_size + stride - 1) // stride
    total = max((out - 1) * stride + k_eff - in_size, 0)
    return total // 2, total - total // 2


def _fused(x: torch.Tensor, act: int) -> torch.Tensor:
    if act == _ACT_NONE:
        return x
    if act == _ACT_RELU:
        return F.relu(x)
    if act == _ACT_RELU_N1:
        return x.clamp(-1.0, 1.0)
    if act == _ACT_RELU6:
        return x.clamp(0.0, 6.0)
    if act == _ACT_TANH:
        return torch.tanh(x)
    raise ValueError(f"未知 fused activation: {act}")


class TfliteModule(nn.Module):
    """由 tflite 图构建的顺序执行模块（NHWC 域）。

    const 张量物化为 fp32 buffer；forward 沿算子序执行（tflite 图为
    拓扑序），中间张量按 index 存于 dict —— 数据无关，可 trace。
    """

    def __init__(self, model_path: str | Path):
        super().__init__()
        buf = Path(model_path).read_bytes()
        self.model = tflite.Model.GetRootAs(buf, 0)
        assert self.model.SubgraphsLength() == 1, "仅支持单子图模型"
        self.graph = self.model.Subgraphs(0)
        g = self.graph
        self.input_indices = list(g.InputsAsNumpy())
        self.output_indices = list(g.OutputsAsNumpy())

        # ---- 物化 const 张量（fp32）----
        self._consts: dict[int, torch.Tensor] = {}
        for ti in range(g.TensorsLength()):
            t = g.Tensors(ti)
            bid = t.Buffer()
            if bid == 0 or bid is None:
                continue
            data = self.model.Buffers(bid).DataAsNumpy()
            # 空 buffer 时 tflite 绑定返回 int 0（而非空数组）
            if data is None or isinstance(data, (int, float)) or len(data) == 0:
                continue
            dtype = _DTYPE_MAP.get(t.Type())
            if dtype is None:
                continue
            arr = np.frombuffer(data.tobytes(), dtype=dtype)
            shape = t.ShapeAsNumpy()
            # 标量/未知形状张量返回 int 而非数组
            if isinstance(shape, int):
                shape = np.array([shape])
            if shape is not None and len(shape) and int(np.prod(shape)) == arr.size:
                arr = arr.reshape([int(s) for s in shape])
            if dtype in (np.float16, np.float32):
                arr = arr.astype(np.float32)
            self._consts[ti] = torch.from_numpy(np.ascontiguousarray(arr))
        # buffer 注册（trace 时随模块迁移设备）
        for ti, v in self._consts.items():
            self.register_buffer(f"c{ti}", v)

        # ---- 静态形状表（tflite 图声明）----
        # trace 时读取运行时 .shape 会被记录成 aten::size + NumToTensor +
        # int()（每次前向 D2H 同步，face_landmarks 实测 141 次/帧 ≈ 19ms
        # 纯同步开销）。所有 op 处理器改查本表 → 形状算术全部退化为常量。
        self._tshape: dict[int, tuple[int, ...]] = {}
        for ti in range(g.TensorsLength()):
            ts = g.Tensors(ti).ShapeAsNumpy()
            if ts is None or isinstance(ts, (int, float)) or len(ts) == 0:
                continue
            dims = [int(v) for v in ts]
            if all(d > 0 for d in dims):
                self._tshape[ti] = tuple(dims)

        # ---- 编译算子表 ----
        self.ops: list[tuple] = []
        for oi in range(g.OperatorsLength()):
            op = g.Operators(oi)
            code = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            custom = self.model.OperatorCodes(op.OpcodeIndex()).CustomCode()
            ins = [int(i) for i in op.InputsAsNumpy()]
            outs = [int(i) for i in op.OutputsAsNumpy()]
            self.ops.append((code, custom, ins, outs, op))

    # ---- forward ----

    def forward(self, *inputs):
        t: dict[int, torch.Tensor] = {}
        for idx, x in zip(self.input_indices, inputs):
            t[int(idx)] = x
        for ti, v in self._consts.items():
            t[ti] = v
        for code, custom, ins, outs, op in self.ops:
            self._exec(t, code, custom, ins, outs, op)
        if len(self.output_indices) == 1:
            return t[int(self.output_indices[0])]
        return tuple(t[int(i)] for i in self.output_indices)

    def _exec(self, t, code, custom, ins, outs, op) -> None:
        name = None
        for k in dir(tflite.BuiltinOperator):
            if k.isupper() and getattr(tflite.BuiltinOperator, k) == code:
                name = k
                break

        # 自定义算子以 CustomCode 判定（builtin code 在 tflite 2.18 里
        # 不再是固定 127，CustomCode() 非 None 即自定义）
        if custom is not None:
            if custom == b"Convolution2DTransposeBias":
                self._conv2d_transpose_bias(t, ins, outs)
                return
            raise NotImplementedError(f"未知自定义算子: {custom}")

        # 枚举名 CONV_2D 与方法名 _op_conv2d 的命名差异：先按原名小写找，
        # 找不到再按去掉下划线的形式找（STRIDED_SLICE→_op_stridedslice、
        # HARD_SWISH→_op_hardswish、RESIZE_BILINEAR→_op_resizebilinear 等）
        lower = name.lower()
        handler = getattr(self, f"_op_{lower}", None) \
            or getattr(self, f"_op_{lower.replace('_', '')}", None)
        if handler is None:
            raise NotImplementedError(f"未实现算子: {name}")
        handler(t, ins, outs, op)

    # ---- 逐算子实现 ----

    def _sz(self, ti, x, d: int) -> int:
        """张量 ti 第 d 维的静态尺寸；无声明/动态维时退回运行时值。"""
        s = self._tshape.get(int(ti))
        if s is not None and len(s) > d and s[d] > 0:
            return int(s[d])
        return x.shape[d]

    def _op_conv2d(self, t, ins, outs, op):
        o = _opts(op, tflite.Conv2DOptions)
        x = t[ins[0]].permute(0, 3, 1, 2)                      # NHWC→NCHW
        w = t[ins[1]]                                          # (O,H,W,I)
        wt = w.permute(0, 3, 1, 2).contiguous()                # → (O,I,H,W)
        b = t[ins[2]] if len(ins) > 2 and ins[2] >= 0 else None
        sh, sw = o.StrideH(), o.StrideW()
        dh, dw = max(o.DilationWFactor(), 1), max(o.DilationHFactor(), 1)
        # 注意：tflite 的 dilation 因子字段名为 W/H 交换的历史问题——
        # 按 H 对 dh、W 对 dw 取用（两值在本项目模型中恒为 1，无实差）
        w_in = t[ins[1]]
        ph = _pad_same(self._sz(ins[0], x, 1), self._sz(ins[1], w_in, 1), sh, dh) if o.Padding() == _PAD_SAME else (0, 0)
        pw = _pad_same(self._sz(ins[0], x, 2), self._sz(ins[1], w_in, 2), sw, dw) if o.Padding() == _PAD_SAME else (0, 0)
        y = F.conv2d(F.pad(x, (pw[0], pw[1], ph[0], ph[1])), wt, b,
                     stride=(sh, sw), dilation=(dh, dw))
        y = _fused(y, o.FusedActivationFunction())
        t[outs[0]] = y.permute(0, 2, 3, 1)                      # 回 NHWC

    def _op_depthwiseconv2d(self, t, ins, outs, op):
        o = _opts(op, tflite.DepthwiseConv2DOptions)
        x = t[ins[0]].permute(0, 3, 1, 2)
        w = t[ins[1]]                                          # (1,H,W,C*M)
        c = self._sz(ins[1], w, 3) // max(o.DepthMultiplier(), 1)
        m = self._sz(ins[1], w, 3) // c
        wt = w.reshape(self._sz(ins[1], w, 1), self._sz(ins[1], w, 2), c, m) \
            .permute(2, 3, 0, 1).contiguous()                  # (C,M,H,W)
        wt = wt.reshape(-1, 1, self._sz(ins[1], w, 1),
                        self._sz(ins[1], w, 2))                 # (C*M,1,H,W)
        b = t[ins[2]] if len(ins) > 2 and ins[2] >= 0 else None
        sh, sw = o.StrideH(), o.StrideW()
        ph = _pad_same(self._sz(ins[0], x, 1), self._sz(ins[1], w, 1), sh, 1) if o.Padding() == _PAD_SAME else (0, 0)
        pw = _pad_same(self._sz(ins[0], x, 2), self._sz(ins[1], w, 2), sw, 1) if o.Padding() == _PAD_SAME else (0, 0)
        y = F.conv2d(F.pad(x, (pw[0], pw[1], ph[0], ph[1])), wt, b,
                     stride=(sh, sw), groups=c)
        y = _fused(y, o.FusedActivationFunction())
        t[outs[0]] = y.permute(0, 2, 3, 1)

    @staticmethod
    def _transpose_conv_same_pad(in_size: int, k: int, s: int
                                 ) -> tuple[int, int, int]:
        """tflite / MediaPipe 转置卷积 SAME padding（对照 transpose_conv_bias.cc）。

        total = max(0, k - (in-1)%s - 1)；两侧各 floor(total/2)；
        输出宽 = s*(in-1) + k - total（total 为奇数时 torch 会多 1 列，调用方裁掉）。
        """
        total = max(0, k - (in_size - 1) % s - 1)
        pad = total // 2
        target = s * (in_size - 1) + k - total
        return pad, target, total - 2 * pad

    def _conv2d_transpose_bias(self, t, ins, outs):
        """MediaPipe 自定义算子 Convolution2DTransposeBias。

        对照官方 transpose_conv_bias.cc：输入序 [data, weights, bias]，
        权重 OHWI（w[3]==输入通道、w[0]==bias 数==输出通道），
        普通全连接转置卷积 + bias 预置，SAME padding 按上述公式。
        """
        x = t[ins[0]].permute(0, 3, 1, 2)
        w = t[ins[1]]                                          # (O,kH,kW,I)
        b = t[ins[2]] if len(ins) > 2 and ins[2] >= 0 else None
        wt = w.permute(3, 0, 1, 2).contiguous()                # → (I,O,kH,kW)
        s = 2
        k = self._sz(ins[1], w, 1)
        in_h = self._sz(ins[0], x, 1)
        in_w = self._sz(ins[0], x, 2)
        pad, target, extra = self._transpose_conv_same_pad(in_h, k, s)
        pad_w, target_w, extra_w = self._transpose_conv_same_pad(in_w, k, s)
        y = F.conv_transpose2d(x, wt, None, stride=s, padding=(pad, pad_w))
        # total 为奇数时 torch 两侧对称 pad 会比目标多 extra 行/列，裁掉
        if extra > 0 or extra_w > 0:
            y_h = s * (in_h - 1) + k - 2 * pad
            y_w = s * (in_w - 1) + k - 2 * pad_w
            y = y[:, :, :target if target < y_h else y_h,
                  :target_w if target_w < y_w else y_w]
        if b is not None:
            y = y + b.view(1, -1, 1, 1)
        t[outs[0]] = y.permute(0, 2, 3, 1)

    def _op_fullyconnected(self, t, ins, outs, op):
        o = _opts(op, tflite.FullyConnectedOptions)
        x = t[ins[0]]
        w = t[ins[1]]
        if w.dim() > 2:
            w = w.reshape(-1, self._sz(ins[1], w, w.dim() - 1))
        # tflite FC 权重 (O,I)：按输入末维判定方向
        x_last = self._sz(ins[0], x, x.dim() - 1)
        wt = w if (w.dim() == 2 and x_last == self._sz(ins[1], w, 1)) else w.t()
        b = t[ins[2]] if len(ins) > 2 and ins[2] >= 0 else None
        y = F.linear(x, wt, b)
        y = _fused(y, o.FusedActivationFunction())
        t[outs[0]] = y

    def _op_prelu(self, t, ins, outs, op):
        # tflite 的 PReLU 斜率按通道广播（NHWC 末维），而 F.prelu 假定
        # NCHW（dim1 是通道）—— 手写广播版避免布局假设
        x = t[ins[0]]
        slope = t[ins[1]].flatten()
        view = (1,) * (x.dim() - 1) + (slope.numel(),)
        a = slope.view(view)
        neg = (x < 0).to(x.dtype)
        t[outs[0]] = x + neg * (a * x - x)

    def _op_relu(self, t, ins, outs, op):
        t[outs[0]] = F.relu(t[ins[0]])

    def _op_relu6(self, t, ins, outs, op):
        t[outs[0]] = t[ins[0]].clamp(0.0, 6.0)

    def _op_logistic(self, t, ins, outs, op):
        t[outs[0]] = torch.sigmoid(t[ins[0]])

    def _op_hardswish(self, t, ins, outs, op):
        t[outs[0]] = F.hardswish(t[ins[0]])

    def _op_tanh(self, t, ins, outs, op):
        t[outs[0]] = torch.tanh(t[ins[0]])

    def _op_add(self, t, ins, outs, op):
        o = _opts(op, tflite.AddOptions)
        y = t[ins[0]] + t[ins[1]]
        t[outs[0]] = _fused(y, o.FusedActivationFunction() if o else 0)

    def _op_sub(self, t, ins, outs, op):
        o = _opts(op, tflite.SubOptions)
        y = t[ins[0]] - t[ins[1]]
        t[outs[0]] = _fused(y, o.FusedActivationFunction() if o else 0)

    def _op_mul(self, t, ins, outs, op):
        o = _opts(op, tflite.MulOptions)
        y = t[ins[0]] * t[ins[1]]
        t[outs[0]] = _fused(y, o.FusedActivationFunction() if o else 0)

    def _op_div(self, t, ins, outs, op):
        o = _opts(op, tflite.DivOptions)
        y = t[ins[0]] / t[ins[1]]
        t[outs[0]] = _fused(y, o.FusedActivationFunction() if o else 0)

    def _op_neg(self, t, ins, outs, op):
        t[outs[0]] = -t[ins[0]]

    def _op_sqrt(self, t, ins, outs, op):
        t[outs[0]] = torch.sqrt(t[ins[0]])

    def _op_rsqrt(self, t, ins, outs, op):
        t[outs[0]] = torch.rsqrt(t[ins[0]])

    def _op_squared_difference(self, t, ins, outs, op):
        d = t[ins[0]] - t[ins[1]]
        t[outs[0]] = d * d

    def _op_concatenation(self, t, ins, outs, op):
        o = _opts(op, tflite.ConcatenationOptions)
        axis = o.Axis() if o else -1
        xs = [t[i] for i in ins if i >= 0]
        t[outs[0]] = torch.cat(xs, dim=axis if axis >= 0 else axis)

    def _op_reshape(self, t, ins, outs, op):
        o = _opts(op, tflite.ReshapeOptions)
        x = t[ins[0]]
        if o is not None and o.NewShapeAsNumpy() is not None \
                and len(o.NewShapeAsNumpy()):
            shape = [int(s) for s in o.NewShapeAsNumpy()]
        elif len(ins) > 1 and ins[1] in t:
            shape = [int(s) for s in t[ins[1]].long().tolist()]
        else:
            shape = list(x.shape)
        # 0 = 沿用输入同位维；-1 = 推断（至多一个）—— 优先全静态尺寸
        in_shp = self._tshape.get(int(ins[0]))
        if in_shp is not None:
            if len(in_shp) == len(shape):
                shape = [in_shp[i] if s == 0 else s
                         for i, s in enumerate(shape)]
            if any(s == -1 for s in shape):
                known = int(np.prod([s for s in shape if s > 0]))
                total = int(np.prod(in_shp))
                shape = [total // known if s == -1 else s for s in shape]
            t[outs[0]] = x.reshape([int(s) for s in shape])
            return
        shape = [x.shape[i] if s == 0 else s for i, s in enumerate(shape)]
        if any(s == -1 for s in shape):
            known = int(np.prod([s for s in shape if s > 0]))
            shape = [x.numel() // known if s == -1 else s for s in shape]
        t[outs[0]] = x.reshape(shape)

    def _op_squeeze(self, t, ins, outs, op):
        o = _opts(op, tflite.SqueezeOptions)
        ax = list(o.SqueezeDimsAsNumpy()) if o is not None and o.SqueezeDimsAsNumpy() is not None else None
        t[outs[0]] = t[ins[0]].squeeze() if not ax else t[ins[0]].squeeze(dim=tuple(ax))

    def _op_stridedslice(self, t, ins, outs, op):
        o = _opts(op, tflite.StridedSliceOptions)
        x = t[ins[0]]
        begins = [int(v) for v in t[ins[1]].long().tolist()]
        ends = [int(v) for v in t[ins[2]].long().tolist()]
        strides = [int(v) for v in t[ins[3]].long().tolist()]
        bm = o.BeginMask()
        em = o.EndMask()
        na = o.NewAxisMask()
        el = o.EllipsisMask()
        assert el == 0 and na == 0, "StridedSlice 掩码组合未支持"
        slices = []
        for d in range(len(begins)):
            b, e, s = begins[d], ends[d], strides[d]
            xd = self._sz(ins[0], x, d)
            if bm & (1 << d):
                b = 0 if s > 0 else xd
            if em & (1 << d):
                e = xd if s > 0 else -xd - 1
            if e > 2 ** 30:
                e = xd if s > 0 else -xd - 1
            if b < -2 ** 30:
                b = 0 if s > 0 else xd
            slices.append(slice(b, e, s))
        t[outs[0]] = x[tuple(slices)]

    def _op_slice(self, t, ins, outs, op):
        self._op_stridedslice(t, ins[:4], outs, op)

    def _op_transpose(self, t, ins, outs, op):
        perm = [int(p) for p in t[ins[1]].long().tolist()] \
            if len(ins) > 1 and ins[1] in t else None
        t[outs[0]] = t[ins[0]].permute(*perm) if perm else t[ins[0]].transpose(-2, -1)

    def _op_pad(self, t, ins, outs, op):
        o = _opts(op, tflite.PadOptions)
        pads = t[ins[1]].long().tolist()                       # (rank, 2)
        x = t[ins[0]]
        # torch pad 按 (末维前,末维后,…) 逆序
        pad_args = []
        for d in range(len(pads) - 1, -1, -1):
            pad_args += [int(pads[d][0]), int(pads[d][1])]
        const_val = 0.0
        if o is not None and len(ins) > 2 and ins[2] in t:
            const_val = float(t[ins[2]].flatten()[0])
        mode = "constant"
        if len(pads) == 4 and pads[2] == [0, 0] and pads[3] == [0, 0] \
                and pads[0] == [0, 0] and pads[1][0] > 16:
            # MediaPipe 反射填充形态（(0,0),(H,H),(W,W),(0,0) 之外的镜像
            # 填充在 tflite 里用 PAD + 特殊值数组实现，本项目模型未使用）
            mode = "constant"
        t[outs[0]] = F.pad(x, pad_args, mode=mode, value=const_val)

    def _op_maxpool2d(self, t, ins, outs, op):
        o = _opts(op, tflite.Pool2DOptions)
        x = t[ins[0]].permute(0, 3, 1, 2)
        kh, kw = o.FilterHeight(), o.FilterWidth()
        sh, sw = o.StrideH(), o.StrideW()
        ph = _pad_same(self._sz(ins[0], x, 1), kh, sh, 1) if o.Padding() == _PAD_SAME else (0, 0)
        pw = _pad_same(self._sz(ins[0], x, 2), kw, sw, 1) if o.Padding() == _PAD_SAME else (0, 0)
        y = F.max_pool2d(x, (kh, kw), (sh, sw),
                         padding=(ph[0], pw[0]))
        t[outs[0]] = y.permute(0, 2, 3, 1)

    def _op_averagepool2d(self, t, ins, outs, op):
        o = _opts(op, tflite.Pool2DOptions)
        x = t[ins[0]].permute(0, 3, 1, 2)
        kh, kw = o.FilterHeight(), o.FilterWidth()
        sh, sw = o.StrideH(), o.StrideW()
        # TF SAME 平均池化：floor((in-1)/s)+1 的输出且 padding 区不计均值
        # —— 等价于无 pad 池化 + 右/下边缘补齐元素复制。本项目模型的
        # SAME 池化均为 k==s（全局下采样），无 pad 即数值一致。
        y = F.avg_pool2d(x, (kh, kw), (sh, sw), padding=(0, 0),
                         count_include_pad=False)
        xh, xw = self._sz(ins[0], x, 1), self._sz(ins[0], x, 2)
        want_h = (xh + sh - 1 - 1) // sh + 1 if o.Padding() == _PAD_SAME \
            else (xh - kh) // sh + 1
        want_w = (xw + sw - 1 - 1) // sw + 1 if o.Padding() == _PAD_SAME \
            else (xw - kw) // sw + 1
        y_h = (xh - kh) // sh + 1      # 无 pad 池化的实际输出尺寸
        y_w = (xw - kw) // sw + 1
        if y_h != want_h or y_w != want_w:
            y = y[:, :want_h, :want_w]
        t[outs[0]] = y.permute(0, 2, 3, 1)

    def _op_resizebilinear(self, t, ins, outs, op):
        o = _opts(op, tflite.ResizeBilinearOptions)
        x = t[ins[0]]
        size = [int(v) for v in t[ins[1]].long().tolist()]     # (H,W)
        xc = x.permute(0, 3, 1, 2)                             # NHWC→NCHW
        if o is not None and o.AlignCorners():
            y = F.interpolate(xc, size=size, mode="bilinear",
                              align_corners=True)
        elif o is not None and o.HalfPixelCenters():
            y = F.interpolate(xc, size=size, mode="bilinear",
                              align_corners=False)
        else:
            # tflite 默认 (asymmetric)：src = dst * scale，无半像素偏移
            y = _resize_asymmetric(xc, size,
                                   (self._sz(ins[0], x, 1),
                                    self._sz(ins[0], x, 2)))
        t[outs[0]] = y.permute(0, 2, 3, 1)                     # 回 NHWC

    def _op_resizenearestneighbor(self, t, ins, outs, op):
        o = _opts(op, tflite.ResizeNearestNeighborOptions)
        x = t[ins[0]]
        size = [int(v) for v in t[ins[1]].long().tolist()]
        ac = o is not None and o.AlignCorners()
        hpc = o is not None and o.HalfPixelCenters()
        xc = x.permute(0, 3, 1, 2)                             # NHWC→NCHW
        if not ac and not hpc:
            # tflite 默认：src = floor(dst * scale)
            y = _resize_nn_asymmetric(xc, size,
                                      (self._sz(ins[0], x, 1),
                                       self._sz(ins[0], x, 2)))
        else:
            y = F.interpolate(xc, size=size, mode="nearest")
        t[outs[0]] = y.permute(0, 2, 3, 1)

    def _op_mean(self, t, ins, outs, op):
        axes = [int(a) for a in t[ins[1]].long().tolist()]
        t[outs[0]] = t[ins[0]].mean(dim=tuple(axes), keepdim=True)

    def _op_sum(self, t, ins, outs, op):
        axes = [int(a) for a in t[ins[1]].long().tolist()]
        t[outs[0]] = t[ins[0]].sum(dim=tuple(axes), keepdim=True)

    def _op_softmax(self, t, ins, outs, op):
        o = _opts(op, tflite.SoftmaxOptions)
        beta = o.Beta() if o else 1.0
        x = t[ins[0]] * beta
        t[outs[0]] = F.softmax(x, dim=-1)

    def _op_dequantize(self, t, ins, outs, op):
        t[outs[0]] = t[ins[0]]     # const 已统一 fp32

    def _op_transposeconv(self, t, ins, outs, op):
        o = _opts(op, tflite.TransposeConvOptions)
        # tflite 输入序：[output_shape(N,H,W,C), weights, input_data, bias]
        out_shape = [int(v) for v in t[ins[0]].long().tolist()]
        w = t[ins[1]]                                          # (O,kH,kW,I)
        x = t[ins[2]].permute(0, 3, 1, 2)
        b = t[ins[3]] if len(ins) > 3 and ins[3] >= 0 else None
        wt = w.permute(3, 0, 1, 2).contiguous()                # → (I,O,kH,kW)
        sh, sw = o.StrideH(), o.StrideW()
        k = wt.shape[2]
        pad_h, tgt_h, extra_h = self._transpose_conv_same_pad(self._sz(ins[2], x, 1), k, sh)
        pad_w, tgt_w, extra_w = self._transpose_conv_same_pad(self._sz(ins[2], x, 2), k, sw)
        y = F.conv_transpose2d(x, wt, None, stride=(sh, sw),
                               padding=(pad_h, pad_w))
        if extra_h > 0 or extra_w > 0:
            y = y[:, :, :tgt_h if tgt_h < y.shape[2] else y.shape[2],
                  :tgt_w if tgt_w < y.shape[3] else y.shape[3]]
        if b is not None:
            y = y + b.view(1, -1, 1, 1)
        assert y.shape[1] == out_shape[3], \
            f"TransposeConv 输出通道 {y.shape[1]} != 声明 {out_shape[3]}"
        assert y.shape[2] == out_shape[1] and y.shape[3] == out_shape[2], \
            f"TransposeConv 输出 {tuple(y.shape)} != 声明 {out_shape}"
        t[outs[0]] = y.permute(0, 2, 3, 1)

    def _op_gather(self, t, ins, outs, op):
        o = _opts(op, tflite.GatherOptions)
        t[outs[0]] = torch.gather(t[ins[0]], o.Axis() if o else 0,
                                  t[ins[1]].long())

    def _op_rsqrt_activation(self, t, ins, outs, op):   # 兜底命名冲突
        self._op_rsqrt(t, ins, outs, op)

    # ---- 元信息 ----

    def input_shapes(self) -> list[list[int]]:
        shapes = []
        for i in self.input_indices:
            s = self.graph.Tensors(int(i)).ShapeAsNumpy()
            shapes.append([int(v) for v in s] if s is not None else [])
        return shapes

    def output_shapes(self) -> list[list[int]]:
        shapes = []
        for i in self.output_indices:
            s = self.graph.Tensors(int(i)).ShapeAsNumpy()
            shapes.append([int(v) for v in s] if s is not None else [])
        return shapes


def _resize_asymmetric(x: torch.Tensor, size: list[int],
                       hw: tuple[int, int] | None = None) -> torch.Tensor:
    """tflite 默认双线性（src = dst·scale，越界钳制到边缘）。NCHW 输入。"""
    oy, ox = size
    ih, iw = hw if hw is not None else (x.shape[2], x.shape[3])
    ys = torch.arange(oy, device=x.device, dtype=torch.float32) * (ih / oy)
    xs = torch.arange(ox, device=x.device, dtype=torch.float32) * (iw / ox)
    grid_y = ys.view(1, oy, 1).expand(1, oy, ox, 1)
    grid_x = xs.view(1, 1, ox).expand(1, oy, ox, 1)
    grid = torch.cat([grid_x * 2 / max(iw - 1, 1) - 1,
                      grid_y * 2 / max(ih - 1, 1) - 1], dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)


def _resize_nn_asymmetric(x: torch.Tensor, size: list[int],
                          hw: tuple[int, int] | None = None) -> torch.Tensor:
    """tflite 默认最近邻（src = floor(dst·scale)）。NCHW 输入。"""
    oy, ox = size
    ih, iw = hw if hw is not None else (x.shape[2], x.shape[3])
    ys = (torch.arange(oy, device=x.device, dtype=torch.float32)
          * (ih / oy)).floor().clamp_(0, ih - 1)
    xs = (torch.arange(ox, device=x.device, dtype=torch.float32)
          * (iw / ox)).floor().clamp_(0, iw - 1)
    idx_y = ys.long().view(1, oy, 1, 1).expand(-1, -1, ox, -1)
    idx_x = xs.long().view(1, 1, ox, 1).expand(-1, oy, -1, -1)
    return x[:, :, idx_y[0], idx_x[0]]


def convert_tflite_module(path: str | Path) -> TfliteModule:
    m = TfliteModule(path)
    m = m.float().eval()
    return m
