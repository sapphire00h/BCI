"""EEGNet 及其他模型的定义。

新模型可在本文件中定义为 nn.Module 子类，再由 create_model 按类名创建。
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
"""
搭建模型：
是对 EEGNet 的 PyTorch 原生复现，在 22×1000 的 MI-EEG 上采用“时间-空间-可分离”三阶段卷积，
参数量仅数千即可达到 4 类运动想象 75 % 左右准确率；与论文差异主要源于采样率、kernel 尺寸及滤波设置。
——————————————————————————————————————————————————————————————————————————————————————————————————————————-
| 代码变量          | 论文符号 | 论文推荐(128 Hz) | 代码默认(1000 Hz) | 备注                               |
| -------------    | ----    | ------------     | ------------- | -------------------------             |
| samples          | T       | 128×4=512        | 1000          | 与采样率、epoch 长度相关                |
| kernelLength     | k1      | 64               | 32            | 时间卷积核，代码按 1000 Hz 等比例缩小   |
| kernelLength2    | k2      | 16               | 16            | 深度可分离卷积核                       |
| F1               | F1      | 8                | 16            | 时间滤波器个数                         |
| D                | D       | 2                | 2             | 深度乘子 → Block2 输出通道 = F1×D       |
| F2               | F2      | 16               | 32            | SeparableConv 输出通道                  |
| dropoutRate      | —       | 0.5              | 0.5           | 一致                                   |
——————————————————————————————————————————————————————————————————————————————————————————————————————-————
模型结构网格图：
        Input: (batch,   1, C=22, T≈1000)
           │
    ┌─── Block1 ───┐  Temporal Conv + BN
    │   (F1=16)    │
           │
    ┌─── Block2 ───┐  Depthwise Spatial Conv + BN + ELU + AvgPool(1,8) + Dropout
    │  (D=2 → 32)  │
           │
    ┌─── Block3 ───┐  Separable Conv (Depthwise+Pointwise) + BN + ELU + AvgPool(1,16) + Dropout
    │   (F2=32)    │
           │
    ┌─── Classifier ───┐ Global Flatten → Dense → Softmax
    │   (n_classes=4)  │
           │
        Output: (batch, 4)
—————————————————————————————————————————————————————————————————————————————————————————————————————————————
"""
class EEGNet(nn.Module):
    def __init__(self, n_classes=4, channels=22, samples=1000,
                 dropoutRate=0.5, kernelLength=32, kernelLength2=16,
                 F1=16, D=2, F2=32):
        super(EEGNet, self).__init__()
        self.n_classes = n_classes
        self.channels = channels
        self.samples = samples
        self.drop_out = dropoutRate
        self.kernelLength = kernelLength
        self.kernelLength2 = kernelLength2
        self.F1 = F1
        self.D = D
        self.F2 = F2

        block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.F1,
                      kernel_size=(1,self.kernelLength), stride=1,
                      padding=(0,self.kernelLength//2),
                      bias=False
        ),
            nn.BatchNorm2d(num_features=self.F1)
        )

        block2 = nn.Sequential(
            nn.Conv2d(in_channels=self.F1,out_channels=self.F1*self.D,
                      kernel_size=(self.channels,1), groups=self.F1,
                      bias=False
                      ),
            nn.BatchNorm2d(num_features=self.F1*self.D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,8)),
            nn.Dropout(p=self.drop_out)
        )

        block3 = nn.Sequential(
            nn.Conv2d(in_channels=self.F1*self.D,out_channels=self.F2,
                      kernel_size=(1,self.kernelLength2), stride=1,
                      padding=(0,self.kernelLength2//2),groups=self.F1*self.D,
                      bias=False
                      ),
            nn.Conv2d(in_channels=self.F2,out_channels=self.F2,
                      kernel_size=(1,1), stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,16)),
            nn.Dropout(p=self.drop_out)
        )

        self.EEGNetLayer = nn.Sequential(block1, block2, block3)
        # self.FlattenLayer = nn.Flatten()
        self.Linear = nn.Linear(in_features=self.F2 * round((self.samples // 8) // 16),
                                                       out_features=self.n_classes,
                                                       bias=False)
        # self.model = nn.Sequential(self.EEGNetLayer, self.FlattenLayer, self.ClassifierBlock)
    def forward(self, x):
        if len(x.shape) != 4:
            x = torch.unsqueeze(x,1)
        x = self.EEGNetLayer(x)
        x = x.view(x.size()[0], -1)  # Flatten # [N, self.F2*1*T/32]
        x = self.Linear(x)
        # x = self.model(x)

        return x










# 以下 EEG-TCNet 及其辅助类来自 是馒头阿（mantoudamahou）的复现代码。
# 来源：https://blog.csdn.net/mantoudamahou/article/details/134990549
# 原文及本段改编代码遵循 CC BY-SA 4.0：
# https://creativecommons.org/licenses/by-sa/4.0/
# 改动：合并模型所需定义到本文件，将 My_Model 命名为 EEGTCNet，
# 并统一部分构造参数名称；保留原模型结构、默认参数和前向计算。
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

    def __call__(self, *input, **kwargs):
        return super()._call_impl(*input, **kwargs)


class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)
        if self.bias:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 dropout=0.2, bias=False, WeightNorm=False, max_norm=1.):
        super(TemporalBlock, self).__init__()
        self.conv1 = Conv1dWithConstraint(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(num_features=n_outputs)
        self.relu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = Conv1dWithConstraint(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(num_features=n_outputs)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        out = out+res
        out = self.relu(out)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2,
                 bias=False, WeightNorm=False, max_norm=1.):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                dropout=dropout, bias=bias, WeightNorm=WeightNorm, max_norm=max_norm
            )]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalInception(nn.Module):
    def __init__(self, in_chan=1, kerSize_1=(1,3), kerSize_2=(1,5), kerSize_3=(1,7),
                 kerStr=1, out_chan=4, pool_ker=(1,3), pool_str=1, bias=False, max_norm=1.):
        super(TemporalInception, self).__init__()
        self.conv1 = Conv2dWithConstraint(
            in_channels=in_chan, out_channels=out_chan, kernel_size=kerSize_1,
            stride=kerStr, padding='same', groups=out_chan, bias=bias, max_norm=max_norm
        )
        self.conv2 = Conv2dWithConstraint(
            in_channels=in_chan, out_channels=out_chan, kernel_size=kerSize_2,
            stride=kerStr, padding='same', groups=out_chan, bias=bias, max_norm=max_norm
        )
        self.conv3 = Conv2dWithConstraint(
            in_channels=in_chan, out_channels=out_chan, kernel_size=kerSize_3,
            stride=kerStr, padding='same', groups=out_chan, bias=bias, max_norm=max_norm
        )
        self.pool4 = nn.MaxPool2d(
            kernel_size=pool_ker, stride=pool_str,
            padding=(round(pool_ker[0]/2+0.1)-1,round(pool_ker[1]/2+0.1)-1)
        )
        self.conv4 = Conv2dWithConstraint(
            in_channels=in_chan, out_channels=out_chan, kernel_size=1, stride=1,
            groups=out_chan, bias=bias, max_norm=max_norm
        )

    def forward(self, x):
        p1 = self.conv1(x)
        p2 = self.conv2(x)
        p3 = self.conv3(x)
        p4 = self.conv4(self.pool4(x))
        out = torch.cat((p1,p2,p3,p4), dim=1)
        return out


class EEGTCNet(nn.Module):
    """网页 My_Model：仅统一类名和部分构造参数名称。

    kerSize -> kernelLength，eeg_chans -> channels，
    kerSize_Tem -> kernelLength2，dropout_dep -> dropoutRate，
    dropout_temp -> dropoutRate2。dropout_atten 沿用原代码的未使用参数。
    """
    def __init__(self, F1=32, D=2, kernelLength=32, channels=22, poolSize=8,
                 kernelLength2=4, dropoutRate=0.5, dropoutRate2=0.5,
                 dropout_atten=0.3, tcn_filters=64, tcn_kernelSize=4,
                 tcn_dropout=0.3, n_classes=4):
        super(EEGTCNet, self).__init__()
        self.F2 = F1*D

        self.sincConv = nn.Conv2d(
            in_channels=1, out_channels=F1, kernel_size=(1, kernelLength),
            stride=1, padding='same', bias=False
        )
        self.bn_sinc = nn.BatchNorm2d(num_features=F1)

        self.conv_depth = Conv2dWithConstraint(
            in_channels=F1, out_channels=F1*D, kernel_size=(channels,1),
            groups=F1, bias=False, max_norm=1.
        )
        self.bn_depth = nn.BatchNorm2d(num_features=self.F2)
        self.act_depth = nn.ELU()
        self.avgpool_depth = nn.AvgPool2d(
            kernel_size=(1,poolSize), stride=(1,poolSize)
        )
        self.drop_depth = nn.Dropout(p=dropoutRate)
        self.incept_temp = TemporalInception(
            in_chan=self.F2,
            kerSize_1=(1,kernelLength2*4),
            kerSize_2=(1,kernelLength2*2),
            kerSize_3=(1,kernelLength2),
            kerStr=1, out_chan=self.F2//4,
            pool_ker=(3,3), pool_str=1, bias=False, max_norm=.5
        )
        self.bn_temp = nn.BatchNorm2d(num_features=self.F2)
        self.act_temp = nn.ELU()
        self.avgpool_temp = nn.AvgPool2d(
            kernel_size=(1,poolSize), stride=(1,poolSize)
        )
        self.drop_temp = nn.Dropout(p=dropoutRate2)

        self.tcn_block = TemporalConvNet(
            num_inputs=self.F2,
            num_channels=[tcn_filters, tcn_filters],
            kernel_size=tcn_kernelSize,
            dropout=tcn_dropout, bias=False, WeightNorm=True, max_norm=.5
        )

        self.flatten = nn.Flatten()
        self.liner_cla = LinearWithConstraint(
            in_features=tcn_filters, out_features=n_classes, max_norm=.25
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)

        x = self.sincConv(x)
        x = self.bn_sinc(x)

        x = self.conv_depth(x)
        x = self.drop_depth(self.avgpool_depth(self.act_depth(self.bn_depth(x))))

        x = self.incept_temp(x)
        x = self.drop_temp(self.avgpool_temp(self.act_temp(self.bn_temp(x))))

        x = torch.squeeze(x, dim=2)
        x = self.tcn_block(x)
        x = x[:, :, -1]

        x = self.flatten(x)
        x = self.liner_cla(x)
        # out = self.softmax(x)
        out=x
        return out










class EEGNex(nn.Module):
    def __init__(self, channels=22, samples=1001, n_classes=4,
                 F1=8, F2=32, D=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, F1, (1, 128), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.Conv2d(F1, F2, (1, 128), padding="same", bias=False),
            nn.BatchNorm2d(F2),

            # Block 2
            nn.Conv2d(F2, F2 * D, (channels, 1),
                      groups=F2, bias=False),
            nn.BatchNorm2d(F2 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(F2 * D, F2, (1, 32), dilation=(1, 2),
                      padding="same", bias=False),
            nn.BatchNorm2d(F2),
            nn.Conv2d(F2, F1, (1, 32), dilation=(1, 4),
                      padding="same", bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout2d(0.25),

            nn.Flatten(),
            nn.Linear(F1 * (samples // 32), n_classes, bias=False),
        )

    def forward(self, x):
        return self.features(x)










# Adapted from https://github.com/Altaheri/TCFormer
# Original copyright (c) 2025 Hamdi Altaheri
# Original project license: MIT
def _itcf_xavier_init(module):
    for layer in module.modules():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class _ITCFGroupAttention(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.groups = groups
        self.group_size = channels // groups
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // 4, 1, groups=groups, bias=False
        )
        self.fc2 = nn.Conv2d(
            channels // 4, groups, 1, groups=groups, bias=False
        )

    def forward(self, x):
        batch, _, height, width = x.shape
        weights = torch.sigmoid(self.fc2(F.relu(self.fc1(self.pool(x)))))
        weights = weights.view(batch, self.groups, 1, 1, 1)
        grouped = x.reshape(
            batch, self.groups, self.group_size, height, width
        )
        return (grouped * weights).reshape_as(x)


class _ITCFConvStem(nn.Module):
    def __init__(
        self,
        channels,
        temporal_kernels,
        f1,
        depth_multiplier,
        group_dim,
        pool1,
        pool2,
        dropout,
    ):
        super().__init__()
        groups = len(temporal_kernels)
        stem_channels = f1 * groups
        spatial_channels = stem_channels * depth_multiplier
        model_dim = group_dim * groups

        self.temporal = nn.ModuleList()
        for kernel in temporal_kernels:
            left = (kernel - 1) // 2
            right = kernel - 1 - left
            self.temporal.append(
                nn.Sequential(
                    nn.ConstantPad2d((left, right, 0, 0), 0),
                    nn.Conv2d(1, f1, (1, kernel), bias=False),
                    nn.BatchNorm2d(f1),
                )
            )

        self.spatial = nn.Sequential(
            nn.Conv2d(
                stem_channels,
                spatial_channels,
                (channels, 1),
                groups=stem_channels,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_channels),
            nn.ELU(),
        )
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(dropout)

        self.reduction = nn.Sequential(
            nn.Conv2d(
                spatial_channels,
                model_dim,
                1,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(model_dim),
        )
        self.grouped_temporal = nn.Sequential(
            nn.Conv2d(
                model_dim,
                model_dim,
                (1, 16),
                padding="same",
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(model_dim),
            nn.ELU(),
        )
        self.group_attention = _ITCFGroupAttention(model_dim, groups)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop2 = nn.Dropout(dropout)

        _itcf_xavier_init(self)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = torch.cat([branch(x) for branch in self.temporal], dim=1)
        x = self.drop1(self.pool1(self.spatial(x)))
        x = self.grouped_temporal(self.reduction(x))
        x = x + self.group_attention(x)
        return self.drop2(self.pool2(x)).squeeze(2)


class _ITCFDropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (keep + torch.rand(shape, device=x.device, dtype=x.dtype)).floor()
        return x * mask / keep


def _itcf_rotary_cache(head_dim, length, device):
    frequencies = 1.0 / (
        10000 ** (
            torch.arange(0, head_dim, 2, device=device).float() / head_dim
        )
    )
    positions = torch.arange(length, device=device).float()
    angles = torch.outer(positions, frequencies)
    angles = torch.repeat_interleave(
        angles, repeats=2, dim=-1
    )
    # angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


def _itcf_rotate(x):
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class _ITCFGroupedQueryAttention(nn.Module):
    def __init__(self, model_dim, query_heads, kv_heads, dropout):
        super().__init__()
        if model_dim % query_heads or query_heads % kv_heads:
            raise ValueError("model_dim、query_heads、kv_heads 不兼容")

        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = model_dim // query_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.kv_proj = nn.Linear(
            model_dim, 2 * kv_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        _itcf_xavier_init(self)

    def forward(self, x, cosine, sine):
        batch, length, model_dim = x.shape
        q = self.q_proj(x).reshape(
            batch, length, self.query_heads, self.head_dim
        ).transpose(1, 2)

        kv = self.kv_proj(x).reshape(
            batch, length, self.kv_heads, 2, self.head_dim
        )
        k = kv[..., 0, :].transpose(1, 2)
        v = kv[..., 1, :].transpose(1, 2)

        repeat = self.query_heads // self.kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        cosine = cosine[:length]
        sine = sine[:length]
        q = q * cosine + _itcf_rotate(q) * sine
        k = k * cosine + _itcf_rotate(k) * sine

        weights = (q @ k.transpose(-2, -1)) * self.scale
        weights = self.dropout(weights.softmax(dim=-1))
        result = weights @ v
        result = result.transpose(1, 2).contiguous()
        result = result.reshape(batch, length, model_dim)
        return self.out_proj(result)


class _ITCFTransformerBlock(nn.Module):
    def __init__(
        self, model_dim, query_heads, kv_heads, dropout, drop_path
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attention = _ITCFGroupedQueryAttention(
            model_dim, query_heads, kv_heads, dropout
        )
        self.drop_path = _ITCFDropPath(drop_path)
        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, 2 * model_dim),
            nn.GELU(),
            nn.Linear(2 * model_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cosine, sine):
        x = x + self.drop_path(
            self.attention(self.norm1(x), cosine, sine)
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _ITCFCausalConv1d(nn.Conv1d):
    def forward(self, x):
        left = (self.kernel_size[0] - 1) * self.dilation[0]
        return super().forward(F.pad(x, (left, 0)))


class _ITCFTCNBlock(nn.Module):
    def __init__(self, features, groups, kernel, dilation, dropout):
        super().__init__()
        self.conv1 = _ITCFCausalConv1d(
            features, features, kernel,
            dilation=dilation, groups=groups
        )
        self.bn1 = nn.BatchNorm1d(features)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = _ITCFCausalConv1d(
            features, features, kernel,
            dilation=dilation, groups=groups
        )
        self.bn2 = nn.BatchNorm1d(features)
        self.drop2 = nn.Dropout(dropout)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        residual = x
        x = self.drop1(F.elu(self.bn1(self.conv1(x))))
        x = self.drop2(F.elu(self.bn2(self.conv2(x))))
        return F.elu(residual + x)


class _ITCFMaxNormConv1d(nn.Conv1d):
    def __init__(self, *args, max_norm=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        with torch.no_grad():
            self.weight.copy_(
                torch.renorm(
                    self.weight, p=2, dim=0, maxnorm=self.max_norm
                )
            )
        return super().forward(x)


class TCFormer(nn.Module):
    """
    输入: (B, channels, T) 或 (B, 1, channels, T)
    输出: (B, n_classes) 原始 logits
    """

    def __init__(
        self,
        n_classes=4,
        channels=22,
        samples=1001,
        temporal_kernels=(20, 32, 64),
        f1=32,
        depth_multiplier=2,
        group_dim=16,
        pool1=8,
        pool2=7,
        conv_dropout=0.4,
        trans_depth=2,
        q_heads=4,
        kv_heads=2,
        trans_dropout=0.4,
        drop_path_max=0.25,
        tcn_depth=2,
        tcn_kernel=4,
        tcn_dropout=0.3,
    ):
        super().__init__()
        if trans_depth < 1 or tcn_depth < 1:
            raise ValueError("trans_depth 和 tcn_depth 必须 >= 1")
        if len(temporal_kernels) != 3:
            raise ValueError("论文配置使用 3 个时间卷积分支")

        self.channels = channels
        self.samples = samples
        self.n_classes = n_classes
        self.groups = len(temporal_kernels)
        model_dim = group_dim * self.groups

        self.stem = _ITCFConvStem(
            channels, temporal_kernels, f1, depth_multiplier,
            group_dim, pool1, pool2, conv_dropout
        )
        self.mix = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 1, bias=False),
            nn.BatchNorm1d(model_dim),
            nn.SiLU(),
        )

        rates = torch.linspace(0, 1, trans_depth).square() * drop_path_max
        self.transformer = nn.ModuleList([
            _ITCFTransformerBlock(
                model_dim, q_heads, kv_heads,
                trans_dropout, rates[i].item()
            )
            for i in range(trans_depth)
        ])
        self.reduce = nn.Sequential(
            nn.Conv1d(model_dim, group_dim, 1, bias=False),
            nn.BatchNorm1d(group_dim),
            nn.SiLU(),
        )

        features = group_dim * (self.groups + 1)
        tcn_groups = self.groups + 1
        self.tcn = nn.Sequential(*[
            _ITCFTCNBlock(
                features, tcn_groups, tcn_kernel,
                dilation=2 ** i, dropout=tcn_dropout
            )
            for i in range(tcn_depth)
        ])
        self.classifier = _ITCFMaxNormConv1d(
            features, n_classes * tcn_groups, 1,
            groups=tcn_groups, max_norm=0.25
        )

        self.register_buffer("_rope_cos", None, persistent=False)
        self.register_buffer("_rope_sin", None, persistent=False)

    def forward(self, x):
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(
                f"输入必须是 (B,{self.channels},T) "
                f"或 (B,1,{self.channels},T)"
            )
        if x.shape[-1] < 56:
            raise ValueError("时间长度 T 必须 >= pool1 * pool2 = 56")

        conv_features = self.stem(x)
        tokens = self.mix(conv_features).transpose(1, 2)

        length = tokens.shape[1]
        head_dim = self.transformer[0].attention.head_dim
        if (
            self._rope_cos is None
            or self._rope_cos.shape[0] < length
            or self._rope_cos.device != tokens.device
        ):
            self._rope_cos, self._rope_sin = _itcf_rotary_cache(
                head_dim, length, tokens.device
            )

        for block in self.transformer:
            tokens = block(tokens, self._rope_cos, self._rope_sin)

        transformer_features = self.reduce(tokens.transpose(1, 2))
        features = torch.cat(
            (conv_features, transformer_features), dim=1
        )
        features = self.tcn(features)[:, :, -1:]
        logits = self.classifier(features)
        logits = logits.squeeze(-1)
        return logits.reshape(
            x.shape[0], self.groups + 1, self.n_classes
        ).mean(dim=1)










# MCTGNet复现
# ============================================================
# 1. Same Padding Temporal Convolution
# ============================================================

class SamePadTemporalConv2d(nn.Module):
    """
    在时间维度进行二维卷积，保持时间长度不变。

    Input : (B, Cin, H, T)
    Output: (B, Cout, H, T)
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size, bias=False):
        super().__init__()

        left = (kernel_size - 1) // 2
        right = kernel_size - 1 - left

        self.net = nn.Sequential(
            nn.ZeroPad2d((left, right, 0, 0)),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size),
                bias=bias
            )
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 2. Group Rational Activation
# ============================================================

class GroupRationalActivation(nn.Module):
    """
    GR-KAN 的分组有理函数激活。

    P(x) = a0 + a1*x + ... + a5*x^5

    Q(x) = 1 + |b0|*|x| + ... + |b3|*|x|^4

    Activation(x) = P(x) / Q(x)

    各组共享分子系数，分别学习分母系数。
    初始化参考 ikan 的 GELU rational coefficients。
    """

    def __init__(self, features, num_groups=8):
        super().__init__()

        if features % num_groups != 0:
            raise ValueError(
                f"features={features} must be divisible "
                f"by num_groups={num_groups}"
            )

        self.num_groups = num_groups

        # GELU rational initialization
        self.numerator = nn.Parameter(
            torch.tensor([
                -0.00153969,
                0.51692871,
                0.44075827,
                0.09725530,
                -0.00884418,
                -0.00378675
            ], dtype=torch.float32)
        )

        denominator_init = torch.tensor([
            -0.14171056,
            -0.06481231,
            -0.04446950,
            0.01283776
        ], dtype=torch.float32)

        self.denominator = nn.Parameter(
            denominator_init.repeat(num_groups, 1)
        )

    def forward(self, x):

        original_shape = x.shape

        # (..., features)
        #     ↓
        # (..., num_groups, features_per_group)
        z = x.reshape(
            *original_shape[:-1],
            self.num_groups,
            original_shape[-1] // self.num_groups
        )

        a = self.numerator

        # Horner polynomial evaluation
        num = (
            (
                (
                    (
                        (a[5] * z + a[4]) * z + a[3]
                    ) * z + a[2]
                ) * z + a[1]
            ) * z + a[0]
        )

        # Broadcast grouped denominator coefficients
        coef_shape = (
            [1] * (z.ndim - 2)
            + [self.num_groups, 4]
        )

        coefs = self.denominator.abs().view(*coef_shape)

        b0, b1, b2, b3 = coefs.unbind(dim=-1)

        t = z.abs()

        den = (
            (
                (
                    b3.unsqueeze(-1) * t
                    + b2.unsqueeze(-1)
                ) * t
                + b1.unsqueeze(-1)
            ) * t
            + b0.unsqueeze(-1)
        ) * t + 1.0

        out = num / den

        return out.reshape(original_shape)


# ============================================================
# 3. GroupKAN Linear
# ============================================================

class GroupKANLinear(nn.Module):
    """
    GroupKANLinear:

    Input
      ↓
    Group Rational Activation
      ↓
    Dropout
      ↓
    Linear
      ↓
    Output
    """

    def __init__(self, in_features, out_features,
                 num_groups=8, dropout=0.0):
        super().__init__()

        self.act = GroupRationalActivation(
            in_features,
            num_groups=num_groups
        )

        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(
            in_features,
            out_features
        )

    def forward(self, x):
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x


# ============================================================
# 4. Multi-Scale CNN Feature Extraction
# ============================================================

class MultiScaleCNN(nn.Module):
    """
    Input: (B, 1, 22, 1000)

    Three parallel branches:
        Temporal Conv
        Spatial Depthwise Conv
        AvgPool
        Temporal Conv
        AvgPool

    Concat -> 1x1 Conv -> Rearrange

    Output: (B, 15, 16)
    """

    def __init__(
        self,
        channels=22,
        f1=8,
        D=2,
        kernels=(32, 64, 96),
        temporal_kernel=16,
        pool1=8,
        pool2=8,
        dropout=0.3
    ):
        super().__init__()

        f2 = f1 * D

        self.branches = nn.ModuleList()

        for kernel in kernels:

            branch = nn.Sequential(

                # Temporal Convolution
                SamePadTemporalConv2d(
                    1, f1, kernel
                ),

                nn.BatchNorm2d(f1),

                # Depthwise Spatial Convolution
                nn.Conv2d(
                    in_channels=f1,
                    out_channels=f2,
                    kernel_size=(channels, 1),
                    groups=f1,
                    bias=False
                ),

                nn.BatchNorm2d(f2),
                nn.ELU(),

                # First temporal downsampling
                nn.AvgPool2d((1, pool1)),
                nn.Dropout(dropout),

                # Additional Temporal Convolution
                SamePadTemporalConv2d(
                    f2, f2, temporal_kernel
                ),

                nn.BatchNorm2d(f2),
                nn.ELU(),

                # Second temporal downsampling
                nn.AvgPool2d((1, pool2)),
                nn.Dropout(dropout)
            )

            self.branches.append(branch)

        # Fuse the three branches
        self.fuse = nn.Conv2d(
            in_channels=f2 * len(kernels),
            out_channels=f2,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):

        outputs = [
            branch(x) for branch in self.branches
        ]

        # Concatenate on feature channel dimension
        x = torch.cat(outputs, dim=1)

        # Feature fusion
        x = self.fuse(x)

        # (B, F2, 1, T') -> (B, F2, T')
        x = x.squeeze(2)

        # (B, F2, T') -> (B, T', F2)
        x = x.transpose(1, 2).contiguous()

        return x


# ============================================================
# 5. Local-Global Multi-Head Attention
# ============================================================

class LocalGlobalAttention(nn.Module):
    """
    Local-Global Attention:

    Input (B, L, E)
       |
       |-- Q projection
       |
       |-- Local K:
       |      Conv1d(k=3)
       |      Conv1d(k=5)
       |      Concat -> BN -> Dropout -> GR-KAN
       |
       |-- Global K:
       |      GR-KAN
       |
       |-- V projection
       |
       |-- Local Attention
       |-- Global Attention
       |
       |-- Sum -> Output projection

    Output (B, L, E)
    """

    def __init__(
        self,
        dim=16,
        num_heads=4,
        num_groups=8,
        attention_dropout=0.5,
        local_dropout=0.5
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                "dim must be divisible by num_heads"
            )

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Local multi-scale temporal convolutions
        self.local_convs = nn.ModuleList([
            nn.Conv1d(
                dim, dim,
                kernel_size=3,
                padding=1
            ),
            nn.Conv1d(
                dim, dim,
                kernel_size=5,
                padding=2
            )
        ])

        self.local_bn = nn.BatchNorm1d(2 * dim)

        self.local_dropout = nn.Dropout(
            local_dropout
        )

        # GR-KAN projections
        self.w_q = GroupKANLinear(
            dim, dim, num_groups
        )

        self.w_k_local = GroupKANLinear(
            2 * dim, dim, num_groups
        )

        self.w_k_global = GroupKANLinear(
            dim, dim, num_groups
        )

        self.w_v = GroupKANLinear(
            dim, dim, num_groups
        )

        self.w_out = GroupKANLinear(
            dim, dim, num_groups
        )

        self.attn_dropout = nn.Dropout(
            attention_dropout
        )

    def split_heads(self, x):

        B, L, E = x.shape

        return (
            x.reshape(
                B, L,
                self.num_heads,
                self.head_dim
            )
            .transpose(1, 2)
        )

    def forward(self, x):

        B, L, E = x.shape

        # ----------------------------------------
        # Local Key Extraction
        # ----------------------------------------

        local_input = x.transpose(1, 2)

        local_key = torch.cat(
            [
                conv(local_input)
                for conv in self.local_convs
            ],
            dim=1
        )

        local_key = self.local_bn(local_key)
        local_key = self.local_dropout(local_key)

        local_key = local_key.transpose(1, 2)

        # ----------------------------------------
        # Q, K, V Projections
        # ----------------------------------------

        q = self.split_heads(self.w_q(x))

        k_local = self.split_heads(
            self.w_k_local(local_key)
        )

        k_global = self.split_heads(
            self.w_k_global(x)
        )

        v = self.split_heads(self.w_v(x))

        # All: (B, heads, L, head_dim)

        # ----------------------------------------
        # Attention Calculation
        # ----------------------------------------

        def attend(k):

            scores = torch.matmul(
                q,
                k.transpose(-1, -2)
            ) / math.sqrt(self.head_dim)

            weights = F.softmax(
                scores,
                dim=-1
            )

            weights = self.attn_dropout(weights)

            return torch.matmul(weights, v)

        # Local attention
        x_local = attend(k_local)

        # Global attention
        x_global = attend(k_global)

        # ----------------------------------------
        # Feature Fusion
        # ----------------------------------------

        x = x_local + x_global

        # Merge heads
        x = (
            x.transpose(1, 2)
            .contiguous()
            .reshape(B, L, E)
        )

        # Output projection
        x = self.w_out(x)

        return x


# ============================================================
# 6. LG-KAT Transformer Encoder Block
# ============================================================

class LGKATBlock(nn.Module):
    """
    Pre-LN Transformer:

    Input
      ↓
    LayerNorm
      ↓
    Local-Global Attention
      ↓
    Residual Add
      ↓
    LayerNorm
      ↓
    GR-KAN FFN
      ↓
    Residual Add
    """

    def __init__(
        self,
        dim=16,
        heads=4,
        num_groups=8,
        ff_ratio=2,
        attn_dropout=0.5,
        ff_dropout=0.5
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = LocalGlobalAttention(
            dim=dim,
            num_heads=heads,
            num_groups=num_groups,
            attention_dropout=attn_dropout
        )

        self.norm2 = nn.LayerNorm(dim)

        # GR-KAN Feed Forward Network
        self.ffn = nn.Sequential(

            GroupKANLinear(
                dim,
                dim * ff_ratio,
                num_groups
            ),

            nn.Dropout(ff_dropout),

            GroupKANLinear(
                dim * ff_ratio,
                dim,
                num_groups
            ),

            nn.Dropout(ff_dropout)
        )

    def forward(self, x):

        # Attention residual
        x = x + self.attn(self.norm1(x))

        # Feed-forward residual
        x = x + self.ffn(self.norm2(x))

        return x


# ============================================================
# 7. Complete MCTGNet Model
# ============================================================

class MCTGNet(nn.Module):
    """
    MCTGNet for BCI Competition IV 2a.

    Input:
        (B, 1, 22, 1000)
        or (B, 22, 1000)

    Output:
        (B, 4) raw logits

    Architecture:
        Multi-Scale CNN
            ↓
        LG-KAT Encoder
            ↓
        Global Residual
            ↓
        Flatten
            ↓
        GroupKAN Classifier
    """

    def __init__(
        self,
        n_classes=4,
        channels=22,
        samples=1000,

        # CNN
        F1=8,
        D=2,
        kernel_sizes=(32, 64, 96),
        kernelLength2=16,
        pool1=8,
        pool2=8,
        dropoutRate=0.3,

        # Transformer
        emb_size=16,
        heads=4,
        depth=1,
        ff_ratio=2,
        num_groups=8,
        attn_dropout=0.5,
        ff_dropout=0.5,

        # Classifier
        classifier_dropout=0.5
    ):
        super().__init__()

        if F1 * D != emb_size:
            raise ValueError(
                "emb_size must equal F1 * D"
            )

        seq_len = samples // pool1 // pool2

        if seq_len < 1:
            raise ValueError(
                "samples is too short for pooling"
            )

        self.n_classes = n_classes
        self.channels = channels
        self.samples = samples
        self.emb_size = emb_size

        # ----------------------------------------
        # Stage 1: Multi-Scale CNN
        # ----------------------------------------

        self.cnn = MultiScaleCNN(
            channels=channels,
            f1=F1,
            D=D,
            kernels=kernel_sizes,
            temporal_kernel=kernelLength2,
            pool1=pool1,
            pool2=pool2,
            dropout=dropoutRate
        )

        # ----------------------------------------
        # Stage 2: LG-KAT Encoder
        # ----------------------------------------

        self.encoder = nn.Sequential(
            *[
                LGKATBlock(
                    dim=emb_size,
                    heads=heads,
                    num_groups=num_groups,
                    ff_ratio=ff_ratio,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout
                )
                for _ in range(depth)
            ]
        )

        # ----------------------------------------
        # Stage 3: Classification
        # ----------------------------------------

        self.flatten = nn.Flatten()

        self.classifier = GroupKANLinear(
            in_features=emb_size * seq_len,
            out_features=n_classes,
            num_groups=num_groups,
            dropout=classifier_dropout
        )

    def forward(self, x):

        # Accept 3D EEG input
        if x.ndim == 3:
            x = x.unsqueeze(1)

        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                "Expected input (B,1,C,T) or (B,C,T)"
            )

        if (
            x.shape[2] != self.channels
            or x.shape[3] != self.samples
        ):
            raise ValueError(
                f"Expected EEG shape (*, 1, "
                f"{self.channels}, {self.samples}), "
                f"got {tuple(x.shape)}"
            )

        # Stage 1
        cnn_features = self.cnn(x)

        # Stage 2
        transformer_features = self.encoder(
            cnn_features
        )

        # Global residual connection
        features = (
            cnn_features + transformer_features
        )

        # Stage 3
        features = self.flatten(features)

        logits = self.classifier(features)

        # Return raw logits for CrossEntropyLoss
        return logits











def create_model(name, **kwargs):
    """按本模块中的类名创建模型，供 Notebook 选择不同模型。"""
    model_class = globals().get(name)
    if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
        raise ValueError(f"未找到模型 {name!r}；请在 eegnet_model.py 中定义对应的 nn.Module 子类")
    return model_class(**kwargs)