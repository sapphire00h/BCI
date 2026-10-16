"""EEGNet 及其他模型的定义。

新模型可在本文件中定义为 nn.Module 子类，再由 create_model 按类名创建。
"""
import torch
import torch.nn as nn

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




def create_model(name, **kwargs):
    """按本模块中的类名创建模型，供 Notebook 选择不同模型。"""
    model_class = globals().get(name)
    if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
        raise ValueError(f"未找到模型 {name!r}；请在 eegnet_model.py 中定义对应的 nn.Module 子类")
    return model_class(**kwargs)