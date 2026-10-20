
import torch
from torch import nn
from torch.nn import functional as F


# ============================================================
# Stage 1: Multi-Kernel CNN Stem
# ============================================================

class SameTemporalConv2d(nn.Module):
    """
    时间卷积，使用非对称 Padding 保持时间长度不变。

    Input : (B, Cin, H, T)
    Output: (B, Cout, H, T)
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        left = (kernel_size - 1) // 2
        right = kernel_size - 1 - left

        self.pad = nn.ZeroPad2d((left, right, 0, 0))

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            bias=False,
        )

    def forward(self, x):
        return self.conv(self.pad(x))


class MultiKernelCNNStem(nn.Module):
    """
    多尺度时间卷积 + Depthwise EEG 空间卷积。

    Input : (B, 1, 22, 1000)
    Output: (B, 96, 1, 125)
    """

    def __init__(
        self,
        channels=22,
        kernels=(16, 32, 64),
        filters_per_kernel=16,
        depth_multiplier=2,
        pool_size=8,
        dropout=0.3,
    ):
        super().__init__()

        if not kernels or any(k < 1 for k in kernels):
            raise ValueError("kernels 必须是非空正整数序列")

        # Block 1: 三个并行时间卷积
        self.temporal_branches = nn.ModuleList([
            nn.Sequential(
                SameTemporalConv2d(
                    1, filters_per_kernel, k
                ),
                nn.BatchNorm2d(filters_per_kernel),
            )
            for k in kernels
        ])

        temporal_channels = len(kernels) * filters_per_kernel

        self.out_channels = temporal_channels * depth_multiplier

        # Block 2: Depthwise Spatial Convolution
        self.spatial = nn.Sequential(
            nn.Conv2d(
                temporal_channels,
                self.out_channels,
                kernel_size=(channels, 1),
                groups=temporal_channels,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ELU(),

            nn.AvgPool2d(
                (1, pool_size),
                stride=(1, pool_size),
            ),

            nn.Dropout(dropout),
        )

    def forward(self, x):

        # 三个时间卷积分支:
        # 3 × (B,16,22,1000)
        # -> (B,48,22,1000)
        x = torch.cat(
            [branch(x) for branch in self.temporal_branches],
            dim=1,
        )

        # -> (B,96,1,125)
        return self.spatial(x)


# ============================================================
# Stage 2: Three-Branch Temporal Feature Extraction
# ============================================================

class CausalConv1d(nn.Module):
    """
    因果卷积，只在时间轴左侧补零。

    Input : (B,C,T)
    Output: (B,C,T)
    """

    def __init__(
        self,
        channels,
        kernel_size=4,
        dilation=1,
    ):
        super().__init__()

        self.left_padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        return self.conv(
            F.pad(x, (self.left_padding, 0))
        )


class TCNBlock(nn.Module):
    """
    TCN Block:
    CausalConv -> BN -> ELU -> Dropout
    CausalConv -> BN -> ELU -> Dropout
    Residual Add -> ELU
    """

    def __init__(
        self,
        features,
        kernel_size=4,
        dilation=1,
        dropout=0.3,
    ):
        super().__init__()

        self.body = nn.Sequential(
            CausalConv1d(
                features, kernel_size, dilation
            ),
            nn.BatchNorm1d(features),
            nn.ELU(),
            nn.Dropout(dropout),

            CausalConv1d(
                features, kernel_size, dilation
            ),
            nn.BatchNorm1d(features),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.activation = nn.ELU()

    def forward(self, x):
        return self.activation(
            x + self.body(x)
        )


class InvertedBottleneckBranch(nn.Module):
    """
    Branch A: Dilated Inverted Bottleneck

    Input : (B,96,1,125)
    Output: (B,16,1,125)
    """

    def __init__(
        self,
        in_channels=96,
        out_channels=16,
        expansion=2,
    ):
        super().__init__()

        hidden = in_channels * expansion

        self.layers = nn.Sequential(

            # 96 -> 192
            nn.Conv2d(
                in_channels, hidden, 1, bias=False
            ),
            nn.BatchNorm2d(hidden),
            nn.ELU(),

            # DWConv: dilation=2
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=(1, 5),
                padding=(0, 4),
                dilation=(1, 2),
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ELU(),

            # DWConv: dilation=4
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=(1, 5),
                padding=(0, 8),
                dilation=(1, 4),
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ELU(),

            # Projection: 192 -> 16
            nn.Conv2d(
                hidden, out_channels, 1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class TCNContextGateBranch(nn.Module):
    """
    Branch B: TCN Context Gating

    Input : (B,96,1,125)
    Output: (B,16,1,125)

    使用 TCN 最后一个时间位置生成通道门控权重，
    再调制池化前保存的完整 125 步特征。
    """

    def __init__(
        self,
        in_channels=96,
        out_channels=16,
        pool_size=7,
        tcn_kernel=4,
        dropout=0.3,
    ):
        super().__init__()

        # Channel Projection: 96 -> 16
        self.project = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )

        # Temporal Downsampling: 125 -> 17
        self.pool = nn.AvgPool1d(
            pool_size,
            stride=pool_size,
        )

        # 两个 TCN Block
        self.tcn = nn.Sequential(
            TCNBlock(
                out_channels,
                kernel_size=tcn_kernel,
                dilation=1,
                dropout=dropout,
            ),
            TCNBlock(
                out_channels,
                kernel_size=tcn_kernel,
                dilation=2,
                dropout=dropout,
            ),
        )

        # 通道门控
        self.gate = nn.Sequential(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):

        # (B,96,1,125) -> (B,16,125)
        features = self.project(x).squeeze(2)

        # (B,16,125) -> (B,16,17)
        context = self.pool(features)

        # (B,16,17)
        context = self.tcn(context)

        # 选取最后一个时间位置: (B,16,1)
        context = context[:, :, -1:]

        # 每个通道一个门控权重
        weights = self.gate(context)

        # 广播乘法: (B,16,125)
        gated_features = features * weights

        return gated_features.unsqueeze(2)


class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention。

    Input / Output: (B,48,1,125)
    """

    def __init__(
        self,
        channels=48,
        reduction=4,
    ):
        super().__init__()

        if channels % reduction:
            raise ValueError(
                "channels 必须能被 reduction 整除"
            )

        hidden = channels // reduction

        self.weights = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.weights(x)


class ThreeBranchFeatureExtractor(nn.Module):
    """
    Stage 2:

    Branch A: Inverted Bottleneck + Dilated DWConv
    Branch B: TCN Context Gating
    Branch C: Standard 1x1 Convolution

    Concat -> SE -> Residual Projection -> ELU

    Input : (B,96,1,125)
    Output: (B,48,1,125)
    """

    def __init__(
        self,
        in_channels=96,
        branch_channels=16,
        dropout_tcn=0.3,
    ):
        super().__init__()

        out_channels = branch_channels * 3

        self.branch_a = InvertedBottleneckBranch(
            in_channels,
            branch_channels,
        )

        self.branch_b = TCNContextGateBranch(
            in_channels,
            branch_channels,
            dropout=dropout_tcn,
        )

        self.branch_c = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=1,
                groups=1,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ELU(),
        )

        self.se = SEAttention(
            out_channels,
            reduction=4,
        )

        # Residual: 96 -> 48
        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.activation = nn.ELU()

    def forward(self, x):

        a = self.branch_a(x)
        b = self.branch_b(x)
        c = self.branch_c(x)

        # (B,48,1,125)
        features = torch.cat(
            (a, b, c),
            dim=1,
        )

        # SE attention
        features = self.se(features)

        # Residual fusion
        features = features + self.residual(x)

        return self.activation(features)


# ============================================================
# Stage 3: Downsampling and Feature Mixing
# ============================================================

class DownsampleAndMix(nn.Module):
    """
    AveragePool -> Dropout -> Squeeze
    -> Conv1d -> BN -> SiLU -> Transpose

    Input : (B,48,1,125)
    Output: (B,17,48)
    """

    def __init__(
        self,
        channels=48,
        pool_size=7,
        dropout=0.3,
    ):
        super().__init__()

        self.pool = nn.AvgPool2d(
            (1, pool_size),
            stride=(1, pool_size),
        )

        self.dropout = nn.Dropout(dropout)

        self.mix = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=1,
                groups=1,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
        )

    def forward(self, x):

        # (B,48,1,125) -> (B,48,1,17)
        x = self.pool(x)
        x = self.dropout(x)

        # -> (B,48,17)
        x = x.squeeze(2)

        # Feature Mixing
        x = self.mix(x)

        # -> (B,17,48)
        return x.transpose(1, 2)


# ============================================================
# Stage 4: Transformer Encoder with GQA and RoPE
# ============================================================

class DropPath(nn.Module):
    """Stochastic Depth: 按样本随机丢弃残差路径。"""

    def __init__(self, probability=0.0):
        super().__init__()

        if not 0 <= probability < 1:
            raise ValueError(
                "DropPath probability 必须在 [0,1) 内"
            )

        self.probability = probability

    def forward(self, x):

        if not self.training or self.probability == 0:
            return x

        keep = 1.0 - self.probability

        mask_shape = (
            (x.shape[0],) + (1,) * (x.ndim - 1)
        )

        mask = (
            torch.rand(mask_shape, device=x.device) < keep
        ).to(x.dtype)

        return x * mask / keep


def rotary_cache(seq_len, head_dim, device, dtype):
    """
    构建 Rotary Position Embedding。

    使用相邻偶奇维成对旋转的频率布局。

    Output cos/sin: (1,1,T,head_dim)
    """

    if head_dim % 2:
        raise ValueError("RoPE 要求 head_dim 为偶数")

    inv_freq = 1.0 / (
        10000.0 ** (
            torch.arange(
                0, head_dim, 2, device=device
            ).float() / head_dim
        )
    )

    positions = torch.arange(
        seq_len, device=device
    ).float()

    angles = torch.outer(positions, inv_freq)

    # [f0,f0,f1,f1,...]，与 rotate_half 匹配
    angles = angles.repeat_interleave(2, dim=-1)

    cos = angles.cos().to(dtype=dtype)[None, None, :, :]
    sin = angles.sin().to(dtype=dtype)[None, None, :, :]

    return cos, sin


def rotate_half(x):
    """
    每两个相邻维度做旋转：

    (x0,x1,x2,x3)
    -> (-x1,x0,-x3,x2)
    """

    even = x[..., ::2]
    odd = x[..., 1::2]

    return torch.stack(
        (-odd, even),
        dim=-1,
    ).flatten(-2)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Self-Attention + RoPE。

    Q: (B,8,T,6)
    K: (B,4,T,6)
    V: (B,4,T,6)

    Output: (B,T,48)

    使用双向 self-attention，不添加 causal mask。
    """

    def __init__(
        self,
        d_model=48,
        q_heads=8,
        kv_heads=4,
        dropout=0.4,
    ):
        super().__init__()

        if d_model % q_heads or q_heads % kv_heads:
            raise ValueError(
                "d_model 必须整除 q_heads，"
                "q_heads 必须整除 kv_heads"
            )

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = d_model // q_heads

        if self.head_dim % 2:
            raise ValueError(
                "RoPE 的 head_dim 必须为偶数"
            )

        self.q_proj = nn.Linear(
            d_model,
            q_heads * self.head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            d_model,
            kv_heads * self.head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            d_model,
            kv_heads * self.head_dim,
            bias=False,
        )

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):

        b, t, d = x.shape

        # Q: (B,8,T,6)
        q = self.q_proj(x).reshape(
            b, t, self.q_heads, self.head_dim
        ).transpose(1, 2)

        # K: (B,4,T,6)
        k = self.k_proj(x).reshape(
            b, t, self.kv_heads, self.head_dim
        ).transpose(1, 2)

        # V: (B,4,T,6)
        v = self.v_proj(x).reshape(
            b, t, self.kv_heads, self.head_dim
        ).transpose(1, 2)

        # RoPE 只作用于 Q 和 K
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA: 4 KV heads -> 8 heads
        repeats = self.q_heads // self.kv_heads

        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        # Attention Scores: (B,8,T,T)
        scores = (
            q @ k.transpose(-2, -1)
        ) * (self.head_dim ** -0.5)

        attention = scores.softmax(dim=-1)
        attention = self.attn_dropout(attention)

        # Attention Output: (B,8,T,6)
        values = attention @ v

        # 合并 Heads: (B,T,48)
        values = values.transpose(
            1, 2
        ).contiguous().reshape(b, t, d)

        return self.out_proj(values)


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer Encoder。

    LN -> GQA + RoPE -> DropPath -> Residual
    LN -> FFN -> Dropout -> DropPath -> Residual
    """

    def __init__(
        self,
        d_model=48,
        q_heads=8,
        kv_heads=4,
        dropout=0.4,
        drop_path=0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attention = GroupedQueryAttention(
            d_model,
            q_heads,
            kv_heads,
            dropout,
        )

        self.path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )

        self.path2 = DropPath(drop_path)

    def forward(self, x, cos, sin):

        # Attention Residual
        x = x + self.path1(
            self.attention(
                self.norm1(x),
                cos,
                sin,
            )
        )

        # FFN Residual
        x = x + self.path2(
            self.ffn(self.norm2(x))
        )

        return x


# ============================================================
# Stage 5: Temporal Feature Aggregation
# ============================================================

class TemporalFeatureAggregation(nn.Module):
    """
    Attention Pooling + GAP + Learnable Weighted Fusion。

    Input : (B,17,48)
    Output: (B,48)

    Fusion:
        lambda = sigmoid(theta)
        output = lambda * attended
               + (1-lambda) * averaged
    """

    def __init__(
        self,
        d_model=48,
        attention_hidden=16,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        self.scorer = nn.Sequential(
            nn.Linear(d_model, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )

        # 初始 lambda = sigmoid(0) = 0.5
        self.fusion_logit = nn.Parameter(
            torch.tensor(0.0)
        )

    def forward(self, x):

        # (B,17,48)
        x = self.norm(x)

        # Attention Scoring: (B,17,1)
        scores = self.scorer(x)

        # 沿时间维度 Softmax
        weights = scores.softmax(dim=1)

        # Weighted Sum: (B,48)
        attended = (weights * x).sum(dim=1)

        # Global Average Pool: (B,48)
        averaged = x.mean(dim=1)

        # 可学习标量 lambda
        lam = torch.sigmoid(self.fusion_logit)

        # Weighted Fusion: (B,48)
        return (
            lam * attended
            + (1.0 - lam) * averaged
        )


# ============================================================
# Stage 6: Complete Improved TCFormer V1.0
# ============================================================

class ImprovedTCFormerV1(nn.Module):
    """
    改进版 TCFormer V1.0。

    Stage 1: Multi-Kernel CNN Stem
    Stage 2: Three-Branch Temporal Extraction + SE
    Stage 3: Downsampling and Feature Mixing
    Stage 4: Transformer Encoder x5 (GQA + RoPE)
    Stage 5: Attention Pooling + GAP Fusion
    Stage 6: Classification Head

    Input:
        (B,1,22,1000)
        或 (B,22,1000)

    Output:
        (B,4) logits
    """

    def __init__(
        self,
        n_classes=4,
        channels=22,
        samples=1000,

        # Stage 1
        temporal_kernels=(16, 32, 64),
        f1=16,
        depth_multiplier=2,
        pool1=8,

        # Stage 2
        branch_channels=16,
        tcn_dropout=0.3,

        # Stage 3
        pool2=7,
        conv_dropout=0.3,

        # Stage 4
        d_model=48,
        trans_depth=5,
        q_heads=8,
        kv_heads=4,
        trans_dropout=0.4,
        drop_path_max=0.25,

        # Stage 6
        classifier_dropout=0.3,
    ):
        super().__init__()

        self.channels = channels
        self.samples = samples

        stem_channels = (
            len(temporal_kernels)
            * f1
            * depth_multiplier
        )

        feature_channels = branch_channels * 3

        if feature_channels != d_model:
            raise ValueError(
                "branch_channels * 3 必须等于 d_model"
            )

        if trans_depth < 1:
            raise ValueError(
                "trans_depth 必须 >= 1"
            )

        # ----------------------------------------------------
        # Stage 1: Multi-Kernel CNN Stem
        # ----------------------------------------------------

        self.stem = MultiKernelCNNStem(
            channels=channels,
            kernels=temporal_kernels,
            filters_per_kernel=f1,
            depth_multiplier=depth_multiplier,
            pool_size=pool1,
            dropout=conv_dropout,
        )

        # ----------------------------------------------------
        # Stage 2: Three-Branch Feature Extraction
        # ----------------------------------------------------

        self.three_branch = ThreeBranchFeatureExtractor(
            in_channels=stem_channels,
            branch_channels=branch_channels,
            dropout_tcn=tcn_dropout,
        )

        # ----------------------------------------------------
        # Stage 3: Downsampling & Feature Mixing
        # ----------------------------------------------------

        self.downsample_mix = DownsampleAndMix(
            channels=d_model,
            pool_size=pool2,
            dropout=conv_dropout,
        )

        # ----------------------------------------------------
        # Stage 4: Transformer Encoder x5
        # ----------------------------------------------------

        drop_rates = (
            torch.linspace(0, 1, trans_depth) ** 2
            * drop_path_max
        ).tolist()

        self.transformer = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                q_heads=q_heads,
                kv_heads=kv_heads,
                dropout=trans_dropout,
                drop_path=drop_rates[i],
            )
            for i in range(trans_depth)
        ])

        # ----------------------------------------------------
        # Stage 5: Attention Pooling + GAP
        # ----------------------------------------------------

        self.aggregation = TemporalFeatureAggregation(
            d_model=d_model,
            attention_hidden=16,
        )

        # ----------------------------------------------------
        # Stage 6: Classification Head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(classifier_dropout),
            nn.Linear(d_model, n_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络参数。"""

        for module in self.modules():

            if isinstance(
                module,
                (nn.Conv1d, nn.Conv2d, nn.Linear)
            ):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(
                module,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.LayerNorm,
                )
            ):
                if module.weight is not None:
                    nn.init.ones_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x):
        """
        返回各 Stage 中间特征，便于检查形状、
        提取特征以及进行消融实验。
        """

        # 兼容两种 EEG 输入形式
        if x.ndim == 3:
            x = x.unsqueeze(1)

        if (
            x.ndim != 4
            or x.shape[1] != 1
            or x.shape[2] != self.channels
        ):
            raise ValueError(
                f"期待 (B,1,{self.channels},T) "
                f"或 (B,{self.channels},T)"
            )

        if x.shape[-1] < 56:
            raise ValueError(
                "输入时间长度必须 >= pool1*pool2 = 56"
            )

        # Stage 1: (B,96,1,125)
        stem = self.stem(x)

        # Stage 2: (B,48,1,125)
        extracted = self.three_branch(stem)

        # Stage 3: (B,17,48)
        tokens = self.downsample_mix(extracted)
        mixed_tokens = tokens

        # 生成 RoPE
        head_dim = self.transformer[0].attention.head_dim

        cos, sin = rotary_cache(
            tokens.shape[1],
            head_dim,
            tokens.device,
            tokens.dtype,
        )

        # Stage 4: Transformer Encoder x5
        for layer in self.transformer:
            tokens = layer(tokens, cos, sin)

        # Stage 5: (B,48)
        pooled = self.aggregation(tokens)

        return {
            "stage1": stem,
            "stage2": extracted,
            "stage3": mixed_tokens,
            "stage4": tokens,
            "stage5": pooled,
        }

    def forward(self, x):

        features = self.forward_features(x)

        # Stage 6: (B,4)
        logits = self.classifier(features["stage5"])

        return logits


# ============================================================
# Model Test
# ============================================================

def demo():

    torch.manual_seed(42)

    model = ImprovedTCFormerV1()

    # 模拟 BCI Competition IV-2a EEG 数据
    x = torch.randn(2, 1, 22, 1000)

    # 0~3 四分类标签
    y = torch.tensor([0, 3], dtype=torch.long)

    # --------------------------------------------------------
    # 1. 前向传播与形状检查
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        features = model.forward_features(x)

        for stage, output in features.items():
            print(stage, tuple(output.shape))

        logits = model(x)

        print("logits:", tuple(logits.shape))

        print(
            "probabilities:",
            logits.softmax(dim=-1),
        )

    # --------------------------------------------------------
    # 2. 训练模式与反向传播测试
    # --------------------------------------------------------

    model.train()

    logits = model(x)

    criterion = nn.CrossEntropyLoss()

    loss = criterion(logits, y)

    loss.backward()

    print("loss:", loss.item())

    print(
        "classifier gradient valid:",
        model.classifier[-1].weight.grad is not None,
    )

    print(
        "trainable params:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
    )


if __name__ == "__main__":
    demo()
