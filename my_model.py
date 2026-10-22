
import torch
from torch import nn
from torch.nn import functional as F


def _itcf_xavier_init(module):
    for layer in module.modules():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class _ITCFGroupAttention(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        if channels % groups or (channels // 4) % groups:
            raise ValueError("GroupAttention requires channels and channels//4 divisible by groups")
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


class _ITCFTemporalSpatialBranch(nn.Module):
    """A branch: temporal 1->8->32, depthwise spatial 32->64.

    First temporal conv: T -> T-1 (symmetric padding for even kernels).
    Second temporal conv: T-1 -> T-1 (asymmetric padding for even kernels).
    Depthwise spatial conv: 22 EEG channels -> spatial height 1.
    """

    def __init__(self, channels, kernel, temporal_hidden, f1, depth_multiplier):
        super().__init__()
        if kernel < 2:
            raise ValueError("Each temporal kernel must be >= 2")
        if temporal_hidden < 1 or f1 < 1 or depth_multiplier < 1:
            raise ValueError("temporal_hidden, f1, and depth_multiplier must be positive")

        # A stride=1 kernel of width K needs total padding K-2 to reduce T by 1.
        first_left = (kernel - 2) // 2
        first_right = (kernel - 2) - first_left
        # Total padding K-1 preserves the input length at stride=1.
        second_left = (kernel - 1) // 2
        second_right = (kernel - 1) - second_left

        self.temporal1 = nn.Sequential(
            nn.ConstantPad2d((first_left, first_right, 0, 0), 0),
            nn.Conv2d(1, temporal_hidden, (1, kernel), bias=False),
            nn.BatchNorm2d(temporal_hidden),
            nn.ELU(),
        )
        self.temporal2 = nn.Sequential(
            nn.ConstantPad2d((second_left, second_right, 0, 0), 0),
            nn.Conv2d(temporal_hidden, f1, (1, kernel), bias=False),
            nn.BatchNorm2d(f1),
            nn.ELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                f1,
                f1 * depth_multiplier,
                kernel_size=(channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
        )

    def forward(self, x):
        x = self.temporal1(x)
        x = self.temporal2(x)
        return self.spatial(x)


class _ITCFConvStem(nn.Module):
    """Stage 1 plus the original Stage 2; output is (B,model_dim,L)."""

    def __init__(
        self,
        channels,
        temporal_kernels,
        temporal_hidden,
        f1,
        depth_multiplier,
        group_dim,
        pool1,
        pool2,
        dropout,
    ):
        super().__init__()
        groups = len(temporal_kernels)
        spatial_channels = f1 * depth_multiplier * groups
        model_dim = group_dim * groups

        self.branches = nn.ModuleList([
            _ITCFTemporalSpatialBranch(
                channels=channels,
                kernel=kernel,
                temporal_hidden=temporal_hidden,
                f1=f1,
                depth_multiplier=depth_multiplier,
            )
            for kernel in temporal_kernels
        ])
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(dropout)

        # Original Grouped Reduction + Temporal Conv + Group Attention.
        self.reduction = nn.Sequential(
            nn.Conv2d(
                spatial_channels, model_dim, 1, groups=groups, bias=False
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
        x = x.unsqueeze(1)  # (B, C, T) -> (B, 1, C, T)
        branches = [branch(x) for branch in self.branches]
        x = torch.cat(branches, dim=1)  # (B,192,1,1000)
        x = self.drop1(self.pool1(x))   # (B,192,1,125)
        x = self.grouped_temporal(self.reduction(x))
        x = x + self.group_attention(x)
        x = self.drop2(self.pool2(x))   # (B,48,1,17)
        return x.squeeze(2)             # (B,48,17)


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
    # _itcf_rotate pairs adjacent (even, odd) coordinates, so repeat adjacent angles.
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
    return angles.cos(), angles.sin()


def _itcf_rotate(x):
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class _ITCFGroupedQueryAttention(nn.Module):
    def __init__(self, model_dim, query_heads, kv_heads, dropout):
        super().__init__()
        if model_dim % query_heads or query_heads % kv_heads:
            raise ValueError("model_dim, query_heads, and kv_heads are incompatible")
        if (model_dim // query_heads) % 2:
            raise ValueError("RoPE requires an even head_dim")

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


class ImprovedTCFormerV1(nn.Module):
    """Four-class EEG classifier returning (B,n_classes) raw logits.

    Supported inputs:
      (B, channels, T), (B, 1, channels, T), (B, channels, 1, T)
    """

    def __init__(
        self,
        n_classes=4,
        channels=22,
        samples=1001,
        temporal_kernels=(16, 32, 64),
        temporal_hidden=8,
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
            raise ValueError("trans_depth and tcn_depth must be >= 1")
        if len(temporal_kernels) != 3:
            raise ValueError("Exactly 3 temporal convolution branches are required")
        if pool1 < 1 or pool2 < 1:
            raise ValueError("pool1 and pool2 must be positive")

        self.channels = channels
        self.samples = samples
        self.n_classes = n_classes
        self.groups = len(temporal_kernels)
        self.pool1 = pool1
        self.pool2 = pool2
        model_dim = group_dim * self.groups

        self.stem = _ITCFConvStem(
            channels, temporal_kernels, temporal_hidden, f1,
            depth_multiplier, group_dim, pool1, pool2, conv_dropout
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
        if x.ndim == 4:
            if x.shape[1] == 1:               # (B,1,C,T)
                x = x.squeeze(1)
            elif x.shape[2] == 1:             # (B,C,1,T)
                x = x.squeeze(2)
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(
                f"Input must have shape (B,{self.channels},T), "
                f"(B,1,{self.channels},T), or (B,{self.channels},1,T); "
                f"received {tuple(x.shape)}"
            )
        # First temporal conv decreases T by one; pool1 and pool2 each floor-divide.
        if x.shape[-1] < self.pool1 * self.pool2 + 1:
            raise ValueError(
                f"T must be >= {self.pool1 * self.pool2 + 1} "
                f"for pool1={self.pool1}, pool2={self.pool2}"
            )

        conv_features = self.stem(x)                     # (B,48,17)
        tokens = self.mix(conv_features).transpose(1, 2)  # (B,17,48)

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
        )                                                    # (B,64,17)
        features = self.tcn(features)[:, :, -1:]              # (B,64,1)
        logits = self.classifier(features).squeeze(-1)        # (B,16)
        return logits.reshape(
            x.shape[0], self.groups + 1, self.n_classes
        ).mean(dim=1)                                        # (B,4)










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
