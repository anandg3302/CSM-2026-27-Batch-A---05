"""
Standalone pure PyTorch reconstruction of the RealForensics 3D-CSN model architecture.

This file provides a clean, dependency-free reference implementation of:
- CSN-101 (Channel-Separated Convolutional Network, ResNet-101 backbone)
- MeanLinear classification head (with optional cosine-normalized linear layer)

Reference:
- "Leveraging Real Talking Faces via Self-Supervision for Robust Forgery Detection"
  Alexandros Haliassos, Rodrigo Mira, Stavros Petridis, Maja Pantic (CVPR 2022)
- Official repository: https://github.com/ahaliassos/RealForensics
"""

import math
from typing import Callable, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaModule(nn.Module):
    def __init__(self, lambd: Callable):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd(x)


class ResNetBasicStem(nn.Module):
    def __init__(
        self,
        conv: nn.Module,
        norm: nn.Module = None,
        activation: nn.Module = None,
        pool: nn.Module = None,
    ):
        super().__init__()
        self.conv = conv
        self.norm = norm
        self.activation = activation
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        conv_a: nn.Module,
        norm_a: nn.Module,
        act_a: nn.Module,
        conv_b: nn.Module,
        norm_b: nn.Module,
        act_b: nn.Module,
        conv_c: nn.Module,
        norm_c: nn.Module,
    ):
        super().__init__()
        self.conv_a = conv_a
        self.norm_a = norm_a
        self.act_a = act_a
        self.conv_b = conv_b
        self.norm_b = norm_b
        self.act_b = act_b
        self.conv_c = conv_c
        self.norm_c = norm_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_a(x)
        if self.norm_a is not None:
            x = self.norm_a(x)
        if self.act_a is not None:
            x = self.act_a(x)
        x = self.conv_b(x)
        if self.norm_b is not None:
            x = self.norm_b(x)
        if self.act_b is not None:
            x = self.act_b(x)
        x = self.conv_c(x)
        if self.norm_c is not None:
            x = self.norm_c(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        branch1_conv: nn.Module = None,
        branch1_norm: nn.Module = None,
        branch2: nn.Module = None,
        activation: nn.Module = None,
    ):
        super().__init__()
        self.branch1_conv = branch1_conv
        self.branch1_norm = branch1_norm
        self.branch2 = branch2
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.branch1_conv is None:
            shortcut = x
        else:
            shortcut = self.branch1_conv(x)
            if self.branch1_norm is not None:
                shortcut = self.branch1_norm(shortcut)
        out = shortcut + self.branch2(x)
        if self.activation is not None:
            out = self.activation(out)
        return out


class ResStage(nn.Module):
    def __init__(self, res_blocks: nn.ModuleList):
        super().__init__()
        self.res_blocks = res_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x)
        return x


class Net(nn.Module):
    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def create_bottleneck(
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    conv_a_kernel_size=(1, 1, 1),
    conv_a_stride=(1, 1, 1),
    conv_a_padding=(0, 0, 0),
    conv_b_kernel_size=(3, 3, 3),
    conv_b_stride=(1, 1, 1),
    conv_b_padding=(1, 1, 1),
    conv_b_num_groups=1,
    norm_eps=1e-5,
    norm_momentum=0.1,
) -> BottleneckBlock:
    conv_a = nn.Conv3d(
        dim_in,
        dim_inner,
        kernel_size=conv_a_kernel_size,
        stride=conv_a_stride,
        padding=conv_a_padding,
        bias=False,
    )
    norm_a = nn.BatchNorm3d(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
    act_a = nn.ReLU()
    conv_b = nn.Conv3d(
        dim_inner,
        dim_inner,
        kernel_size=conv_b_kernel_size,
        stride=conv_b_stride,
        padding=conv_b_padding,
        groups=conv_b_num_groups,
        bias=False,
    )
    norm_b = nn.BatchNorm3d(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
    act_b = nn.ReLU()
    conv_c = nn.Conv3d(dim_inner, dim_out, kernel_size=(1, 1, 1), bias=False)
    norm_c = nn.BatchNorm3d(num_features=dim_out, eps=norm_eps, momentum=norm_momentum)
    return BottleneckBlock(conv_a, norm_a, act_a, conv_b, norm_b, act_b, conv_c, norm_c)


def build_csn_r101(input_channel: int = 3) -> Net:
    """
    Constructs the ir-CSN-101 backbone according to the official RealForensics configuration.
    Stage depths: (3, 4, 23, 3).
    Temporal strides are set to 1 throughout the stages (csn_temporal_no_head).
    """
    stage_depths = (3, 4, 23, 3)
    bottleneck_ratio = 4

    stem_conv = nn.Conv3d(
        input_channel,
        64,
        kernel_size=(3, 7, 7),
        stride=(1, 2, 2),
        padding=(1, 3, 3),
        bias=False,
    )
    stem_norm = nn.BatchNorm3d(num_features=64, eps=1e-5, momentum=0.1)
    stem_act = nn.ReLU()
    stem_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
    stem = ResNetBasicStem(conv=stem_conv, norm=stem_norm, activation=stem_act, pool=stem_pool)

    blocks = [stem]
    stage_dim_in = 64
    stage_dim_out = stage_dim_in * 4  # 256

    stage_spatial_strides = (1, 2, 2, 2)
    stage_temporal_strides = (1, 1, 1, 1)

    for idx, depth in enumerate(stage_depths):
        stage_dim_inner = stage_dim_out // bottleneck_ratio
        stage_conv_b_stride = (
            stage_temporal_strides[idx],
            stage_spatial_strides[idx],
            stage_spatial_strides[idx],
        )
        conv_b_groups = stage_dim_inner // 1  # Channel-separated convolution (group size 1)

        res_blocks = []
        for b_idx in range(depth):
            d_in = stage_dim_in if b_idx == 0 else stage_dim_out
            d_out = stage_dim_out
            b_stride = stage_conv_b_stride if b_idx == 0 else (1, 1, 1)

            # Shortcut branch
            branch1_conv_stride = b_stride
            has_shortcut = (d_in != d_out) or (branch1_conv_stride != (1, 1, 1))
            if has_shortcut:
                branch1_conv = nn.Conv3d(
                    d_in,
                    d_out,
                    kernel_size=(1, 1, 1),
                    stride=branch1_conv_stride,
                    bias=False,
                )
                branch1_norm = nn.BatchNorm3d(num_features=d_out, eps=1e-5, momentum=0.1)
            else:
                branch1_conv = None
                branch1_norm = None

            branch2 = create_bottleneck(
                dim_in=d_in,
                dim_inner=stage_dim_inner,
                dim_out=d_out,
                conv_a_kernel_size=(1, 1, 1),
                conv_a_stride=(1, 1, 1),
                conv_a_padding=(0, 0, 0),
                conv_b_kernel_size=(3, 3, 3),
                conv_b_stride=b_stride,
                conv_b_padding=(1, 1, 1),
                conv_b_num_groups=conv_b_groups,
            )
            res_block = ResBlock(branch1_conv, branch1_norm, branch2, activation=nn.ReLU())
            res_blocks.append(res_block)

        blocks.append(ResStage(nn.ModuleList(res_blocks)))
        stage_dim_in = stage_dim_out
        stage_dim_out = stage_dim_out * 2

    blocks.append(LambdaModule(lambda x: x.mean(dim=(3, 4))))
    return Net(nn.ModuleList(blocks))


class MeanLinear(nn.Module):
    """
    Mean pooling classification head.
    Averages features across the temporal dimension, then computes cosine similarity
    against the normalized weight vector multiplied by scale factor.
    """
    def __init__(self, in_dim: int = 2048, out_dim: int = 1, norm_linear: bool = True, scale: float = 64.0):
        super().__init__()
        self.scale = scale
        self.norm_linear = norm_linear
        if norm_linear:
            self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 2048, T)
        x = x.mean(-1)  # Average over temporal dimension: (B, 2048)
        if self.norm_linear:
            return F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1)) * self.scale
        return self.linear(x)


class RealForensicsModel(nn.Module):
    """
    Complete RealForensics classification model for binary face-forgery detection.
    """
    def __init__(self):
        super().__init__()
        self.backbone = build_csn_r101(input_channel=3)
        self.df_head = MeanLinear(in_dim=2048, out_dim=1, norm_linear=True, scale=64.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 3, T, H, W)
        features = self.backbone(x)  # (B, 2048, T)
        logits = self.df_head(features)  # (B, 1)
        return logits
