"""Cross-Modal Spatial Gating (LSKA) networks for CBCT/MRI -> CT synthesis.

This module contains only the model components used by the experiments in this
repository:

* `Pix2PixRRDB_LSKA` -- RRDB generator with a Large Separable Kernel Attention
  (LSKA) gate applied to the concatenated multi-modal input.
* `Pix2PixRRDB_MultiModalLSKA` -- same idea but with independent modality stems
  and a reliability-aware fusion gate (LSKA + residual SE) before the RRDB trunk.

A minimal, self-contained BCE PatchAdversarialLoss is included so that no
external MONAI-Generative package is required.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia
from kornia.filters import SpatialGradient


# ---------------------------------------------------------------------------
# Minimal Patch adversarial loss (BCE), self-contained.
# ---------------------------------------------------------------------------
class PatchAdversarialLoss(nn.Module):
    """BCE patch adversarial loss compatible with the MONAI-Generative API."""

    def __init__(self, criterion: str = "bce"):
        super().__init__()
        if criterion.lower() not in ("bce", "least_squares", "hinge"):
            raise ValueError(f"Unrecognised adversarial criterion: {criterion}")
        self.criterion = criterion.lower()
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        fill = 1.0 if target_is_real else 0.0
        return torch.full_like(input, fill)

    def forward(self, input, target_is_real: bool, for_discriminator: bool):
        if isinstance(input, (list, tuple)):
            return sum(
                self.forward(x, target_is_real, for_discriminator) for x in input
            ) / len(input)
        target = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target)


adversarial_loss = PatchAdversarialLoss(criterion="bce")


# ---------------------------------------------------------------------------
# Patch discriminator
# ---------------------------------------------------------------------------
class L1SSIMLoss(nn.Module):
    """Weighted combination of L1 and SSIM losses (inputs assumed in [-1, 1])."""

    def __init__(self, alpha=0.84):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = kornia.losses.SSIMLoss(window_size=11, reduction='mean')
        self.alpha = alpha

    def forward(self, x, y):
        l1_loss = self.l1(x, y)
        ssim_loss = self.ssim((x + 1) / 2, (y + 1) / 2)
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels, num_filters=64, num_layers=3, strides=None):
        super().__init__()
        if strides is None:
            strides = [2] * (num_layers - 2) + [1, 1]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=strides[0], padding=1))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))
        for i in range(1, num_layers - 1):
            self.layers.append(nn.Conv2d(num_filters * 2 ** (i - 1), num_filters * 2 ** i, kernel_size=4, stride=strides[i], padding=1))
            self.layers.append(nn.InstanceNorm2d(num_filters * 2 ** i))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers.append(nn.Conv2d(num_filters * 2 ** (num_layers - 2), 1, kernel_size=4, stride=strides[-1], padding=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels, num_d=3, num_filters=64, num_layers_d=3):
        super().__init__()
        self.num_d = num_d
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.discriminators = nn.ModuleList(
            [PatchGANDiscriminator(in_channels, num_filters, num_layers_d) for _ in range(num_d)]
        )

    def forward(self, x):
        outputs = []
        for discriminator in self.discriminators:
            outputs.append(discriminator(x))
            x = self.downsample(x)
        return outputs


# ---------------------------------------------------------------------------
# RRDB building blocks
# ---------------------------------------------------------------------------
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return torch.cat([x, out], 1)


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_dense_layers):
        super().__init__()
        self.dense_layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_dense_layers):
            self.dense_layers.append(DenseBlock(current_channels, growth_rate))
            current_channels += growth_rate
        self.conv1x1 = nn.Conv2d(current_channels, in_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        for layer in self.dense_layers:
            x = layer(x)
        return self.conv1x1(x) + identity


class RRDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_dense_layers, num_rdb):
        super().__init__()
        self.rdb_layers = nn.ModuleList(
            [ResidualDenseBlock(in_channels, growth_rate, num_dense_layers) for _ in range(num_rdb)]
        )
        self.conv_final = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        for layer in self.rdb_layers:
            x = layer(x)
        return self.conv_final(x) * 0.2 + identity


# ---------------------------------------------------------------------------
# LSKA attention
# ---------------------------------------------------------------------------
class LSKA(nn.Module):
    """Large Separable Kernel Attention.

    Provides a massive effective receptive field (23x23) to capture whole streak
    artifacts while scaling linearly to protect VRAM.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv0_h = nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 2), groups=channels)
        self.conv0_v = nn.Conv2d(channels, channels, kernel_size=(5, 1), padding=(2, 0), groups=channels)
        self.conv_spatial_h = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 9), groups=channels, dilation=(1, 3))
        self.conv_spatial_v = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(9, 0), groups=channels, dilation=(3, 1))
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        attn = self.conv0_h(x)
        attn = self.conv0_v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        return self.conv1(attn)


class LSKAGate(nn.Module):
    """Applies LSKA to the concatenated multi-modal input (per-channel sigmoid gating)."""

    def __init__(self, in_channels):
        super().__init__()
        self.lska = LSKA(in_channels)

    def forward(self, x):
        spatial_weights = torch.sigmoid(self.lska(x))
        return x * spatial_weights


class ResidualSE(nn.Module):
    """Channel-wise Squeeze-and-Excitation with a residual design around 1.0."""

    def __init__(self, channels, reduction=16, se_scale=0.5):
        super().__init__()
        self.se_scale = se_scale
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        logits = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        weights = 1.0 + self.se_scale * torch.tanh(logits)
        return x * weights


class ReliabilityAwareFusionGate(nn.Module):
    """Combines spatial gating (LSKA) and channel recalibration (residual SE)."""

    def __init__(self, channels, se_scale=0.5, spatial_scale=0.5):
        super().__init__()
        self.spatial_scale = spatial_scale
        self.se = ResidualSE(channels, reduction=16, se_scale=se_scale)
        self.lska = LSKA(channels)
        self.spatial_proj = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.spatial_proj.weight)
        nn.init.zeros_(self.spatial_proj.bias)

    def forward(self, x):
        x_se = self.se(x)
        spatial_logits = self.spatial_proj(self.lska(x_se))
        spatial_weights = 1.0 + self.spatial_scale * torch.tanh(spatial_logits)
        return x_se * spatial_weights


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
class RRDBGenerator_LSKA(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, num_rrdb=23, num_dense_layers=3,
                 growth_rate=32, feature_channels=64):
        super().__init__()
        self.lska_gate = LSKAGate(in_channels)
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.rrdb_trunk = nn.Sequential(
            *[RRDB(feature_channels, growth_rate, num_dense_layers, num_rdb=3) for _ in range(num_rrdb)]
        )
        self.conv_trunk = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_out = nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        gated_x = self.lska_gate(x)
        initial_features = self.conv_in(gated_x)
        trunk_output = self.conv_trunk(self.rrdb_trunk(initial_features))
        return self.conv_out(initial_features + trunk_output)


class MultiModalRRDBGenerator_LSKA(nn.Module):
    def __init__(self, modalities, out_channels=1, num_rrdb=23, num_dense_layers=3,
                 growth_rate=32, feature_channels=64, stem_feature_channels=16):
        super().__init__()
        self.modalities = modalities
        num_modalities = len(modalities)
        self.stems = nn.ModuleDict()
        for mod_name in modalities:
            self.stems[mod_name] = nn.Sequential(
                nn.Conv2d(1, stem_feature_channels, kernel_size=3, padding=1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(stem_feature_channels, stem_feature_channels, kernel_size=3, padding=1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
            )
        fusion_channels = stem_feature_channels * num_modalities
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(fusion_channels, feature_channels, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.reliability_gate = ReliabilityAwareFusionGate(feature_channels, se_scale=0.5, spatial_scale=0.5)
        self.rrdb_trunk = nn.Sequential(
            *[RRDB(feature_channels, growth_rate, num_dense_layers, num_rdb=3) for _ in range(num_rrdb)]
        )
        self.conv_trunk = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_out = nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == len(self.modalities), (
            f"Input channels ({C}) do not match the expected number of modalities ({len(self.modalities)})."
        )
        stem_outputs = []
        for i, mod_name in enumerate(self.modalities):
            stem_outputs.append(self.stems[mod_name](x[:, i:i + 1, :, :]))
        fused = torch.cat(stem_outputs, dim=1)
        projected = self.fusion_proj(fused)
        gated = self.reliability_gate(projected)
        trunk_output = self.conv_trunk(self.rrdb_trunk(gated))
        return self.conv_out(gated + trunk_output)


# ---------------------------------------------------------------------------
# Full models
# ---------------------------------------------------------------------------
class Pix2PixRRDB(nn.Module):
    """Base RRDB pix2pix model sharing the loss API used by the LSKA variants."""

    def __init__(self, in_channels, out_channels, num_d=1, num_filters_d=64, num_layers_d=3,
                 num_rrdb_G=23, num_dense_layers_G=3, growth_rate_G=32, feature_channels_G=64,
                 use_se=False):
        super().__init__()
        self.discriminator_B = MultiScaleDiscriminator(
            in_channels=out_channels, num_d=num_d, num_filters=num_filters_d, num_layers_d=num_layers_d
        )
        self.criterionL1 = nn.L1Loss()
        self.criterionMS_SSIM_L1 = kornia.losses.MS_SSIMLoss()
        self.spatial_gradient = SpatialGradient()

    # -- losses -----------------------------------------------------------
    def calculate_edge_loss(self, img1, img2, alpha_NGF):
        grad_src = self.spatial_gradient(img1)
        grad_tgt = self.spatial_gradient(img2)
        src_x, src_y = grad_src[:, :, 0], grad_src[:, :, 1]
        tgt_x, tgt_y = grad_tgt[:, :, 0], grad_tgt[:, :, 1]
        gradmag_src = torch.sqrt(torch.pow(src_x, 2) + torch.pow(src_y, 2) + alpha_NGF ** 2)
        gradmag_tgt = torch.sqrt(torch.pow(tgt_x, 2) + torch.pow(tgt_y, 2) + alpha_NGF ** 2)
        eps = 1e-8
        NGF = 1 - 0.5 * torch.pow(
            src_x / (gradmag_src + eps) * tgt_x / (gradmag_tgt + eps)
            + src_y / (gradmag_src + eps) * tgt_y / (gradmag_tgt + eps), 2
        )
        return torch.mean(NGF)

    def calculate_edge_loss_nonorm(self, img1, img2):
        grad_src = self.spatial_gradient(img1)
        grad_tgt = self.spatial_gradient(img2)
        src_x, src_y = grad_src[:, :, 0], grad_src[:, :, 1]
        tgt_x, tgt_y = grad_tgt[:, :, 0], grad_tgt[:, :, 1]
        GF = 1 - 0.5 * torch.pow((src_x * tgt_x + src_y * tgt_y), 2)
        return torch.mean(GF)

    def calculate_sobel_loss(self, img1, img2):
        grad_src = self.spatial_gradient(img1)
        grad_tgt = self.spatial_gradient(img2)
        src_x, src_y = grad_src[:, :, 0], grad_src[:, :, 1]
        tgt_x, tgt_y = grad_tgt[:, :, 0], grad_tgt[:, :, 1]
        return torch.mean(torch.abs(src_x - tgt_x) + torch.abs(src_y - tgt_y))

    def compute_l1_loss(self, fake_B, real_B):
        return self.criterionL1(real_B, fake_B)

    def compute_l1_ssim_loss(self, fake_B, real_B, alpha=0.84):
        return L1SSIMLoss(alpha=alpha)(fake_B, real_B)

    def compute_l1_mssim_loss(self, fake_B, real_B):
        return self.criterionMS_SSIM_L1(fake_B, real_B)

    def compute_NGF_loss(self, fake_B, real_B, alpha_NGF):
        return self.calculate_edge_loss(real_B, fake_B, alpha_NGF)

    def compute_GF_loss(self, fake_B, real_B):
        return self.calculate_edge_loss_nonorm(real_B, fake_B)

    def compute_sobel_loss(self, fake_B, real_B):
        return self.calculate_sobel_loss(real_B, fake_B)

    def compute_adv_loss(self, pred_fake_B):
        return adversarial_loss(pred_fake_B, target_is_real=True, for_discriminator=False)

    def compute_identity_loss(self, real_B):
        identity_B = self.generator_A_to_B(real_B)
        return self.criterionL1(real_B, identity_B)

    def compute_discriminator_loss(self, real_B, fake_B):
        pred_real_B = self.discriminator_B(real_B)
        pred_fake_B = self.discriminator_B(fake_B.detach())
        d_real = adversarial_loss(pred_real_B, target_is_real=True, for_discriminator=True)
        d_fake = adversarial_loss(pred_fake_B, target_is_real=False, for_discriminator=True)
        return d_real + d_fake


class Pix2PixRRDB_LSKA(Pix2PixRRDB):
    """LSKA-gated RRDB generator (single shared gate over concatenated input)."""

    def __init__(self, in_channels=4, out_channels=1, num_d=1, num_filters_d=64, num_layers_d=3,
                 num_rrdb_G=23, num_dense_layers_G=3, growth_rate_G=32, feature_channels_G=64):
        super().__init__(in_channels=in_channels, out_channels=out_channels, num_d=num_d,
                         num_filters_d=num_filters_d, num_layers_d=num_layers_d,
                         num_rrdb_G=num_rrdb_G, num_dense_layers_G=num_dense_layers_G,
                         growth_rate_G=growth_rate_G, feature_channels_G=feature_channels_G,
                         use_se=False)
        self.generator_A_to_B = RRDBGenerator_LSKA(
            in_channels=in_channels, out_channels=out_channels, num_rrdb=num_rrdb_G,
            num_dense_layers=num_dense_layers_G, growth_rate=growth_rate_G,
            feature_channels=feature_channels_G,
        )

    def forward(self, real_A, real_B=None, is_training=True):
        fake_B = self.generator_A_to_B(real_A)
        if is_training:
            identity_B = self.generator_A_to_B(real_B) if real_A.shape[1] == real_B.shape[1] else None
        pred_fake_B = self.discriminator_B(fake_B)
        if is_training:
            return fake_B, identity_B, pred_fake_B
        return fake_B, pred_fake_B


class Pix2PixRRDB_MultiModalLSKA(Pix2PixRRDB):
    """Multi-modal RRDB generator with independent modality stems + reliability gate."""

    def __init__(self, modalities, out_channels=1, num_d=1, num_filters_d=64, num_layers_d=3,
                 num_rrdb_G=23, num_dense_layers_G=3, growth_rate_G=32, feature_channels_G=64,
                 stem_feature_channels_G=16):
        in_channels = len(modalities)
        super().__init__(in_channels=in_channels, out_channels=out_channels, num_d=num_d,
                         num_filters_d=num_filters_d, num_layers_d=num_layers_d,
                         num_rrdb_G=num_rrdb_G, num_dense_layers_G=num_dense_layers_G,
                         growth_rate_G=growth_rate_G, feature_channels_G=feature_channels_G,
                         use_se=False)
        self.generator_A_to_B = MultiModalRRDBGenerator_LSKA(
            modalities=modalities, out_channels=out_channels, num_rrdb=num_rrdb_G,
            num_dense_layers=num_dense_layers_G, growth_rate=growth_rate_G,
            feature_channels=feature_channels_G, stem_feature_channels=stem_feature_channels_G,
        )

    def forward(self, real_A, real_B=None, is_training=True):
        fake_B = self.generator_A_to_B(real_A)
        if is_training:
            identity_B = self.generator_A_to_B(real_B) if real_A.shape[1] == real_B.shape[1] else None
        pred_fake_B = self.discriminator_B(fake_B)
        if is_training:
            return fake_B, identity_B, pred_fake_B
        return fake_B, pred_fake_B