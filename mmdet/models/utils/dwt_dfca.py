from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet_process import TransBasicConv2d


class DWTDFCA(nn.Module):
    """Cross-stage RGB calibration with a device-agnostic Haar transform."""

    def __init__(self,
                 channels: int,
                 residual_scale: float = 0.1,
                 eps: float = 1e-6):
        super().__init__()
        if channels <= 0:
            raise ValueError('channels must be positive')
        self.channels = channels
        self.residual_scale = residual_scale
        self.eps = eps
        self.deconv = TransBasicConv2d(channels, channels)
        self.q_proj = self._make_projection(channels)
        self.k_proj = self._make_projection(channels)
        self.v_proj = self._make_projection(channels)
        self.low_gate = nn.Conv2d(1, 1, 3, padding=1)
        self.mid_gate = nn.Conv2d(1, 1, 3, padding=1)
        self.high_gate = nn.Conv2d(1, 1, 3, padding=1)
        self.band_router = nn.Linear(3, 3)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self._diagnostics: Dict[str, torch.Tensor] = {}
        self._init_dfca_weights()

    @staticmethod
    def _make_projection(channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False))

    def _init_dfca_weights(self) -> None:
        for projection in (self.q_proj, self.k_proj, self.v_proj):
            for layer in projection:
                nn.init.kaiming_normal_(
                    layer.weight, mode='fan_out', nonlinearity='relu')
        for gate in (self.low_gate, self.mid_gate, self.high_gate):
            nn.init.normal_(gate.weight, std=1e-3)
            nn.init.zeros_(gate.bias)
        nn.init.zeros_(self.band_router.weight)
        nn.init.zeros_(self.band_router.bias)
        nn.init.normal_(self.out_proj.weight, std=1e-3)

    @staticmethod
    def haar_dwt(
            tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if tensor.ndim != 4:
            raise ValueError('Haar DWT expects a 4D tensor')
        if tensor.shape[-2] % 2 or tensor.shape[-1] % 2:
            raise ValueError('Haar DWT expects even spatial dimensions')

        x00 = tensor[..., 0::2, 0::2]
        x01 = tensor[..., 0::2, 1::2]
        x10 = tensor[..., 1::2, 0::2]
        x11 = tensor[..., 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    @staticmethod
    def haar_idwt(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor,
                  hh: torch.Tensor) -> torch.Tensor:
        shapes = {tuple(component.shape) for component in (ll, lh, hl, hh)}
        if len(shapes) != 1:
            raise ValueError('Haar IDWT components must share one shape')

        output = ll.new_empty(
            (*ll.shape[:-2], ll.shape[-2] * 2, ll.shape[-1] * 2))
        output[..., 0::2, 0::2] = (ll - lh - hl + hh) * 0.5
        output[..., 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
        output[..., 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
        output[..., 1::2, 1::2] = (ll + lh + hl + hh) * 0.5
        return output

    def _validate_inputs(self, thermal: torch.Tensor,
                         visible: torch.Tensor) -> None:
        if thermal.ndim != 4 or visible.ndim != 4:
            raise ValueError('DWTDFCA expects 4D Thermal and RGB tensors')
        if thermal.shape[0] != visible.shape[0]:
            raise ValueError('Thermal and RGB batch sizes must match')
        if thermal.shape[1] != self.channels or visible.shape[1] != self.channels:
            raise ValueError(
                f'Thermal and RGB channel counts must both equal {self.channels}')

    def _band_correlation(self, query: torch.Tensor,
                          key: torch.Tensor) -> torch.Tensor:
        query = F.normalize(query, p=2, dim=1, eps=self.eps)
        key = F.normalize(key, p=2, dim=1, eps=self.eps)
        return (query * key).sum(dim=1, keepdim=True)

    def forward(self, thermal: torch.Tensor,
                visible: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(thermal, visible)
        visible_up = self.deconv(visible)
        if visible_up.shape != thermal.shape:
            raise ValueError(
                'RGB post-DeConv shape must match Thermal shape, got '
                f'{tuple(visible_up.shape)} and {tuple(thermal.shape)}')

        query_bands = self.haar_dwt(self.q_proj(thermal))
        key_bands = self.haar_dwt(self.k_proj(visible_up))
        value_bands = self.haar_dwt(self.v_proj(visible_up))
        correlations = tuple(
            self._band_correlation(query, key)
            for query, key in zip(query_bands, key_bands))

        low_gate = torch.sigmoid(self.low_gate(correlations[0]))
        lh_gate = torch.sigmoid(self.mid_gate(correlations[1]))
        hl_gate = torch.sigmoid(self.mid_gate(correlations[2]))
        high_gate = torch.sigmoid(self.high_gate(correlations[3]))
        gates = (low_gate, lh_gate, hl_gate, high_gate)

        low_stat = correlations[0].abs().mean(dim=(1, 2, 3))
        mid_stat = 0.5 * (
            correlations[1].abs().mean(dim=(1, 2, 3)) +
            correlations[2].abs().mean(dim=(1, 2, 3)))
        high_stat = correlations[3].abs().mean(dim=(1, 2, 3))
        band_stats = torch.stack((low_stat, mid_stat, high_stat), dim=1)
        band_weights = torch.softmax(self.band_router(band_stats), dim=1)

        low_weight = band_weights[:, 0].view(-1, 1, 1, 1)
        mid_weight = band_weights[:, 1].view(-1, 1, 1, 1)
        high_weight = band_weights[:, 2].view(-1, 1, 1, 1)
        weighted_values = (
            low_weight * low_gate * value_bands[0],
            mid_weight * lh_gate * value_bands[1],
            mid_weight * hl_gate * value_bands[2],
            high_weight * high_gate * value_bands[3],
        )
        reconstructed = self.haar_idwt(*weighted_values)
        delta = self.residual_scale * torch.tanh(
            self.out_proj(reconstructed))
        output = visible_up + delta

        self._diagnostics = {
            'w_low': band_weights[:, 0].mean().detach(),
            'w_mid': band_weights[:, 1].mean().detach(),
            'w_high': band_weights[:, 2].mean().detach(),
            'gate_mean': torch.stack(
                [gate.mean() for gate in gates]).mean().detach(),
            'delta_ratio': (delta.abs().mean() /
                            (visible_up.abs().mean() + self.eps)).detach(),
        }
        return output

    def get_diagnostics(self) -> Dict[str, torch.Tensor]:
        return dict(self._diagnostics)
