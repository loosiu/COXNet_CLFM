from typing import Dict, Tuple

import torch
import torch.nn as nn

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
        self._diagnostics: Dict[str, torch.Tensor] = {}

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

    def forward(self, thermal: torch.Tensor,
                visible: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(thermal, visible)
        visible_up = self.deconv(visible)
        if visible_up.shape != thermal.shape:
            raise ValueError(
                'RGB post-DeConv shape must match Thermal shape, got '
                f'{tuple(visible_up.shape)} and {tuple(thermal.shape)}')
        return visible_up

    def get_diagnostics(self) -> Dict[str, torch.Tensor]:
        return dict(self._diagnostics)
