from __future__ import annotations

import copy
from typing import Iterable, Optional

import numpy as np

import torch
import torch.nn as nn


class Noisy_Inference(torch.autograd.Function):
    noise_sd = 1e-1

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        weight = input_tensor.clone()
        delta_w = 2 * torch.abs(weight).max()
        noise = torch.randn_like(weight) * (Noisy_Inference.noise_sd * delta_w)
        return torch.add(weight, noise)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


noiser = Noisy_Inference.apply


class NoisyLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        noise_inference: bool = False,
        noise_training: Optional[bool] = None,
        noise_sd: float = 0.1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.noise_inference = noise_inference
        self.noise_training = noise_inference if noise_training is None else noise_training
        self.noise_sd = noise_sd

    @property
    def noise_enabled(self) -> bool:
        return self.noise_inference and self.noise_training

    @noise_enabled.setter
    def noise_enabled(self, enabled: bool) -> None:
        self.noise_inference = enabled
        self.noise_training = enabled

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        use_noise = self.noise_training if self.training else self.noise_inference
        if not use_noise:
            return nn.functional.linear(input_tensor, self.weight, self.bias)

        Noisy_Inference.noise_sd = self.noise_sd
        noisy_weight = noiser(self.weight)
        noisy_bias = noiser(self.bias) if self.bias is not None else self.bias
        return nn.functional.linear(input_tensor, noisy_weight, noisy_bias)

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return (
            f"{base}, noise_inference={self.noise_inference}, "
            f"noise_training={self.noise_training}, noise_sd={self.noise_sd}"
        )


class NoisyConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias: bool = True,
        padding_mode: str = "zeros",
        noise_inference: bool = False,
        noise_training: Optional[bool] = None,
        noise_sd: float = 0.1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.noise_inference = noise_inference
        self.noise_training = noise_inference if noise_training is None else noise_training
        self.noise_sd = noise_sd

    @property
    def noise_enabled(self) -> bool:
        return self.noise_inference and self.noise_training

    @noise_enabled.setter
    def noise_enabled(self, enabled: bool) -> None:
        self.noise_inference = enabled
        self.noise_training = enabled

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        use_noise = self.noise_training if self.training else self.noise_inference
        if not use_noise:
            return self._conv_forward(input_tensor, self.weight, self.bias)

        Noisy_Inference.noise_sd = self.noise_sd
        noisy_weight = noiser(self.weight)
        noisy_bias = noiser(self.bias) if self.bias is not None else self.bias
        return self._conv_forward(input_tensor, noisy_weight, noisy_bias)

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return (
            f"{base}, noise_inference={self.noise_inference}, "
            f"noise_training={self.noise_training}, noise_sd={self.noise_sd}"
        )

def quantize_tensor(
    parameters: torch.Tensor,
    levels: Optional[torch.Tensor] = None,
    num_levels: int = 15,
    quantile: float = 0.01,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Quantize a tensor by snapping each value to the nearest discrete level.

    Quantization levels are either provided explicitly or derived automatically
    from the tensor's own distribution, clipping outliers via quantiles before
    computing evenly-spaced levels across the remaining range.

    Args:
        parameters: The tensor to quantize.
        levels:     Explicit quantization levels to use. If None, levels are
                    computed automatically from the tensor using ``num_levels``
                    and ``quantile``.
        num_levels: Number of evenly-spaced quantization levels to generate
                    when ``levels`` is None. Ignored if ``levels`` is provided.
        quantile:   Fraction of values to clip from each tail of the 
                    distribution before computing levels. For example, 0.01
                    clips the bottom 1% and top 1%, making levels robust to 
                    outliers. Must be in [0, 0.5].
        device:     Device to place the levels and bins tensors on. Should
                    match the device of ``parameters``.

    Returns:
        A tensor of the same shape as ``parameters`` where each value has been
        replaced by the nearest quantization level.
    """
    if levels is None:
        upper_w = torch.quantile(parameters, np.clip(1 - quantile, 0, 1)).item()
        lower_w = torch.quantile(parameters, np.clip(quantile, 0, 1)).item()
        levels = torch.linspace(lower_w, upper_w, num_levels).to(device)

    bins = torch.tensor(
        [levels[i] + (levels[i] - levels[i + 1]).abs() / 2 for i in range(num_levels - 1)],
        device=device,
    )
    idx = torch.bucketize(parameters, boundaries=bins)
    return levels[idx]

def quantize_symmetric(
    parameters: torch.Tensor,
    num_bits: int = 8,
) -> torch.Tensor:
    """Quantize a tensor using symmetric uniform quantization.

    Divides the range [-max, max] into evenly-spaced levels determined by the
    number of bits, mimicking the behavior of fixed bit-width hardware such as
    INT8 inference engines. Unlike distribution-based quantization, the grid is
    anchored to the tensor's absolute maximum rather than its quantile range.

    Args:
        parameters: The tensor to quantize.
        num_bits:   Number of bits to simulate. Determines the number of
                    quantization levels as ``2^num_bits - 1``. For example,
                    8 bits gives 255 levels.

    Returns:
        A tensor of the same shape as ``parameters`` with values snapped to
        the nearest symmetric quantization level.
    """
    n_levels = 2 ** num_bits - 1
    max_val = parameters.abs().max()
    scale = max_val / (n_levels // 2)
    return torch.clamp(torch.round(parameters / scale), -(n_levels // 2), n_levels // 2) * scale

def quantize_stochastic(
    parameters: torch.Tensor,
    num_bits: int = 8,
) -> torch.Tensor:
    """Quantize a tensor using stochastic rounding.

    Rather than always rounding to the nearest level, each value is rounded up
    or down randomly with probability proportional to its distance from the
    two adjacent levels. This produces an unbiased estimator of the original
    value in expectation, which makes it better suited for use during training
    than deterministic rounding, which can introduce systematic bias that
    accumulates across layers.

    Args:
        parameters: The tensor to quantize.
        num_bits:   Number of bits to simulate. Determines the number of
                    quantization levels as ``2^num_bits``.

    Returns:
        A tensor of the same shape as ``parameters`` with values stochastically
        rounded to quantization levels.
    """
    n_levels = 2 ** num_bits - 1
    max_val = parameters.abs().max()
    scale = max_val / n_levels
    scaled = parameters / scale
    floored = torch.floor(scaled)
    prob = scaled - floored
    return (floored + torch.bernoulli(prob)) * scale

def quantize_log(
    parameters: torch.Tensor,
    num_bits: int = 4,
) -> torch.Tensor:
    """Quantize a tensor using logarithmically-spaced levels.

    Places quantization levels on a log scale rather than a linear one,
    giving finer resolution near zero and coarser resolution in the tails.
    This is well suited for weight tensors that follow a roughly normal
    distribution, where most values are clustered near zero. Signs are
    preserved by quantizing in log-space and reapplying them afterward.

    Args:
        parameters: The tensor to quantize. Zero values are clamped to a
                    small epsilon (1e-8) before log transformation to avoid
                    -inf.
        num_bits:   Number of bits to simulate. Determines the number of
                    log-spaced levels as ``2^num_bits``.

    Returns:
        A tensor of the same shape as ``parameters`` with values snapped to
        the nearest log-spaced quantization level, with original signs
        restored.
    """
    signs = torch.sign(parameters)
    log_vals = torch.log2(parameters.abs().clamp(min=1e-8))
    min_log, max_log = log_vals.min(), log_vals.max()
    levels = torch.linspace(min_log, max_log, 2 ** num_bits, device=parameters.device)
    quantized_log = quantize_tensor(log_vals, levels=levels, num_levels=2 ** num_bits)
    return signs * (2 ** quantized_log)

def _copy_conv2d_to_noisy(
    module: nn.Conv2d,
    noise_inference: bool,
    noise_sd: float,
    noise_training: Optional[bool] = None,
) -> NoisyConv2d:
    noisy = NoisyConv2d(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode,
        noise_inference=noise_inference,
        noise_training=noise_training,
        noise_sd=noise_sd,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    noisy.weight.data.copy_(module.weight.data)
    if module.bias is not None:
        noisy.bias.data.copy_(module.bias.data)
    return noisy


def _copy_linear_to_noisy(
    module: nn.Linear,
    noise_inference: bool,
    noise_sd: float,
    noise_training: Optional[bool] = None,
) -> NoisyLinear:
    noisy = NoisyLinear(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
        noise_inference=noise_inference,
        noise_training=noise_training,
        noise_sd=noise_sd,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    noisy.weight.data.copy_(module.weight.data)
    if module.bias is not None:
        noisy.bias.data.copy_(module.bias.data)
    return noisy


def _convert_to_noisy_layers(
    model: nn.Module,
    noise_inference: bool = True,
    noise_training: Optional[bool] = None,
    noise_sd: float = 0.05,
) -> nn.Module:
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d) and not isinstance(child, NoisyConv2d):
            setattr(
                model,
                name,
                _copy_conv2d_to_noisy(
                    child,
                    noise_inference=noise_inference,
                    noise_training=noise_training,
                    noise_sd=noise_sd,
                ),
            )
            continue
        if isinstance(child, nn.Linear) and not isinstance(child, NoisyLinear):
            setattr(
                model,
                name,
                _copy_linear_to_noisy(
                    child,
                    noise_inference=noise_inference,
                    noise_training=noise_training,
                    noise_sd=noise_sd,
                ),
            )
            continue
        _convert_to_noisy_layers(
            child,
            noise_inference=noise_inference,
            noise_training=noise_training,
            noise_sd=noise_sd,
        )
    return model


def _set_noise_mode(
    model: nn.Module,
    enabled: bool,
    noise_sd: Optional[float] = None,
    include_name_contains: Optional[Iterable[str]] = None,
    exclude_name_contains: Optional[Iterable[str]] = None,
    set_noise_inference: bool = True,
    set_noise_training: bool = True,
) -> None:
    include_name_contains = tuple(include_name_contains or [])
    exclude_name_contains = tuple(exclude_name_contains or [])

    for name, module in model.named_modules():
        if not hasattr(module, "noise_inference"):
            continue

        if include_name_contains and not any(k in name for k in include_name_contains):
            continue
        if exclude_name_contains and any(k in name for k in exclude_name_contains):
            continue

        if set_noise_inference and hasattr(module, "noise_inference"):
            module.noise_inference = enabled
        if set_noise_training and hasattr(module, "noise_training"):
            module.noise_training = enabled
        if noise_sd is not None and hasattr(module, "noise_sd"):
            module.noise_sd = noise_sd

def clone_with_parameter_noise(
    model: nn.Module,
    add_quantization: bool = True,
    add_noise: bool = True,
    quantize_fn=None,
    num_levels: int = 15,
    noise_sd: float = 1e-2,
) -> nn.Module:
    """Create a one-time noisy copy of a model.

    This does NOT attach persistent noise behavior. It deep-copies the model,
    applies quantization/noise once to copied parameters, and returns the copy.
    """
    with torch.no_grad():
        model_noisy = copy.deepcopy(model)
        for parameter in model_noisy.parameters():
            if add_quantization and quantize_fn is not None:
                parameter.copy_(quantize_fn(parameter, num_levels=num_levels))
            if add_noise:
                delta_w = 2 * parameter.abs().max()
                perturbation = torch.randn_like(parameter) * (noise_sd * delta_w)
                parameter.copy_(parameter + perturbation)
    return model_noisy

def clone_with_noisy_layers(
    model: nn.Module,
    noise_inference: bool = True,
    noise_training: Optional[bool] = None,
    noise_sd: float = 1e-2,
    add_one_time_noise: bool = False,
    add_quantization: bool = False,
    quantize_fn=None,
    quantize_kwargs: Optional[dict] = None,
    include_name_contains: Optional[Iterable[str]] = None,
    exclude_name_contains: Optional[Iterable[str]] = None,
) -> nn.Module:
    """Deep-copy a model and convert its layers to noisy variants.

    Unlike clone_with_parameter_noise(), this attaches persistent NoisyLinear/
    NoisyConv2d layers so noise behavior can be toggled later via set_noise_mode().
    Optionally also applies a one-time parameter perturbation and/or quantization
    on top of the persistent noise setup.

    Args:
        model:                  The model to clone.
        noise_inference:        Whether to enable noise at inference time.
        noise_training:         Whether to enable noise at training time.
                                Defaults to noise_inference if None.
        noise_sd:               Noise standard deviation for persistent layers.
        add_one_time_noise:     Also perturb parameters once at clone time.
        add_quantization:       Apply one-time quantization at clone time.
        quantize_fn:            Quantization function to apply if add_quantization
                                is True. Compatible with quantize_tensor,
                                quantize_symmetric, quantize_stochastic, and
                                quantize_log. If None, no quantization is applied.
        quantize_kwargs:        Keyword arguments forwarded to quantize_fn.
                                For example, ``{"num_levels": 15}`` for
                                quantize_tensor or ``{"num_bits": 8}`` for
                                quantize_symmetric/stochastic/log. If None,
                                the quantize_fn defaults are used.
        include_name_contains:  Only noisify layers whose name contains one of these.
        exclude_name_contains:  Skip layers whose name contains one of these.

    Returns:
        A deep copy of the model with noisy layers and optional one-time
        quantization/noise applied.
    """
    cloned = copy.deepcopy(model)
    quantize_kwargs = quantize_kwargs or {}

    if add_one_time_noise or add_quantization:
        include = tuple(include_name_contains or [])
        exclude = tuple(exclude_name_contains or [])
        with torch.no_grad():
            for name, parameter in cloned.named_parameters():
                if include and not any(k in name for k in include):
                    continue
                if exclude and any(k in name for k in exclude):
                    continue
                if add_quantization and quantize_fn is not None:
                    parameter.copy_(quantize_fn(parameter, num_levels))
                if add_one_time_noise:
                    delta_w = 2 * parameter.abs().max()
                    parameter.copy_(parameter + torch.randn_like(parameter) * (noise_sd * delta_w))

    _convert_to_noisy_layers(
        cloned,
        noise_inference=noise_inference,
        noise_training=noise_training,
        noise_sd=noise_sd,
    )

    if include_name_contains or exclude_name_contains:
        _set_noise_mode(cloned, enabled=False)
        _set_noise_mode(
            cloned,
            enabled=True,
            noise_sd=noise_sd,
            include_name_contains=include_name_contains,
            exclude_name_contains=exclude_name_contains,
        )

    return cloned