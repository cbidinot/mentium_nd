from __future__ import annotations

import copy
from typing import Dict, Iterable, Optional

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


def convert_to_noisy_layers(
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
        convert_to_noisy_layers(
            child,
            noise_inference=noise_inference,
            noise_training=noise_training,
            noise_sd=noise_sd,
        )
    return model


def set_noise_mode(
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


def set_noise_map(
    model: nn.Module,
    layer_noise_map: Dict[str, bool],
    noise_sd: Optional[float] = None,
    exact_match: bool = False,
    set_noise_inference: bool = True,
    set_noise_training: bool = True,
) -> None:
    for name, module in model.named_modules():
        if not hasattr(module, "noise_inference"):
            continue
        for key, enabled in layer_noise_map.items():
            matched = name == key if exact_match else key in name
            if matched:
                if set_noise_inference and hasattr(module, "noise_inference"):
                    module.noise_inference = enabled
                if set_noise_training and hasattr(module, "noise_training"):
                    module.noise_training = enabled
                if noise_sd is not None and hasattr(module, "noise_sd"):
                    module.noise_sd = noise_sd
                break


def set_noise_enabled(
    model: nn.Module,
    enabled: bool,
    noise_sd: Optional[float] = None,
    include_name_contains: Optional[Iterable[str]] = None,
    exclude_name_contains: Optional[Iterable[str]] = None,
) -> None:
    set_noise_mode(
        model=model,
        enabled=enabled,
        noise_sd=noise_sd,
        include_name_contains=include_name_contains,
        exclude_name_contains=exclude_name_contains,
        set_noise_inference=True,
        set_noise_training=True,
    )


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


def clone_with_parameter_noise_once(
    model: nn.Module,
    add_quantization: bool = True,
    add_noise: bool = True,
    quantize_fn=None,
    num_levels: int = 15,
    noise_sd: float = 1e-2,
) -> nn.Module:
    return clone_with_parameter_noise(
        model=model,
        add_quantization=add_quantization,
        add_noise=add_noise,
        quantize_fn=quantize_fn,
        num_levels=num_levels,
        noise_sd=noise_sd,
    )