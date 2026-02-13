"""This file is a modified version of minLoRA.

https://github.com/cccntu/minLoRA/tree/main

The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



References:
1) the official LoRA implementation released by Microsoft:
https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
"""
# mypy: ignore-errors
from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn


class LoRAParametrization(nn.Module):
    """LoRAParametrization module."""

    def __init__(
        self, fan_in, fan_out, fan_in_fan_out=False, rank=4, lora_dropout_p=0.0, lora_alpha=1
    ):
        """Initialize LoRAParametrization module.

        Args:
            fan_in (int): Number of input features.
            fan_out (int): Number of output features.
            fan_in_fan_out (bool, optional): Whether to swap fan_in and fan_out. Defaults to False.
            rank (int, optional): Rank of the LoRA decomposition. Defaults to 4.
            lora_dropout_p (float, optional): Dropout probability for LoRA. Defaults to 0.0.
            lora_alpha (int, optional): Scaling factor for LoRA. Defaults to 1.
        """
        super().__init__()
        # if weight is stored as (fan_out, fan_in), the memory layout of A & B follows (W + BA)x
        # otherwise, it's x(W + AB). This allows us to tie the weights between linear layers and embeddings
        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)
        self.lora_A = nn.Parameter(torch.zeros(self.swap((rank, fan_in))))
        self.lora_B = nn.Parameter(torch.zeros(self.swap((fan_out, rank))))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_alpha, self.rank = lora_alpha, rank
        self.scaling = lora_alpha / rank
        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if lora_dropout_p > 0 else lambda x: x
        self.register_buffer(
            "lora_dropout_mask", torch.ones(self.swap((1, fan_in)), dtype=self.lora_A.dtype)
        )
        self.forward_fn = self.lora_forward

    @classmethod
    def from_linear(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        """Create LoRAParametrization from a linear layer.

        Args:
            layer (nn.Linear): Linear layer to transform.
            rank (int, optional): Rank of the LoRA decomposition. Defaults to 4.
            lora_dropout_p (float, optional): Dropout probability for LoRA. Defaults to 0.0.
            lora_alpha (int, optional): Scaling factor for LoRA. Defaults to 1.

        Returns:
            LoRAParametrization: Transformed LoRAParametrization layer.
        """
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in,
            fan_out,
            fan_in_fan_out=False,
            rank=rank,
            lora_dropout_p=lora_dropout_p,
            lora_alpha=lora_alpha,
        )

    @classmethod
    def from_conv2d(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        """Create LoRAParametrization from a Conv2d layer.

        Args:
            layer (nn.Conv2d): Conv2d layer to transform.
            rank (int, optional): Rank of the LoRA decomposition. Defaults to 4.
            lora_dropout_p (float, optional): Dropout probability for LoRA. Defaults to 0.0.
            lora_alpha (int, optional): Scaling factor for LoRA. Defaults to 1.

        Returns:
            LoRAParametrization: Transformed LoRAParametrization layer.
        """
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        return cls(
            fan_in,
            fan_out,
            fan_in_fan_out=False,
            rank=rank,
            lora_dropout_p=lora_dropout_p,
            lora_alpha=lora_alpha,
        )

    @classmethod
    def from_embedding(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        """Create LoRAParametrization from an embedding layer.

        Args:
            layer (nn.Embedding): Embedding layer to transform.
            rank (int, optional): Rank of the LoRA decomposition. Defaults to 4.
            lora_dropout_p (float, optional): Dropout probability for LoRA. Defaults to 0.0.
            lora_alpha (int, optional): Scaling factor for LoRA. Defaults to 1.

        Returns:
            LoRAParametrization: Transformed LoRAParametrization layer.
        """
        fan_in, fan_out = layer.weight.shape
        return cls(
            fan_in,
            fan_out,
            fan_in_fan_out=True,
            rank=rank,
            lora_dropout_p=lora_dropout_p,
            lora_alpha=lora_alpha,
        )

    def lora_forward(self, x):
        """Forward pass with LoRA applied.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying LoRA.
        """
        return (
            x
            + torch.matmul(*self.swap((self.lora_B, self.dropout_fn(self.lora_A)))).view(x.shape)
            * self.scaling
        )

    def forward(self, x):
        """Forward pass of the module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.forward_fn(x)

    def enable_lora(self):
        """Enable LoRA for the forward pass."""
        self.forward_fn = self.lora_forward

    def disable_lora(self):
        """Disable LoRA for the forward pass."""
        self.forward_fn = lambda x: x

    def _dropout(self, a):
        """Apply dropout to the tensor.

        Args:
            a (Tensor): Input tensor.

        Returns:
            Tensor: Tensor after applying dropout.
        """
        # to mimic the original implementation: A @ dropout(x), we do (A * dropout(ones)) @ x
        return a * self.lora_dropout(self.lora_dropout_mask)


default_lora_config = {  # specify which layers to add lora to, by default only add to linear layers
    nn.Linear: {
        "weight": partial(LoRAParametrization.from_linear, rank=4),
    },
}


def apply_lora(layer, register=True, merge=False, lora_config=default_lora_config):
    """Add lora parametrization to a layer, designed to be used with model.apply."""
    if register:
        if type(layer) in lora_config:
            for attr_name, parametrization in lora_config[type(layer)].items():
                parametrize.register_parametrization(layer, attr_name, parametrization(layer))
    else:  # this will remove all parametrizations, use with caution
        if hasattr(layer, "parametrizations"):
            for attr_name in layer.parametrizations.keys():
                parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=merge)


# def add_lora(model, lora_config=default_lora_config) -> None:
def add_lora(model: nn.Module, lora_config: dict = default_lora_config) -> None:
    """Add lora parametrization to all layers in a model. Calling it twice will add lora twice."""
    model.apply(partial(apply_lora, lora_config=lora_config))


def add_lora_by_name(model, target_module_names, lora_config=default_lora_config):
    """Add LoRA parameterization to specific layers in a model by names."""
    for name, layer in model.named_modules():
        if any([m in name for m in target_module_names]):
            add_lora(layer, lora_config=lora_config)


def merge_lora(model):
    """Merge lora parametrization to all layers in a model. This will remove all parametrization."""
    model.apply(partial(apply_lora, register=False, merge=True))


def remove_lora(model):
    """Remove lora parametrization to all layers in a model. This will remove all parametrization."""
    model.apply(partial(apply_lora, register=False, merge=False))


#################
### Utilities ###
#################


def apply_to_lora(fn):
    """Apply a function to LoRAParametrization layers, designed to be used with model.apply."""

    def apply_fn(layer):
        if isinstance(layer, LoRAParametrization):
            fn(layer)

    return apply_fn


# enable_lora = lambda model: model.apply(apply_to_lora(lambda x: x.enable_lora()))
def enable_lora(model):
    """Enable LoRA for the forward pass."""
    model.apply(apply_to_lora(lambda x: x.enable_lora()))


# disable_lora = lambda model: model.apply(apply_to_lora(lambda x: x.disable_lora()))
def disable_lora(model):
    """Disable LoRA for the forward pass."""
    model.apply(apply_to_lora(lambda x: x.disable_lora()))


# ------------------- helper function for collecting parameters for training/saving -------------------


def name_is_lora(name):
    """Check if the name is a LoRA parameterization."""
    return (
        len(name.split(".")) >= 4
        and (name.split(".")[-4]) == "parametrizations"
        and name.split(".")[-1] in ["lora_A", "lora_B"]
    )


def name_is_bias(name):
    """Check if the name is a bias parameter."""
    return name.split(".")[-1] == "bias"


def get_params_by_name(model, print_shapes=False, name_filter=None):
    """Get parameters of the model by name."""
    for n, p in model.named_parameters():
        if name_filter is None or name_filter(n):
            if print_shapes:
                print(n, p.shape)
            yield p


def get_lora_params(model, print_shapes=False):
    """Get LoRA parameters of the model."""
    return get_params_by_name(model, print_shapes=print_shapes, name_filter=name_is_lora)


def get_bias_params(model, print_shapes=False):
    """Get bias parameters of the model."""
    return get_params_by_name(model, print_shapes=print_shapes, name_filter=name_is_bias)


def get_lora_state_dict(model):
    """Get LoRA state dict of the model."""
    return {k: v for k, v in model.state_dict().items() if name_is_lora(k)}


# ------------------- helper function for inferencing with multiple lora -------------------


def _prepare_for_multiple_lora(lora_layer):
    """Prepare for multiple LoRA layers."""
    lora_layer.lora_As = []
    lora_layer.lora_Bs = []


def _append_lora(lora_layer):
    """Append LoRA layers."""
    lora_layer.lora_As.append(nn.Parameter(lora_layer.lora_A.clone()))
    lora_layer.lora_Bs.append(nn.Parameter(lora_layer.lora_B.clone()))


def load_multiple_lora(model, lora_state_dicts):
    """Load multiple LoRA state dicts to the model."""
    model.apply(apply_to_lora(_prepare_for_multiple_lora))
    for state_dict in lora_state_dicts:
        _ = model.load_state_dict(state_dict, strict=False)
        model.apply(apply_to_lora(_append_lora))
    return model


def _select_lora(lora_layer, index):
    """Select LoRA layer."""
    lora_layer.lora_A = lora_layer.lora_As[index]
    lora_layer.lora_B = lora_layer.lora_Bs[index]


def select_lora(model, index):
    """Select LoRA model."""
    model.apply(apply_to_lora(lambda x: _select_lora(x, index)))
    return model


# ------------------- helper function for tying and untieing weights -------------------


def tie_weights(linear: nn.Linear, embedding: nn.Embedding):
    """Tie the weights of the linear layer and the embedding layer both with the same lora."""
    # this line below is optional if the original is already tied
    embedding.parametrizations.weight.original = linear.parametrizations.weight.original
    embedding.parametrizations.weight[0].lora_A = linear.parametrizations.weight[0].lora_B
    embedding.parametrizations.weight[0].lora_B = linear.parametrizations.weight[0].lora_A


def untie_weights(linear: nn.Linear, embedding: nn.Embedding):
    """Untie the weights of the linear layer and the embedding layer."""
    embedding.parametrizations.weight.original = nn.Parameter(embedding.weight.original.clone())
    embedding.parametrizations.weight[0].lora_A = nn.Parameter(
        embedding.parametrizations.weight[0].lora_A.clone()
    )
    embedding.parametrizations.weight[0].lora_B = nn.Parameter(
        embedding.parametrizations.weight[0].lora_B.clone()
    )
