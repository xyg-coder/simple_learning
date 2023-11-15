from __future__ import annotations

from bitsandbytes.optim import AdamW
from torch import nn

from .utils import get_parameter_names

ALL_LAYERNORM_LAYERS = [nn.LayerNorm]


def get_decay_parameter_names(model):
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def create_adamw_paged_32_bit_optimizer(
    model,
    training_args,
):
    decay_parameters = get_decay_parameter_names(model)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "lr": training_args.learning_rate,
        "is_paged": True,
        "optim_bits": 32,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
    return AdamW(
        optimizer_grouped_parameters,
        **optimizer_kwargs,
    )
