import torch


def get_extended_attention_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    1. extend the attention_mask of [n_batch, sequence_length] to [n_batch, 1, 1, sequence_length]
    to simplify calculation in similarities calculation
    2. map 1 to 0, 0 to -inf
    """
    # [n_batch, 1, 1, sequence_length]
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = (1 - extended_attention_mask) * torch.finfo(torch.float32).min
    return extended_attention_mask
