from dataclasses import dataclass


@dataclass
class BertConfig:
    vocab_size: int
    max_position: int
    token_types_size: int
    hidden_size: int
    layer_norm_eps: float
    dropout_p: float
    attention_head_size: int
    intermediate_size: int

    # number of bertLayers in the encoder
    num_hidden_layers: int
