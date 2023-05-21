from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from own_transformers.bert_config import BertConfig
from own_transformers.model.modeling_utils import get_extended_attention_mask


class BertEmbedding(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position, config.hidden_size)
        self.token_embedding = nn.Embedding(config.token_types_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_p)
        self.register_buffer('position_ids', torch.arange(config.max_position).view(1, -1))
        self.register_buffer('token_type_ids', torch.zeros(1, config.max_position, dtype=torch.long))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError('input_ids do not exist')
        _, sequence_length = input_ids.shape
        if position_ids is None:
            position_ids = self.position_ids[:, :sequence_length]
        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, :sequence_length]
        embedding = self.word_embedding(input_ids) + self.position_embedding(position_ids) + self.token_embedding(token_type_ids)
        output = self.layer_norm(embedding)
        return self.dropout(output)


class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        if config.hidden_size % config.attention_head_size != 0:
            raise ValueError("The hidden size is not a multiple of the attention_head_size")
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_p)
        self.attention_head_size = config.attention_head_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # q,k,v: [n, n_sequence, hidden_states]
        queries = self.query(hidden_states)
        keys = self.key(hidden_states)
        values = self.key(hidden_states)
        n_batch, n_sequence, hidden_state_size = queries.shape

        queries = torch.reshape(queries, (n_batch, n_sequence, self.attention_head_size, -1))
        keys = torch.reshape(keys, (n_batch, n_sequence, self.attention_head_size, -1))
        values = torch.reshape(values, (n_batch, n_sequence, self.attention_head_size, -1))
        # head_queries: [n_batch, attention_head_size, n_sequence, sub_head_size]
        headed_queries = torch.permute(queries, (0, 2, 1, 3))
        # head_keys: [n_batch, attention_head_size, sub_head_size, n_sequence]
        headed_keys = torch.permute(keys, (0, 2, 3, 1))
        # head_values: [n_batch, attention_head_size, n_sequence, sub_head_size]
        headed_values = torch.permute(values, (0, 2, 1, 3))
        # similarities: [n_batch, attention_head_size, n_sequence, n_sequence]
        similarities = torch.matmul(headed_queries, headed_keys)
        if attention_mask is not None:
            # attention_mask: [n_batch, 1, 1, n_sequence]
            similarities = similarities + attention_mask
        similarities = nn.functional.softmax(similarities, dim=-1)
        # headed_attention: [n_batch, attention_head_size, n_sequence, sub_head_size]
        headed_attention = torch.matmul(similarities, headed_values)
        # headed_attention: [n_batch, n_sequence, attention_head_size, sub_head_size]
        headed_attention = torch.permute(headed_attention, (0, 2, 1, 3))
        attention = torch.reshape(headed_attention, (n_batch, n_sequence, hidden_state_size))
        return self.dropout(attention)


class BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.self_attention = BertSelfAttention(config)
        self.self_output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention_output = self.self_attention(hidden_states, attention_mask)
        output = self.self_output(attention_output, hidden_states)
        return output


class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return self.act(hidden_states)


class BertOutput(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.bert_attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.bert_output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.bert_attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(hidden_states)
        bert_output = self.bert_output(intermediate_output, hidden_states)
        return bert_output


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for bert_layer in self.layer:
            hidden_states = bert_layer(hidden_states, attention_mask)
        # hidden_states: [n_batch, n_sequence, hidden_size]
        return hidden_states

class BertPooler(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activate = nn.Tanh()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """pick the first token value for final output

        :param hidden_states: result from BertEncoder. [n_batch, n_sequence, hidden_size]
        :return: [n_batch, hidden_size]
        """
        first_token = hidden_states[:, 0, :]
        output = self.dense(first_token)
        return self.activate(output)


class BertModel(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.bert_embedding = BertEmbedding(config)
        self.bert_encoder = BertEncoder(config)
        self.bert_pooler = BertPooler(config)

    def forward(
        self,
        input_tensor: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedding = self.bert_embedding(
            input_tensor,
            position_ids,
            token_type_ids,
        )
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask)
        encoder_result = self.bert_encoder(embedding, attention_mask)
        return self.bert_pooler(encoder_result)
