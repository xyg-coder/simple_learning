from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
PROMPT = "question"
PROMPT_INPUT_IDS = "prompt_input_ids"
PROMPT_ATTENTION_MASK = "prompt_attention_mask"
CHOSEN = "response_j"
CHOSEN_INPUT_IDS = "chosen_input_ids"
CHOSEN_ATTENTION_MASK = "chosen_attention_mask"
CHOSEN_LABELS = "chosen_labels"
REJECTED = "response_k"
REJECTED_INPUT_IDS = "rejected_input_ids"
REJECTED_ATTENTION_MASK = "rejected_attention_mask"
REJECTED_LABELS = "rejected_labels"
CONCATENATED_INPUT_IDS = "concatenated_input_ids"
CONCATENATED_ATTENTION_MASK = "concatenated_attention_mask"
CONCATENATED_LABELS = "concatenated_labels"


@dataclass
class FinetuneDataCollator:
    tokenizer: AutoTokenizer
    max_length: int
    label_pad_token_id: int
    max_prompt_length: int
    collate_func: Callable

    def _get_new_attention_mask(
        self,
        eos_token_id: int,
        input_ids: List[int],
        original_attention_mask: List[int],
    ):
        eos_pos = [i for i, v in enumerate(input_ids) if v == eos_token_id]
        return [0 if i in eos_pos else v for i, v in enumerate(original_attention_mask)]

    def tokenize_one_single_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict:
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
        rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)

        eos_token_id = self.tokenizer.eos_token_id
        prompt_tokens[ATTENTION_MASK] = self._get_new_attention_mask(
            eos_token_id,
            prompt_tokens[INPUT_IDS],
            prompt_tokens[ATTENTION_MASK],
        )
        chosen_tokens[ATTENTION_MASK] = self._get_new_attention_mask(
            eos_token_id,
            chosen_tokens[INPUT_IDS],
            chosen_tokens[ATTENTION_MASK],
        )
        rejected_tokens[ATTENTION_MASK] = self._get_new_attention_mask(
            eos_token_id,
            rejected_tokens[INPUT_IDS],
            rejected_tokens[ATTENTION_MASK],
        )

        chosen_tokens[INPUT_IDS].append(eos_token_id)
        chosen_tokens[ATTENTION_MASK].append(1)
        rejected_tokens[INPUT_IDS].append(eos_token_id)
        rejected_tokens[ATTENTION_MASK].append(1)

        # now make sure the concatenated input length is less than max_length
        answer_max_len = max(len(chosen_tokens[INPUT_IDS]), len(rejected_tokens[INPUT_IDS]))
        if len(prompt_tokens[INPUT_IDS]) + answer_max_len > self.max_length:
            prompt_tokens[INPUT_IDS] = prompt_tokens[INPUT_IDS][-self.max_prompt_length :]
            prompt_tokens[ATTENTION_MASK] = prompt_tokens[ATTENTION_MASK][-self.max_prompt_length :]

        # cut the length of responses if necessary
        if len(prompt_tokens[INPUT_IDS]) + answer_max_len > self.max_length:
            length_for_answer = self.max_length - len(prompt_tokens[INPUT_IDS])
            chosen_tokens[INPUT_IDS] = chosen_tokens[INPUT_IDS][:length_for_answer]
            chosen_tokens[ATTENTION_MASK] = chosen_tokens[ATTENTION_MASK][:length_for_answer]
            rejected_tokens[INPUT_IDS] = rejected_tokens[INPUT_IDS][:length_for_answer]
            rejected_tokens[ATTENTION_MASK] = rejected_tokens[ATTENTION_MASK][:length_for_answer]

        batch = {
            CHOSEN_INPUT_IDS: prompt_tokens[INPUT_IDS] + chosen_tokens[INPUT_IDS],
            CHOSEN_ATTENTION_MASK: prompt_tokens[ATTENTION_MASK] + chosen_tokens[ATTENTION_MASK],
            REJECTED_INPUT_IDS: prompt_tokens[INPUT_IDS] + rejected_tokens[INPUT_IDS],
            REJECTED_ATTENTION_MASK: prompt_tokens[ATTENTION_MASK] + rejected_tokens[ATTENTION_MASK],
        }
        # create labels
        batch[CHOSEN_LABELS] = batch[CHOSEN_INPUT_IDS][:]
        batch[CHOSEN_LABELS][: len(prompt_tokens[INPUT_IDS])] = [self.label_pad_token_id] * len(
            prompt_tokens[INPUT_IDS]
        )
        batch[REJECTED_LABELS] = batch[REJECTED_INPUT_IDS][:]
        batch[REJECTED_LABELS][: len(prompt_tokens[INPUT_IDS])] = [self.label_pad_token_id] * len(
            prompt_tokens[INPUT_IDS]
        )
        return batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Args:
            features (List[Dict[str, Any]]): keys should include 'prompt', 'chosen' and rejected

        Returns:
            Dict[str, Any]: tokenized batched data
                {
                    prompt_input_ids: batched tensor, (n_batch, n_sequence_size, n_dim)
                    chosen_input_ids: prompt + chosen result. (n_batch, n_sequence_size, n_dim)
                    chosen_attention_mask: prompt + chosen result. (n_batch, n_sequence_size, n_dim)
                    chosen_labels: prompt + chosen result. (n_batch, n_sequence_size, n_dim)
                    rejected_input_ids: prompt + rejected result. (n_batch, n_sequence_size, n_dim)
                    rejected_attention_mask: prompt + rejected result. (n_batch, n_sequence_size, n_dim)
                    rejected_labels: prompt + rejected result. (n_batch, n_sequence_size, n_dim)
                    prompt: prompt strings
                    chosen: chosen strings
                    rejected: rejected strings
                }
        """
        prompt_strs = [feature[PROMPT] for feature in features]
        chosen_strs = [feature[CHOSEN] for feature in features]
        rejected_strs = [feature[REJECTED] for feature in features]

        batch = []
        for prompt_str, chosen_str, rejected_str in zip(prompt_strs, chosen_strs, rejected_strs):
            batch.append(
                self.tokenize_one_single_element(
                    prompt_str,
                    chosen_str,
                    rejected_str,
                )
            )

        return self.collate_func(self.tokenizer, self.label_pad_token_id, batch)


def finetune_collate(tokenizer: AutoTokenizer, label_pad_token_id, batch: List[Dict]) -> Dict:
    if len(batch) == 0:
        return {}

    result = {}
    # https://stackoverflow.com/questions/73256206
    for key in batch[0].keys():
        if "input_id" in key:
            padding_value = tokenizer.pad_token_id
        elif "attention_mask" in key:
            padding_value = 0
        elif "label" in key:
            padding_value = label_pad_token_id
        else:
            raise Exception("key is not supported")
        elements = [sample[key] for sample in batch]
        padded = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(element) for element in elements],
            batch_first=True,
            padding_value=padding_value,
        )
        result[key] = padded
    return result


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1):
    if tensor.shape[dim] >= length:
        return tensor

    pad_size = list(tensor.shape)
    pad_size[dim] = length - tensor.shape[dim]
    return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


def dpo_collate(tokenizer: AutoTokenizer, label_pad_token_id, batch: List[Dict]) -> Dict:
    """return the {concatenated_input_ids, concatenated_labels, concatenated_attention_mask}

    Args:
        tokenizer (AutoTokenizer): _description_
        label_pad_token_id (_type_): _description_
        batch (List[Dict]): _description_

    Returns:
        Dict: _description_
    """
    pre_concatenated_batch = finetune_collate(tokenizer, label_pad_token_id, batch)
    if not pre_concatenated_batch or len(pre_concatenated_batch) == 0:
        return {}

    result = {}
    # concatenate input_ids, labels and attention_masks
    assert (
        pre_concatenated_batch[CHOSEN_INPUT_IDS].shape[1] == pre_concatenated_batch[CHOSEN_ATTENTION_MASK].shape[1]
        and pre_concatenated_batch[CHOSEN_INPUT_IDS].shape[1] == pre_concatenated_batch[CHOSEN_LABELS].shape[1]
    )
    assert (
        pre_concatenated_batch[REJECTED_INPUT_IDS].shape[1] == pre_concatenated_batch[REJECTED_INPUT_IDS].shape[1]
        and pre_concatenated_batch[REJECTED_INPUT_IDS].shape[1] == pre_concatenated_batch[REJECTED_LABELS].shape[1]
    )
    max_length = max(
        pre_concatenated_batch[CHOSEN_INPUT_IDS].shape[1], pre_concatenated_batch[REJECTED_INPUT_IDS].shape[1]
    )

    result[CONCATENATED_INPUT_IDS] = torch.cat(
        [
            pad_to_length(pre_concatenated_batch[CHOSEN_INPUT_IDS], max_length, tokenizer.pad_token_id, dim=1),
            pad_to_length(pre_concatenated_batch[REJECTED_INPUT_IDS], max_length, tokenizer.pad_token_id, dim=1),
        ],
        dim=0,
    )
    result[CONCATENATED_ATTENTION_MASK] = torch.cat(
        [
            pad_to_length(pre_concatenated_batch[CHOSEN_ATTENTION_MASK], max_length, 0, dim=1),
            pad_to_length(pre_concatenated_batch[REJECTED_ATTENTION_MASK], max_length, 0, dim=1),
        ],
        dim=0,
    )
    result[CONCATENATED_LABELS] = torch.cat(
        [
            pad_to_length(pre_concatenated_batch[CHOSEN_LABELS], max_length, 0, dim=1),
            pad_to_length(pre_concatenated_batch[REJECTED_LABELS], max_length, 0, dim=1),
        ],
        dim=0,
    )
    return result
