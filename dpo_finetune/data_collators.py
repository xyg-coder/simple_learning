from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List

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


@dataclass
class DpoFinetuneDataCollator:
    tokenizer: AutoTokenizer
    max_length: int
    label_pad_token_id: int = -100
    max_prompt_length: int = 512

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

    def collate(self, batch: List[Dict]) -> Dict:
        if len(batch) == 0:
            return {}

        result = {}
        # https://stackoverflow.com/questions/73256206
        for key in batch[0].keys():
            if "input_id" in key:
                padding_value = self.tokenizer.pad_token_id
            elif "attention_mask" in key:
                padding_value = 0
            elif "label" in key:
                padding_value = self.label_pad_token_id
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

        return self.collate(batch)
