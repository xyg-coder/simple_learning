from __future__ import annotations

from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import inspect

import torch

"""to add:
HammingDiversityLogitsProcessor
RepetitionPenaltyLogitsProcessor
NoBadWordsLogitsProcessor
"""


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class HammingDiversityLogitsProcessor(LogitsProcessor):
    def __init__(self, diversity_penalty: float, num_beams: int, num_beam_groups: int):
        if not isinstance(diversity_penalty, float) or (not diversity_penalty > 0.0):
            raise ValueError("`diversity_penalty` should be a float strictly larger than 0.")
        self._diversity_penalty = diversity_penalty
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError("`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError("`num_beam_groups` should be an integer strictly larger than 1.")
        if num_beam_groups > num_beams:
            raise ValueError("`beam_groups` has to be smaller or equal to `num_beams`.")
        self._num_sub_beams = num_beams // num_beam_groups

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        current_tokens: torch.LongTensor,
        beam_group_idx: int,
    ) -> torch.FloatTensor:
        """_summary_

        Args:
            input_ids (torch.LongTensor): input_ids of shape `(batch_size * n_beams, sequence_length)`
            scores (torch.FloatTensor): scores for the next token `(batch_size * group_size, n_vocab)`
            current_tokens (torch.LongTensor): Indices of input sequence tokens in the vocabulary, corresponding to the tokens selected by the other
                beam groups in the current generation step. shape `(batch_size * n_beams)`
            beam_group_idx (int): The index of the beam group currently being processed.

        Returns:
            torch.FloatTensor: processed scores for the next token. This step aims to update the only this beam-group
        """
        batch_size = input_ids.shape[0] // self._num_beams
        group_start = beam_group_idx * self._num_sub_beams
        n_vocab = scores.shape[1]
        group_end = min(self._num_beams, beam_group_idx * self._num_sub_beams + self._num_sub_beams)
        group_size = group_end - group_start

        if group_start == 0:
            return scores

        # handdle each batch differently
        for batch_idx in range(batch_size):
            batch_start = batch_idx * self._num_beams
            batch_group_start = batch_idx * self._num_beams + group_start
            token_frequencies = torch.bincount(current_tokens[batch_start:batch_group_start], minlength=n_vocab).to(
                scores.device
            )
            scores[batch_idx * group_size : batch_idx * group_size + group_size] -= (
                self._diversity_penalty * token_frequencies
            )
        return scores


class SequenceBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, sequence_bias: Dict[Tuple[int], float]) -> None:
        self.sequence_bias = sequence_bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """apply bias to the matching sequence. The bias is applied to the last token

        Args:
            input_ids (torch.LongTensor): previous chosen token of shape (n_batch, n_sequence)
            scores (torch.FloatTensor): (n_batch, n_vocab)

        Returns:
            torch.FloatTensor: updated score
        """
        bias = torch.zeros_like(scores, dtype=scores.dtype, device=scores.device)
        n_vocab = scores.shape[-1]
        for sequence_ids, bias_score in self.sequence_bias.items():
            for sequence_id in sequence_ids:
                assert sequence_id < n_vocab, "invalid token id in the sequence bias"
            if len(sequence_ids) == 0 or len(sequence_ids) - 1 > input_ids.shape[-1]:
                continue
            elif len(sequence_ids) == 1:
                bias[:, sequence_ids[-1]] += bias_score
            else:
                prefix_length = len(sequence_ids) - 1
                prefix_tensor = torch.Tensor(sequence_ids[:-1], dtype=input_ids.dtype, device=input_ids.device)
                # matching_row: (n_batch)
                matching_row = torch.eq(
                    input_ids[:, -prefix_length:],
                    prefix_tensor,
                ).prod(dim=1)
                bias[:, sequence_ids[-1]] += bias_score * matching_row
        return scores + bias


class NoBadWordsLogitsProcessor(SequenceBiasLogitsProcessor):
    def __init__(self, bad_words_ids: List[List[int]], eos_token_id: Union[int, List[int]]):
        self.bad_word_ids = bad_words_ids
        if eos_token_id is None:
            eos_token_id = []
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        bad_words_ids = list(
            filter(lambda bad_token_seq: all(bad_token_seq != [i] for i in eos_token_id), bad_words_ids)
        )

        # Forbidding a sequence is equivalent to setting its bias to -inf
        sequence_bias = {tuple(sequence): float("-inf") for sequence in bad_words_ids}
        super().__init__(sequence_bias=sequence_bias)


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """
    Post diversities to input_ids based on the frequency
    see https://arxiv.org/pdf/1909.05858.pdf for more details
    """

    def __init__(self, penalty: float) -> None:
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """_summary_

        Args:
            input_ids (torch.LongTensor): (batch_size, n_sequence_before_score)
            scores (torch.FloatTensor): (batch_size, n_vocab)

        Returns:
            torch.FloatTensor: scores (torch.FloatTensor): (batch_size, n_vocab)
        """
        # if scores are less than 0, we should div by penalty instead
        scores_to_penalty = torch.gather(scores, 1, input_ids)
        penalty_scores = torch.where(
            scores_to_penalty < 0, scores_to_penalty * self.penalty, scores_to_penalty / self.penalty
        )
        scores = scores.scatter_(1, input_ids, penalty_scores)
        return scores


class LogitsProcessorList(list):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores
