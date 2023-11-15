from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import heapq
import itertools

import torch


class BeamHypotheses:
    def __init__(
        self,
        length_penalty: float,
        group_size: int,
        stop_early: bool = False,
    ) -> None:
        self.length_penalty = length_penalty
        self.group_size = group_size
        self.stop_early = stop_early
        self.sorted_beams = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.sorted_beams)

    def add(
        self,
        input_ids: torch.LongTensor,
        next_token: int,
        logsoftmax_sum: float,
        next_beam_index: int,
        beam_indices: Optional[torch.LongTensor],
    ):
        """add a completed sentence to the storage

        Args:
            input_ids (torch.LongTensor): (n_sequence) - Note the eos_token is not included in this input_ids
            logsoftmax_sum (float)
            beam_indices (torch.LongTensor): (n_sequence)
            next_beam_index: int
        """
        sequence_length = input_ids.shape[0]
        score = logsoftmax_sum / (sequence_length**self.length_penalty)
        if len(self) >= self.group_size and score <= self.worst_score:
            return

        if beam_indices is not None:
            concatenated_beam_indices = torch.cat(
                (
                    beam_indices,
                    torch.tensor(next_beam_index, dtype=beam_indices.dtype, device=beam_indices.device).unsqueeze(0),
                )
            )
        else:
            concatenated_beam_indices = torch.tensor(
                next_beam_index, dtype=beam_indices.dtype, device=beam_indices.device
            ).unsqueeze(0)
        heapq.heappush(self.sorted_beams, (score, input_ids, concatenated_beam_indices))
        if len(self) > self.group_size:
            heapq.heappop(self.sorted_beams)
        self.worst_score = self.sorted_beams[0][2]

    def is_done(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        max_length: int,
    ) -> bool:
        """check if this beamHypotheses can stop early

        Args:
            input_ids (torch.LongTensor): input_ids of shape (group_size, n_sequence). Next token is not included
            next_scores (torch.FloatTensor): (group_size)
            max_length (int): the max_length

        Returns:
            bool: do we want to stop early
        """
        if len(self) < self.group_size:
            return False
        if self.stop_early:
            return True

        cur_len = input_ids.shape[1] + 1
        max_score = torch.max(next_scores).item()
        # because logsoftmax are always negative
        if self.length_penalty > 0.0:
            highest_possible_score = max_score / (max_length**self.length_penalty)
        else:
            highest_possible_score = max_score / (cur_len**self.length_penalty)
        return self.worst_score >= highest_possible_score

    def get_sorted_inputs(self) -> List[Tuple[float, torch.LongTensor, torch.LongTensor]]:
        return sorted(self.sorted_beams, reverse=True)


class BeamSearchScorer:
    def __init__(
        self,
        batch_size: int,
        n_beams: int,
        n_beam_groups: int,
        device: torch.device,
        eos_token_ids: Union[list[int], int],
        pad_token_id: int,
        length_penalty: Optional[float] = 1.0,
        num_beam_hyps_to_keep: Optional[int] = 1,
        max_length: Optional[int] = None,
    ) -> None:
        self.batch_size = batch_size
        self.n_beams = n_beams
        self.n_beam_groups = n_beam_groups
        self.group_size = self.n_beams // self.n_beam_groups
        self.device = device
        self.length_penalty = length_penalty
        self.num_beams_hypes_to_keep = num_beam_hyps_to_keep
        self.max_length = max_length
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        self.eos_token_ids = eos_token_ids
        self.pad_token_id = pad_token_id
        self.beam_hypothesis_list = [
            BeamHypotheses(
                length_penalty,
                self.group_size,
                stop_early=False,
            )
            for _ in range(self.batch_size * self.n_beam_groups)
        ]
        self._done = torch.tensor(
            [False for _ in range(batch_size * self.n_beam_groups)], dtype=torch.bool, device=self.device
        )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor,
        input_beam_scores: torch.FloatTensor,
        input_beam_tokens: torch.LongTensor,
        input_beam_indices: torch.LongTensor,
        beam_group_index: int,
        beam_indices: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:
        """process one beam_group

        Args:
            input_ids (torch.LongTensor): (n_batch * group_size, n_sequence)
            input_beam_scores (torch.FloatTensor): beam scores for the sampled data (n_batch, n_sampled)
            input_beam_tokens (torch.LongTensor): beam tokens for the sampled data (n_batch, n_sampled)
            input_beam_indices (torch.LongTensor): which beam does this sample belongs to (n_batch, n_sampled)
            beam_group_index (int): which beam group are we processing
            beam_indices (torch.LongTensor): beam_indices of previous sequence. (n_batch * group_size, n_sequence)
        Returns:
            Dict[str, torch.Tensor]: {beam_tokens, beam_scores, beam_indices}
        """
        n_sampled = input_beam_scores.shape[1]
        output_beam_tokens = torch.zeros(
            (self.batch_size, self.group_size), dtype=input_beam_tokens.dtype, device=input_beam_tokens.device
        )
        output_beam_scores = torch.zeros(
            (self.batch_size, self.group_size), dtype=input_beam_scores.dtype, device=input_beam_scores.device
        )
        output_beam_indices = torch.zeros(
            (self.batch_size, self.group_size), dtype=input_beam_indices.dtype, device=input_beam_indices.device
        )

        for i in range(self.batch_size):
            beam_index = i * self.n_beam_groups + beam_group_index
            insert_index = 0
            if self._done[beam_index]:
                output_beam_tokens[i, :] = self.pad_token_id
                output_beam_scores[i, :] = 0
                output_beam_indices[i, :] = 0
                continue
            for j in range(n_sampled):
                score = input_beam_scores[i, j].item()
                token = input_beam_tokens[i, j].item()
                index = input_beam_indices[i, j].item()
                if token in self.eos_token_ids:
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    if j < self.group_size:
                        self.beam_hypothesis_list[beam_index].add(
                            input_ids[i * self.group_size + index],
                            token,
                            score,
                            beam_indices,
                            index,
                        )
                else:
                    output_beam_scores[i, insert_index] = score
                    output_beam_tokens[i, insert_index] = token
                    # the index in the input_id
                    output_beam_indices[i, insert_index] = i * self.group_size + index
                    ++insert_index
                if insert_index >= self.group_size:
                    break

            self.is_done[beam_index] = self.is_done[beam_index] or self.beam_hypothesis_list[beam_index].is_done(
                input_ids[i * self.group_size : i * self.group_size + self.group_size, :],
                output_beam_scores[i],
                self.max_length,
            )

        return {
            "beam_tokens": output_beam_tokens,
            "beam_scores": output_beam_scores,
            "beam_indices": output_beam_indices,
        }

    def finalize(
        self,
        input_ids: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:
        """get the output decoded

        Args:
            input_ids (torch.LongTensor): (n_batch * n_beam, n_sequence)
            input_beam_scores (torch.FloatTensor): beam scores for the sampled data (n_batch * n_beam_groups, n_sampled)
            input_beam_tokens (torch.LongTensor): beam tokens for the sampled data (n_batch * n_beam_groups, n_sampled)
            input_beam_indices (torch.LongTensor): which beam does this sample belongs to (n_batch * n_beam_groups, n_sampled)

        Returns:
            Dict[str, torch.Tensor]: {sequences, sequence_scores, beam_indices}
        """
        # 1. firstly handle the case that we still have some hype_beams unfinished yet
        for i in range(self.batch_size * self.n_beam_groups):
            if self.is_done[i]:
                continue
            # this means it ends due to StoppingCriteria, gracefully handle
            self.beam_hypothesis_list[i].add()

        # 2. loop over the beam_groups and get the decoded results
        beam_scores = torch.zeros(
            (self.batch_size * self.num_beams_hypes_to_keep),
            dtype=torch.float,
            device=input_ids.device,
        )
        max_result_length = 0
        best_hypos = []
        for i in range(self.batch_size):
            beam_hyp_list_for_this_batch = self.beam_hypothesis_list[
                i * self.n_beam_groups : i * self.n_beam_groups + self.n_beam_groups
            ]
            list_of_ordered_list = [beam_hyp.get_sorted_inputs() for beam_hyp in beam_hyp_list_for_this_batch]
            merged_list = list(itertools.chain.from_iterable(list_of_ordered_list))
            sorted_list = sorted(merged_list, key=lambda x: x[0], reverse=True)
            sorted_list = sorted_list[: self.num_beams_hypes_to_keep]
            best_hypos.add(sorted_list)

            for j, (score, input_ids, _) in enumerate(sorted_list):
                beam_scores[i * self.num_beams_hypes_to_keep + j] = score
                # leave space for the eos_token_id
                max_result_length = max(max_result_length, input_ids.shape[0] + 1)

        max_result_length = max(max_result_length, self.max_length)
        result_sequences = torch.full(
            (self.batch_size * self.num_beams_hypes_to_keep, max_result_length),
            self.pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        result_beam_indices = torch.full(
            (self.batch_size * self.num_beams_hypes_to_keep, max_result_length),
            -1,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        for i, (score, input_ids, beam_indices) in enumerate(best_hypos):
            # we have max_length stopping criteria, so we can make sure input_ids size is smaller or equal to max_length
            result_sequences[i, : input_ids.shape[0]] = input_ids
            if input_ids.shape[0] < max_result_length:
                result_sequences[i, input_ids.shape[0]] = self.eos_token_ids[0]
            result_beam_indices[i, : beam_indices.shape[0]] = beam_indices

        return {
            "sequences": result_sequences,
            "sequence_scores": beam_scores,
            "beam_indices": result_beam_indices,
        }
