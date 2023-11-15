from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict

import copy
from enum import Enum

import torch

from .beam_search import BeamSearchScorer
from .logits_process import HammingDiversityLogitsProcessor
from .logits_process import LogitsProcessorList
from .logits_process import RepetitionPenaltyLogitsProcessor
from .logits_process import SequenceBiasLogitsProcessor
from .stopping_criteria import MaxLengthCriteria
from .stopping_criteria import StoppingCriteriaList

if TYPE_CHECKING:
    from .generation_arguments import GenerationArguments


class GenerationMode(str, Enum):
    """
    Possible generation modes, downstream of the [`~generation.GenerationMixin.generate`] method.
    """

    # Non-beam methods
    CONTRASTIVE_SEARCH = "contrastive_search"
    GREEDY_SEARCH = "greedy_search"
    SAMPLE = "sample"
    ASSISTED_GENERATION = "assisted_generation"
    # Beam methods
    BEAM_SEARCH = "beam_search"
    BEAM_SAMPLE = "beam_sample"
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
    GROUP_BEAM_SEARCH = "group_beam_search"


SUPPORTED_GENERATION_MODES = {
    GenerationMode.GREEDY_SEARCH,
    GenerationMode.BEAM_SEARCH,
    GenerationMode.GROUP_BEAM_SEARCH,
}


class ModelGenerator:
    def __init__(self, generation_config: GenerationArguments, model) -> None:
        self.generation_config = generation_config
        self.model = model

    def _prepare_attention_mask(
        self,
        input_ids: torch.LongTensor,
        generation_config: GenerationArguments,
    ) -> torch.Tensor:
        """return the attention_mask used for inference. And confirm the input is not right padding.

        Args:
            input_ids (torch.LongTensor): tensor of shape [n_batch, n_sequence]

        Returns:
            torch.Tensor: attention_mask of shape [n_batch, n_sequence]
        """
        attention_mask = torch.ne(input_ids, generation_config.pad_token_id).long()
        # right side should all be 1 in mask
        any_right_padding = torch.sum(attention_mask[:, -1]).item()
        assert any_right_padding == input_ids.shape[0], "input_ids should be left padding only"
        return attention_mask

    def _validate_max_length_param(
        self,
        generation_config: GenerationArguments,
        input_ids_length,
    ):
        assert input_ids_length < generation_config.max_length, "input length is more than max_length"

    def _get_generation_mode(self, generation_config: GenerationArguments) -> GenerationMode:
        if generation_config.num_beams > 1:
            if generation_config.num_beams_groups > 1:
                return GenerationMode.GROUP_BEAM_SEARCH
            else:
                return GenerationMode.BEAM_SEARCH
        elif generation_config.top_k > 1:
            return GenerationMode.CONTRASTIVE_SEARCH
        else:
            return GenerationMode.GREEDY_SEARCH

    def _get_logits_processor(self, generation_config: GenerationArguments) -> LogitsProcessorList:
        processors = LogitsProcessorList()
        if generation_config.sequence_bias is not None:
            processors.append(SequenceBiasLogitsProcessor(generation_config.sequence_bias))
        if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    generation_config.diversity_penalty,
                    num_beams=generation_config.num_beams,
                    num_beam_groups=generation_config.num_beams_groups,
                )
            )
        if generation_config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(generation_config.repetition_penalty))
        return processors

    def _get_stopping_criteria_list(self, generation_config: GenerationArguments) -> StoppingCriteriaList:
        criteria_list = StoppingCriteriaList()
        if generation_config.max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            criteria_list.append(MaxLengthCriteria(generation_config.max_length, max_position_embeddings))
        return criteria_list

    def _extract_past_from_model_outputs(self, model_outputs, standardize_cache_format: bool = False):
        past_key_values = None
        if "past_key_values" in model_outputs:
            past_key_values = model_outputs.past_key_values
        elif "mems" in model_outputs:
            past_key_values = model_outputs.mems
        elif "past_buckets_states" in model_outputs:
            past_key_values = model_outputs.past_buckets_states

        # Bloom fix: standardizes the cache format when requested
        if standardize_cache_format and hasattr(self.model, "_convert_to_standard_cache"):
            batch_size = model_outputs.logits.shape[0]
            past_key_values = self.model._convert_to_standard_cache(past_key_values, batch_size=batch_size)
        return past_key_values

    def _update_model_kwargs(
        self, model_outputs, next_tokens: torch.LongTensor, standardize_cache_format: bool = False, **model_kwargs
    ):
        """update input_ids, past_key_values, attention_mask of model_kwargs and return the new model_kwargs"""
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            model_outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(model_outputs, "state", None) is not None:
            model_kwargs["state"] = model_outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention_mask
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
        )

        input_ids = model_kwargs["input_ids"]
        model_kwargs["input_ids"] = torch.cat([input_ids, next_tokens], dim=-1)

        return model_kwargs

    def greedy_search(
        self,
        logits_processors: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        **model_kwargs,
    ) -> torch.Tensor:
        # eos_token_id can be a list of tokens
        eos_token_id = model_kwargs["eos_token_id"]
        padding_token_id = model_kwargs["padding_token_id"]
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(self.model.device)
        unfinished_sequences = torch.ones(
            model_kwargs["input_ids"].shape[0], dtype=torch.long, device=model_kwargs["input_ids"].device
        )
        while True:
            model_inputs = self.model.prepare_inputs_for_generation(model_kwargs)
            # model_output: (n_batch, n_sequence, n_vocab)
            model_outputs = self.model(
                **model_inputs,
                return_dict=True,
            )
            old_input_ids = model_kwargs["input_ids"]
            next_token_scores = logits_processors(model_kwargs["input_ids"], model_outputs.logits[:, -1, :])
            # next_tokens: (n_batch)
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            # set padding_token_id for the finished sequences
            next_tokens = unfinished_sequences * next_tokens + (1 - unfinished_sequences) * padding_token_id
            model_kwargs = self._update_model_kwargs(
                model_outputs=model_outputs, next_tokens=next_tokens, **model_kwargs
            )
            if stopping_criteria(old_input_ids, next_token_scores):
                return model_kwargs["input_ids"]
            # check eos_token_id, output is (n_batch), indicating whether there is any match with eos_token_ids
            next_token_ne_eos = torch.ne(next_tokens.unsqueeze(1), eos_token_id_tensor.unsqueeze(0)).int().prod(dim=1)
            unfinished_sequences = unfinished_sequences * next_token_ne_eos
            if torch.max(unfinished_sequences) == 0:
                return model_kwargs["input_ids"]

    def _expand_tensors(self, n_beams: int, input_tensors_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """expand tensor from [n_batch,*] to [n_batch * n_beams, *]"""
        output = {}
        for tensor_name, tensor in input_tensors_dict.items():
            original_shape = tensor.shape[0]
            if len(tensor.shape) == 1:
                output_tensor = tensor.unsqueeze(-1).repeat(1, n_beams).view(original_shape * n_beams, -1).squeeze(-1)
                output[tensor_name] = output_tensor
            elif len(tensor.shape) == 2:
                output_tensor = tensor.repeat(1, n_beams).view(original_shape * n_beams, -1)
                output[tensor_name] = output_tensor
            else:
                raise Exception("shape of tensor is not supported")
        return output

    def beam_search(
        self,
        logits_processors: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        beam_search_scorer: BeamSearchScorer,
        **model_kwargs,
    ) -> torch.Tensor:
        # eos_token_id can be a list of tokens
        eos_token_id = model_kwargs["eos_token_id"]
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        n_beams = model_kwargs["num_beams"]
        n_beams_groups = model_kwargs["num_beams_groups"]
        if n_beams_groups <= n_beams or n_beams % n_beams_groups == 0:
            raise Exception("num_beams should be larger than num_beams_groups and should have mod 0")

        group_size = n_beams // n_beams_groups
        batch_size = model_kwargs["input_ids"].shape[0]
        tensor_to_expand = {
            "input_ids": model_kwargs["input_ids"],
            "attention_mask": model_kwargs["attention_mask"],
        }
        expanded_tensor_dict = self._expand_tensors(n_beams, tensor_to_expand)
        model_kwargs.update(expanded_tensor_dict)

        # this can make sure that we pick the first beam in the first topk sampling. So the first selection is different
        beam_score = torch.full(
            (batch_size * n_beams, 1), -1e9, dtype=torch.float, device=model_kwargs["input_ids"].device
        )
        beam_score[::n_beams] = 0
        beam_indices = None

        while True:
            model_inputs = self.model.prepare_inputs_for_generation(model_kwargs)
            # model_output: (n_batch, n_sequence, n_vocab)
            model_outputs = self.model(
                **model_inputs,
                return_dict=True,
            )
            n_vocab = model_outputs.logits.shape[-1]
            previous_picked = torch.zeros(
                (batch_size * n_beams), dtype=torch.long, device=model_kwargs["input_ids"].device
            )
            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(
                batch_size * n_beams, dtype=torch.long, device=model_kwargs["input_ids"].device
            )
            for group_index in range(n_beams_groups):
                # the indices in the whole batch_size * n_beams
                batch_group_indices = []
                for batch_index in range(batch_size):
                    start_index = batch_index * n_beams + group_index * group_size
                    end_index = batch_index * n_beams + group_index * group_size + group_size
                    batch_group_indices.extend([i for i in range(start_index, end_index)])
                group_input_ids = model_kwargs["input_ids"][batch_group_indices]
                output_scores = model_outputs.logits[batch_group_indices]
                # next_token_scores: [batch_size * group_size, n_vocab]
                next_token_scores = logits_processors(
                    model_kwargs["input_ids"],
                    output_scores,
                    current_tokens=previous_picked,
                    beam_group_idx=group_index,
                )
                token_scores = next_token_scores + beam_score[batch_group_indices]
                token_scores = token_scores.view(batch_size, group_size * n_vocab)
                # make sure we always have at least group_size sequences to continue (not eos_token_id)
                sampled_scores, sampled_indices = torch.topk(
                    token_scores,
                    max(2, 1 + len(eos_token_id)) * group_size,
                    dim=-1,
                    largest=True,
                    sorted=True,
                )
                input_beam_tokens = sampled_indices % n_vocab
                input_beam_indices = torch.div(sampled_indices, n_vocab, rounding_mode="floor")
                scorer_output = beam_search_scorer.process(
                    input_ids=group_input_ids,
                    input_beam_scores=sampled_scores,
                    input_beam_tokens=input_beam_tokens,
                    input_beam_indices=input_beam_indices,
                    beam_group_index=group_index,
                    beam_indices=beam_indices,
                )
                beam_score[batch_group_indices] = scorer_output["beam_scores"].view(-1)
                beam_indices = scorer_output["beam_indices"]
                # reorder based on the new indices
                model_kwargs["input_ids"][batch_group_indices] = group_input_ids[beam_indices]
                previous_picked[batch_group_indices] = scorer_output["beam_tokens"].view(-1)
                # reordering includes the indices in the whole data (batch_size * n_beam)
                # new_index = batch_start_index + group_start_index + group_offset
                # batch_start_index = torch.div(beam_indices, group_size) * n_beam
                # group_start_index = group_size * group_index
                # group_offset = beam_indices % group_size
                reordering_indices[batch_group_indices] = (
                    torch.div(beam_indices, group_size, rounding_mode="floor") * n_beams
                    + group_size * group_index
                    + (beam_indices % group_size)
                )

            self._update_model_kwargs(model_outputs=model_outputs, next_tokens=previous_picked, **model_kwargs)
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self.model._reorder_cache(
                    model_kwargs["past_key_values"], reordering_indices
                )

            if beam_search_scorer.is_done or stopping_criteria(model_kwargs["input_ids"], beam_score):
                break

        sequence_outputs = beam_search_scorer.finalize(
            input_ids=model_kwargs["input_ids"],
        )
        return sequence_outputs["sequences"]

    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, **kwargs) -> torch.Tensor:
        """_summary_

        Args:
            input_ids (torch.Tensor): input_ids tensor of shape (n_batch, len_sequence)
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model
        Returns:
            torch.Tensor: result tensor of shape (n_batch, len_sequence)
        """
        generation_config = copy.deepcopy(self.generation_config)
        model_kwargs = generation_config.update(**kwargs)
        # 1. prepare model_kward used for inference
        model_kwargs["input_ids"] = input_ids
        model_kwargs["use_cache"] = True
        model_kwargs["attention_mask"] = self._prepare_attention_mask(
            input_ids,
            generation_config,
        )

        # 2. prepare max_length depending on the stop criteria
        input_ids_length = input_ids.shape[-1]
        self._validate_max_length_param(model_kwargs, input_ids_length)

        # 3. generation_mode
        generation_mode = self._get_generation_mode(generation_config)
        assert generation_mode in SUPPORTED_GENERATION_MODES, "generation mode is not supported"

        assert self.model.device.type == input_ids.device.type, "input_ids are not in the same device as model"

        logit_processors = self._get_logits_processor(generation_config)
        stopping_criteria = self._get_stopping_criteria_list(generation_config)

        if generation_mode == GenerationMode.GREEDY_SEARCH:
            return self.greedy_search(
                logit_processors,
                stopping_criteria,
                **model_kwargs,
            )
        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH or generation_mode == GenerationMode.BEAM_SEARCH:
            beam_search_scorer = BeamSearchScorer(
                batch_size=input_ids.shape[0],
                n_beams=generation_config.num_beams,
                n_beam_groups=generation_config.num_beams_groups,
                device=input_ids.device,
                eos_token_ids=generation_config.eos_token_id,
                pad_token_id=generation_config.pad_token_id,
                num_beam_hyps_to_keep=generation_config.num_beam_hypes,
                max_length=generation_config.max_length,
            )
            self.beam_search(
                logit_processors,
                stopping_criteria,
                beam_search_scorer,
                **model_kwargs,
            )

        raise Exception("generation mode is not supported")
