from __future__ import annotations

from typing import Dict

import os

import torch
import torch.nn as nn
from accelerate import Accelerator
from peft import PeftModel
from peft import get_peft_model
from peft import prepare_model_for_kbit_training
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from .training_arguments import FinetuneArguments
from .utils import WEIGHTS_NAME
from .utils import unwrap_model


class BaseTrainer:
    def __init__(
        self,
        model,
        data_collator,
        dataset,
        eval_dataset,
        training_args: FinetuneArguments,
        optimizer,
        lr_scheduler,
        accelerator,
        input_id_key,
        attention_mask_key,
        label_key,
        label_ignore_index=-100,
        peft_config=None,
    ) -> None:
        self.accelerator = accelerator if accelerator else Accelerator()
        self.model = model
        if peft_config and not isinstance(self.model, PeftModel):
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                model = prepare_model_for_kbit_training(model)
            self.model = get_peft_model(model, peft_config)
        self.train_dataloader = self.get_train_dataloader(
            dataset,
            batch_size=training_args.batch_size,
            data_collator=data_collator,
            train_shuffle=training_args.train_shuffle,
            num_proc=training_args.data_loader_worker,
        )
        self.eval_dataloader = self.get_eval_dataloader(
            eval_dataset,
            batch_size=training_args.eval_batch_size,
            data_collator=data_collator,
            num_proc=training_args.data_loader_worker,
        )

        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model,
            optimizer,
            lr_scheduler,
        )
        self.gradient_accumulation_steps = training_args.gradient_accumulation_steps
        self.train_steps = training_args.max_train_steps
        self.train_epochs = training_args.train_epochs
        self.training_args = training_args
        self.input_id_key = input_id_key
        self.attention_mask_key = attention_mask_key
        self.label_key = label_key
        self.label_smoother_epsilon = training_args.label_smoother_epsilon
        self.label_ignore_index = label_ignore_index
        self.eval_steps = training_args.eval_steps
        self.log_steps = training_args.log_steps
        self.eval_data_num = training_args.eval_nums
        print(f"number of parameters that require gradients = {model.num_parameters(only_trainable=True)}")

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        logits = self.model(batch[self.input_id_key], attention_mask=batch[self.attention_mask_key]).logits
        return self._compute_label_smoothing_loss(batch, logits)

    def _compute_label_smoothing_loss(self, batch: Dict, logits: torch.Tensor) -> torch.Tensor:
        """compute loss using label smoothing to avoid over-confidence

        Args:
            batch (Dict): {chosen_inputs, chosen_attention_mask, chosen_labels}
            logits (torch.Tensor): shape: (n_batch, sequence_length, n_dim)

        Returns:
            torch.Tensor: loss
        """
        logits = logits[..., :-1, :].contiguous()
        labels = batch[self.label_key][..., 1:].contiguous()
        label_mask = labels.eq(self.label_ignore_index)
        # to avoid negative label_ignore_index will throw error
        labels = torch.clamp(label_mask, min=0)
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)
        normal_loss = log_probs.gather(dim=-1, index=labels)
        normal_loss = torch.squeeze(normal_loss, dim=-1)
        # [n_batch, sequence_length]
        smooth_loss = torch.sum(log_probs, dim=-1, keepdim=False)
        normal_loss.masked_fill(label_mask, 0.0)
        smooth_loss.masked_fill(label_mask, 0.0)

        num_active_elements = label_mask.numel() - label_mask.long().sum()
        normal_loss = normal_loss.sum() / num_active_elements
        smooth_loss = smooth_loss.sum() / (num_active_elements * logits.shape[-1])
        return (1 - self.label_smoother_epsilon) * normal_loss + self.label_smoother_epsilon * smooth_loss

    def train(self) -> None:
        self.model.train()
        completed_steps = 0
        max_steps = min(self.train_steps, len(self.train_dataloader))
        print(f"max train steps={max_steps}")
        for epoch in range(self.train_epochs):
            aver_loss = 0
            for step, batch in tqdm(enumerate(self.train_dataloader, start=1), total=max_steps):
                loss = self.compute_loss(batch)
                loss = loss / self.gradient_accumulation_steps
                aver_loss += loss.item()
                self.accelerator.backward(loss)

                # step here starts from 1
                if step % self.gradient_accumulation_steps == 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.training_args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    completed_steps += 1
                if step % self.log_steps == 0:
                    print(f"epoch={epoch}, step={step}, loss={aver_loss / self.log_steps}")
                    aver_loss = 0
                if self.eval_steps and self.eval_steps > 0 and step % self.eval_steps == 0:
                    self.evaluate(epoch, step)
                if step > max_steps:
                    if self.training_args.eval_at_the_end:
                        self.evaluate(epoch, step)
                    break

    @torch.no_grad()
    def evaluate(self, train_epoch, train_step):
        loss_sum = 0
        for step, batch in tqdm(enumerate(self.eval_dataloader, start=1), total=self.eval_data_num):
            logits = self.model(batch[self.input_id_key], attention_mask=batch[self.attention_mask_key]).logits
            loss_sum += self._compute_label_smoothing_loss(batch, logits).item()
            if step > self.eval_data_num:
                print(f"epoch={train_epoch}, step={train_step}, eval loss={loss_sum / step}")
                return

    def get_eval_dataloader(
        self,
        eval_dataset,
        batch_size,
        data_collator,
        num_proc,
        should_shuffle=False,
    ) -> DataLoader:
        dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            num_workers=num_proc,
            collate_fn=data_collator,
        )
        return self.accelerator.prepare(dataloader)

    def get_train_dataloader(
        self,
        dataset,
        batch_size,
        data_collator,
        train_shuffle,
        num_proc,
    ) -> DataLoader:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=num_proc,
            collate_fn=data_collator,
        )
        return self.accelerator.prepare(dataloader)

    def save_model(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        print(f"saving model checkpoint to {output_dir}")
        supported_classes = (PreTrainedModel, PeftModel)
        state_dict = self.model.state_dict()
        if not isinstance(self.model, supported_classes):
            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                )
            else:
                print("trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir,
                state_dict,
            )

    def merge_and_unload(self):
        return self.model.merge_and_unload()


class DpoTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        reference_model,
        data_collator,
        dataset,
        eval_dataset,
        training_args: FinetuneArguments,
        optimizer,
        lr_scheduler,
        accelerator,
        input_id_key,
        attention_mask_key,
        label_key,
        label_ignore_index=-100,
        peft_config=None,
    ) -> None:
        super(DpoTrainer, self).__init__(
            model,
            data_collator=data_collator,
            dataset=dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            input_id_key=input_id_key,
            attention_mask_key=attention_mask_key,
            label_key=label_key,
            label_ignore_index=label_ignore_index,
            peft_config=peft_config,
        )
        self.reference_model = self.accelerator.prepare(
            reference_model,
        )
        assert (
            self.reference_model.num_parameters(only_trainable=True) == 0
        ), "reference model should have no parameters that require grad"
        self.dpo_beta = training_args.dpo_beta
        self.split_chosen_rejected = training_args.dpo_split_chosen_rejected

    def _compute_log_softmax(
        self,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model,
    ) -> torch.Tensor:
        """_summary_

        Args:
            label (torch.Tensor): (n_batch, len_sequence)
            input_ids (torch.Tensor): (n_batch, len_sequence, n_dim)
            attention_mask: (n_batch, len_sequence)
            model

        Returns:
            torch.Tensor: log_softmax of shape (n_batch, len_sequence)
        """
        logits = model(input_ids, attention_mask=attention_mask).logits
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., :-1].contiguous()
        log_softmax = nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_softmax.dim() - 1:
            labels = labels.unsqueeze(-1)
        labels = torch.clamp(labels, min=0)
        log_softmax = log_softmax.gather(dim=-1, index=labels)
        log_softmax = torch.squeeze(log_softmax, dim=-1)
        return log_softmax

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        input_ids = batch[self.input_id_key]
        attention_mask = batch[self.attention_mask_key]
        labels = batch[self.label_key]

        half_size = int(input_ids.shape[0] / 2)
        if self.split_chosen_rejected:
            log_softmax = torch.cat(
                [
                    self._compute_log_softmax(
                        labels[:half_size], input_ids[:half_size], attention_mask[:half_size], self.model
                    ),
                    self._compute_log_softmax(
                        labels[half_size:], input_ids[half_size:], attention_mask[half_size:], self.model
                    ),
                ],
                dim=0,
            )
        else:
            log_softmax = self._compute_log_softmax(labels, input_ids, attention_mask, self.model)

        with torch.no_grad():
            if self.split_chosen_rejected:
                ref_log_softmax = torch.cat(
                    [
                        self._compute_log_softmax(
                            labels[:half_size], input_ids[:half_size], attention_mask[:half_size], self.reference_model
                        ),
                        self._compute_log_softmax(
                            labels[half_size:], input_ids[half_size:], attention_mask[half_size:], self.reference_model
                        ),
                    ],
                    dim=0,
                )
            else:
                ref_log_softmax = self._compute_log_softmax(labels, input_ids, attention_mask, self.reference_model)

        # to align with the operation inside _compute_log_softmax
        labels = labels[..., :-1].contiguous()
        label_mask = labels.eq(self.label_ignore_index)
        # to avoid negative label_ignore_index will throw error
        labels = torch.clamp(label_mask, min=0)
        chosen_logits = log_softmax[:half_size]
        chosen_ref_logits = ref_log_softmax[half_size:]
        rejected_logits = ref_log_softmax[:half_size]
        rejected_ref_logits = ref_log_softmax[half_size:]
        pi_logratios = chosen_logits - rejected_logits
        ref_logratios = chosen_ref_logits - rejected_ref_logits
        logits = pi_logratios - ref_logratios
        loss = -nn.functional.logsigmoid(self.dpo_beta * logits)
        loss.masked_fill(label_mask, 0.0)
        return loss.mean()

    def evaluate(self, train_epoch, train_step):
        pass
