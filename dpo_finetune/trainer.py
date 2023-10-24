from __future__ import annotations

from typing import Dict

import os
import time

import torch
from accelerate import Accelerator
from peft import get_peft_model
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .training_arguments import FinetuneArguments


class BaseTrainer:
    def __init__(
        self,
        model,
        data_collator,
        dataset,
        training_args: FinetuneArguments,
        optimizer,
        lr_scheduler,
        accelerator,
        input_id_key,
        attention_mask_key,
        label_key,
        label_smoother_epsilon=0.1,
        label_ignore_index=-100,
        peft_config=None,
    ) -> None:
        self.accelerator = accelerator if accelerator else Accelerator()
        self.model = model
        if peft_config:
            self.model = get_peft_model(model, peft_config)
            # peft will freeze lm_head
            self.model.base_model.model.lm_head.weight.requires_grad = True
        self.train_dataloader = self.get_train_dataloader(
            dataset,
            batch_size=training_args.batch_size,
            data_collator=data_collator,
            train_shuffle=training_args.train_shuffle,
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
        self.label_smoother_epsilon = label_smoother_epsilon
        self.label_ignore_index = label_ignore_index
        print(f"number of parameters that require gradients = {model.num_parameters(only_trainable=True)}")

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
                logits = self.model(
                    batch[self.input_id_key],
                    attention_mask=batch[self.attention_mask_key],
                ).logits
                loss = self._compute_label_smoothing_loss(batch, logits)
                loss = loss / self.gradient_accumulation_steps
                aver_loss += loss.item()
                self.accelerator.backward(loss)

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.training_args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    print(f"epoch={epoch}, step={step}, loss={aver_loss}")
                    aver_loss = 0
                    completed_steps += 1
                if step > max_steps:
                    break

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
