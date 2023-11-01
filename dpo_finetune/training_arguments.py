from __future__ import annotations

from typing import Optional

from dataclasses import dataclass
from dataclasses import field


@dataclass
class FinetuneArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"})
    finetune_data_dir: Optional[str] = field(
        default="data/finetune", metadata={"help": "the data dir for finetune dataloader"}
    )
    batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    eval_batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size for evaluation"})
    eval_at_the_end: Optional[bool] = field(default=True, metadata={"help": "do we eval at the end of each epoch"})
    train_shuffle: Optional[bool] = field(default=True, metadata={"help": "whether to shuffle the training data"})
    data_loader_worker: Optional[int] = field(default=20, metadata={"help": "the number of data loader workers"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of steps before we update the parameters"}
    )
    max_train_steps: Optional[int] = field(default=500, metadata={"help": "the maximum training steps"})
    train_epochs: Optional[int] = field(default=20, metadata={"help": "the maximum training epochs"})
    max_grad_norm: Optional[float] = field(default=1.0, metadata={"help": "the maximum gradient norm value"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum length of the tokenized tensor"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "lr scheduler type"})
    num_warm_up_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "learning rate"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    eval_steps: Optional[int] = field(default=600, metadata={"help": "the evaluation frequency"})
    log_steps: Optional[int] = field(default=50, metadata={"help": "the log frequency"})
    eval_nums: Optional[int] = field(default=100, metadata={"help": "how many batch do we evaluate"})
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label pad token id"})
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "max prompt length"})
    label_smoother_epsilon: Optional[float] = field(default=0.1, metadata={"help": "label smoother epsilon"})
    dpo_beta: Optional[float] = field(default=0.1, metadata={"help": "DPO beta"})
    dpo_split_chosen_rejected: Optional[bool] = field(
        default=True, metadata={"help": "whether to split the chosen and rejected to save memory"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    finetune_output_dir: Optional[str] = field(
        default="/tmp/model_dir/finetune/", metadata={"help": "the model save dir for finetune"}
    )


test_arguments = FinetuneArguments(
    train_epochs=1,
    max_train_steps=100,
    eval_at_the_end=False,
    # reduce the max length for dpo otherwise DPO might cause OOM
    max_length=512,
    max_prompt_length=256,
)

train_arguments = FinetuneArguments(
    # ignore the eval for now
    eval_steps=0,
)
