from __future__ import annotations

import os

import torch

# otherwise it might throw "module 'torch.utils' has no attribute 'checkpoint'" error
import torch.utils.checkpoint
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import get_scheduler

from .data_collators import CHOSEN_ATTENTION_MASK
from .data_collators import CHOSEN_INPUT_IDS
from .data_collators import CHOSEN_LABELS
from .data_collators import FinetuneDataCollator
from .data_collators import finetune_collate
from .optimization import create_adamw_paged_32_bit_optimizer
from .trainer import BaseTrainer
from .training_arguments import test_arguments

"""
pip install -q xformers wandb datasets gradio tyro>=0.5.7 &&
pip install accelerate -U &&
pip install bitsandbytes -U &&
pip install git+https://github.com/huggingface/transformers &&
pip install git+https://github.com/huggingface/peft &&
pip install --upgrade 'urllib3<2'

Note: pip upgrade is due to
urllib3 cannot import name 'DEFAULT_CIPHERS' from 'urllib3.util.ssl_'

NotImplementedError: Loading a dataset cached in a LocalFileSystem is not supported.
    sudo vim /usr/local/lib/python3.8/site-packages/datasets/filesystems/__init__.py
    change to -> if fs is not None and fs.protocol[0] != "file":

huggingface-cli login
python -m trainer.ppytorch.mlenv.dpo_finetune.finetune_script.py


current finding:
1. AutoModelForCausalLM.from_pretrained with quantization_config will use Linear4bit to replace the Linear layers of the attentions.
This would cause a big reduction of the trainable parameters.
"""

finetune_args = test_arguments

"""firstly finetune the model
"""
accelerator = Accelerator()
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    finetune_args.model_name,
    quantization_config=bnb_config,
    device_map={"": Accelerator().local_process_index},
    trust_remote_code=True,
    use_auth_token=True,
)
# in the training, we don't need decoder attention cache
base_model.config.use_cache = False

peft_config = LoraConfig(
    r=finetune_args.lora_r,
    lora_alpha=finetune_args.lora_alpha,
    lora_dropout=finetune_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(
    finetune_args.model_name,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# dataset: a dict of {qid', 'question', 'date', 'metadata', 'response_j', 'response_k'} as the keys
dataset = load_dataset(
    finetune_args.dataset_name,
    data_dir=finetune_args.finetune_data_dir,
    split="train",
    use_auth_token=True,
)

dataset = dataset.train_test_split(test_size=0.005, seed=None)
train_data = dataset["train"]
valid_data = dataset["test"]
print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

finetune_data_collator = FinetuneDataCollator(
    tokenizer=tokenizer,
    max_length=finetune_args.max_length,
    label_pad_token_id=finetune_args.label_pad_token_id,
    max_prompt_length=finetune_args.max_prompt_length,
    collate_func=finetune_collate,
)


# use paged_adamw_32bit as optimizer
optimizer = create_adamw_paged_32_bit_optimizer(
    base_model,
    finetune_args,
)

lr_scheduler = get_scheduler(
    finetune_args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=finetune_args.num_warm_up_steps,
    num_training_steps=finetune_args.max_train_steps,
)

ft_trainer = BaseTrainer(
    model=base_model,
    data_collator=finetune_data_collator,
    dataset=train_data,
    eval_dataset=valid_data,
    training_args=finetune_args,
    accelerator=accelerator,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    input_id_key=CHOSEN_INPUT_IDS,
    attention_mask_key=CHOSEN_ATTENTION_MASK,
    label_key=CHOSEN_LABELS,
    peft_config=peft_config,
)

ft_trainer.train()
ft_trainer.save_model(finetune_args.finetune_output_dir)

model = AutoPeftModelForCausalLM.from_pretrained(
    finetune_args.finetune_output_dir, device_map="auto", torch_dtype=torch.bfloat16
)
merged_model = model.merge_and_unload()

output_merged_dir = os.path.join(finetune_args.finetune_output_dir, "final_merged_checkpoint")
merged_model.save_pretrained(output_merged_dir)
