from __future__ import annotations

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from transformers import AdamW
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import get_scheduler

from .data_collators import CHOSEN_ATTENTION_MASK
from .data_collators import CHOSEN_INPUT_IDS
from .data_collators import CHOSEN_LABELS
from .data_collators import DpoFinetuneDataCollator
from .optimization import create_adamw_paged_32_bit_optimizer
from .trainer import BaseTrainer
from .training_arguments import FinetuneArguments

"""
pip install -q xformers wandb datasets gradio tyro>=0.5.7
pip install accelerate -U
pip install bitsandbytes -U
pip install git+https://github.com/huggingface/transformers
pip install git+https://github.com/huggingface/peft
huggingface-cli login
python -m trainer.ppytorch.mlenv.dpo_finetune.finetune_script.py


if see urllib3 cannot import name 'DEFAULT_CIPHERS' from 'urllib3.util.ssl_'
 pip install --upgrade 'urllib3<2'

NotImplementedError: Loading a dataset cached in a LocalFileSystem is not supported.
    sudo vim /usr/local/lib/python3.8/site-packages/datasets/filesystems/__init__.py
    change to -> if fs is not None and fs.protocol[0] != "file":

current finding:
1. AutoModelForCausalLM.from_pretrained with quantization_config will use Linear4bit to replace the Linear layers of the attentions.
This would cause a big reduction of the trainable parameters.
2. Peft config will add lora adapters to the attention layers. But would freeze the last linearLayer. Not sure if this is expected
What we can try is use peft + quantization, but have the last linearLayer require grad
"""

finetune_args = FinetuneArguments()

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
finetune_dataset = load_dataset(
    finetune_args.dataset_name,
    data_dir=finetune_args.finetune_data_dir,
    split="train",
    use_auth_token=True,
)

finetune_data_collator = DpoFinetuneDataCollator(
    tokenizer=tokenizer,
    max_length=finetune_args.max_length,
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
    dataset=finetune_dataset,
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
