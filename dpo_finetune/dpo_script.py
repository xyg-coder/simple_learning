"""
python -m trainer.ppytorch.mlenv.dpo_finetune.dpo_script.py
"""
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from peft import LoraConfig
from transformers import AutoTokenizer
from transformers import get_scheduler

from .data_collators import CONCATENATED_ATTENTION_MASK
from .data_collators import CONCATENATED_INPUT_IDS
from .data_collators import CONCATENATED_LABELS
from .data_collators import FinetuneDataCollator
from .data_collators import dpo_collate
from .optimization import create_adamw_paged_32_bit_optimizer
from .trainer import DpoTrainer
from .training_arguments import test_arguments

"""
python -m trainer.ppytorch.mlenv.dpo_finetune.dpo_script.py
"""

finetune_args = test_arguments
accelerator = Accelerator()
model = AutoPeftModelForCausalLM.from_pretrained(
    test_arguments.finetune_output_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    is_trainable=True,
    load_in_4bit=True,
)
reference_model = AutoPeftModelForCausalLM.from_pretrained(
    test_arguments.finetune_output_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    is_trainable=False,
)
model.config.use_cache = False
reference_model.config.use_cache = False

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
    split="train",
    use_auth_token=True,
)

dataset = dataset.train_test_split(test_size=0.005, seed=None)
train_data = dataset["train"]
valid_data = dataset["test"]
print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

dpo_collator = FinetuneDataCollator(
    tokenizer=tokenizer,
    max_length=finetune_args.max_length,
    label_pad_token_id=finetune_args.label_pad_token_id,
    max_prompt_length=finetune_args.max_prompt_length,
    collate_func=dpo_collate,
)

optimizer = create_adamw_paged_32_bit_optimizer(
    model,
    finetune_args,
)

lr_scheduler = get_scheduler(
    finetune_args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=finetune_args.num_warm_up_steps,
    num_training_steps=finetune_args.max_train_steps,
)


dpo_trainer = DpoTrainer(
    model=model,
    reference_model=reference_model,
    data_collator=dpo_collator,
    dataset=train_data,
    eval_dataset=valid_data,
    training_args=finetune_args,
    accelerator=accelerator,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    input_id_key=CONCATENATED_INPUT_IDS,
    attention_mask_key=CONCATENATED_ATTENTION_MASK,
    label_key=CONCATENATED_LABELS,
    peft_config=peft_config,
)

dpo_trainer.train()
