from __future__ import annotations

from transformers import AutoModelForCausalLM

from .generation_arguments import GenerationArguments

"""
python -m trainer.ppytorch.mlenv.dpo_finetune.generation_script.py
"""

model = AutoModelForCausalLM.from_pretrained(GenerationArguments.model_name)
print(model.generation_config)
