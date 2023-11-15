from __future__ import annotations

from typing import Optional

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

"""
for reference, the default generation config for llama2 is

GenerationConfig {
  "bos_token_id": 1,
  "do_sample": true,
  "eos_token_id": 2,
  "max_length": 4096,
  "pad_token_id": 0,
  "temperature": 0.6,
  "top_p": 0.9
}
"""


@dataclass
class GenerationArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    loading_path: Optional[str] = field(default=None, metadata={"help": "the model name"})

    pad_token_id: Optional[int] = field(default=0, metadata={"help": "pad token id"})
    bos_token_id: Optional[int] = field(default=1, metadata={"help": "begining of the sequence token_id"})
    eos_token_id: Optional[int] = field(default=2, metadata={"help": "end of the sequence token_id"})
    max_length: Optional[int] = field(default=4096, metadata={"help": "maximum length of the generated result"})

    num_beams: Optional[int] = field(default=1, metadata={"help": "number of beams for search"})
    num_beams_groups: Optional[int] = field(default=1, metadata={"help": "beams groups number"})
    num_beam_hypes: Optional[int] = field(
        default=1, metadata={"help": "how many hypothesis do we keep for each beam group"}
    )
    do_sample: Optional[bool] = field(default=False, metadata={"help": "whether to do sampling"})
    top_k: Optional[int] = field(default=1, metadata={"help": "top k to remain"})
    sequence_bias: Optional[dict] = field(default=None, metadata={"help": "sequence bias dict"})
    diversity_penalty: Optional[float] = field(default=0.0, metadata={"help", "penalty for diversity"})
    repetition_penalty: Optional[float] = field(default=1.0, metadata={"help", "penalty for repitition"})

    def get_dict(self):
        return asdict(self)

    def update(self, **kwargs) -> dict:
        self_dict = asdict(self)
        self_dict.update(**kwargs)
        return self_dict
