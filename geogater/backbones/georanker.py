"""GeoRanker reward model (inference only).

Qwen2-VL with a scalar value head, loaded with the released GeoRanker LoRA
adapter. From the official GeoRanker implementation
(https://github.com/Applied-Machine-Learning-Lab/GeoRanker, Apache-2.0), modified:
training-only code paths were removed.
"""

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoConfig, Qwen2VLForConditionalGeneration


def _reward_model_class(base_cls, value_head_prefix: str = "value_head"):
    class RewardModel(base_cls):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))
            self.value_head_prefix = value_head_prefix
            self.normalize_reward = getattr(config, "normalize_reward", False)
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids,
            attention_mask=None,
            pixel_values=None,
            return_output=False,
            **kwargs,
        ):
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
                **kwargs,
            )
            last_hidden = outputs.hidden_states[-1]
            values = getattr(self, self.value_head_prefix)(last_hidden).squeeze(-1)
            reward = values[:, -1]  # left padding -> last token is the EOS
            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std
            return (reward, outputs) if return_output else reward

    return RewardModel


def load_georanker(
    base_model: str = "Qwen/Qwen2-VL-7B-Instruct",
    lora_path: str = "checkpoints/georanker_lora",
    use_flash_attention_2: bool = True,
    device_map="auto",
):
    """Load the frozen GeoRanker reward model."""
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    config.normalize_reward = False
    config._attn_implementation = (
        "flash_attention_2" if use_flash_attention_2 else "eager"
    )
    cls = _reward_model_class(Qwen2VLForConditionalGeneration)
    model = cls.from_pretrained(
        base_model,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
    model.config.use_cache = False
    model.eval()
    return model
