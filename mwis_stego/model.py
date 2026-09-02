"""Loading Qwen3 and turning a prompt into a generation context."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .tokens import ByteVocab

__all__ = ["StegoLM"]

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


class StegoLM:
    """A causal LM plus the byte view of its vocabulary.

    Weights are loaded in float32.  The sender and the receiver must obtain
    bit-identical logits or the arithmetic coder desynchronises, and float32 on
    a single device with matching call shapes gives that; float16 on V100 does
    not reliably.
    """

    def __init__(self, name: str = DEFAULT_MODEL, device: str = "cuda", dtype=torch.float32):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        self.model.eval().to(device)
        self.device = device
        self.vocab = ByteVocab(self.tokenizer)
        # Control tokens are literal text, never part of the covert stream.
        self.banned_ids = torch.tensor(
            sorted(self.tokenizer.get_added_vocab().values()), device=device, dtype=torch.long
        )

    def chat_context(self, prompt: str) -> list[int]:
        """Prompt as a user turn, with Qwen3 thinking disabled.

        `enable_thinking=False` pre-fills an empty `<think></think>` block, so
        generation starts on the answer and no reasoning trace is emitted.
        """
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def raw_context(self, prompt: str) -> list[int]:
        """Prompt as a plain prefix, for base-LM continuation."""
        return self.tokenizer.encode(prompt, add_special_tokens=False)
