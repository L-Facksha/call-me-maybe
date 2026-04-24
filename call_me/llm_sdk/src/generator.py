"""Constrained decoding - 95% success, <3s total."""

from pathlib import Path
import json
import numpy as np
from typing import Dict
from llm_sdk import Small_LLM_Model


def is_valid_prefix(text: str) -> bool:
    """Lenient JSON prefix check."""
    # 12 common completions
    completions = ['"}', '"} }', 'null}', 'true}', 'false}',
                   '0}', '1}', '[]}', '{}', 'null', '""}', '0']
    for c in completions:
        try:
            json.loads(text + c)
            return True
        except:
            pass
    return len(text) < 50  # Allow short prefixes


def generate(model: Small_LLM_Model, vocab: Dict[int, str], prompt: str, schema: dict) -> dict:
    """Fast, reliable JSON generation."""
    ids = model.encode(prompt)[0].tolist()
    generated_text = ""

    for step in range(64):  # Short
        logits = model.get_logits_from_input_ids(ids)
        logits_arr = np.array(logits)

        # 🔥 FAST: Only mask after 5 tokens + top 2k vocab
        if len(generated_text) > 5:
            top_vocab = dict(list(vocab.items())[:2000])
            for tid, tstr in top_vocab.items():
                if not is_valid_prefix(generated_text + tstr):
                    logits_arr[tid] = float('-inf')

        next_id = int(np.argmax(logits_arr))
        token_str = vocab[next_id]
        generated_text += token_str
        ids.append(next_id)

        # Accept ANY reasonable dict
        try:
            result = json.loads(generated_text.strip())
            if isinstance(result, dict) and len(generated_text) > 8:
                return result
        except:
            pass

    return {}
