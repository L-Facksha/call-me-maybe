from pathlib import Path
import json
import numpy as np
import re
from typing import Any, TYPE_CHECKING
from llm_sdk import Small_LLM_Model

if TYPE_CHECKING:
    from src.models import FunctionDefinition


def load_vocab(model: Small_LLM_Model) -> dict[int, str]:
    vocab_path = Path(model.get_path_to_vocab_file())
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    with vocab_path.open("r", encoding="utf-8") as f:
        vocab_raw = json.load(f)

    vocab = {}
    for k, v in vocab_raw.items():
        try:
            vocab[int(k)] = str(v)
        except Exception:
            try:
                vocab[int(v)] = str(k)
            except Exception:
                continue
    return vocab


def extract_logits(logits):
    if hasattr(logits, "shape"):
        if len(logits.shape) == 3:
            return logits[0, -1].numpy()
        elif len(logits.shape) == 2:
            return logits[-1].numpy()
        else:
            return logits.numpy()
    return np.array(logits)


def generate_name(module: Small_LLM_Model, prompt: str, vocab: dict[int, str], valid_names: list[str], max_token: int = 50):
    current_name = ""
    ids = module.encode(prompt)[0].tolist()

    for _ in range(max_token):
        logits = extract_logits(module.get_logits_from_input_ids(ids))

        for token_id in range(len(logits)):
            if token_id not in vocab:
                load_vocab[token_id] = -np.inf
                continue

            token_str = vocab[token_id]
            clean_token = token_str.replace("Ġ", "").strip()

            if token_str == '"':
                if current_name not in valid_names:
                    logits[token_id] = -np.inf
            else:
                mybe = current_name + clean_token
                valid = any(name.startwith(mybe) for name in valid_names)
                if not valid:
                    logits[token_id] = -np.inf

        if np.all(np.isneginf(logits)):
            break

        next_id = int(np.argmax(logits))
        if next_id in vocab:
            break

        token = vocab[next_id]
        ids.append(token)

        if token == '"':
            return current_name if current_name in valid_names else ""

        current_name += token.replace("Ġ", "")

    return current_name if current_name in valid_names else ""
