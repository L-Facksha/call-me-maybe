"""Constrained decoding generator for function calling."""

from pathlib import Path
import json
import re
import numpy as np
from typing import Any, TYPE_CHECKING
from llm_sdk.llm_sdk import Small_LLM_Model

if TYPE_CHECKING:
    from src.models import FunctionDefinition


def load_vocab(model: Small_LLM_Model) -> dict[int, str]:
    vocab_path = Path(model.get_path_to_vocab_file())
    with vocab_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    vocab: dict[int, str] = {}
    for k, v in raw.items():
        try:
            vocab[int(k)] = str(v)
        except:
            try:
                vocab[int(v)] = str(k)
            except:
                continue
    return vocab


def extract_logits(logits: Any) -> np.ndarray:
    """Safely convert logits to a numpy array and handle indexing."""
    # Convert list to numpy array if it isn't one
    if isinstance(logits, list):
        logits = np.array(logits)

    # Check if we have shape (batch, seq_len, vocab) or (seq_len, vocab)
    if len(logits.shape) == 3:
        return logits[0, -1]
    if len(logits.shape) == 2:
        return logits[-1]
    return logits


def _clean(token: str) -> str:
    # Ġ is space, Ċ is newline
    return token.replace("Ġ", " ").replace("▁", "").replace("Ċ", "\n")


def generate_name(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    valid_names: list[str],
    max_token: int = 15,
) -> str:
    current = ""
    ids = model.encode(prompt)[0].tolist()

    for _ in range(max_token):
        raw_logits = model.get_logits_from_input_ids(ids)
        logits = extract_logits(raw_logits)
        mask = np.full_like(logits, -np.inf)

        for tid, token_raw in vocab.items():
            if tid >= len(logits):
                continue
            clean = _clean(token_raw)

            if '"' in clean:
                if current.strip() in valid_names:
                    mask[tid] = logits[tid]
            else:
                potential = (current + clean).strip()
                if any(name.startswith(potential) for name in valid_names) or potential == "":
                    mask[tid] = logits[tid]

        if np.all(np.isneginf(mask)):
            break
        next_id = int(np.argmax(mask))
        ids.append(next_id)
        tok = _clean(vocab[next_id])
        if '"' in tok:
            break
        current += tok
    return current.strip()


def generate_string(model, vocab, prompt, max_token=150) -> str:
    current = ""
    ids = model.encode(prompt)[0].tolist()
    stop_sequences = ["\n", "Answer:", "Request:", "Value:"]

    for _ in range(max_token):
        raw_logits = model.get_logits_from_input_ids(ids)
        logits = extract_logits(raw_logits)
        mask = np.full_like(logits, -np.inf)

        for tid, token_raw in vocab.items():
            if tid >= len(logits):
                continue
            clean = _clean(token_raw)
            if any(stop in clean for stop in stop_sequences):
                continue
            mask[tid] = logits[tid]

        if np.all(np.isneginf(mask)):
            break
        next_id = int(np.argmax(mask))
        ids.append(next_id)
        tok = _clean(vocab[next_id])

        if '"' in tok:
            current += tok.split('"')[0]
            break
        current += tok
    return current.strip()


def generate_number(model, vocab, prompt, max_token=25) -> float:
    current = ""
    ids = model.encode(prompt)[0].tolist()
    allowed = set("0123456789.+- eE")

    for _ in range(max_token):
        raw_logits = model.get_logits_from_input_ids(ids)
        logits = extract_logits(raw_logits)
        mask = np.full_like(logits, -np.inf)

        for tid, token_raw in vocab.items():
            if tid >= len(logits):
                continue
            clean = _clean(token_raw)
            if clean and all(c in allowed for c in clean):
                mask[tid] = logits[tid]

        if np.all(np.isneginf(mask)):
            break
        next_id = int(np.argmax(mask))
        ids.append(next_id)
        tok = _clean(vocab[next_id])

        if "\n" in tok or (current.strip() and tok.isspace()):
            break
        current += tok

    try:
        # Final regex to grab only the valid number part
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", current)
        return float(nums[0]) if nums else 0.0
    except:
        return 0.0


def generate_args(model, vocab, user_prompt, func) -> dict[str, Any]:
    parameters = {}
    for p_name, p_def in func.parameters.items():
        if p_def.type == "number":
            # Simplified prompt for zero-shot number extraction
            prompt = f"User Request: {user_prompt}\nTarget: {func.name}\nValue of {p_name}: "
            parameters[p_name] = generate_number(model, vocab, prompt)
        else:
            prompt = f"User Request: {user_prompt}\nTarget: {func.name}\nValue of {p_name}: \""
            parameters[p_name] = generate_string(model, vocab, prompt)
    return parameters
