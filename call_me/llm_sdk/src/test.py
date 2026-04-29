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


def gnerate_number(model: Small_LLM_Model, prompt: str, vocab: dict[int, str], max_token: int = 15):
    number_text = ""
    ids = model.encode(prompt)[0].tolist()
    dot_used = False

    for _ in range(max_token):
        logits = extract_logits(model.get_logits_from_input_ids(ids))

        for token_id in range(len(logits)):
            if token_id not in vocab:
                logits[token_id] = -np.inf
                continue

            token_str = vocab[token_id]
            stripped_token = token_str.strip()

            is_valid = all(c.isdigit() for c in stripped_token) or (
                stripped_token == '.' and not dot_used)

            if not is_valid:
                logits[token_id] = -np.inf

        if np.all(np.isneginf(logits)):
            break

        next_id = int(np.argmax(logits))
        if next_id not in vocab:
            break

        token = vocab[next_id].strip()
        if not token:
            break

        if token == '.':
            dot_used = True

        number_text += token
        ids.append(next_id)

    try:
        return float(number_text) if number_text else 0.0
    except Exception:
        return 0.0


def generate_string(model: Small_LLM_Model, vocab: dict[int, str], prompt: str, max_token: int = 50) -> str:
    string_text = ""
    ids = model.encode(prompt)[0].tolist()

    for _ in range(max_token):
        logits = extract_logits(model.get_logits_from_input_ids(ids))

        for token_id in logits:
            if token_id not in vocab:
                logits[token_id] = -np.inf
                continue

            token_str = vocab[token_id]

            if any(c in token_str for c in ['{', '}', '[', ']', ':', '.', '\n', '\t']):
                logits[token_id] = -np.inf
            elif token_str.count(' ') > 1:
                logits[token_id] = -np.inf

        if np.all(np.isneginf(logits)):
            break

        next_id = np.argmax(logits)
        if next_id not in vocab:
            break

        token = vocab[next_id]
        clear_token = token.replace('Ġ', ' ')
        ids.append(clear_token)

        if '"' in token or "'" in token:
            return string_text.strip()

        if "  " in (string_text + clear_token):
            return string_text.strip()

        string_text += clear_token

        return string_text.strip()


def generate_args(model: Small_LLM_Model, vocab: dict[int, str], user_prompt: str, func: FunctionDefinition) -> dict[str: Any]:
    parameters: dict[str, Any] = {}

    numbers = re.findall(r"-?\d+\.\d*", user_prompt)
    strings = re.findall(r"['\"]([^'\"]*)['\"]", user_prompt)

    num_index = 0
    str_index = 0

    for param_name, param_def in func.parameters.items():
        if param_name.type == "number":
            if num_index < len(numbers):
                value = float(numbers(num_index))
                num_index += 1
        else:
            value = 0.0
            
        parameters[param_name] = value

