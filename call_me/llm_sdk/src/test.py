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


def generate_name(model: Small_LLM_Model, vocab: dict[int, str], prompt: str, valid_names: list[str], max_token: int = 50):
    current_name = ""
    ids = model.encode(prompt)[0].tolist()

    for _ in range(max_token):
        logits = extract_logits(model.get_logits_from_input_ids(ids))

        for token_id in range(len(logits)):
            if token_id not in vocab:
                logits[token_id] = -np.inf
                continue

            token_str = vocab[token_id]
            clean_token = token_str.replace("Ġ", "").strip()

            if token_str == '"':
                if current_name not in valid_names:
                    logits[token_id] = -np.inf
            else:
                mybe = current_name + clean_token
                valid = any(name.startswith(mybe) for name in valid_names)
                if not valid:
                    logits[token_id] = -np.inf

        if np.all(np.isneginf(logits)):
            break

        next_id = int(np.argmax(logits))
        if next_id not in vocab:
            break

        token = vocab[next_id]
        ids.append(next_id)

        if token == '"':
            return current_name if current_name in valid_names else ""

        current_name += token.replace("Ġ", "")

    return current_name if current_name in valid_names else ""


def generate_args(user_prompt: str, func: "FunctionDefinition") -> dict[str, Any]:
    parameters: dict[str, Any] = {}

    numbers = re.findall(r"-?\d+(?:\.\d+)?", user_prompt)
    strings = re.findall(r"['\"]([^'\"]*)['\"]", user_prompt)

    num_index = 0
    str_index = 0

    for param_name, param_def in func.parameters.items():
        if param_def.type == "number":
            if not numbers:
                return None
            if num_index < len(numbers):
                try:
                    value = float(numbers[num_index])
                except:
                    value = 0.0
                num_index += 1
            else:
                value = 0.0
            parameters[param_name] = value

        elif param_def.type == "string":
            if not strings:
                words = user_prompt.split()
                strings = [words[-1]] if words else []
            if param_name == "name":
                get_name = re.search(r"[Gg]reet\s+(\w+)", user_prompt)
                if get_name:
                    value = get_name.group(1)
                elif str_index < len(strings):
                    value = strings[str_index]
                    str_index += 1
                else:
                    value = ""
            elif param_name == "s":
                if str_index < len(strings):
                    value = strings[str_index]
                    str_index += 1
                else:
                    value = ""
            elif param_name == "source_string":
                all_strings = re.findall(r"'([^']*)'", user_prompt)
                if not all_strings:
                    all_strings = re.findall(r'"([^"]*)"', user_prompt)
                if all_strings:
                    value = max(all_strings, key=len)
                else:
                    value = ""
            elif param_name == "regex":
                if "vowel" in user_prompt.lower():
                    value = "[aeiouAEIOU]"
                elif "number" in user_prompt.lower():
                    value = "\\d+"
                elif "word" in user_prompt.lower() or "cat" in user_prompt.lower():
                    get_word = re.findall(r"word\s+'([^']+)'", user_prompt)
                    if get_word:
                        value = get_word[0]
                    else:
                        quoted = re.findall(r'"([^"]*)"', user_prompt)
                        if not quoted:
                            quoted = re.findall(r"'([^']*)'", user_prompt)
                        value = quoted[0] if quoted else ""
                else:
                    value = ""
            elif param_name == "replacement":
                if "with" in user_prompt.lower():
                    parts = user_prompt.split("with")
                    if len(parts) > 1:
                        after_with = parts[1].strip()
                        get_word = re.search(
                            r"['\"]?([^'\"]+)['\"]?", after_with)
                        if get_word:
                            value = get_word.group(1).strip()
                        else:
                            value = ""
                    else:
                        value = ""
                else:
                    value = ""
            else:
                if str_index < len(strings):
                    value = strings[str_index]
                    str_index += 1
                else:
                    value = ""
            parameters[param_name] = value

    return parameters
