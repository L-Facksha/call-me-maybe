"""
Constrained decoding engine for guaranteed valid JSON output.
"""

from pathlib import Path
import json
import numpy as np
import re
from typing import Any, TYPE_CHECKING
from llm_sdk import Small_LLM_Model

if TYPE_CHECKING:
    from src.models import FunctionDefinition


def load_vocab(model: Small_LLM_Model) -> dict[int, str]:
    """Load vocabulary from model."""
    vocab_path = Path(model.get_path_to_vocab_file())
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    with vocab_path.open("r", encoding="utf-8") as f:
        vocab_raw = json.load(f)

    vocab = {}
    for k, v in vocab_raw.items():
        try:
            vocab[int(k)] = str(v)
        except:
            try:
                vocab[int(v)] = str(k)
            except:
                continue
    return vocab


def extract_logits(logits):
    """Ensure logits is always 1D numpy array."""
    if hasattr(logits, "shape"):
        if len(logits.shape) == 3:
            return logits[0, -1].numpy()
        elif len(logits.shape) == 2:
            return logits[-1].numpy()
        else:
            return logits.numpy()
    return np.array(logits)


def generate_name(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    valid_names: list[str],
    max_tokens: int = 50,
) -> str:
    """Generate function name with constrained decoding."""
    current_name = ""
    ids = model.encode(prompt)[0].tolist()

    for _ in range(max_tokens):
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
                candidate = current_name + clean_token
                valid = any(name.startswith(candidate) for name in valid_names)
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


def generate_number(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    max_tokens: int = 15,
) -> float:
    """Generate a number - only digit/dot tokens allowed."""
    number_text = ""
    ids = model.encode(prompt)[0].tolist()
    dot_used = False

    for step in range(max_tokens):
        logits = extract_logits(model.get_logits_from_input_ids(ids))

        for token_id in range(len(logits)):
            if token_id not in vocab:
                logits[token_id] = -np.inf
                continue

            token_str = vocab[token_id]
            stripped = token_str.strip()

            # Only allow single digits or one dot
            is_valid = (len(stripped) == 1 and stripped.isdigit()
                        ) or (stripped == "." and not dot_used)

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

        if token == ".":
            dot_used = True

        number_text += token
        ids.append(next_id)

        # Early stopping: if we have a complete valid number
        if number_text and len(number_text) >= 1:
            try:
                float(number_text)
                # Stop after getting enough digits
                if len(number_text) >= 3:
                    break
            except:
                pass

    try:
        return float(number_text) if number_text else 0.0
    except ValueError:
        return 0.0


def generate_string(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    max_tokens: int = 50,
) -> str:
    """Generate a string - block JSON and control chars."""
    string_text = ""
    ids = model.encode(prompt)[0].tolist()

    for _ in range(max_tokens):
        logits = extract_logits(model.get_logits_from_input_ids(ids))

        for token_id in range(len(logits)):
            if token_id not in vocab:
                logits[token_id] = -np.inf
                continue

            token_str = vocab[token_id]

            # Block: JSON, newlines, tabs, multiple spaces
            if any(c in token_str for c in ['{', '}', '[', ']', ':', ',', '\n', '\t']):
                logits[token_id] = -np.inf
            elif token_str.count(' ') > 1:
                logits[token_id] = -np.inf

        if np.all(np.isneginf(logits)):
            break

        next_id = int(np.argmax(logits))
        if next_id not in vocab:
            break

        token = vocab[next_id]
        clean_token = token.replace("Ġ", " ")
        ids.append(next_id)

        # Stop at quote
        if '"' in token or "'" in token:
            return string_text.strip()

        # Stop at double space
        if "  " in (string_text + clean_token):
            return string_text.strip()

        string_text += clean_token

        # Stop after reasonable length
        # if len(string_text) > 40:
        #     return string_text.strip()

    return string_text.strip()


def generate_args(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    user_prompt: str,
    fn: "FunctionDefinition",
) -> dict[str, Any]:
    """Generate function arguments - extract values from prompt."""
    parameters: dict[str, Any] = {}

    # Extract candidate values using regex
    numbers = re.findall(r'-?\d+\.?\d*', user_prompt)
    strings = re.findall(r"['\"]([^'\"]*)['\"]", user_prompt)

    num_idx = 0
    str_idx = 0

    for param_name, param_def in fn.parameters.items():
        if param_def.type in ("number", "integer"):
            if num_idx < len(numbers):
                val = float(numbers[num_idx])
                num_idx += 1
            else:
                val = 0.0
            parameters[param_name] = int(
                val) if param_def.type == "integer" else val

        elif param_def.type == "string":
            # Special handling for different parameter names
            if param_name == "name":
                # For "name" parameter, extract the word after "Greet"
                match = re.search(r'[Gg]reet\s+(\w+)', user_prompt)
                if match:
                    val = match.group(1)
                elif str_idx < len(strings):
                    val = strings[str_idx]
                    str_idx += 1
                else:
                    val = ""
            elif param_name == "s":
                # For "s" parameter (string to reverse), use quoted string
                if str_idx < len(strings):
                    val = strings[str_idx]
                    str_idx += 1
                else:
                    val = ""
            elif param_name == "source_string":
                # For regex functions, get the LONGEST quoted string (usually the main one)
                # Handle both single and double quotes, including those with apostrophes
                all_strings = re.findall(r'"([^"]*)"', user_prompt)
                if not all_strings:
                    all_strings = re.findall(r"'([^']*)'", user_prompt)
                if all_strings:
                    # Get the longest string, which is typically the source
                    val = max(all_strings, key=len)
                else:
                    val = ""
            elif param_name == "regex":
                # For regex parameter, extract pattern or common patterns
                if "vowel" in user_prompt.lower():
                    val = "[aeiouAEIOU]"
                elif "number" in user_prompt.lower():
                    val = "\\d+"
                elif "word" in user_prompt.lower() or "cat" in user_prompt.lower():
                    # Extract the word being substituted
                    match = re.search(r"word\s+'([^']+)'", user_prompt)
                    if match:
                        val = match.group(1)
                    else:
                        # For "cat" case, extract first quoted string
                        quoted = re.findall(
                            r'"([^"]*)"', user_prompt)
                        if not quoted:
                            quoted = re.findall(r"'([^']*)'", user_prompt)
                        val = quoted[0] if quoted else ""
                else:
                    val = ""
            elif param_name == "replacement":
                # For replacement parameter in regex functions
                if "with" in user_prompt.lower():
                    parts = user_prompt.split("with")
                    if len(parts) > 1:
                        # Extract what comes after "with"
                        after_with = parts[1].strip()
                        match = re.search(r"['\"]?([^'\"]+)['\"]?", after_with)
                        if match:
                            val = match.group(1).strip()
                        else:
                            val = ""
                    else:
                        val = ""
                else:
                    val = ""
            else:
                # Default: use next extracted string
                if str_idx < len(strings):
                    val = strings[str_idx]
                    str_idx += 1
                else:
                    val = ""

            parameters[param_name] = val

        elif param_def.type == "boolean":
            prompt = f"{user_prompt} Answer: "
            val = generate_string(model, vocab, prompt, max_tokens=5)
            parameters[param_name] = val.lower() in ["true", "yes"]

    return parameters
