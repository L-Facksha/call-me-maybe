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


# =========================
# VOCAB LOADING (ROBUST)
# =========================
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
        except:
            try:
                vocab[int(v)] = str(k)
            except:
                continue

    return vocab


# =========================
# HELPERS
# =========================
def is_valid_name_prefix(partial: str, valid_names: list[str]) -> bool:
    return any(name.startswith(partial) for name in valid_names)


def extract_logits(logits):
    """Ensure logits is always 1D numpy array"""
    if hasattr(logits, "shape"):
        if len(logits.shape) == 3:
            return logits[0, -1].numpy()
        elif len(logits.shape) == 2:
            return logits[-1].numpy()
        else:
            return logits.numpy()
    return np.array(logits)


# =========================
# NAME GENERATION
# =========================
def generate_name(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    valid_names: list[str],
    max_tokens: int = 50,
) -> str:

    current_name = ""
    ids = model.encode(prompt)[0].tolist()

    for _ in range(max_tokens):
        logits = extract_logits(model.get_logits_from_input_ids(ids))

        for token_id, token_str in vocab.items():
            if token_id >= len(logits):
                continue

            clean_token = token_str.replace("Ġ", "").strip()

            # Check if adding this token keeps us on a valid path
            candidate = (current_name + clean_token).strip()

            # Allow quote to end the name
            if token_str == '"':
                if current_name in valid_names:
                    # Slightly boost ending quote
                    logits[token_id] = -np.inf + 1e-9
                else:
                    logits[token_id] = -np.inf
            # For other tokens, ensure they lead toward a valid name
            elif candidate and not any(name.startswith(candidate) for name in valid_names):
                logits[token_id] = -np.inf

        if np.all(np.isneginf(logits)):
            break

        next_id = int(np.argmax(logits))
        if next_id not in vocab:
            break

        token = vocab[next_id]
        ids.append(next_id)

        if token == '"':
            if current_name in valid_names:
                return current_name
            break

        current_name += token.replace("Ġ", "")

    return "" if current_name not in valid_names else current_name


def generate_function_call(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    user_prompt: str,
    fn: "FunctionDefinition",
) -> dict[str, Any]:
    """Generate a complete function call object with name and parameters."""

    result = {
        "prompt": user_prompt,
        "name": generate_name(model, vocab, f'Function name: ', [fn.name]),
        "parameters": generate_args(model, vocab, user_prompt, fn)
    }

    return result


# =========================
# NUMBER GENERATION
# =========================
def generate_number(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    max_tokens: int = 30,
) -> float:
    """Generate a number with constrained decoding - only allow digits and dots."""

    number_text = ""
    ids = model.encode(prompt)[0].tolist()
    dot_used = False

    for step in range(max_tokens):
        logits = extract_logits(model.get_logits_from_input_ids(ids))

        # Mask all tokens that aren't valid number characters
        for token_id, token_str in vocab.items():
            if token_id >= len(logits):
                continue

            stripped = token_str.strip()

            # Empty tokens invalid
            if not stripped:
                logits[token_id] = -np.inf
                continue

            # Only allow single digits
            if len(stripped) == 1 and stripped.isdigit():
                logits[token_id] = logits[token_id]  # Keep valid
            # Allow dot only once
            elif stripped == "." and not dot_used:
                logits[token_id] = logits[token_id]
            else:
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

        # Early stop: if we have a valid number and next token would break it
        if number_text and len(number_text) > 0:
            try:
                float(number_text)
                # Check if we should stop (reasonable length for a number)
                if len(number_text) >= 4:
                    break
            except:
                pass

    try:
        result = float(number_text) if number_text else 0.0
        return result
    except:
        return 0.0


# =========================
# STRING GENERATION
# =========================
def generate_string(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    max_tokens: int = 50,
) -> str:
    """Generate a string with constrained decoding - extract from quotes/context."""

    string_text = ""
    ids = model.encode(prompt)[0].tolist()
    in_quotes = False

    for _ in range(max_tokens):
        logits = extract_logits(model.get_logits_from_input_ids(ids))

        # Mask JSON special chars and control chars
        for token_id, token_str in vocab.items():
            if token_id >= len(logits):
                continue

            # Hard mask: JSON structural chars
            if any(c in token_str for c in ['{', '}', '[', ']', ':', ',']):
                logits[token_id] = -np.inf
                continue

            # Mask newlines and excessive whitespace
            if token_str in ['\n', '\t'] or token_str.count(' ') > 1:
                logits[token_id] = -np.inf

        if np.all(np.isneginf(logits)):
            break

        next_id = int(np.argmax(logits))
        if next_id not in vocab:
            break

        token = vocab[next_id]
        clean_token = token.replace("Ġ", " ")

        ids.append(next_id)

        # Handle quotes
        if '"' in clean_token or "'" in clean_token:
            if not in_quotes:
                in_quotes = True
            else:
                string_text += clean_token.split('"')[0].split("'")[0]
                return string_text.strip()

        # Stop at excessive whitespace when we have text
        if string_text and clean_token.strip() == "":
            return string_text.strip()

        string_text += clean_token

        # Reasonable max length for extracted string
        if len(string_text) > 100:
            return string_text.strip()

    return string_text.strip()


# =========================
# ARGUMENT GENERATION
# =========================
def generate_args(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    user_prompt: str,
    fn: "FunctionDefinition",
) -> dict[str, Any]:
    """Generate function arguments by extracting from user prompt context."""

    result: dict[str, Any] = {}

    for i, (param_name, param_def) in enumerate(fn.parameters.items()):

        if param_def.type in ("number", "integer"):
            # Use regex to extract numbers from user_prompt as guidance
            numbers = re.findall(r'-?\d+\.?\d*', user_prompt)

            prompt = (
                f"Extract {param_name} from: {user_prompt}\n"
                f"Number: "
            )

            val = generate_number(model, vocab, prompt, max_tokens=15)

            # If model generated 0, try to extract from prompt directly
            if val == 0.0 and numbers:
                val = float(numbers[i % len(numbers)])

            result[param_name] = int(
                val) if param_def.type == "integer" else val

        elif param_def.type == "string":
            # Extract strings in quotes
            strings = re.findall(r"['\"]([^'\"]*)['\"]", user_prompt)

            prompt = (
                f"Extract {param_name} from: {user_prompt}\n"
                f"String: "
            )

            result[param_name] = generate_string(
                model, vocab, prompt, max_tokens=30)

            # If model generated empty, use regex extraction
            if not result[param_name] and strings:
                result[param_name] = strings[i % len(strings)]

        elif param_def.type == "boolean":
            prompt = (
                f"Question: {user_prompt}\n"
                f"Is {param_name} true? Answer: "
            )

            val = generate_string(model, vocab, prompt, max_tokens=10)
            result[param_name] = val.lower() in ["true", "yes", "1"]

    for key in fn.parameters.keys():
        if key not in result:
            result[key] = None

    return result
