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
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    with vocab_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    vocab: dict[int, str] = {}
    for k, v in raw.items():
        try:
            vocab[int(k)] = str(v)
        except Exception:
            try:
                vocab[int(v)] = str(k)
            except Exception:
                continue
    return vocab


def extract_logits(logits: Any) -> np.ndarray:
    if hasattr(logits, "shape"):
        if len(logits.shape) == 3:
            return logits[0, -1].numpy()
        elif len(logits.shape) == 2:
            return logits[-1].numpy()
        else:
            return logits.numpy()
    return np.array(logits)


def _clean(token: str) -> str:
    return token.replace("Ġ", " ").replace("▁", " ").replace("Ċ", " ")


def _extract_string_for_param(
    user_prompt: str,
    param_index: int,
    total_string_params: int,
) -> str | None:
    """Extract the correct string value for a parameter from the user prompt.

    Handles two cases:
    - "... X ... in 'SOURCE'" pattern: the string after "in" is the source
      (first param), the others fill remaining params in order.
    - Normal positional: first quote -> first param, second -> second, etc.
    Returns None when the param cannot be extracted and the LLM should be used.
    """
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', user_prompt)
    candidates = [a if a else b for a, b in quoted]

    if not candidates:
        # No quotes: single-param function takes the last word.
        if total_string_params == 1 and param_index == 0:
            words = user_prompt.strip().split()
            return words[-1] if words else None
        return None

    if len(candidates) == total_string_params:
        # Check for "... in 'SOURCE'" pattern.
        in_match = re.search(
            r"\bin\s+['\"]([^'\"]+)['\"]", user_prompt, re.IGNORECASE)
        if in_match and total_string_params >= 2:
            source_val = in_match.group(1)
            rest = [c for c in candidates if c != source_val]
            ordered = [source_val] + rest
            return ordered[param_index] if param_index < len(ordered) else None
        # Normal positional assignment.
        return candidates[param_index] if param_index < len(candidates) else None

    # Fewer candidates than params: assign what we have, leave rest to LLM.
    return candidates[param_index] if param_index < len(candidates) else None


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
        logits = extract_logits(model.get_logits_from_input_ids(ids))

        for tid in range(len(logits)):
            if tid not in vocab:
                logits[tid] = -np.inf
                continue
            clean = _clean(vocab[tid]).strip()
            if clean == '"':
                if current not in valid_names:
                    logits[tid] = -np.inf
            else:
                maybe = current + clean
                if not any(name.startswith(maybe) for name in valid_names):
                    logits[tid] = -np.inf

        if np.all(np.isneginf(logits)):
            break

        next_id = int(np.argmax(logits))
        if next_id not in vocab:
            break

        clean = _clean(vocab[next_id]).strip()
        ids.append(next_id)

        if clean == '"':
            return current if current in valid_names else ""
        current += clean

    return current if current in valid_names else ""


def generate_number(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    user_prompt: str,
    param_index: int,
    max_token: int = 15,
) -> float:
    numbers = [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", user_prompt)]
    if param_index < len(numbers):
        return numbers[param_index]

    # ids = model.encode(prompt)[0].tolist()
    # current = ""

    # for _ in range(max_token):
    #     logits = extract_logits(model.get_logits_from_input_ids(ids))

    #     for tid in range(len(logits)):
    #         if tid not in vocab:
    #             logits[tid] = -np.inf
    #             continue
    #         clean = _clean(vocab[tid]).strip()
    #         if not clean:
    #             logits[tid] = -np.inf
    #             continue
    #         if not current and clean[0] not in "0123456789-+.":
    #             logits[tid] = -np.inf
    #         elif current and not all(c in "0123456789.eE+-" for c in clean):
    #             logits[tid] = -np.inf

    #     if np.all(np.isneginf(logits)):
    #         break

    #     next_id = int(np.argmax(logits))
    #     if next_id not in vocab:
    #         break

    #     clean = _clean(vocab[next_id]).strip()
    #     ids.append(next_id)
    #     current += clean

    # try:
    #     return float(current) if current else 0.0
    # except ValueError:
    #     return 0.0


def generate_string(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    user_prompt: str,
    param_index: int,
    total_string_params: int,
    max_token: int = 60,
) -> str:
    # Try direct extraction first.
    direct = _extract_string_for_param(
        user_prompt, param_index, total_string_params)
    if direct is not None:
        return direct

    # LLM fallback with constrained decoding.
    ids = model.encode(prompt)[0].tolist()
    current = ""

    for _ in range(max_token):
        logits = extract_logits(model.get_logits_from_input_ids(ids))

        for tid in range(len(logits)):
            if tid not in vocab:
                logits[tid] = -np.inf
                continue
            clean = _clean(vocab[tid])
            if "\n" in clean:
                logits[tid] = -np.inf
                continue
            if any(c in clean for c in ["{", "}", "[", "]"]):
                logits[tid] = -np.inf

        if np.all(np.isneginf(logits)):
            break

        next_id = int(np.argmax(logits))
        if next_id not in vocab:
            break

        clean = _clean(vocab[next_id])
        ids.append(next_id)

        if '"' in clean:
            current += clean.split('"')[0]
            return current.strip()

        current += clean

        if len(current) > 80:
            return current.strip()

    return current.strip()


def generate_args(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    user_prompt: str,
    func: "FunctionDefinition",
) -> dict[str, Any]:
    parameters: dict[str, Any] = {}

    string_params = [
        n for n, p in func.parameters.items()
        if p.type == "string"
    ]
    num_idx = 0
    str_idx = 0

    for param_name, param_def in func.parameters.items():

        if param_def.type == "number":
            prompt = (
                f"Request: \"{user_prompt}\"\n"
                f"Function: {func.name} - {func.description}\n"
                f"What is the numeric value of parameter '{param_name}'?\n"
                f"Write only the number: "
            )
            parameters[param_name] = generate_number(
                model, vocab, prompt, user_prompt, num_idx
            )
            num_idx += 1

        elif param_def.type == "string":
            prompt = (
                "Extract the parameter value from the user request.\n"
                "Return ONLY the value.\n"

                "Example:\n"
                "User request: What is the sum of 2 and 3?\n"
                "Parameter: a\n"
                "Value: 2\n"

                "Example:\n"
                "User request: What is the sum of 2 and 3?\n"
                "Parameter: b\n"
                "Value: 3\n"

                "Example:\n"
                "User request: Greet john\n"
                "Parameter: name\n"
                "Value: john\n"

                "Example:\n"
                "User request: Reverse the string 'hello'\n"
                "Parameter: s\n"
                "Value: hello\n"

                "Example:\n"
                "User request: Replace all numbers in "
                "'Hello 34 I'm 233 years old' with NUMBERS\n"
                "Parameter: source_string\n"
                "Value: Hello 34 I'm 233 years old\n\n"

                "Example:\n"
                "User request: Replace all numbers in "
                "'Hello 34 I'm 233 years old' with NUMBERS\n"
                "Parameter: regex\n"
                "Value: \\d+\n"

                "Example:\n"
                "User request: Replace all numbers in "
                "'Hello 34 I'm 233 years old' with NUMBERS\n"
                "Parameter: replacement\n"
                "Value: NUMBERS\n"

                "Example:\n"
                "User request: Replace all vowels in "
                "'Programming is fun' with asterisks\n"
                "Parameter: regex\n"
                "Value: [aeiouAEIOU]\n"

                "Example:\n"
                "User request: Replace all vowels in "
                "'Programming is fun' with asterisks\n"
                "Parameter: replacement\n"
                "Value: *\n"

                f"User request: {user_prompt}\n"
                f"Parameter: {param_name}\n"
                "Value:"
            )
            parameters[param_name] = generate_string(
                model, vocab, prompt, user_prompt,
                str_idx, len(string_params)
            )
            str_idx += 1

        elif param_def.type == "boolean":
            parameters[param_name] = False

    return parameters
