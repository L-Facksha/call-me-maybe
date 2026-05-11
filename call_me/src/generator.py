"""Constrained decoding generator for function calling."""

from pathlib import Path
import json
import re
import numpy as np
from typing import Any, TYPE_CHECKING
from llm_sdk.llm_sdk import Small_LLM_Model

if TYPE_CHECKING:
    from src.models import FunctionDefinition


def load_vocab(model: Small_LLM_Model) -> dict[str, int]:
    vocab_path = Path(model.get_path_to_vocab_file())
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    return vocab


def encode_prompt(prompt: str, vocab: dict[str, int]) -> list[int]:
    prompt = prompt.replace(" ", "Ġ")

    sorted_vocab = sorted(vocab.items(), key=lambda x: len(x[0]), reverse=True)
    i = 0
    ids = []
    while i < len(prompt):
        best_match = None
        for token, tid in sorted_vocab:
            if prompt.startswith(token, i):
                best_match = (tid, token)
                break
        if best_match is None:
            i += 1
            continue

        tid, token = best_match
        ids.append(tid)
        i += len(token)

    return ids


def decode_prompt(ids: list[int], vocab: dict[str, int]):
    reverce_vocab = {tid: token for token, tid in vocab.items()}

    text = ""
    for tid in ids:
        if tid not in reverce_vocab:
            continue
        text += reverce_vocab[tid]
    return text


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
    return token.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")


def generate_name(
    model: Small_LLM_Model,
    vocab: dict[str, int],
    prompt: str,
    valid_names: list[str],
    max_token: int = 15,
) -> str:
    current = ""

    ids = encode_prompt(prompt, vocab)

    id_to_token = {
        tid: token
        for token, tid in vocab.items()
    }

    for _ in range(max_token):
        logits = extract_logits(model.get_logits_from_input_ids(ids))

        for tid in range(len(logits)):

            if tid not in id_to_token:
                logits[tid] = -np.inf
                continue
            clean = _clean(id_to_token[tid]).strip()
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
        if next_id not in id_to_token:
            break

        clean = _clean(id_to_token[next_id]).strip()
        ids.append(next_id)

        if clean == '"':
            return current if current in valid_names else ""
        current += clean

    return current if current in valid_names else ""


def generate_number(
    model: Small_LLM_Model,
    vocab: dict[str, int],
    prompt: str,
    user_prompt: str,
    param_index: int,
) -> float:
    """Extract a number using constrained decoding.

    The LLM reads the prompt and generates the number token by token.
    At each step, only tokens that keep the accumulated string a valid
    prefix of the target number are allowed.
    """

    candidates = [
        m.group()
        for m in re.finditer(r"-?\d+(?:\.\d+)?", user_prompt)
    ]

    if not candidates:
        return 0.0

    target = (
        candidates[param_index]
        if param_index < len(candidates)
        else candidates[0]
    )
    max_token = len(target)
    print(max_token)
    ids = encode_prompt(prompt, vocab)
    id_to_token = {
        tid: token
        for token, tid in vocab.items()
    }

    current = ""

    for _ in range(max_token):
        logits = extract_logits(model.get_logits_from_input_ids(ids))

        for tid in range(len(logits)):

            if tid not in id_to_token:
                logits[tid] = -np.inf
                continue

            tok = _clean(id_to_token[tid]).strip()

            if not tok:
                logits[tid] = -np.inf
                continue

            if not target.startswith(current + tok):
                logits[tid] = -np.inf

        if np.all(np.isneginf(logits)):
            break

        next_id = int(np.argmax(logits))

        if next_id not in id_to_token:
            break

        tok = _clean(id_to_token[next_id]).strip()

        ids.append(next_id)
        current += tok

        if current == target:
            break

    try:
        return float(current)
    except (ValueError, TypeError):
        return 0.0


def _extract_string_for_param(
    user_prompt: str,
    param_index: int,
    total_string_params: int,
) -> str | None:
    """Extract the correct string value for a parameter from the user prompt.

    - "... X ... in 'SOURCE'" pattern: string after "in" is the source (first param).
    - Normal positional: first quote → first param, second → second, etc.
    - No quotes + single param: take the last word.
    - Returns None when the param cannot be extracted (LLM fallback used).
    """
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', user_prompt)
    candidates = [a if a else b for a, b in quoted]

    if not candidates:
        if total_string_params == 1 and param_index == 0:
            words = user_prompt.strip().split()
            return words[-1] if words else None
        return None

    if len(candidates) == total_string_params:
        in_match = re.search(
            r"\bin\s+['\"]([^'\"]+)['\"]", user_prompt, re.IGNORECASE)
        if in_match and total_string_params >= 2:
            source_val = in_match.group(1)
            rest = [c for c in candidates if c != source_val]
            ordered = [source_val] + rest
            return ordered[param_index] if param_index < len(ordered) else None
        return candidates[param_index] if param_index < len(candidates) else None

    return candidates[param_index] if param_index < len(candidates) else None


def generate_string(
    model: Small_LLM_Model,
    vocab: dict[str, int],
    prompt: str,
    max_token: int = 30,
) -> str:

    ids = encode_prompt(prompt, vocab)

    id_to_token = {
        tid: token
        for token, tid in vocab.items()
    }

    current = ""

    for _ in range(max_token):

        logits = extract_logits(
            model.get_logits_from_input_ids(ids)
        )

        for tid in range(len(logits)):

            if tid not in id_to_token:
                logits[tid] = -np.inf
                continue

            token = _clean(id_to_token[tid])

            # stop hallucinations
            if "\n" in token:
                logits[tid] = -np.inf
                continue

            if any(c in token for c in ["{", "}", "[", "]"]):
                logits[tid] = -np.inf
                continue

        if np.all(np.isneginf(logits)):
            break

        next_id = int(np.argmax(logits))

        if next_id not in id_to_token:
            break

        token = _clean(id_to_token[next_id])

        ids.append(next_id)

        # stop on quote
        if '"' in token or "'" in token:
            break

        # stop if model starts explaining
        if "User" in token or "Parameter" in token:
            break

        current += token

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
                "Extract the numeric value from the user request.\n"
                "Return ONLY the number.\n"
                "No explanation. No extra text.\n\n"

                "Rules:\n"
                "- keep negative sign if present\n"
                "- keep decimal numbers\n"
                "- ignore all words\n"
                "- output must be a valid number\n\n"

                "Examples:\n\n"

                "Request: What is 2 + 3?\n"
                "Value: 2\n\n"

                "Request: Add 10 and -7\n"
                "Value: 10\n\n"

                "Request: Add 10 and -7\n"
                "Value: -7\n\n"

                "Request: 5.5 multiplied by 2\n"
                "Value: 5.5\n\n"

                "Request: square root of 144\n"
                "Value: 144\n\n"

                f"Request: {user_prompt}\n"
                f"Parameter: {param_name}\n"
                "Value:"
            )
            parameters[param_name] = generate_number(
                model, vocab, prompt, user_prompt, num_idx
            )
            num_idx += 1

        elif param_def.type == "string":
            prompt = (
                "Extract the parameter value from the user request.\n"
                "Return ONLY the value, nothing else.\n\n"
                "User request: Greet john\n"
                "Parameter: name\n"
                "Value: john\n\n"
                "User request: Reverse the string 'hello'\n"
                "Parameter: s\n"
                "Value: hello\n\n"
                "User request: Replace all numbers in 'Hello 34' with NUMBERS\n"
                "Parameter: regex\n"
                "Value: \\d+\n\n"
                "User request: Replace all numbers in 'Hello 34' with NUMBERS\n"
                "Parameter: replacement\n"
                "Value: NUMBERS\n\n"
                "User request: Replace all vowels in 'abc' with asterisks\n"
                "Parameter: regex\n"
                "Value: /aeiou/\n\n"
                "User request: Replace all vowels in 'abc' with asterisks\n"
                "Parameter: replacement\n"
                "Value: *\n\n"
                f"User request: {user_prompt}\n"
                f"Parameter: {param_name}\n"
                "Value:"
            )
            parameters[param_name] = generate_string(
                model, vocab, prompt
            )
            str_idx += 1

        elif param_def.type == "boolean":
            parameters[param_name] = False

    return parameters
