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


def generate_string(
    model: Small_LLM_Model,
    vocab: dict[str, int],
    prompt: str,
    max_token: int = 60,
) -> str:
    """Generate a string value using constrained decoding.

    The few-shot prompt format uses newlines as value terminators:
        Value: john\n
        Value: hello\n
    So we stop when the model generates a newline token — that is the
    natural end of the value. Quotes are NOT stop signals because they
    are valid content characters (e.g. "I'm", "Hello 34 I'm 233...",
    'cat', 'dog'). Blocking quotes would truncate these values.

    Parameters
    ----------
    model : Small_LLM_Model
        The loaded LLM model instance.
    vocab : dict[str, int]
        Token string to ID mapping.
    prompt : str
        Extraction prompt ending just before the value.
    max_token : int
        Hard cap on decoding steps.

    Returns
    -------
    str
        The extracted string value.
    """
    ids = encode_prompt(prompt, vocab)
    id_to_token = {tid: token for token, tid in vocab.items()}
    current = ""

    for _ in range(max_token):
        logits = extract_logits(model.get_logits_from_input_ids(ids))

        for tid in range(len(logits)):
            if tid not in id_to_token:
                logits[tid] = -np.inf
                continue

            token = _clean(id_to_token[tid])

            # Newline = end of value in the few-shot prompt format
            if "\n" in token:
                logits[tid] = -np.inf
                continue

            # Block structural JSON chars
            if any(c in token for c in ["{", "}", "[", "]"]):
                logits[tid] = -np.inf
                continue

            # Block digit-only tokens after non-numeric content
            # (prevents "shrek24", "name16" suffixes)
            if (current
                    and token.strip().isdigit()
                    and not current.strip().lstrip("-").replace(".", "").isdigit()):
                logits[tid] = -np.inf

        if np.all(np.isneginf(logits)):
            break

        next_id = int(np.argmax(logits))
        if next_id not in id_to_token:
            break

        token = _clean(id_to_token[next_id])

        # Stop when the model generates a newline — value is complete
        if "\n" in token:
            # Grab anything before the newline on the same token
            current += token.split("\n")[0]
            break

        ids.append(next_id)
        current += token

        # Stop if model starts writing the next prompt line
        for word in ("Parameter", "User", "Request"):
            if word in current:
                current = current.split(word)[0]
                return current.strip()

        # Single special char is a complete value (e.g. "*")
        if current.strip() in ("*", "+", "?", "!", "#", "@", "^", "~"):
            break

        # Stop on double-space (model is padding/rambling)
        if "  " in current:
            current = current.split("  ")[0]
            break

        if len(current) > 80:
            break

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
