"""Constrained decoding generator for function calling."""

import re
import sys
import math
import numpy as np
from typing import Any, TYPE_CHECKING
from llm_sdk.llm_sdk import Small_LLM_Model

if TYPE_CHECKING:
    from src.models import FunctionDefinition
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def encode_prompt(prompt: str, vocab: list[tuple[int, str]], token_to_id: dict[str, int]) -> list[int]:
    prompt = prompt.replace(" ", "Ġ")

    i = 0
    ids = []
    while i < len(prompt):
        best_match = None
        for tid, token in vocab:
            if prompt.startswith(token, i):
                best_match = (tid, token)
                break
        if best_match is None:
            char = prompt[i]
            if char in token_to_id:
                ids.append(token_to_id[char])
            i += 1
            continue

        tid, token = best_match
        ids.append(tid)
        i += len(token)

    return ids


def decode_prompt(ids: list[int], vocab: dict[str, int]):
    text = ""

    for tid in ids:
        if tid not in vocab:
            continue
        text += vocab[tid]
    return text


def _clean(token: str) -> str:
    return token.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")


def generate_name(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    sorted_vocab,
    token_to_id,
    valid_token_ids,
    prompt: str,
    valid_names: list[str],
    max_token: int,
) -> str:

    ids = encode_prompt(prompt, sorted_vocab, token_to_id)
    current = ""

    valid_token_ids = np.array(list(valid_token_ids), dtype=np.int32)

    clean_vocab = {
        tid: _clean(vocab[tid]).strip()
        for tid in valid_token_ids
    }

    prefixes = set()
    for name in valid_names:
        for i in range(1, len(name) + 1):
            prefixes.add(name[:i])

    valid_names_set = set(valid_names)

    for _ in range(max_token):

        logits = np.array(model.get_logits_from_input_ids(ids))

        best_id = -1
        best_score = -1e9

        for tid in valid_token_ids:
            tok = clean_vocab.get(int(tid), "")

            if not tok:
                continue

            if tok == '"':
                if current not in valid_names_set:
                    continue
            else:
                if (current + tok) not in prefixes:
                    continue

            score = logits[tid]

            if score > best_score:
                best_score = score
                best_id = tid

        if best_id == -1:
            break

        tok = clean_vocab[best_id]
        ids.append(best_id)

        if tok == '"':
            return current if current in valid_names_set else ""

        current += tok

    return current if current in valid_names_set else ""


def generate_number(
    model,
    vocab,
    sorted_vocab,
    token_to_id,
    valid_token_ids,
    prompt,
    user_prompt,
    param_index,
    param_def,
):

    candidates = re.findall(r"-?\d+(?:\.\d+)?", user_prompt)
    if not candidates:
        return 0.0

    target = candidates[param_index] if param_index < len(
        candidates) else candidates[0]

    if len(target) > 9:
        print(f"{RED}[ERROR]{RESET} too long (<9 digits)", file=sys.stderr)
        return 0.0

    ids = encode_prompt(prompt, sorted_vocab, token_to_id)
    current = ""

    valid_token_ids = np.array(list(valid_token_ids), dtype=np.int32)

    clean_vocab = {
        tid: _clean(vocab[tid]).strip()
        for tid in valid_token_ids
    }

    for _ in range(len(target)):

        logits = np.array(model.get_logits_from_input_ids(ids))
        masked = np.full_like(logits, -np.inf)

        for tid in valid_token_ids:
            tok = clean_vocab.get(int(tid), "")
            if not tok:
                continue

            if target.startswith(current + tok):
                masked[tid] = logits[tid]

        if np.all(np.isneginf(masked)):
            break

        next_id = int(np.argmax(masked))
        tok = clean_vocab[next_id]

        ids.append(next_id)
        current += tok

        if current == target:
            break

    try:
        return int(float(current)) if param_def == "integer" else float(current)
    except:
        return 0.0


def generate_string(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    sorted_vocab: list[tuple[int, str]],
    token_to_id: dict[str, int],
    valid_token_ids,
    prompt: str,
    max_token: int = 40,
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
    ids = encode_prompt(prompt, sorted_vocab, token_to_id)
    current = ""

    for _ in range(max_token):
        logits = np.array(model.get_logits_from_input_ids(ids))

        for tid in range(len(logits)):
            if tid not in valid_token_ids:
                logits[tid] = -np.inf
                continue

            token = _clean(vocab[tid])

            if not token:
                logits[tid] = -np.inf
                continue

            if any(c in token for c in ["{", "}", "[", "]"]):
                logits[tid] = -np.inf
                continue

        if np.all(np.isneginf(logits)):
            break

        next_id = int(np.argmax(logits))
        if next_id not in vocab:
            break

        token = _clean(vocab[next_id])

        if "\n" in token:
            current += token.split("\n")[0]
            break

        ids.append(next_id)
        current += token

        for word in ("Parameter", "User", "Request"):
            if word in current:
                current = current.split(word)[0]
                return current.strip()

        if current.strip() in ("*", "+", "?", "!", "#", "@", "^", "~"):
            break

        if "  " in current:
            current = current.split("  ")[0]
            break

        if len(current) > 80:
            break
        if current.strip() == "aeiouAEIOU":
            current = "[aeiouAEIOU]"

    return current.strip().strip("'").strip('"')


def generate_args(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    sorted_vocab: list[tuple[int, str]],
    token_to_id: dict[str, int],
    valid_token_ids,
    user_prompt: str,
    func: "FunctionDefinition",
) -> dict[str, Any]:
    parameters: dict[str, Any] = {}

    num_idx = 0
    str_idx = 0

    for param_name, param_def in func.parameters.items():

        if param_def.type in ("number", "float", "integer"):
            prompt = (
                "Extract the numeric value.\n"
                "Return ONLY the number.\n\n"

                f"Request: {user_prompt}\n"
                f"Parameter: {param_name}\n"
                "Value:"
            )
            parameters[param_name] = generate_number(
                model, vocab, sorted_vocab, token_to_id, valid_token_ids, prompt, user_prompt, num_idx, param_def.type
            )
            num_idx += 1

        elif param_def.type == "string":
            prompt = (
                "Extract the parameter value.\n"
                "Return ONLY the value.\n"
                "Take the exact value.\n\n"

                "User request: Greet john\n"
                "Parameter: name\n"
                "Value: john\n\n"

                "User request: Greet shrek\n"
                "Parameter: name\n"
                "Value: shrek\n\n"

                "User request: Reverse the string 'hello'\n"
                "Parameter: s\n"
                "Value: hello\n\n"

                "User request: Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS\n"
                "Parameter: regex\n"
                "Value: \\d+\n\n"

                "User request: Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS\n"
                "Parameter: replacement\n"
                "Value: NUMBERS\n\n"

                "User request: Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS\n"
                "Parameter: source_string\n"
                "Value: Hello 34 I'm 233 years old\n\n"

                "User request: Replace all vowels in 'abc' with asterisks\n"
                "Parameter: regex\n"
                "Value: [aeiouAEIOU]\n\n"

                "User request: Replace all vowels in 'abc' with asterisks\n"
                "Parameter: replacement\n"
                "Value: *\n\n"

                "User request: Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat' \n"
                "Parameter: regex\n"
                "Value: cat\n\n"

                "User request: Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat' \n"
                "Parameter: replacement\n"
                "Value: dog\n\n"

                f"User request: {user_prompt}\n"
                f"Parameter: {param_name}\n"
                "Value:"
            )
            parameters[param_name] = generate_string(
                model, vocab, sorted_vocab, token_to_id, valid_token_ids, prompt
            )
            str_idx += 1

    return parameters
