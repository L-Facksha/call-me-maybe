"""Constrained decoding generator for function calling."""

import re
import numpy as np
from typing import Any
from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def encode_prompt(prompt: str, vocab: dict[int, str]) -> list[int]:
    """Encode a prompt string into a list of token IDs.

    Parameters
    ----------
    prompt : str
        The raw text to encode.
    vocab : dict[int, str]
        A mapping from token IDs to token strings used to match against the /
        prompt.

    Returns
    -------
    list[int]
        The sequence of token IDs representing the prompt.
    """
    prompt = prompt.replace(" ", "Ġ")

    i = 0
    ids = []
    while i < len(prompt):
        best_match = None
        for tid, token in vocab.items():
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


def decode_prompt(token_id: int, vocab: dict[int, str]) -> str:
    """Decode a single token ID back into a string.

    Parameters
    ----------
    token_id : int
        The token ID to decode.
    vocab : dict[int, str]
        A mapping from token IDs to token strings.

    Returns
    -------
    str
        The corresponding token string, or an empty string if not found.
    """

    if token_id not in vocab:
        return ""

    return vocab[token_id]


def _clean(token: str) -> str:
    """Replace special whitespace tokens with their readable equivalents.

    Parameters
    ----------
    token : str
        A raw token string possibly containing special characters.

    Returns
    -------
    str
        The cleaned token with whitespace characters substituted.
    """
    return token.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")


def generate_name(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    valid_names: list[str],
    max_token: int,
) -> str:
    """Select a valid function name using constrained decoding.

    At each step, only tokens that keep the accumulated string a valid
    prefix of at least one name in ``valid_names`` are allowed. Generation
    stops when a closing quote is produced and the accumulated string is a
    complete valid name.

    Parameters
    ----------
    model : Small_LLM_Model
        The language model used to produce logits.
    vocab : dict[int, str]
        A mapping from token IDs to token strings.
    prompt : str
        The encoded prompt used as the initial context.
    valid_names : list[str]
        The set of function names the output is constrained to.
    max_token : int
        Maximum number of tokens to generate.

    Returns
    -------
    str
        The selected function name, or an empty string if none matched.
    """

    current = ""

    ids = encode_prompt(prompt, vocab)

    for _ in range(max_token):
        logits = np.array(model.get_logits_from_input_ids(ids))

        for tid in range(len(logits)):

            if tid not in vocab:
                logits[tid] = -np.inf
                continue
            clean = _clean(decode_prompt(tid, vocab)).strip()
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

        clean = _clean(decode_prompt(next_id, vocab)).strip()
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
    param_def: str,
    param_index: int,
) -> float:
    """Extract a number using constrained decoding.

    The LLM reads the prompt and generates the number token by token.
    At each step, only tokens that keep the accumulated string a valid
    prefix of the target number are allowed.

    Parameters
    ----------
    model : Small_LLM_Model
        The language model used to produce logits.
    vocab : dict[int, str]
        A mapping from token IDs to token strings.
    prompt : str
        The encoded prompt used as the initial context.
    user_prompt : str
        The original natural language request used to extract candidate
        numbers.
    param_def : str
        The parameter type, either ``"integer"`` or ``"number"``.
    param_index : int
        The index of the target number among all candidates found in the
        prompt.

    Returns
    -------
    float
        The extracted numeric value, cast to ``int`` if ``param_def`` is
        ``"integer"``, or ``0.0`` on failure.
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

    ids = encode_prompt(prompt, vocab)

    current = ""

    for _ in range(max_token):
        logits = np.array(model.get_logits_from_input_ids(ids))

        for tid in range(len(logits)):

            if tid not in vocab:
                logits[tid] = -np.inf
                continue

            tok = _clean(decode_prompt(tid, vocab)).strip()

            if not tok:
                logits[tid] = -np.inf
                continue

            if not target.startswith(current + tok):
                logits[tid] = -np.inf

        if np.all(np.isneginf(logits)):
            break

        next_id = int(np.argmax(logits))

        if next_id not in vocab:
            break

        tok = _clean(decode_prompt(next_id, vocab)).strip()

        ids.append(next_id)
        current += tok

        if current == target:
            break

    try:
        if param_def == "integer":
            return int(float(current))
        return float(current)
    except (ValueError, TypeError):
        return 0.0


def generate_string(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    max_token: int = 15,
) -> str:
    """Extract a string parameter value using greedy decoding.

    Generation stops early on a newline, a sentinel word such as
    ``Parameter`` or ``User``, repeated whitespace, or when the
    accumulated output exceeds 80 characters.

    Parameters
    ----------
    model : Small_LLM_Model
        The language model used to produce logits.
    vocab : dict[int, str]
        A mapping from token IDs to token strings.
    prompt : str
        The encoded prompt used as the initial context.
    max_token : int, optional
        Maximum number of tokens to generate, by default 15.

    Returns
    -------
    str
        The extracted string value with surrounding quotes stripped.
    """

    ids = encode_prompt(prompt, vocab)

    current = ""

    for _ in range(max_token):
        logits = np.array(model.get_logits_from_input_ids(ids))

        for tid in range(len(logits)):
            if tid not in vocab:
                logits[tid] = -np.inf
                continue

            token = _clean(decode_prompt(tid, vocab))

            if not token:
                logits[tid] = -np.inf
                continue

        if np.all(np.isneginf(logits)):
            break

        next_id = int(np.argmax(logits))
        if next_id not in vocab:
            break

        token = _clean(decode_prompt(next_id, vocab))

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

    return current.strip().strip("'").strip('"')


def generate_args(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    user_prompt: str,
    func: "FunctionDefinition",
) -> dict[str, Any]:
    """Extract all function arguments from a natural language prompt.

    Iterates over each parameter in ``func`` and dispatches to the
    appropriate constrained generator based on the parameter type.

    Parameters
    ----------
    model : Small_LLM_Model
        The language model used to produce logits.
    vocab : dict[int, str]
        A mapping from token IDs to token strings.
    user_prompt : str
        The original natural language request from the user.
    func : FunctionDefinition
        The function definition describing expected parameters and types.

    Returns
    -------
    dict[str, Any]
        A mapping from parameter names to their extracted values.
    """
    parameters: dict[str, Any] = {}

    num_idx = 0

    for param_name, param_def in func.parameters.items():

        if param_def.type in ('number', 'integer'):
            prompt = (
                "Extract the numeric value from the user request.\n"
                "Return ONLY the number.\n"
                "No explanation. No extra text.\n\n"

                "Rules:\n"
                "- keep negative sign if present\n"
                "- keep decimal numbers\n"
                "- ignore all words\n"
                "- output must be a valid number\n\n"

                f"Request: {user_prompt}\n"
                f"Parameter: {param_name}\n"
                "Value:"
            )
            parameters[param_name] = generate_number(
                model, vocab, prompt, user_prompt, param_def.type, num_idx
            )
            num_idx += 1

        elif param_def.type == "string":
            prompt = (
                "Extract the parameter value from the user request.\n"
                "Return ONLY the value, nothing else.\n\n"
                "User request: Greet shrek\n"
                "Parameter: name\n"
                "Value: shrek\n\n"

                "User request: Reverse the string 'hello'\n"
                "Parameter: s\n"
                "Value: hello\n\n"

                "User request: What is the square root of 16?\n"
                "Parameter: s\n"
                "Value: 16.0\n\n"

                "User request: Replace all numbers in \
                    \"Hello 34 I'm 233 years old\" with NUMBERS\n"
                "Parameter: regex\n"
                "Value: \\d+\n\n"

                "User request: Replace all numbers in \
                    \"Hello 34 I'm 233 years old\" with NUMBERS\n"
                "Parameter: replacement\n"
                "Value: NUMBERS\n\n"

                "User request: Replace all numbers in \
                    \"Hello 34 I'm 233 years old\" with NUMBERS\n"
                "Parameter: source_string\n"
                "Value: Hello 34 I'm 233 years old\n\n"

                "User request: Replace all vowels in 'abc' with asterisks\n"
                "Parameter: regex\n"
                "Value: [aeiouAEIOU]\n\n"

                "User request: Replace all vowels in 'abc' with asterisks\n"
                "Parameter: replacement\n"
                "Value: *\n\n"

                "User request: Substitute the word 'cat' with 'dog' in \
                    'The cat sat on the mat with another cat' \n"
                "Parameter: regex\n"
                "Value: cat\n\n"

                "User request: Substitute the word 'cat' with 'dog' in \
                    'The cat sat on the mat with another cat' \n"
                "Parameter: replacement\n"
                "Value: dog\n\n"

                f"User request: {user_prompt}\n"
                f"Parameter: {param_name}\n"
                "Value:"
            )
            parameters[param_name] = generate_string(
                model, vocab, prompt
            )

    return parameters
