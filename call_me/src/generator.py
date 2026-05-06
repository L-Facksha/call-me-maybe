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
    """Load the vocabulary mapping from token IDs to token strings.

    Parameters
    ----------
    model : Small_LLM_Model
        The loaded LLM model instance.

    Returns
    -------
    dict[int, str]
        Mapping of token ID (int) to token string.
    """
    vocab_path = Path(model.get_path_to_vocab_file())
    with vocab_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    vocab: dict[int, str] = {}
    for k, v in raw.items():
        try:
            vocab[int(k)] = str(v)
        except (ValueError, KeyError):
            try:
                vocab[int(v)] = str(k)
            except (ValueError, KeyError):
                continue
    return vocab


def extract_logits(logits: Any) -> np.ndarray:
    """Extract the last-token logits vector from model output.

    Parameters
    ----------
    logits : Any
        Raw logits returned by the model (tensor or array).

    Returns
    -------
    np.ndarray
        1-D array of logit scores for each vocab token.
    """
    if hasattr(logits, "numpy"):
        logits = logits.numpy()
    else:
        logits = np.array(logits)
    if len(logits.shape) == 3:
        return logits[0, -1]
    if len(logits.shape) == 2:
        return logits[-1]
    return logits


def _clean(token: str) -> str:
    """Normalize a raw vocab token to plain text.

    Parameters
    ----------
    token : str
        Raw token string from the vocabulary.

    Returns
    -------
    str
        Human-readable token string.
    """
    return token.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")


def _encode_prompt(model: Small_LLM_Model, prompt: str) -> list[int]:
    """Encode a prompt string into a flat list of token IDs.

    Parameters
    ----------
    model : Small_LLM_Model
        The loaded LLM model instance.
    prompt : str
        The text prompt to encode.

    Returns
    -------
    list[int]
        Flat list of integer token IDs.
    """
    encoded = model.encode(prompt)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(t) for t in encoded]


# =========================
# NUMBER EXTRACTION
# =========================

def _extract_numbers_from_text(text: str) -> list[float]:
    """Extract all numbers from a natural language string.

    Parameters
    ----------
    text : str
        The user prompt text.

    Returns
    -------
    list[float]
        All numeric values found in the text, in order of appearance.
    """
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return [float(m) for m in matches]


def _pick_number_by_position(
    user_prompt: str,
    param_name: str,
    param_index: int,
    all_params: list[str],
) -> float | None:
    """Try to pick the right number from the user prompt by param position.

    For functions with multiple numeric params (a, b), param index 0 gets
    the first number found in the prompt, index 1 gets the second, etc.

    Parameters
    ----------
    user_prompt : str
        The original natural language request.
    param_name : str
        The name of the parameter being extracted.
    param_index : int
        The 0-based position of this param among all numeric params.
    all_params : list[str]
        Names of all numeric parameters for this function, in order.

    Returns
    -------
    float | None
        The picked number, or None if not enough numbers found.
    """
    numbers = _extract_numbers_from_text(user_prompt)
    if not numbers:
        return None
    # If there is only one number but multiple params, reuse it.
    if len(numbers) == 1:
        return numbers[0]
    # Otherwise map by position.
    if param_index < len(numbers):
        return numbers[param_index]
    return numbers[-1]


# =========================
# CORE GENERATORS
# =========================


def generate_number(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    user_prompt: str,
    param_name: str,
    param_index: int,
    all_numeric_params: list[str],
    max_token: int = 15,
) -> float:
    """Extract a numeric argument, preferring direct regex over the model.

    Strategy:
    1. Try to extract numbers directly from ``user_prompt`` with regex and
       pick the one at ``param_index``.  This is 100% reliable for simple
       prompts like "sum of 2 and 3".
    2. Only fall back to constrained LLM decoding when regex yields no
       usable result (e.g. the number is written out in words).

    Parameters
    ----------
    model : Small_LLM_Model
        The loaded LLM model instance.
    vocab : dict[int, str]
        Token ID to string mapping.
    prompt : str
        Extraction prompt ending just before the number.
    user_prompt : str
        The original natural language request (used for regex extraction).
    param_name : str
        Name of the parameter being extracted.
    param_index : int
        0-based index of this param among all numeric params.
    all_numeric_params : list[str]
        All numeric parameter names for this function, in order.
    max_token : int
        Maximum LLM tokens to generate (fallback only).

    Returns
    -------
    float
        The extracted number, or 0.0 if nothing found.
    """
    # --- Strategy 1: direct regex extraction (fast & exact) ---
    result = _pick_number_by_position(
        user_prompt, param_name, param_index, all_numeric_params
    )
    if result is not None:
        return result

    # --- Strategy 2: constrained LLM decoding (fallback) ---
    ids = _encode_prompt(model, prompt)
    current = ""
    dot_seen = False

    for _ in range(max_token):
        logits = extract_logits(model.get_logits_from_input_ids(ids))

        valid_ids: list[int] = []
        for tid, tok_raw in vocab.items():
            if tid >= len(logits):
                continue
            tok = _clean(tok_raw).strip()
            if not tok:
                continue
            # Decimal point: allowed only once, only after digits.
            if tok == "." and not dot_seen and re.fullmatch(r"\d+", current.strip()):
                valid_ids.append(tid)
            # Pure digit token.
            elif re.fullmatch(r"\d+", tok):
                valid_ids.append(tid)

        if not valid_ids:
            break

        next_id = max(valid_ids, key=lambda i: logits[i])
        tok_str = _clean(vocab[next_id]).strip()
        if not tok_str:
            break

        if tok_str == ".":
            dot_seen = True

        current += tok_str
        ids.append(next_id)

        # Early-stop once we have a valid complete number.
        if re.fullmatch(r"\d+(\.\d+)?", current.strip()):
            next_logits = extract_logits(model.get_logits_from_input_ids(ids))
            best_next = int(np.argmax(next_logits))
            best_clean = _clean(vocab.get(best_next, "")).strip()
            if not re.fullmatch(r"\d+", best_clean) and best_clean != ".":
                break

    match = re.search(r"\d*\.?\d+", current)
    return float(match.group()) if match else 0.0


def generate_string(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    max_token: int = 60,
) -> str:
    """Generate a string value using constrained decoding.

    The prompt must end with an opening double-quote. Generation stops
    as soon as the model emits a closing double-quote token.

    Parameters
    ----------
    model : Small_LLM_Model
        The loaded LLM model instance.
    vocab : dict[int, str]
        Token ID to string mapping.
    prompt : str
        Extraction prompt ending with an opening double-quote.
    max_token : int
        Maximum number of tokens to generate.

    Returns
    -------
    str
        The extracted string value.
    """
    ids = _encode_prompt(model, prompt)
    current = ""

    # Pre-compute closing-quote token IDs.
    closing_quote_ids: set[int] = set()
    for tid, tok_raw in vocab.items():
        if '"' in _clean(tok_raw):
            closing_quote_ids.add(tid)

    # Patterns that signal prompt leakage rather than the real value.
    _NOISE = re.compile(
        r"(task:|rules:|example|user:|output:|parameter:|function:|"
        r"description:|if the|if missing|\(if)",
        re.IGNORECASE,
    )

    for _ in range(max_token):
        logits = extract_logits(model.get_logits_from_input_ids(ids))
        mask = np.full_like(logits, -np.inf)

        for tid, tok_raw in vocab.items():
            if tid >= len(logits):
                continue
            clean = _clean(tok_raw)

            if tid in closing_quote_ids:
                mask[tid] = logits[tid]
                continue

            if "\n" in clean or _NOISE.search(clean):
                continue

            mask[tid] = logits[tid]

        if np.all(np.isneginf(mask)):
            break

        next_id = int(np.argmax(mask))

        if next_id in closing_quote_ids:
            current += _clean(vocab[next_id]).split('"')[0]
            break

        tok_str = _clean(vocab[next_id])
        current += tok_str
        ids.append(next_id)

    return current.strip().strip("'").strip('"')


def generate_name(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    valid_names: list[str],
    max_token: int = 20,
) -> str:
    """Generate a function name using prefix-constrained decoding.

    At every step only tokens that extend the current prefix toward at
    least one valid function name are allowed.

    Parameters
    ----------
    model : Small_LLM_Model
        The loaded LLM model instance.
    vocab : dict[int, str]
        Token ID to string mapping.
    prompt : str
        Selection prompt ending just before the function name.
    valid_names : list[str]
        All recognised function names.
    max_token : int
        Maximum number of tokens to generate.

    Returns
    -------
    str
        The selected function name, or an empty string if none matched.
    """
    ids = _encode_prompt(model, prompt)
    current = ""

    for _ in range(max_token):
        logits = extract_logits(model.get_logits_from_input_ids(ids))
        mask = np.full_like(logits, -np.inf)

        for tid, tok_raw in vocab.items():
            if tid >= len(logits):
                continue
            clean = _clean(tok_raw)
            potential = (current + clean).strip().strip('"')
            if any(n.startswith(potential) for n in valid_names) or potential == "":
                mask[tid] = logits[tid]

        if np.all(np.isneginf(mask)):
            break

        next_id = int(np.argmax(mask))
        tok_str = _clean(vocab[next_id])
        current += tok_str
        ids.append(next_id)

        if current.strip().strip('"') in valid_names:
            break

    return current.strip().strip('"')


# =========================
# MAIN ARGUMENT HANDLER
# =========================


def generate_args(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    user_prompt: str,
    func: "FunctionDefinition",
) -> dict[str, Any]:
    """Extract all arguments for a function call from a user prompt.

    Parameters
    ----------
    model : Small_LLM_Model
        The loaded LLM model instance.
    vocab : dict[int, str]
        Token ID to string mapping.
    user_prompt : str
        The original natural language request.
    func : FunctionDefinition
        The function whose parameters must be extracted.

    Returns
    -------
    dict[str, Any]
        Mapping of parameter name to extracted value.
    """
    from src.prompts import build_extraction_prompt
    parameters: dict[str, Any] = {}

    # Pre-compute the ordered list of numeric params so generate_number
    # can pick the correct number by position (first param → first number).
    numeric_params: list[str] = [
        n for n, p in func.parameters.items() if p.type == "number"
    ]

    numeric_index = 0
    for name, p in func.parameters.items():
        prompt = build_extraction_prompt(
            user_prompt=user_prompt,
            func=func,
            param_name=name,
            param_type=p.type,
        )

        if not prompt:
            parameters[name] = 0.0 if p.type == "number" else ""
            if p.type == "number":
                numeric_index += 1
            continue

        if p.type == "number":
            parameters[name] = generate_number(
                model=model,
                vocab=vocab,
                prompt=prompt + " ",
                user_prompt=user_prompt,
                param_name=name,
                param_index=numeric_index,
                all_numeric_params=numeric_params,
            )
            numeric_index += 1
        else:
            parameters[name] = generate_string(
                model, vocab, prompt + ' "')

    return parameters
