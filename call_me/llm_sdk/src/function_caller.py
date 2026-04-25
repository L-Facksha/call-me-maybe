"""Two-stage function calling pipeline."""

import sys
from typing import Any
from src.models import FunctionDefinition, FunctionCallResult
from src.generator import load_vocab, generate_name, generate_args
from llm_sdk import Small_LLM_Model


def build_name_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Build prompt for function name selection.

    Parameters
    ----------
    user_prompt : str
        The natural language request.
    functions : list[FunctionDefinition]
        Available function definitions.

    Returns
    -------
    str
        Full prompt string ending with opening quote.
    """
    fn_list = "\n".join(
        f"- {fn.name}: {fn.description}" for fn in functions
    )
    return (
        f"You are a function selector.\n"
        f"Available functions:\n{fn_list}\n\n"
        f"User request: \"{user_prompt}\"\n"
        f"Reply with the function name. JSON: {{\"name\": \""
    )


def build_args_prompt(
    user_prompt: str,
    fn: FunctionDefinition,
) -> str:
    """Build prompt for argument extraction.

    Parameters
    ----------
    user_prompt : str
        The natural language request.
    fn : FunctionDefinition
        The selected function definition.

    Returns
    -------
    str
        Full prompt string ending with opening brace.
    """
    params = ", ".join(
        f"{name}: {pdef.type}" for name, pdef in fn.parameters.items()
    )
    return (
        f"Extract arguments for {fn.name}({params}).\n"
        f"Description: {fn.description}\n"
        f"User request: \"{user_prompt}\"\n"
        f"Reply with only JSON arguments. JSON: {{"
    )


def coerce_args(
    raw: dict[str, Any],
    fn: FunctionDefinition,
) -> dict[str, Any]:
    """Coerce extracted arguments to their declared types.

    Parameters
    ----------
    raw : dict[str, Any]
        Raw parsed argument dict.
    fn : FunctionDefinition
        Function definition with declared parameter types.

    Returns
    -------
    dict[str, Any]
        Coerced argument dict.
    """
    result: dict[str, Any] = {}
    for name, pdef in fn.parameters.items():
        val = raw.get(name)
        if val is None:
            result[name] = val
            continue
        try:
            if pdef.type == "number":
                result[name] = float(val)
            elif pdef.type == "integer":
                result[name] = int(val)
            elif pdef.type == "string":
                result[name] = str(val)
            elif pdef.type == "boolean":
                if isinstance(val, bool):
                    result[name] = val
                else:
                    result[name] = str(val).lower() in ("true", "1", "yes")
            else:
                result[name] = val
        except (ValueError, TypeError) as e:
            print(f"[WARNING] Could not coerce {name}={val!r}: {e}",
                  file=sys.stderr)
            result[name] = val
    return result


def process_prompt(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> FunctionCallResult | None:
    """Run the two-stage pipeline for a single prompt.

    Parameters
    ----------
    model : Small_LLM_Model
        The loaded LLM model instance.
    vocab : dict[int, str]
        Mapping from token ID to token string.
    user_prompt : str
        The natural language request.
    functions : list[FunctionDefinition]
        Available function definitions.

    Returns
    -------
    FunctionCallResult or None
        The result or None if pipeline fails.
    """
    valid_names = [fn.name for fn in functions]

    print(f"[INFO] Selecting function for: {user_prompt!r}", file=sys.stderr)
    name_prompt = build_name_prompt(user_prompt, functions)
    fn_name = generate_name(model, vocab, name_prompt, valid_names)

    if not fn_name:
        print(f"[ERROR] Failed to select function for: {user_prompt!r}",
              file=sys.stderr)
        return None

    print(f"[INFO] Selected: {fn_name}", file=sys.stderr)

    fn_def = next((fn for fn in functions if fn.name == fn_name), None)
    if fn_def is None:
        print(f"[ERROR] Function {fn_name!r} not found.", file=sys.stderr)
        return None

    if not fn_def.parameters:
        return FunctionCallResult(
            prompt=user_prompt,
            name=fn_name,
            parameters={},
        )

    print(f"[INFO] Extracting arguments for {fn_name}...", file=sys.stderr)

    raw_args = generate_args(model, vocab, user_prompt, fn_def)

    if not raw_args:
        print(f"[ERROR] Failed to extract arguments for: {user_prompt!r}",
              file=sys.stderr)
        return None

    return FunctionCallResult(
        prompt=user_prompt,
        name=fn_name,
        parameters=raw_args,
    )


def run_pipeline(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    prompts: list[Any],
) -> list[dict[str, Any]]:
    """Run the full pipeline over all prompts.

    Parameters
    ----------
    model : Small_LLM_Model
        The loaded LLM model instance.
    functions : list[FunctionDefinition]
        Available function definitions.
    prompts : list
        List of TestPrompt objects.

    Returns
    -------
    list[dict[str, Any]]
        List of result dicts for JSON serialization.
    """
    vocab = load_vocab(model)
    results: list[dict[str, Any]] = []

    for i, test_prompt in enumerate(prompts):
        print(f"\n[INFO] Prompt {i + 1}/{len(prompts)}: {test_prompt.prompt!r}",
              file=sys.stderr)
        try:
            result = process_prompt(
                model, vocab, test_prompt.prompt, functions)
            if result is not None:
                results.append(result.model_dump())
            else:
                results.append({
                    "prompt": test_prompt.prompt,
                    "name": "unknown",
                    "parameters": {},
                })
        except Exception as e:
            print(f"[ERROR] Prompt {i + 1} failed: {e}", file=sys.stderr)
            results.append({
                "prompt": test_prompt.prompt,
                "name": "unknown",
                "parameters": {},
            })

    return results
