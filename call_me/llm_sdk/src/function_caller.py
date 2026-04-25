"""Function calling pipeline with constrained decoding."""

import sys
from typing import Any
from src.models import FunctionDefinition, FunctionCallResult
from src.generator import load_vocab, generate_name, generate_args
from llm_sdk import Small_LLM_Model


def process_prompt(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> FunctionCallResult | None:
    """Process a single prompt using constrained decoding."""
    try:
        # Build context with all function names
        valid_names = [fn.name for fn in functions]
        fn_descriptions = "\n".join(
            f"{fn.name}: {fn.description}" for fn in functions
        )

        # Create prompt with function context
        name_prompt = (
            f"Available functions:\n{fn_descriptions}\n"
            f"User request: {user_prompt}\n"
            f"Selected function: \""
        )

        fn_name = generate_name(model, vocab, name_prompt, valid_names)

        if not fn_name:
            print(f"[WARNING] Could not select function for: {user_prompt!r}",
                  file=sys.stderr)
            return None

        # Find function definition
        fn_def = next((fn for fn in functions if fn.name == fn_name), None)
        if not fn_def:
            print(f"[WARNING] Function not found: {fn_name}", file=sys.stderr)
            return None

        # Extract arguments
        raw_args = generate_args(model, vocab, user_prompt, fn_def)

        return FunctionCallResult(
            prompt=user_prompt,
            name=fn_name,
            parameters=raw_args,
        )

    except Exception as e:
        print(f"[ERROR] Process failed: {e}", file=sys.stderr)
        return None


def run_pipeline(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    prompts: list[Any],
) -> list[dict[str, Any]]:
    """Run the pipeline over all prompts."""
    vocab = load_vocab(model)
    results: list[dict[str, Any]] = []

    for i, test_prompt in enumerate(prompts):
        print(f"[INFO] {i + 1}/{len(prompts)}: {test_prompt.prompt!r}",
              file=sys.stderr)

        result = process_prompt(model, vocab, test_prompt.prompt, functions)

        if result:
            results.append(result.model_dump())
        else:
            results.append({
                "prompt": test_prompt.prompt,
                "name": "",
                "parameters": {},
            })

    return results
