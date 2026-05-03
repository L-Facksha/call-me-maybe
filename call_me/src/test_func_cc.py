import sys
from typing import Any
from src.models import FunctionDefinition, FunctionCallResult
from src.generator import load_vocab, generate_args, generate_name
from llm_sdk import Small_LLM_Model


def process_prompt(model: Small_LLM_Model, vocab: dict[int, str], user_prompt: str, functions: list[FunctionDefinition]) -> FunctionCallResult | None:
    try:
        valid_names = [fn.name for fn in functions]
        fn_descriptions = "\n".join(
            f"{fn.name}: {fn.description}" for fn in functions
        )

        name_prompt = (
            f"Available functions:\n{fn_descriptions}\n"
            f"User request: {user_prompt}\n"
            f"Selected function: \""
        )

        fn_name = generate_name(model, vocab, name_prompt, valid_names)
        if not fn_name or fn_name not in valid_names:
            print(
                f"[WARNING] Cloud not select function for: {user_prompt!r}", file=sys.stderr)
            return None

        fn_def = next((fn for fn in functions if fn.name == fn_name), None)
        if not fn_def:
            print(f"[WARNING] Function not found: {fn_name}", file=sys.stderr)
            return None

        raw_args = generate_args(user_prompt, fn_def)
        if raw_args is None:
            return None

        return FunctionCallResult(
            prompt=user_prompt,
            name=fn_name,
            parameters=raw_args
        )
    except Exception as error:
        print(f"[ERROR] Process failed: {error}", file=sys.stderr)
        return None


def run_pipeline(model: Small_LLM_Model, functions: list[FunctionDefinition], prompts: list[Any]) -> list[dict[str, Any]]:
    vocab = load_vocab(model)
    results: list[dict[str, Any]] = []

    for i, test_prompt in enumerate(prompts):
        print(
            f"[INFO] {i+1}/{len(prompts)}: {test_prompt.prompt!r}", file=sys.stderr)

        result = process_prompt(model, vocab, test_prompt, functions)

        if result:
            results.append(result.model_dump())
        else:
            results.append({
                'prompt': test_prompt.prompt,
                'name': "",
                'parameters': {}
            })

    return results
