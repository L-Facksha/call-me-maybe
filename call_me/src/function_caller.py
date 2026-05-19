"""Function calling pipeline: selects a function and extracts its arguments."""

import sys
import json
from pathlib import Path
from typing import Any
from src.models import FunctionDefinition, FunctionCallResult
from src.generator import generate_args, generate_name
from llm_sdk import Small_LLM_Model
GREEN = "\033[92m"
ORANGE = "\033[38;5;214m"
RESET = "\033[0m"
RED = "\033[91m"


def load_vocab(model: Small_LLM_Model) -> dict[str, int]:
    vocab_path = Path(model.get_path_to_vocab_file())
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    id_token = {int(tid): token for token, tid in vocab.items()}
    token_to_id = {token: int(tid) for token, tid in vocab.items()}
    sorted_vocab = sorted(
        id_token.items(), key=lambda x: len(x[1]), reverse=True)
    

    return id_token, sorted_vocab, token_to_id


def process_prompt(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    sorted_vocab: list[tuple[int, str]],
    token_to_id: dict[str: int],
    valid_token_ids,
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> FunctionCallResult | None:
    """Translate one natural language prompt into a structured function call.

    Parameters
    ----------
    model : Small_LLM_Model
        The loaded LLM model instance.
    vocab : dict[int, str]
        Token ID → string mapping.
    user_prompt : str
        The natural language request to process.
    functions : list[FunctionDefinition]
        All available function definitions.

    Returns
    -------
    FunctionCallResult | None
        The structured result, or None if no function could be selected.
    """
    try:
        valid_names = [
            fn.name for fn in functions if fn.name and fn.name.strip()]

        valid_names.append("fn_no_function")

        fn_descriptions = "\n".join(
            [f"- {fn.name}: {fn.description}" for fn in functions if fn.name and fn.name.strip()]
            + ["- fn_no_function: if no words in the prompt match the function"]
        )

        name_prompt = (
            "You are a function calling system.\n"
            "Choose ONLY one function name.\n"
            "The prompt must clearly ask for the function's action.\n"
            "If it is a question, greeting, or unrelated request, return fn_no_function.\n\n"

            f"Available functions:\n{fn_descriptions}\n\n"

            "Examples:\n"

            "User request: Greet shrek\n"
            "Best matching function name: fn_greet\n\n"

            "User request: greet john\n"
            "Best matching function name: fn_greet\n\n"

            "User request: say hello to bob\n"
            "Best matching function name: fn_greet\n\n"

            "User request: What is the square root of 16?\n"
            "Best matching function name: fn_get_square_root\n\n"

            "User request: add 2 and 3\n"
            "Best matching function name: fn_add_numbers\n\n"

            "User request: What is the sum of 3 and 5?\n"
            "Best matching function name: fn_add_numbers\n\n"

            "User request: What is the sum or of 6985 and -255?\n"
            "Best matching function name: fn_add_numbers\n\n"

            "User request: tell me a joke\n"
            "Best matching function name: fn_no_function\n\n"

            "User request: what is the weather\n"
            "Best matching function name: fn_no_function\n\n"

            "User request: how many stars in the sky\n"
            "Best matching function name: fn_no_function\n\n"

            "User request: who is the president of france\n"
            "Best matching function name: fn_no_function\n\n"

            "User request: write me a poem\n"
            "Best matching function name: fn_no_function\n\n"

            "User request: who are you\n"
            "Best matching function name: fn_no_function\n\n"

            "User request: what can you do\n"
            "Best matching function name: fn_no_function\n\n"

            "User request: hello\n"
            "Best matching function name: fn_no_function\n\n"

            "User request: how are you\n"
            "Best matching function name: fn_no_function\n\n"

            f"User request: {user_prompt}\n"
            'Best matching function name: "'
        )
        max_token = max(len(func) for func in valid_names)

        fn_name = generate_name(
            model, vocab, sorted_vocab, token_to_id, valid_token_ids, name_prompt, valid_names, max_token)

        if fn_name == "fn_no_function" or fn_name not in valid_names:
            print(
                f"{ORANGE}[WARNING]{RESET} Could not select function for: {user_prompt!r}",
                file=sys.stderr,
            )
            return None

        fn_def = next((fn for fn in functions if fn.name == fn_name), None)
        if fn_def is None:
            print(
                f"{ORANGE}[WARNING]{RESET} Function not found: {fn_name}", file=sys.stderr)
            return None

        args = generate_args(model, vocab, sorted_vocab, token_to_id,
                             valid_token_ids, user_prompt, fn_def)

        return FunctionCallResult(
            prompt=user_prompt,
            name=fn_name,
            parameters=args,
        )

    except Exception as error:
        print(f"{RED}[ERROR]{RESET} Process failed: {error}", file=sys.stderr)
        return None


def run_pipeline(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    prompts: list[Any],
) -> list[dict[str, Any]] | None:

    missing_name = False
    for fn in functions:
        if not fn.name or fn.name.strip() == "":
            missing_name = True
    if missing_name:
        print(
            f"{ORANGE}[WARNING]{RESET} Missing function name in 'functions_definition.json'"
        )
        print(
            f"{ORANGE}[WARNING]{RESET} Function name should start with 'fn_' for better results")
        return None

    vocab, sorted_vocab, token_to_id = load_vocab(model)
    valid_token_ids = set(vocab.keys())

    results: list[dict[str, Any]] = []

    for i, test_prompt in enumerate(prompts):
        print(
            f"{GREEN}[INFO]{RESET} {i + 1}/{len(prompts)}: {test_prompt.prompt!r}",
            file=sys.stderr,
        )
        result = process_prompt(
            model, vocab, sorted_vocab, token_to_id, valid_token_ids, test_prompt.prompt, functions)

        if result and result.name:
            results.append(result.model_dump())
        else:
            results.append({
                "prompt": test_prompt.prompt,
                "name": "",
                "parameters": {},
            })

    return results
