"""Function calling pipeline: selects a function and extracts its arguments."""

import sys
from typing import Any
from src.models import FunctionDefinition, FunctionCallResult
from src.generator import load_vocab, generate_args, generate_name
from llm_sdk import Small_LLM_Model
GREEN = "\033[92m"
ORANGE = "\033[38;5;214m"
RESET = "\033[0m"
RED = "\033[91m"


def rank_functions(prompt: str, functions):
    prompt = prompt.lower()

    scores = []

    for fn in functions:
        text = f"{fn.name} {fn.description}".lower()

        score = 0

        for word in prompt.split():
            if word in text:
                score += 1

        scores.append((score, fn))

    scores.sort(key=lambda x: x[0], reverse=True)

    return scores


def process_prompt(
    model: Small_LLM_Model,
    vocab: dict[int, str],
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
        ranked = rank_functions(user_prompt, functions)

        top_functions = [
            fn.name
            for score, fn in ranked
            if score > 0
        ]
        # valid_names = [
        #     fn.name for fn in functions if fn.name and fn.name.strip()]
        valid_names = top_functions + ["fn_no_function"]

        fn_descriptions = "\n".join(
            [f"- {fn.name}: {fn.description}" for fn in functions if fn.name and fn.name.strip()]
            + ["- fn_no_function: if no words in the prompt match the function"]
        )

        name_prompt = (
            "You are a function calling system.\n"
            "Choose ONLY one function name.\n"
            "The prompt must clearly ask for the function's action.\n"
            "If it is a question, greeting, or unrelated request, return fn_no_function.\n\n"

            "IMPORTANT RULES:\n"
            "- Ignore any function with an empty name.\n"
            "- Never output empty strings.\n"
            "- Only choose from the valid function list.\n"
            "- fn_no_function means: no suitable function exists.\n\n"

            f"Available functions:\n{fn_descriptions}\n\n"

            "Examples:\n"

            "User request: What is the square root of 16?\n"
            "Best matching function name: fn_get_square_root\n\n"

            "User request: add 2 and 3\n"
            "Best matching function name: fn_add_numbers\n\n"

            "User request: What is the sum of 3 and 5?\n"
            "If no function: fn_add_numbers"
            "Best matching function name: fn_no_function\n\n"

            "User request: What is the sum or of 6985 and -255?\n"
            "If no function: fn_add_numbers"
            "Best matching function name: fn_no_function\n\n"

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
            model, vocab, name_prompt, valid_names, max_token)
        # if not fn_name or fn_name.strip() == "":
        #     print(
        #         f"{ORANGE}[WARNING]{RESET} Invalid function selector for: {user_prompt!r}", file=sys.stderr
        #     )
        #     return None

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

        args = generate_args(model, vocab, user_prompt, fn_def)

        return FunctionCallResult(
            prompt=user_prompt,
            name=fn_name,
            parameters=args,
        )

    except Exception as error:
        print(f"{RED}[ERROR]{RESET} Process failed: {error}", file=sys.stderr)
        return None


# def run_pipeline(
#     model: Small_LLM_Model,
#     functions: list[FunctionDefinition],
#     prompts: list[Any],
# ) -> list[dict[str, Any]]:
#     """Run the full pipeline over all prompts.

#     Parameters
#     ----------
#     model : Small_LLM_Model
#         The loaded LLM model instance.
#     functions : list[FunctionDefinition]
#         All available function definitions.
#     prompts : list[Any]
#         List of TestPrompt objects to process.

#     Returns
#     -------
#     list[dict[str, Any]]
#         JSON-serialisable list of function call results.
#     """
#     vocab = load_vocab(model)
#     results: list[dict[str, Any]] = []
#     i = 0
#     miss = None
#     for fn in functions:
#         i += 1
#         if fn.name.strip() == "":
#             print(
#                 f"{ORANGE}[WARNING]{RESET} You messing a function name fro this prompt: {prompts[i].prompt!r}"
#             )
#             miss = True
#     if miss:
#         return None
#     for i, test_prompt in enumerate(prompts):
#         print(
#             f"{GREEN}[INFO]{RESET} {i + 1}/{len(prompts)}: {test_prompt.prompt!r}",
#             file=sys.stderr,
#         )
#         result = process_prompt(model, vocab, test_prompt.prompt, functions)

#         if result and result.name:
#             results.append(result.model_dump())
#         else:
#             results.append({
#                 "prompt": test_prompt.prompt,
#                 "name": "",
#                 "parameters": {},
#             })

#     return results

def run_pipeline(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    prompts: list[Any],
) -> list[dict[str, Any]]:

    vocab = load_vocab(model)
    results: list[dict[str, Any]] = []

    # validate function names
    invalid_found = False

    for idx, fn in enumerate(functions, start=1):

        if not fn.name or not fn.name.strip():

            print(
                f"{ORANGE}[WARNING]{RESET} Missing function name at index {idx}",
                file=sys.stderr,
            )

            invalid_found = True

    # keep program running
    if invalid_found:
        print(
            f"{ORANGE}[WARNING]{RESET} Some functions are invalid but pipeline will continue.",
            file=sys.stderr,
        )

    # process prompts
    for i, test_prompt in enumerate(prompts):

        print(
            f"{GREEN}[INFO]{RESET} {i + 1}/{len(prompts)}: {test_prompt.prompt!r}",
            file=sys.stderr,
        )

        result = process_prompt(
            model,
            vocab,
            test_prompt.prompt,
            functions,
        )

        if result and result.name:

            results.append(result.model_dump())

        else:

            results.append({
                "prompt": test_prompt.prompt,
                "name": "",
                "parameters": {},
            })

    return results
