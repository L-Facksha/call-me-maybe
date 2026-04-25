"""Entry point for the function calling program."""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="/goinfre/azebahad/call_me/call_me/data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/goinfre/azebahad/call_me/call_me/data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/goinfre/azebahad/call_me/call_me/data/output/function_calling_results.json",
    )
    return parser.parse_args()


def main() -> int:
    """Run the function calling pipeline.

    Returns
    -------
    int
        Exit code.
    """
    args = parse_args()

    try:
        from llm_sdk import Small_LLM_Model
    except ImportError as e:
        print(f"[ERROR] Cannot import llm_sdk: {e}", file=sys.stderr)
        return 1

    from src.loader import load_function_definitions, load_test_prompts, save_results
    from src.function_caller import run_pipeline

    print(f"[INFO] Loading functions from: {args.functions_definition}")
    functions = load_function_definitions(args.functions_definition)
    if not functions:
        print("[ERROR] No functions loaded.", file=sys.stderr)
        return 1

    print(f"[INFO] Loading prompts from: {args.input}")
    prompts = load_test_prompts(args.input)
    if not prompts:
        print("[ERROR] No prompts loaded.", file=sys.stderr)
        return 1

    print("[INFO] Loading model...")
    try:
        model = Small_LLM_Model()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}", file=sys.stderr)
        return 1

    print("[INFO] Running pipeline...")
    results = run_pipeline(model, functions, prompts)

    save_results(results, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
