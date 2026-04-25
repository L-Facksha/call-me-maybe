"""Entry point for the function calling program."""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate natural language into function calls."
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
    """Run the function calling pipeline."""
    args = parse_args()

    try:
        from llm_sdk import Small_LLM_Model
        from src.loader import load_function_definitions, load_test_prompts, save_results
        from src.function_caller import run_pipeline
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}", file=sys.stderr)
        return 1

    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_test_prompts(args.input)

        if not functions or not prompts:
            print("[ERROR] No functions or prompts loaded.", file=sys.stderr)
            return 1

        print("[INFO] Loading model...", file=sys.stderr)
        model = Small_LLM_Model()

        print("[INFO] Running pipeline...", file=sys.stderr)
        results = run_pipeline(model, functions, prompts)

        save_results(results, args.output)
        print(f"[INFO] Results saved to {args.output}", file=sys.stderr)
        return 0

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
