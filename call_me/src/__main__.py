import argparse
import sys
import time
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Function calling system using constrained decoding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model to load.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arg()

    try:
        from llm_sdk import Small_LLM_Model
        from src.loader import load_function_definitions, load_test_prompts, save_results
        from src.function_caller import run_pipeline
    except Exception as error:
        print(f"{RED}[ERROR]{RESET} Import failed: {error}", file=sys.stderr)
        return 1

    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_test_prompts(args.input)

        for prompt in prompts:
            if len(prompt.prompt) > 90:
                print(
                    f"{RED}[ERROR] Prompt is to long (max <90 charachter){RESET}: '{prompt.prompt}'")
                return 1

        if not functions or not prompts:
            print(
                f"{RED}[ERROR]{RESET} No functions or prompts loaded!", file=sys.stderr)
            return 1

        print(f"{GREEN}[INFO]{RESET} Loading model...", file=sys.stderr)
        model = Small_LLM_Model(args.model)
        start = time.perf_counter()

        print(f"{GREEN}[INFO]{RESET} Running pipeline...", file=sys.stderr)
        results = run_pipeline(model, functions, prompts)

        save_results(results, args.output)
        print(
            f"{GREEN}[INFO]{RESET} Results saved to {args.output}", file=sys.stderr)
        duration = time.perf_counter() - start
        print(f"[TIME: {duration / 60:.3f}m]", file=sys.stderr)
        return 0

    except Exception as error:
        print(f"{RED}[ERROR]{RESET} {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
