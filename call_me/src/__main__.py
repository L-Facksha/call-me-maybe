import argparse
import sys
import time
GREEN = "\033[92m"
RED = "\033[91m"
ORANGE = "\033[38;5;214m"
RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with the following fields:

        - ``functions_definition`` : path to the functions definition JSON
        file.
        - ``input`` : path to the test prompts JSON file.
        - ``output`` : path where results will be saved.
        - ``model`` : the model name or path to load.
    """
    parser = argparse.ArgumentParser()

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
    """Entry point for the function calling pipeline.

    Loads function definitions and test prompts, initialises the model,
    runs the pipeline, and saves the results.

    Returns
    -------
    int
        Exit code: ``0`` on success, ``1`` on any failure.
    """
    args = parse_args()

    try:
        from llm_sdk import Small_LLM_Model
        from src.loader import load_function_definitions, load_test_prompts, \
            save_results
        from src.function_caller import run_pipeline
    except Exception as error:
        print(f"{RED}[ERROR]{RESET} Import failed: {error}", file=sys.stderr)
        return 1

    try:
        functions = load_function_definitions(args.functions_definition)
        for fn in functions:
            if not fn.name or fn.name.strip() == "":
                print(
                    f"{ORANGE}[WARNING]{RESET} Missing function name in \
                    'functions_definition.json'"
                )
                print(
                    f"{ORANGE}[WARNING]{RESET} Function name should start with\
                        'fn_' for better results"
                )
                return 1
        prompts = load_test_prompts(args.input)

        if not functions or not prompts:
            print(
                f"{RED}[ERROR]{RESET} No functions or prompts loaded!",
                file=sys.stderr)
            return 1

        print(f"{GREEN}[INFO]{RESET} Loading model...", file=sys.stderr)
        model = Small_LLM_Model()
        start = time.perf_counter()

        print(f"{GREEN}[INFO]{RESET} Running pipeline...", file=sys.stderr)
        results = run_pipeline(model, functions, prompts)

        save_results(results, args.output)
        print(
            f"{GREEN}[INFO]{RESET} Results saved to {args.output}",
            file=sys.stderr)
        duration = time.perf_counter() - start
        print(f"[TIME: {duration / 60:.2f}m]")
        return 0

    except Exception as error:
        print(f"{RED}[ERROR]{RESET} {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
