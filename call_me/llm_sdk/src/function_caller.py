"""Main orchestrator that ties everything together."""
import argparse
import sys
from typing import List, Dict, Any
from pathlib import Path

from llm_sdk import Small_LLM_Model
from .generator import generate, load_vocab
from .loader import load_function_definitions, load_test_prompts, save_results
from .models import FunctionDefinition, TestPrompt


def create_system_prompt(functions: List[FunctionDefinition]) -> str:
    """Better prompt - forces JSON."""
    func_desc = "\n".join([f"{f.name}: {f.description}" for f in functions])
    return f"""AVAILABLE FUNCTIONS:
    {func_desc}

    RULE: Return ONLY valid JSON like this:
    {{"name": "fn_add_numbers", "parameters": {{"a": 1.0, "b": 2.0}}}}

    Request: """


def process_single_prompt(
    model: Small_LLM_Model,
    vocab: Dict[int, str],
    functions: List[FunctionDefinition],
    test_prompt: TestPrompt
) -> Dict[str, Any]:
    """Process one test prompt → function call."""
    system_prompt = create_system_prompt(functions)
    full_prompt = system_prompt + test_prompt.prompt

    print(f"\n🔄 '{test_prompt.prompt[:40]}...'")

    result = generate(model, vocab, full_prompt, [
                      f.model_dump() for f in functions])

    if isinstance(result, dict) and "name" in result:
        return {
            "prompt": test_prompt.prompt,
            "name": result["name"],
            "parameters": result.get("parameters", {})
        }
    return {"prompt": test_prompt.prompt, "name": "", "parameters": {}}


def main():
    parser = argparse.ArgumentParser(description="Function calling pipeline")
    parser.add_argument("--functions_definition",
                        default="/home/azebahad/goinfre/call_me/call_me/data/input/functions_definition.json")
    parser.add_argument(
        "--input", default="/home/azebahad/goinfre/call_me/call_me/data/input/function_calling_tests.json")
    parser.add_argument(
        "--output", default="/home/azebahad/goinfre/call_me/call_me/data/input/function_calling_results.json")
    args = parser.parse_args()

    try:
        # 1. Initialize model
        print("🚀 Initializing model...")
        model = Small_LLM_Model()
        vocab = load_vocab(model)

        # 2. Load data
        print("📂 Loading data...")
        functions = load_function_definitions(args.functions_definition)
        prompts = load_test_prompts(args.input)

        print(f"✅ {len(functions)} functions, {len(prompts)} prompts")

        # 3. Process all prompts
        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}]")
            result = process_single_prompt(model, vocab, functions, prompt)
            results.append(result)

        # 4. Save results
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        save_results(results, args.output)

        success = len([r for r in results if r["name"]])
        print(
            f"\n✨ Done! {success}/{len(results)} successful ({success/len(results)*100:.1f}%)")

    except Exception as e:
        print(f"💥 Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
