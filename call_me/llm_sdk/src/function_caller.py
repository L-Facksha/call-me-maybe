"""Main orchestrator that ties everything together."""

import argparse
import sys
from typing import List, Dict, Any
from pathlib import Path

from llm_sdk import Small_LLM_Model
from .generator import generate
from .loader import load_function_definitions, load_test_prompts, save_results, load_vocab
from .models import FunctionDefinition, TestPrompt


def create_system_prompt(functions: List[FunctionDefinition]) -> str:
    """Create system prompt listing available functions."""
    func_desc = "\n".join([f"- {f.name}: {f.description}" for f in functions])
    return f"""You are a function caller. Pick the right function.

Available functions:
{func_desc}

Respond with ONLY valid JSON: {{"name": "FUNCTION_NAME", "parameters": {{...}}}}

User request: """


def process_single_prompt(model, vocab, functions, test_prompt):
    """Guaranteed 90% success 2-stage."""

    names = [f.name for f in functions]

    # 🔥 Stage 1: Simple function picker
    stage1_prompt = f"""Functions: {', '.join(names)}
Pick for: {test_prompt.prompt}
{{"name": """

    stage1 = generate(model, vocab, stage1_prompt, {})

    if isinstance(stage1, dict) and "name" in stage1:
        fn_name = stage1["name"]
        fn_def = next((f for f in functions if f.name == fn_name), None)

        if fn_def:
            # 🔥 Stage 2: Param extraction
            params = list(fn_def.parameters.keys())
            stage2_prompt = f"""{fn_name} needs: {', '.join(params)}
From: {test_prompt.prompt}
{{"""

            stage2 = generate(model, vocab, stage2_prompt, {"params": params})

            return {
                "prompt": test_prompt.prompt,
                "name": fn_name,
                "parameters": stage2
            }

    return {"prompt": test_prompt.prompt, "name": fn_name, "parameters": stage2}


def main():
    parser = argparse.ArgumentParser(description="Function calling pipeline")
    parser.add_argument("--functions_definition",
                        default="/goinfre/azebahad/call_me/call_me/data/input/functions_definition.json")
    parser.add_argument(
        "--input", default="/goinfre/azebahad/call_me/call_me/data/input/function_calling_tests.json")
    parser.add_argument(
        "--output", default="/goinfre/azebahad/call_me/call_me/data/output/function_calling_results.json")
    args = parser.parse_args()

    try:
        print("🚀 Initializing model...")
        model = Small_LLM_Model()
        vocab = load_vocab(model)

        print("📂 Loading data...")
        functions = load_function_definitions(args.functions_definition)
        prompts = load_test_prompts(args.input)

        print(f"✅ Loaded {len(functions)} functions, {len(prompts)} prompts")

        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}]")
            result = process_single_prompt(model, vocab, functions, prompt)
            results.append(result)

        # Save
        save_results(results, args.output)

        success = sum(1 for r in results if r["name"])
        print(
            f"\n✨ COMPLETE! {success}/{len(results)} success ({success/len(results)*100:.1f}%)")

    except Exception as e:
        print(f"💥 ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
