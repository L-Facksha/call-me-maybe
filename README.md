*This project has been created as part of the 42 curriculum by azebahad.*

# Call Me Maybe — Function Calling via Constrained Decoding

## Description

This project builds a function calling system that translates natural language prompts into structured JSON function calls using a small LLM (Qwen3-0.6B) with constrained decoding.

Given a prompt like `"What is the sum of 2 and 3?"`, the system produces:

```json
{
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The key insight is that small language models (0.6B parameters) are unreliable at generating structured output when prompted directly — they succeed only ~30% of the time. Constrained decoding solves this by guiding the model token by token, guaranteeing 100% valid output.

## Instructions

### Requirements

- Python 3.10+
- `uv` package manager

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd call_me

# Install dependencies
uv sync
  +
uv pip install -r requirements.txt
```

The `llm_sdk` package is included in the repository. No additional setup is needed.

### Execution

```bash
# Run with default paths
uv run python -m src

# Run with custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Other Makefile commands

```bash
make install   # install dependencies
make run       # run the program
make debug     # run with pdb debugger
make clean     # remove cache files
make lint      # run flake8 and mypy
```

## Example Usage

Input prompt file `data/input/function_calling_tests.json`:
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet shrek"},
  {"prompt": "Reverse the string 'hello'"}
]
```

Output file `data/output/function_calling_results.json`:
```json
[
  {"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}},
  {"prompt": "Greet shrek", "name": "fn_greet", "parameters": {"name": "shrek"}},
  {"prompt": "Reverse the string 'hello'", "name": "fn_reverse_string", "parameters": {"s": "hello"}}
]
```

## Algorithm Explanation

The system uses a two-stage constrained decoding pipeline:

**Stage 1 — Function Name Selection:**

The model generates the function name token by token. At each step, the logits (raw model scores) are masked: any token that cannot extend the current prefix toward a valid function name is set to `-inf`. Only tokens that keep at least one valid name reachable are allowed. The model naturally selects the most likely valid name given the prompt context.

For example, generating `fn_greet`:
```
step 1: only tokens starting valid prefixes allowed → model picks "fn"
step 2: current="fn" → only "_g", "_a", "_r", "_s" survive → model picks "_g"
step 3: current="fn_g" → only "reet" survives → model picks "reet"
step 4: current="fn_greet" → only '"' survives → return "fn_greet"
```

**Stage 2 — Argument Extraction:**

For number parameters, numbers are extracted directly from the prompt using regex, then the model is constrained to generate exactly those digits in order.

For string parameters, the model uses few-shot prompting with constrained decoding — newline tokens are blocked so the model cannot ramble, and it stops naturally at the end of the value.

**Custom Tokenizer:**

Instead of using `model.encode()`, the project implements a custom tokenizer using `get_path_to_vocab_file()`. The vocabulary is loaded, converted to an ID-to-token mapping, sorted by token length, and encoded using a greedy longest-match strategy.

## Design Decisions

**Why constrained decoding over prompting?**
Small models fail to produce valid JSON reliably when prompted. Constrained decoding guarantees valid output regardless of model size by restricting what the model can generate at each step.

**Why custom tokenizer?**
The bonus requirement asks to avoid `model.encode()`. The custom tokenizer uses `get_path_to_vocab_file()` and implements greedy longest-match BPE encoding. This gives full control over tokenization and avoids SDK dependencies.

**Why few-shot prompting for strings?**
String values are diverse names, regex patterns, replacement strings. Few-shot examples guide the model to understand what each parameter type expects, while constrained decoding prevents hallucination by blocking newlines and structural JSON characters.

**Why `fn_no_function` as a valid name?**
Adding `fn_no_function` to the valid names list lets the constrained decoder select it when no real function matches. This handles bad or irrelevant prompts gracefully without crashing.

## Performance Analysis

**Accuracy:** 100% on the provided test prompts. Function selection and argument extraction both work correctly for all 11 prompts including complex regex substitution cases.

**Speed:** ~3-4 minutes for 11 prompts on CPU. The bottleneck is the LLM inference call itself each token requires a full forward pass through 28 transformer layers. Optimizations include precomputed vocab arrays, vectorized numpy masking, and reduced `max_token` limits.

**Reliability:** 100% valid JSON output guaranteed by constrained decoding, the model cannot generate malformed JSON or invalid function names.

**Edge cases handled:**
- Bad/irrelevant prompts → empty result, no crash
- Functions with empty names → skipped with warning
- Large numbers → extracted via regex, no digit limit
- Special characters in prompts → handled by tokenizer fallback

## Challenges Faced

**Challenge 1 — Custom tokenizer design:**
The main challenge was implementing tokenization without using `model.encode()`. The solution uses greedy longest-match matching over the vocabulary to convert text into token IDs.

**Challenge 2 — Model rambling in string generation:**
Without constraints, the model generates values followed by long explanations. Fixed by blocking newline tokens (`Ċ`) in `generate_string`, the few-shot format uses newlines as natural terminators, so the model stops cleanly.

**Challenge 3 — `fn_no_function` not filtering bad prompts:**
The model sometimes selected `fn_no_function` but the code then tried to find it in the functions list and failed. Fixed by checking `if fn_name == "fn_no_function"` immediately after name generation and returning `None` before looking up the definition.

**Challenge 4 — Regex parameters:**
The model did not know what a regex pattern looks like without guidance. Fixed by adding specific few-shot examples for `regex` and `replacement` parameters that show the expected format.

## Testing Strategy

Testing was done by running the pipeline against the provided test prompts and verifying the output JSON manually:

1. **Function selection** — verified all 11 prompts select the correct function
2. **Argument extraction** — verified values match expected types and content
3. **Edge cases** — tested bad prompts, empty function names, large numbers
4. **JSON validity** — verified output is 100% parseable with `json.loads()`
5. **Robustness** — tested with modified `functions_definition.json` (renamed functions, empty names) to verify the system adapts without hardcoding

## Resources

- [Hugging Face — Constrained Beam Search](https://huggingface.co/blog/constrained-beam-search)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [BPE Tokenization Explained](https://huggingface.co/learn/nlp-course/chapter6/5)
- [JSON Schema Specification](https://json-schema.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

### AI Usage

- **Learning** — explaining how constrained decoding works conceptually and providing resources to go deeper
- **Understanding AI** — explaining how language models work in general: tokenization, embeddings, transformer layers, logits, and token selection
