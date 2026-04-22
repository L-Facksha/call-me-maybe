from pathlib import Path
import json
import numpy as np
from llm_sdk import Small_LLM_Model


def load_vocab(model: Small_LLM_Model) -> dict[int, str]:
    vocab_path = Path(model.get_path_to_vocab_file())

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found {vocab_path}")

    with vocab_path.open("r", encoding="utf-8") as f:
        try:
            vocab = json.load(f)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON file {vocab_path}: {error}")

    id_to_tokens = {int(x): y for y, x in vocab.items()}
    return id_to_tokens


def generate(model: Small_LLM_Model, vocab, prompt, schema) -> dict:
    generate_text = ""
    max_token = 200
    ids = model.encode(prompt)[0].tolist()

    for _ in range(max_token):
        logits = model.get_logits_from_input_ids(ids)
        next_token_id = int(np.argmax(logits))
        ids.append(next_token_id)

        token_str = vocab[next_token_id]
        generate_text += token_str
        print(repr(token_str), repr(generate_text))
        if generate_text.strip().endswith("}"):
            try:
                result = json.loads(generate_text.strip())
                return result
            except json.JSONDecodeError:
                pass
    return {}


if __name__ == "__main__":
    model = Small_LLM_Model()
    vocab = load_vocab(model)
    result = generate(
        model,
        vocab,
        'You are a function selector. Reply with only JSON.\n{"name": "',
        {}
    )
    print(result)
