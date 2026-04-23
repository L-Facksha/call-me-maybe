from pathlib import Path
import json
import numpy as np
import re
from typing import Dict, List, Any
from llm_sdk import Small_LLM_Model


def is_valid_name_prefix(partial: str, valid_names: List[str]) -> bool:
    return any(name.startswith(partial) for name in valid_names)


def load_vocab(model: Small_LLM_Model) -> Dict[int, str]:
    vocab_path = Path(model.get_path_to_vocab_file())
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found {vocab_path}")

    with vocab_path.open("r", encoding="utf-8") as f:
        vocab_raw = json.load(f)

    vocab = {}
    numeric_keys = sum(1 for k in vocab_raw if str(k).isdigit())
    numeric_values = sum(1 for v in vocab_raw.values() if str(v).isdigit())

    if numeric_keys > numeric_values * 2:
        for k, v in vocab_raw.items():
            try:
                vocab[int(k)] = str(v)
            except:
                continue
    else:
        for k, v in vocab_raw.items():
            try:
                vocab[int(v)] = str(k)
            except:
                continue

    return vocab


def get_state(text: str) -> str:
    recent = text.lower()
    if '"parameters"' in recent:
        return "params"
    if '"name"' in recent:
        return "name"
    return "start"


def generate(model: Small_LLM_Model, vocab: Dict[int, str], prompt: str, schema: List[Dict]) -> Dict[str, Any]:
    ids = model.encode(prompt)[0].tolist()
    valid_names = [f["name"] for f in schema]

    for step in range(100):
        logits = model.get_logits_from_input_ids(ids)

        # 🔥 FIX: Ensure 1D array
        if hasattr(logits, 'shape'):
            if len(logits.shape) > 1:
                logits = logits[-1]  # Last token
        else:
            logits = np.array([logits])[-1]

        text = model.decode(ids)

        # Simple masking - FIRST 5000 vocab only (FAST)
        for token_id in range(min(5000, len(logits))):
            token_str = vocab.get(token_id)
            if not token_str:
                continue

            # Name state: force fn_*
            if '"name"' in text[-50:]:
                if not any(name.startswith(token_str[:3]) for name in valid_names):
                    logits[token_id] = logits[token_id] - 50

            # Params: basic JSON
            elif '"parameters"' in text:
                if token_str not in ['{', '}', ':', ',', '"', 'true', 'false', 'null']:
                    logits[token_id] *= 0.1

        next_id = np.argmax(logits)
        ids.append(int(next_id))

        # JSON check
        if step > 20 and '}' in text[-20:]:
            try:
                # Simple last-JSON extraction
                start = text.rfind('{')
                end = text.rfind('}')
                if end > start:
                    candidate = text[start:end+1]
                    parsed = json.loads(candidate)

                    if parsed.get("name") in valid_names:
                        params = parsed.get("parameters", {})
                        # Force float numbers
                        for k, v in params.items():
                            if isinstance(v, (int, np.number)):
                                params[k] = float(v)

                        print(f"✅ {parsed['name']} ({len(params)} params)")
                        return {
                            "name": parsed["name"],
                            "parameters": params
                        }
            except:
                pass

    return {"name": "", "parameters": {}}
