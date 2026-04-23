"""Functions to load and save JSON files for the function calling project."""

from pathlib import Path
import json
from .models import FunctionDefinition, TestPrompt


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    """Load function definitions from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file containing function definitions.

    Returns
    -------
    list[FunctionDefinition]
        Parsed JSON data.
    """
    result = []
    f_path = Path(path)

    if not f_path.exists():
        raise FileNotFoundError(f"File not found {f_path}")
    if not f_path.is_file():
        raise ValueError(f"Expected a file but got a directory: {f_path}")

    with f_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid json file {f_path}: {error}")
    for item in data:
        result.append(FunctionDefinition.model_validate(item))
    return result


def load_test_prompts(path: str) -> list[TestPrompt]:
    """Load tests prompt from a JSON file.

    Parameters
    ----------
    path : str
        path to the JSON file containing tests prompt.

    Returns
    -------
    list[TestPrompt]
        parsed JSON file.
    """
    result = []
    f_path = Path(path)
    if not f_path.exists():
        raise FileNotFoundError(f"File not found {f_path}")
    if not f_path.is_file():
        raise ValueError(f"Expected a file but got a directory: {f_path}")

    with f_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid json file {f_path}: {error}")

    for item in data:
        result.append(TestPrompt.model_validate(item))
    return result


def save_results(results: list[dict], path: str) -> bool:
    """Save function-calling results to a JSON file.

    Parameters
    ----------
    results : list[dict[str, Any]]
        A list of dictionaries representing the function-calling results.
        Each dictionary must follow the expected output schema.

    path : str
        The file path where the JSON output will be written.

    Returns
    -------
    bool
        True if the file is successfully written.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(results, f)
    return True


def load_vocab(model) -> dict[int, str]:
    """Load tokenizer vocabulary from model path."""

    vocab_path = Path(model.get_path_to_vocab_file())

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    with vocab_path.open("r", encoding="utf-8") as f:
        try:
            vocab = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid vocab JSON: {e}")

    # format: token -> id OR id -> token (depends on file)
    if all(isinstance(k, str) and k.isdigit() for k in vocab.keys()):
        return {int(k): v for k, v in vocab.items()}
    else:
        return {int(v): k for k, v in vocab.items()}
