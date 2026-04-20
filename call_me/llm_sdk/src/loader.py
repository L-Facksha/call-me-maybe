from pathlib import Path
import json
from .models import FunctionDefinition


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

    res = []
    f_path = Path(path)

    if not f_path.exists():
        raise FileNotFoundError(f"File not found {f_path}")
    if not f_path.is_file():
        raise ValueError(f"Expected a file but got a directory: {f_path}")

    with f_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        res.append(FunctionDefinition.model_validate(item))
    return res



# def load_test_prompts(path):
