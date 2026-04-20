from pathlib import Path
import json
from typing import List
# import FunctionDefinition


def load_function_definitions(path: str) -> List[dict]:
    """Load function definitions from a JSON file.

    Attributes
    ----------
    path : str
        Path to the JSON file containing function definitions.

    Returns
    -------
    list[dict]
        Parsed JSON data.
    """
    f_path = Path(path)

    if not f_path.exists:
        raise FileNotFoundError(f"File not found {f_path}")
    if not f_path.is_file():
        raise ValueError(f"Expected a file but got a directory: {f_path}")

    with f_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(data)


load_function_definitions(
    "/home/azebahad/goinfre/call_me/call_me/data/input/functions_definition.json")


def load_test_prompts(path):
