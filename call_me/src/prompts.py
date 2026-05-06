from src.models import FunctionDefinition


def build_extraction_prompt(
    user_prompt: str,
    func: FunctionDefinition,
    param_name: str,
    param_type: str,
) -> str:
    """Build a focused prompt for constrained decoding of a single parameter.

    Parameters
    ----------
    user_prompt : str
        The original natural language request from the user.
    func : FunctionDefinition
        The function definition being called.
    param_name : str
        The name of the parameter to extract.
    param_type : str
        The expected type of the parameter ('number' or 'string').

    Returns
    -------
    str
        A prompt string ready for constrained decoding, or an empty
        string if the user_prompt is blank.
    """
    if not user_prompt or not user_prompt.strip():
        return ""

    description = getattr(func, "description", "")

    if param_type == "number":
        # Use the actual user prompt as the only source of truth.
        # The model must copy a number it sees in the request — not invent one.
        # List all params so the model knows which position to target.
        param_list = ", ".join(func.parameters.keys())
        return (
            f"Function: {func.name}({param_list})\n"
            f"Task: {description}\n"
            f"Request: {user_prompt}\n"
            f"Copy the number that maps to [{param_name}] from the request above.\n"
            f"Only output the number, nothing else.\n"
            f"Number:"
        )

    # For strings: the key insight is that the model must echo back
    # a substring it can SEE in the request, not hallucinate.
    # We give one concrete example that mirrors the prompt's own structure.
    return (
        f"Function: {func.name}\n"
        f"Task: {description}\n"
        f"Request: {user_prompt}\n"
        f"Extract the exact value for [{param_name}] from the request.\n"
        f"Output only the value, no quotes, no explanation.\n"
        f"Value:"
    )
