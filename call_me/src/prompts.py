from src.models import FunctionDefinition


def build_extraction_prompt(
    user_prompt: str,
    func: FunctionDefinition,
    param_name: str,
    param_type: str,
) -> str:
    """Build a prompt that extracts a raw argument value from the user text.

    Critical design rule: the prompt must NOT describe what the function
    does, and must NOT include worked examples that show a computed result.
    A 0.6B model will execute the function (reverse the string, add the
    numbers, greet the person) instead of extracting the raw argument if
    the prompt gives it enough context to do so.

    The prompt must only ask the model to COPY a value it can SEE in the
    user request — nothing more.

    Parameters
    ----------
    user_prompt : str
        The original natural language request.
    func : FunctionDefinition
        The function definition being called.
    param_name : str
        Name of the parameter to extract.
    param_type : str
        Type of the parameter ('number' or 'string').

    Returns
    -------
    str
        Ready-to-use prompt ending with a neutral anchor token, or an
        empty string if user_prompt is blank.
    """
    if not user_prompt or not user_prompt.strip():
        return ""

    # Build the ordered parameter list so the model knows the position
    # of each argument (first number, second number, etc.).
    param_list = list(func.parameters.keys())
    param_index = param_list.index(param_name)

    if param_type == "number":
        # State the position explicitly ("1st", "2nd", …) so the model
        # copies the right number without computing anything.
        ordinal = {0: "1st", 1: "2nd", 2: "3rd"}.get(param_index,
                                                        f"{param_index + 1}th")
        return (
            f"Parameters: {', '.join(param_list)}\n"
            f"Input: {user_prompt}\n"
            f"Copy the {ordinal} number from the Input for [{param_name}].\n"
            f"Number:"
        )

    # For strings: name the parameter explicitly and ask the model to
    # copy the substring — never to transform or compute anything.
    return (
        f"Parameters: {', '.join(param_list)}\n"
        f"Input: {user_prompt}\n"
        f"Copy the exact value for [{param_name}] from the Input.\n"
        f"Do not transform, compute, or modify it.\n"
        f"Value:"
    )