"""Pydantic models for the function calling project."""
from pydantic import BaseModel
from typing import Any


class ParameterDefinition(BaseModel):
    """Describe a single parameter in a function definition.

    Attributes
    ----------
    type : str
        The expected data type of the parameter
    """

    type: str


class FunctionDefinition(BaseModel):
    """Represents a complete function definition.

    Attributes
    ----------
    name : str
        The name of the function ex 'fn_add_numbers'.

    description : str
        Human readable description of what the function does.

    parameters : dict[str, ParameterDefinition]
        The parameters of the function mapped by name.

    returns : ParameterDefinition
        The return type of the function.
    """

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


class TestPrompt(BaseModel):
    """Represents a single natural language prompt to process.

    Attributes
    ----------
    prompt : str
        The natural language request ex 'What is the sum of 2 and 3?'.
    """

    prompt: str


class FunctionCallResult(BaseModel):
    """Represents the output result for one prompt.

    Attributes
    ----------
    prompt : str
        The original natural language request.

    name : str
        The name of the selected function to call.

    parameters : dict[str, Any]
        The extracted arguments mapped by parameter name.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
