import json
import __init__ as q
# ---------------ex1----------------------#


def fn_add_numbers(a, b):
    return a + b


text = '{"name": "fn_add_numbers", "parameters": {"a": 10, "b": 5}}'

data = json.loads(text)
result = fn_add_numbers(data['parameters']['a'], data['parameters']['b'])
print("#----------------ex1----------------------#")
print(result)

# ----------------ex2----------------------#"


def fn_greet(name):
    return f"Hello, {name}!"


text = '{"name": "fn_greet", "parameters": {"name": "Alice"}}'
data = json.loads(text)
result = fn_greet(**data['parameters'])
print("#----------------ex2----------------------#")
print(result)

x = json
# ----------------ex3----------------------#"

def fn_add_numbers(a, b):
    return a + b


def fn_greet(name):
    return f"Hello, {name}!"


def fn_reverse_string(s):
    return s[::-1]


registry = {
    "fn_add_numbers": fn_add_numbers,
    "fn_greet": fn_greet,
    "fn_reverse_string": fn_reverse_string
}

text = '{"name": "fn_reverse_string", "parameters": {"s": "hello"}}'

data = json.loads(text)
result = registry[data['name']](**data['parameters'])
print("#----------------ex3----------------------#")
print(result)

# ----------------ex4----------------------#


def fn_add_numbers(a, b):
    return a + b


def fn_greet(name):
    return f"Hello, {name}!"


def fn_reverse_string(s):
    return s[::-1]


registry = {
    "fn_add_numbers": fn_add_numbers,
    "fn_greet": fn_greet,
    "fn_reverse_string": fn_reverse_string
}


def dispatch(json_text: str) -> str:
    data = json.loads(json_text)
    return registry[data['name']](**data['parameters'])


print("#----------------ex4----------------------#")
print(dispatch('{"name": "fn_greet", "parameters": {"name": "Bob"}}'))
print(dispatch('{"name": "fn_add_numbers", "parameters": {"a": 7, "b": 3}}'))
