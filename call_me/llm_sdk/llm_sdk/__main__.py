import json
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


# ----------------ex5----------------------#
try:
    bad_json = '{"name": "fn_greet", "parameters": {"name": "Bob"'
    unknown_fn = '{"name": "fn_fly_to_moon", "parameters": {}}'
    good = '{"name": "fn_greet", "parameters": {"name": "Carol"}}'

    print(dispatch(bad_json))
    print(dispatch(unknown_fn))
    print(dispatch(good))
except json.decoder.JSONDecodeError as error:
    print("#----------------ex5----------------------#")
    print(error)

# ----------------ex6----------------------#


text = [
    '{"name": "fn_greet", "parameters": {"name": "Alice"}}',
    '{"name": "fn_add_numbers", "parameters": {"a": 10, "b": 20}}',
    '{"name": "fn_reverse_string", "parameters": {"s": "world"}}',
    '{"name": "fn_greet", "parameters": {"name": "Bob"}}'
]

print("#----------------ex6----------------------#")

try:
    for x in text:
        print(dispatch(x))
except json.decoder.JSONDecodeError as error:
    print(error)


# ----------------ex7----------------------#

print("#----------------ex7----------------------#")

