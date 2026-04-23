from llm_sdk import Small_LLM_Model
from generator import load_vocab, generate

schema = [
    {"name": "add"},
    {"name": "multiply"},
    {"name": "subtract"}
]

prompt = 'You are a function selector. Reply ONLY JSON.\n{"name": "'

model = Small_LLM_Model()
vocab = load_vocab(model)

result = generate(model, vocab, prompt, schema)

print("\nFINAL RESULT:")
print(result)