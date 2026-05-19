import re

text = "hello 123.54454646 jjk;j 45"

matches = re.findall(r"-?\d+(?:\.\d+)?", text)

# print(matches)
# for i, m in enumerate(matches):
#     matches[i] = round(float(m))
for m in matches:
    print(float(m))

print(len("Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"))
print(int(float("56463")))