import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input")
args = parser.parse_args()
print(args.input)
