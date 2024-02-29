import argparse

# Create the parser
parser = argparse.ArgumentParser(description='An example script.')

# Add an argument
parser.add_argument('--input', type=str, help='The input file.')

# Parse the arguments
args = parser.parse_args()

# Print the input argument
args.input = "test.txt"
print(f"Input file: {args.input}")