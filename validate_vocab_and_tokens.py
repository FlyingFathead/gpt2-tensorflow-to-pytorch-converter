# validate_vocab_and_tokens.py

import json
import sys

def validate_vocab_and_tokens(vocab_json_path, generated_tokens):
    with open(vocab_json_path, 'r', encoding='utf-8') as vocab_file:
        vocab = json.load(vocab_file)
    
    vocab_size = len(vocab)
    print(f"Vocabulary Size: {vocab_size}")

    # Validate token IDs
    invalid_tokens = [token for token in generated_tokens if token not in vocab.values()]
    if invalid_tokens:
        print(f"Invalid token IDs found: {invalid_tokens}")
    else:
        print("All generated token IDs are valid.")

def main():
    if len(sys.argv) != 3:
        print("Usage: python validate_vocab_and_tokens.py <vocab_json_path> <generated_tokens>")
        sys.exit(1)

    vocab_json_path = sys.argv[1]
    generated_tokens = list(map(int, sys.argv[2].split(',')))

    validate_vocab_and_tokens(vocab_json_path, generated_tokens)

if __name__ == "__main__":
    main()
