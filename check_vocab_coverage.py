# check_vocab_coverage.py

import json

def check_vocab_coverage(vocab_path, merges_path):
    try:
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab = json.load(vocab_file)
        
        with open(merges_path, 'r', encoding='utf-8') as merges_file:
            merges = merges_file.readlines()

        vocab_tokens = set(vocab.keys())
        merge_tokens = set()

        for line in merges:
            if not line.startswith("#"):
                tokens = line.strip().split()
                merge_tokens.update(tokens)

        missing_tokens = merge_tokens - vocab_tokens

        if missing_tokens:
            print(f"Missing tokens in vocab.json: {missing_tokens}")
        else:
            print("All tokens from merges.txt are present in vocab.json.")

    except Exception as e:
        print(f"Error checking vocab coverage: {e}")

if __name__ == "__main__":
    vocab_path = "./converted_model/vocab.json"
    merges_path = "./converted_model/merges.txt"
    check_vocab_coverage(vocab_path, merges_path)
