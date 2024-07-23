# verify_encoding.py

import json

def verify_encoding(vocab_path, merges_path):
    try:
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab = json.load(vocab_file)
        
        with open(merges_path, 'r', encoding='utf-8') as merges_file:
            merges = merges_file.readlines()
        
        vocab_tokens = set(vocab.keys())
        merge_tokens = set()
        for line in merges:
            if not line.startswith("#") and line.strip():
                merge_tokens.update(line.strip().split())
        
        for token in merge_tokens:
            if token not in vocab_tokens:
                print(f"Token in merges.txt but not in vocab.json: {token}")

    except Exception as e:
        print(f"Error verifying encoding: {e}")

if __name__ == "__main__":
    vocab_path = "./converted_model/vocab.json"
    merges_path = "./converted_model/cleaned_merges.txt"
    verify_encoding(vocab_path, merges_path)
