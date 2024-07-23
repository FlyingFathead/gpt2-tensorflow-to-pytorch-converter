# inspect_tokenizer_files.py

import json

def inspect_tokenizer_files(vocab_path, merges_path):
    try:
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab = json.load(vocab_file)
            print(f"Vocab size: {len(vocab)}")
            print(f"First 10 vocab entries: {list(vocab.items())[:10]}")

        with open(merges_path, 'r', encoding='utf-8') as merges_file:
            merges = merges_file.readlines()
            print(f"Merges size: {len(merges)}")
            print(f"First 10 merges entries: {merges[:10]}")

    except Exception as e:
        print(f"Error inspecting tokenizer files: {e}")

if __name__ == "__main__":
    vocab_path = "./converted_model/vocab.json"
    merges_path = "./converted_model/merges.txt"
    inspect_tokenizer_files(vocab_path, merges_path)
