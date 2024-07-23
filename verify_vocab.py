# verify_vocab.py

import json

def verify_vocab(vocab_path):
    try:
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab = json.load(vocab_file)
        
        print("First 10 entries of vocab.json:")
        for i, (token, index) in enumerate(vocab.items()):
            print(f"{i}: {token} -> {index}")
            if i >= 9:
                break

    except Exception as e:
        print(f"Error reading vocab.json: {e}")

if __name__ == "__main__":
    vocab_path = "./converted_model/vocab.json"
    verify_vocab(vocab_path)
