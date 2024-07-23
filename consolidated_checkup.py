# consolidated_checkup.py

import json

def check_file_encoding(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file.read()
        print(f"{file_path} is in UTF-8 encoding.")
    except UnicodeDecodeError:
        print(f"{file_path} is not in UTF-8 encoding.")

def print_file_contents(vocab_path, merges_path, num_lines=50):
    try:
        print("First lines of vocab.json:")
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab = json.load(vocab_file)
            for i, (token, index) in enumerate(vocab.items()):
                print(f"{i}: {token} -> {index}")
                if i >= num_lines - 1:
                    break
        
        print("\nFirst lines of cleaned_merges.txt:")
        with open(merges_path, 'r', encoding='utf-8') as merges_file:
            for i, line in enumerate(merges_file):
                print(f"{i}: {line.strip()}")
                if i >= num_lines - 1:
                    break

    except Exception as e:
        print(f"Error reading files: {e}")

def extract_tokens_from_merges(merges_path):
    tokens = set()
    try:
        with open(merges_path, 'r', encoding='utf-8') as merges_file:
            for line in merges_file:
                if not line.startswith("#") and line.strip():
                    tokens.update(line.strip().split())
    except Exception as e:
        print(f"Error reading merges.txt: {e}")
    return tokens

def extract_tokens_from_vocab(vocab_path):
    tokens = set()
    vocab = {}
    try:
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab = json.load(vocab_file)
            tokens = set(vocab.keys())
    except Exception as e:
        print(f"Error reading vocab.json: {e}")
    return tokens, vocab

def add_missing_tokens(vocab_path, merges_tokens, vocab_tokens, vocab):
    missing_tokens = merges_tokens - vocab_tokens
    current_index = max(vocab.values()) + 1

    if missing_tokens:
        try:
            for token in missing_tokens:
                print(f"Adding missing token: {token} -> {current_index}")
                vocab[token] = current_index
                current_index += 1

            with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
                json.dump(vocab, vocab_file, ensure_ascii=False, indent=2)

            print(f"Added {len(missing_tokens)} missing tokens to vocab.json.")
            print(f"Missing tokens added: {missing_tokens}")
        except Exception as e:
            print(f"Error updating vocab.json: {e}")
    else:
        print("No missing tokens found. vocab.json is up to date.")

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

    # Step 1: Check UTF-8 Encoding
    check_file_encoding(vocab_path)
    check_file_encoding(merges_path)

    # Step 2: Print File Contents
    print_file_contents(vocab_path, merges_path)

    # Step 3: Verify Encoding and Synchronize Tokens
    merges_tokens = extract_tokens_from_merges(merges_path)
    vocab_tokens, vocab = extract_tokens_from_vocab(vocab_path)
    add_missing_tokens(vocab_path, merges_tokens, vocab_tokens, vocab)

    # Step 4: Verify Encoding
    verify_encoding(vocab_path, merges_path)
