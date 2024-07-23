# sync_tokens.py

import json

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

def sync_tokens(vocab_path, merges_path, vocab_tokens, merges_tokens, vocab):
    # Identify missing tokens
    missing_in_vocab = merges_tokens - vocab_tokens
    missing_in_merges = vocab_tokens - merges_tokens

    # Add missing tokens to vocab.json
    current_index = max(vocab.values()) + 1
    for token in missing_in_vocab:
        print(f"Adding missing token to vocab.json: {token} -> {current_index}")
        vocab[token] = current_index
        current_index += 1

    with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
        json.dump(vocab, vocab_file, ensure_ascii=False, indent=2)

    print(f"Added {len(missing_in_vocab)} missing tokens to vocab.json.")

    # Add missing tokens to merges.txt
    with open(merges_path, 'a', encoding='utf-8') as merges_file:
        for token in missing_in_merges:
            merges_file.write(f"{token} {token}\n")

    print(f"Added {len(missing_in_merges)} missing tokens to merges.txt.")

def main():
    vocab_path = "./converted_model/vocab.json"
    merges_path = "./converted_model/cleaned_merges.txt"

    merges_tokens = extract_tokens_from_merges(merges_path)
    vocab_tokens, vocab = extract_tokens_from_vocab(vocab_path)

    print(f"Extracted tokens from merges.txt: {list(merges_tokens)[:10]} ...")
    print(f"Extracted tokens from vocab.json: {list(vocab_tokens)[:10]} ...")

    sync_tokens(vocab_path, merges_path, vocab_tokens, merges_tokens, vocab)

    # Verify all tokens are now in both files
    updated_merges_tokens = extract_tokens_from_merges(merges_path)
    updated_vocab_tokens, _ = extract_tokens_from_vocab(vocab_path)

    missing_tokens_after = updated_merges_tokens - updated_vocab_tokens
    if missing_tokens_after:
        print(f"Still missing tokens after update: {missing_tokens_after}")
    else:
        print("All tokens from merges.txt are now in vocab.json and vice versa.")

if __name__ == "__main__":
    main()
