# clean_and_sync_vocab.py

import json

def clean_merges_file(original_merges_path, cleaned_merges_path):
    try:
        with open(original_merges_path, 'r', encoding='utf-8') as merges_file:
            merges = merges_file.readlines()
        
        with open(cleaned_merges_path, 'w', encoding='utf-8') as cleaned_file:
            for line in merges:
                if not line.startswith("#") and line.strip():
                    cleaned_file.write(line)

        print(f"Cleaned merges.txt and saved to {cleaned_merges_path}")
    except Exception as e:
        print(f"Error cleaning merges.txt: {e}")

def sync_vocab_with_cleaned_merges(vocab_path, cleaned_merges_path):
    try:
        # Load vocab.json
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab = json.load(vocab_file)

        # Load cleaned merges.txt
        with open(cleaned_merges_path, 'r', encoding='utf-8') as merges_file:
            merges = merges_file.readlines()

        # Extract all tokens from cleaned merges.txt
        merge_tokens = set()
        for line in merges:
            tokens = line.strip().split()
            merge_tokens.update(tokens)

        # Find missing tokens
        vocab_tokens = set(vocab.keys())
        missing_tokens = merge_tokens - vocab_tokens

        # Add missing tokens to vocab.json
        current_index = max(vocab.values()) + 1
        for token in missing_tokens:
            if token not in vocab:
                vocab[token] = current_index
                current_index += 1

        # Write updated vocab.json back to file
        with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
            json.dump(vocab, vocab_file, ensure_ascii=False, indent=2)

        print(f"Added {len(missing_tokens)} missing tokens to vocab.json.")
        print(f"Missing tokens added: {missing_tokens}")

    except Exception as e:
        print(f"Error updating vocab.json: {e}")

if __name__ == "__main__":
    original_merges_path = "./converted_model/merges.txt"
    cleaned_merges_path = "./converted_model/cleaned_merges.txt"
    vocab_path = "./converted_model/vocab.json"

    clean_merges_file(original_merges_path, cleaned_merges_path)
    sync_vocab_with_cleaned_merges(vocab_path, cleaned_merges_path)
