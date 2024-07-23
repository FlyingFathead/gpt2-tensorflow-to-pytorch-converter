# sync_vocab_merges.py

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

def sync_vocab_with_merges(vocab_path, merges_tokens, vocab_tokens, vocab):
    missing_tokens = merges_tokens - vocab_tokens
    current_index = max(vocab.values()) + 1

    if missing_tokens:
        try:
            for token in missing_tokens:
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

if __name__ == "__main__":
    merges_path = "./converted_model/merges.txt"
    vocab_path = "./converted_model/vocab.json"

    merges_tokens = extract_tokens_from_merges(merges_path)
    vocab_tokens, vocab = extract_tokens_from_vocab(vocab_path)

    print(f"Extracted tokens from merges.txt: {list(merges_tokens)[:10]} ...")
    print(f"Extracted tokens from vocab.json: {list(vocab_tokens)[:10]} ...")

    sync_vocab_with_merges(vocab_path, merges_tokens, vocab_tokens, vocab)


# # sync_vocab_merges.py

# import json

# def sync_vocab_with_merges(vocab_path, merges_path):
#     try:
#         # Load vocab.json
#         with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
#             vocab = json.load(vocab_file)

#         # Load merges.txt
#         with open(merges_path, 'r', encoding='utf-8') as merges_file:
#             merges = merges_file.readlines()

#         # Print first few entries for manual verification
#         print("First 10 entries of vocab.json:")
#         for i, (token, index) in enumerate(vocab.items()):
#             print(f"{i}: {token} -> {index}")
#             if i >= 9:
#                 break

#         print("\nFirst 10 entries of merges.txt:")
#         for i, line in enumerate(merges):
#             print(f"{i}: {line.strip()}")
#             if i >= 9:
#                 break

#         # Extract all tokens from merges.txt, ignoring comment lines
#         merge_tokens = set()
#         for line in merges:
#             if not line.startswith("#") and line.strip():
#                 tokens = line.strip().split()
#                 merge_tokens.update(tokens)

#         # Find missing tokens
#         vocab_tokens = set(vocab.keys())
#         missing_tokens = merge_tokens - vocab_tokens

#         # Add missing tokens to vocab.json
#         current_index = max(vocab.values()) + 1
#         for token in missing_tokens:
#             if token not in vocab:
#                 vocab[token] = current_index
#                 current_index += 1

#         # Write updated vocab.json back to file
#         with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
#             json.dump(vocab, vocab_file, ensure_ascii=False, indent=2)

#         print(f"Added {len(missing_tokens)} missing tokens to vocab.json.")

#     except Exception as e:
#         print(f"Error updating vocab.json: {e}")

# if __name__ == "__main__":
#     vocab_path = "./converted_model/vocab.json"
#     merges_path = "./converted_model/merges.txt"
#     sync_vocab_with_merges(vocab_path, merges_path)
