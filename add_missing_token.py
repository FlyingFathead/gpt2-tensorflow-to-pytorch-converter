# add_missing_token.py

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

if __name__ == "__main__":
    merges_path = "./converted_model/cleaned_merges.txt"
    vocab_path = "./converted_model/vocab.json"

    merges_tokens = extract_tokens_from_merges(merges_path)
    vocab_tokens, vocab = extract_tokens_from_vocab(vocab_path)

    print(f"Extracted tokens from merges.txt: {list(merges_tokens)[:10]} ...")
    print(f"Extracted tokens from vocab.json: {list(vocab_tokens)[:10]} ...")

    add_missing_tokens(vocab_path, merges_tokens, vocab_tokens, vocab)

# import json

# def add_missing_token(vocab_path, token, index):
#     try:
#         with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
#             vocab = json.load(vocab_file)
        
#         if token not in vocab:
#             vocab[token] = index
#             with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
#                 json.dump(vocab, vocab_file, ensure_ascii=False, indent=2)
#             print(f"Added token {token} with index {index} to vocab.json.")
#         else:
#             print(f"Token {token} already exists in vocab.json.")
#     except Exception as e:
#         print(f"Error adding token to vocab.json: {e}")

# if __name__ == "__main__":
#     vocab_path = "./converted_model/vocab.json"
#     missing_token = "Ã¥"
#     index = max(json.load(open(vocab_path)).values()) + 1
#     add_missing_token(vocab_path, missing_token, index)
