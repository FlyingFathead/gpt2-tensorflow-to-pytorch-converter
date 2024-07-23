# check_for_problematic_token.py

import json

def check_for_problematic_token(vocab_path, merges_path, problematic_token):
    try:
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab = json.load(vocab_file)
        
        if problematic_token in vocab:
            print(f"Token {problematic_token} found in vocab.json.")
        else:
            print(f"Token {problematic_token} not found in vocab.json.")
        
        found_in_merges = False
        with open(merges_path, 'r', encoding='utf-8') as merges_file:
            for line in merges_file:
                if problematic_token in line:
                    found_in_merges = True
                    break
        
        if found_in_merges:
            print(f"Token {problematic_token} found in cleaned_merges.txt.")
        else:
            print(f"Token {problematic_token} not found in cleaned_merges.txt.")

    except Exception as e:
        print(f"Error checking for problematic token: {e}")

if __name__ == "__main__":
    problematic_token = "Ã¥"
    vocab_path = "./converted_model/vocab.json"
    merges_path = "./converted_model/cleaned_merges.txt"
    check_for_problematic_token(vocab_path, merges_path, problematic_token)
