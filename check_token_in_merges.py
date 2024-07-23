# check_token_in_merges.py

def check_token_in_merges(merges_path, token):
    try:
        with open(merges_path, 'r', encoding='utf-8') as merges_file:
            for line in merges_file:
                if token in line:
                    print(f"Token {token} found in cleaned_merges.txt.")
                    return
        print(f"Token {token} not found in cleaned_merges.txt.")
    except Exception as e:
        print(f"Error checking for token in merges.txt: {e}")

if __name__ == "__main__":
    merges_path = "./converted_model/cleaned_merges.txt"
    token = "Ã¥"
    check_token_in_merges(merges_path, token)
