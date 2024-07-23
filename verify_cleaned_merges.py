# verify_cleaned_merges.py

def verify_cleaned_merges(cleaned_merges_path):
    try:
        with open(cleaned_merges_path, 'r', encoding='utf-8') as merges_file:
            merges = merges_file.readlines()
        
        print("First 10 entries of cleaned_merges.txt:")
        for i, line in enumerate(merges):
            print(f"{i}: {line.strip()}")
            if i >= 9:
                break

    except Exception as e:
        print(f"Error reading cleaned_merges.txt: {e}")

if __name__ == "__main__":
    cleaned_merges_path = "./converted_model/cleaned_merges.txt"
    verify_cleaned_merges(cleaned_merges_path)