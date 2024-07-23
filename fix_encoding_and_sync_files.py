# fix_encoding_and_sync_files.py

import json

def fix_encoding_and_create_files(bpe_path, encoder_path, vocab_json_path, merges_txt_path):
    # Correct encoding issue by reading the file correctly
    with open(bpe_path, 'r', encoding='utf-8') as bpe_file:
        bpe_lines = bpe_file.readlines()
    
    # Remove BOM if present
    if bpe_lines[0].startswith('\ufeff'):
        bpe_lines[0] = bpe_lines[0][1:]
    
    # Create vocab.json from encoder.json
    with open(encoder_path, 'r', encoding='utf-8') as encoder_file:
        encoder = json.load(encoder_file)
    
    with open(vocab_json_path, 'w', encoding='utf-8') as vocab_file:
        json.dump(encoder, vocab_file, ensure_ascii=False, indent=2)
    
    # Create merges.txt from corrected bpe file
    with open(merges_txt_path, 'w', encoding='utf-8') as merges_file:
        merges_file.write("#version: 0.2\n")
        for line in bpe_lines:
            if not line.startswith("#"):
                merges_file.write(line)
    
    print(f"Created {vocab_json_path} and {merges_txt_path} with corrected encoding.")

def main():
    bpe_path = "/mnt/data/vocab.bpe"
    encoder_path = "/mnt/data/encoder.json"
    vocab_json_path = "./converted_model/vocab.json"
    merges_txt_path = "./converted_model/merges.txt"
    
    fix_encoding_and_create_files(bpe_path, encoder_path, vocab_json_path, merges_txt_path)

if __name__ == "__main__":
    main()
