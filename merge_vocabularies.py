# merge vocabularies
# (this util merges two different vocabulary files from encoder.json)
import json
import requests
import os
import argparse

def download_file(url, dest_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        f.write(response.content)

def merge_encoders(original_encoder_path, new_encoder_url, output_encoder_path):
    with open(original_encoder_path, 'r', encoding='utf-8') as original_file:
        original_encoder = json.load(original_file)

    new_encoder_path = 'new_encoder.json'
    download_file(new_encoder_url, new_encoder_path)

    with open(new_encoder_path, 'r', encoding='utf-8') as new_file:
        new_encoder = json.load(new_file)

    # Merge the two encoders
    merged_encoder = {**new_encoder, **original_encoder}

    with open(output_encoder_path, 'w', encoding='utf-8') as output_file:
        json.dump(merged_encoder, output_file, ensure_ascii=False, indent=2)
    
    os.remove(new_encoder_path)

def merge_bpe_files(original_bpe_path, new_bpe_url, output_bpe_path):
    with open(original_bpe_path, 'r', encoding='utf-8') as original_file:
        original_bpe = original_file.readlines()

    new_bpe_path = 'new_bpe.txt'
    download_file(new_bpe_url, new_bpe_path)

    with open(new_bpe_path, 'r', encoding='utf-8') as new_file:
        new_bpe = new_file.readlines()

    # Merge the two BPE lists, removing duplicates
    merged_bpe = list(dict.fromkeys(original_bpe + new_bpe))

    with open(output_bpe_path, 'w', encoding='utf-8') as output_file:
        output_file.writelines(merged_bpe)
    
    os.remove(new_bpe_path)

def main():
    parser = argparse.ArgumentParser(description="Merge vocabulary files.")
    parser.add_argument("original_encoder_path", type=str, help="Path to the original encoder.json")
    parser.add_argument("original_bpe_path", type=str, help="Path to the original vocab.bpe")
    parser.add_argument("output_encoder_path", type=str, help="Path to save the merged encoder.json")
    parser.add_argument("output_bpe_path", type=str, help="Path to save the merged vocab.bpe")

    args = parser.parse_args()

    new_encoder_url = 'https://huggingface.co/Finnish-NLP/gpt2-finnish/raw/main/vocab.json'
    new_bpe_url = 'https://huggingface.co/Finnish-NLP/gpt2-finnish/raw/main/merges.txt'

    merge_encoders(args.original_encoder_path, new_encoder_url, args.output_encoder_path)
    merge_bpe_files(args.original_bpe_path, new_bpe_url, args.output_bpe_path)

if __name__ == "__main__":
    main()
