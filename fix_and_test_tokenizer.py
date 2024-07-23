#!/usr/bin/env python3
# fix_and_test_tokenizer.py

from transformers import GPT2Tokenizer
import torch

def load_tokenizer(tokenizer_dir):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir)
    # Ensure there is a padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.save_pretrained(tokenizer_dir)
    print("Tokenizer loaded successfully.")
    return tokenizer

def identify_and_fix_problematic_tokens(tokenizer, prompt, tokenizer_dir):
    inputs = tokenizer(prompt, return_tensors=None, padding=True, truncation=True, max_length=512)
    print(f"Tokenized inputs without return_tensors: {inputs}")
    
    prompt_words = prompt.split()
    problematic_tokens = []
    for i, token_id in enumerate(inputs['input_ids']):
        if token_id is None and i < len(prompt_words):
            problematic_tokens.append(prompt_words[i])
            print(f"Problematic token at position {i}: '{prompt_words[i]}'")
    
    if problematic_tokens:
        added_tokens_count = tokenizer.add_tokens(problematic_tokens)
        tokenizer.save_pretrained(tokenizer_dir)
        print(f"Added {added_tokens_count} tokens to the vocabulary.")
        return True
    return False

def test_tokenizer(tokenizer, prompt):
    try:
        # Manually ensure that None values are replaced
        inputs = tokenizer(prompt, return_tensors=None, padding=True, truncation=True)
        inputs['input_ids'] = [id if id is not None else tokenizer.pad_token_id for id in inputs['input_ids']]
        
        # Convert list to tensor manually to ensure correct formatting
        input_ids_tensor = torch.tensor([inputs['input_ids']], dtype=torch.long)
        
        print(f"Tokenized inputs manually converted to tensor: {input_ids_tensor}")
        
        decoded_text = tokenizer.decode(input_ids_tensor[0], skip_special_tokens=True)
        print(f"Decoded text: {decoded_text}")
    except Exception as e:
        print(f"Error in tokenizer test: {e}")

if __name__ == "__main__":
    tokenizer_dir = "./converted_model"
    prompt = "Hei, miten voit?"

    tokenizer = load_tokenizer(tokenizer_dir)
    
    if identify_and_fix_problematic_tokens(tokenizer, prompt, tokenizer_dir):
        tokenizer = load_tokenizer(tokenizer_dir)  # Reload tokenizer after updates
    test_tokenizer(tokenizer, prompt)


# == (old method) ==
# #!/usr/bin/env python3
# # fix_and_test_tokenizer.py

# from transformers import GPT2Tokenizer
# import json

# def load_tokenizer(tokenizer_dir):
#     tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir)
    
#     # Add a padding token if it doesn't exist
#     if tokenizer.pad_token is None:
#         tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#     print("Tokenizer loaded successfully.")
#     return tokenizer

# def identify_problematic_tokens(tokenizer, prompt):
#     inputs = tokenizer(prompt, padding=True, truncation=True, max_length=512, return_tensors=None)
#     print(f"Tokenized inputs without return_tensors: {inputs}")
    
#     problematic_tokens = []
#     prompt_tokens = prompt.split()
#     for i, token_id in enumerate(inputs['input_ids']):
#         if token_id is None:
#             token_position = min(i, len(prompt_tokens) - 1)
#             problematic_token = prompt_tokens[token_position]
#             problematic_tokens.append(problematic_token)
#             print(f"Problematic token at position {i}: '{problematic_token}'")
    
#     return problematic_tokens

# def add_missing_tokens(vocab_path, tokens):
#     try:
#         with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
#             vocab = json.load(vocab_file)
        
#         current_index = max(vocab.values()) + 1

#         for token in tokens:
#             if token not in vocab:
#                 vocab[token] = current_index
#                 print(f"Adding token '{token}' with index {current_index}")
#                 current_index += 1

#         with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
#             json.dump(vocab, vocab_file, ensure_ascii=False, indent=2)
        
#         print(f"Added {len(tokens)} tokens to the vocabulary.")
#     except Exception as e:
#         print(f"Error adding tokens to vocabulary: {e}")

# def test_tokenizer(tokenizer, prompt):
#     try:
#         inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
#         print(f"Tokenized inputs: {inputs}")
        
#         decoded_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
#         print(f"Decoded text: {decoded_text}")

#     except Exception as e:
#         print(f"Error in tokenizer test: {e}")

# if __name__ == "__main__":
#     tokenizer_dir = "./converted_model"  # Path to your tokenizer files directory
#     vocab_path = "./converted_model/vocab.json"
#     prompt = "Hei, miten voit?"

#     tokenizer = load_tokenizer(tokenizer_dir)
#     problematic_tokens = identify_problematic_tokens(tokenizer, prompt)

#     if problematic_tokens:
#         add_missing_tokens(vocab_path, problematic_tokens)
#         tokenizer = load_tokenizer(tokenizer_dir)  # Reload tokenizer after updating vocab
#         test_tokenizer(tokenizer, prompt)
#     else:
#         print("No problematic tokens found.")
#         test_tokenizer(tokenizer, prompt)
