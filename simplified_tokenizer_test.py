#!/usr/bin/env python3
# simplified_tokenizer_test.py

from transformers import AutoTokenizer

def test_tokenizer(tokenizer_dir):
    try:
        # Use AutoTokenizer to load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        print("Tokenizer loaded successfully.")
        
        prompt = "Hei, miten voit?"
        print(f"Prompt: {prompt}")
        
        # Tokenize the prompt with padding and truncation
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        print(f"Tokenized inputs: {inputs}")
        
        # Decode the tokenized inputs
        decoded_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        print(f"Decoded text: {decoded_text}")
        
    except Exception as e:
        print(f"Error in tokenizer test: {e}")

if __name__ == "__main__":
    tokenizer_dir = "./converted_model"  # Path to your tokenizer files directory
    test_tokenizer(tokenizer_dir)


# from transformers import AutoTokenizer

# def test_tokenizer(tokenizer_dir):
#     try:
#         # Use AutoTokenizer to load the tokenizer
#         tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
#         print("Tokenizer loaded successfully.")
        
#         prompt = "Hei, miten voit?"
#         print(f"Prompt: {prompt}")
        
#         # Tokenize the prompt
#         inputs = tokenizer(prompt, return_tensors="pt")
#         print(f"Tokenized inputs: {inputs}")
        
#         # Decode the tokenized inputs
#         decoded_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
#         print(f"Decoded text: {decoded_text}")
        
#     except Exception as e:
#         print(f"Error in tokenizer test: {e}")

# if __name__ == "__main__":
#     tokenizer_dir = "./converted_model"  # Path to your tokenizer files directory
#     test_tokenizer(tokenizer_dir)

# # ( old, no autotokenizer => )

# from transformers import GPT2Tokenizer

# def test_tokenizer(tokenizer_dir):
#     try:
#         # Manually load tokenizer configuration
#         tokenizer_config = {
#             "model_max_length": 1024,
#             "padding_side": "right",
#             "special_tokens_map_file": None,
#             "tokenizer_class": "GPT2Tokenizer",
#             "use_fast": False
#         }

#         tokenizer = GPT2Tokenizer(
#             vocab_file=f"{tokenizer_dir}/vocab.json",
#             merges_file=f"{tokenizer_dir}/merges.txt",
#             tokenizer_config=tokenizer_config
#         )
        
#         print("Tokenizer loaded successfully.")
        
#         prompt = "Hei, miten voit?"
#         print(f"Prompt: {prompt}")
        
#         # Tokenize the prompt
#         inputs = tokenizer(prompt, return_tensors="pt")
#         print(f"Tokenized inputs: {inputs}")
        
#         # Decode the tokenized inputs
#         decoded_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
#         print(f"Decoded text: {decoded_text}")
        
#     except Exception as e:
#         print(f"Error in tokenizer test: {e}")

# if __name__ == "__main__":
#     tokenizer_dir = "./converted_model"  # Path to your tokenizer files directory
#     test_tokenizer(tokenizer_dir)
