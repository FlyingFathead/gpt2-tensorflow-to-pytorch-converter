#!/usr/bin/env python3

from transformers import GPT2Tokenizer
import torch

def test_tokenizer(tokenizer_dir):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir)
        
        # Add a padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        print("Tokenizer loaded successfully.")
        
        prompt = "Hei, miten voit?"
        print(f"Prompt: {prompt}")
        
        # Tokenize the prompt without return_tensors initially
        inputs = tokenizer(prompt, padding=True, truncation=True, max_length=512)
        print(f"Tokenized inputs without return_tensors: {inputs}")
        
        # Now tokenize with return_tensors="pt"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        print(f"Tokenized inputs with return_tensors='pt': {inputs}")
        
        # Check the contents of the inputs
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        print(f"Input IDs: {input_ids}")
        print(f"Attention Mask: {attention_mask}")

        # Check the types of input_ids and attention_mask
        print(f"Input IDs Type: {type(input_ids)}")
        print(f"Attention Mask Type: {type(attention_mask)}")

        # Check the shapes of input_ids and attention_mask
        print(f"Input IDs Shape: {input_ids.shape}")
        print(f"Attention Mask Shape: {attention_mask.shape}")

        # Decode the tokenized inputs
        decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"Decoded text: {decoded_text}")

    except Exception as e:
        print(f"Error in tokenizer test: {e}")

if __name__ == "__main__":
    tokenizer_dir = "./converted_model"  # Path to your tokenizer files directory
    test_tokenizer(tokenizer_dir)
