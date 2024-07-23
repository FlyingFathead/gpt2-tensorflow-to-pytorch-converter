#!/usr/bin/env python3

from transformers import GPT2Tokenizer

def test_tokenizer(tokenizer_dir):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir)
        print("Tokenizer loaded successfully.")
        
        prompt = "Hei, miten voit?"
        print(f"Prompt: {prompt}")
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        print(f"Tokenized inputs: {inputs}")
        
        # Decode the tokenized inputs
        decoded_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        print(f"Decoded text: {decoded_text}")
        
    except Exception as e:
        print(f"Error in tokenizer test: {e}")

if __name__ == "__main__":
    tokenizer_dir = "./converted_model"  # Path to your tokenizer files directory
    test_tokenizer(tokenizer_dir)
