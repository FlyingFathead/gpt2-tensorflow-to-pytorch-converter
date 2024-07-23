#!/usr/bin/env python3

# test_pytorch_model.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://www.github.com/FlyingFathead/gpt2-tensorflow-to-pytorch-converter
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (Ghostcode via ChaosWhisperer)

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model(model_dir):
    try:
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def generate_text(model, tokenizer, prompt, max_length=50):
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=max_length)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        print(f"Error generating text: {e}")
        return None

if __name__ == "__main__":
    model_dir = "./converted_model"  # Path to your converted model directory
    model, tokenizer = load_model(model_dir)
    
    if model and tokenizer:
        prompt = "Hei, miten voit?"  # Example prompt in Finnish
        print("Prompt:", prompt)
        
        generated_text = generate_text(model, tokenizer, prompt)
        
        if generated_text:
            print("Generated Text:")
            print(generated_text)
        else:
            print("Failed to generate text.")
    else:
        print("Model or tokenizer not loaded. Please check the paths and try again.")
