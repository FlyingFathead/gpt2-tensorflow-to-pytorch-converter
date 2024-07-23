#!/usr/bin/env python3

# convert_model.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://www.github.com/FlyingFathead/gpt2-tensorflow-to-pytorch-converter
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (Ghostcode via ChaosWhisperer)

import os
import re
import sys
import shutil
import tensorflow as tf
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import json
import numpy as np

def get_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.match(r'model-\d+\.index', f)]
    if not checkpoint_files:
        raise ValueError("No checkpoint files found in the directory.")
    
    checkpoint_numbers = [int(re.search(r'model-(\d+)\.index', f).group(1)) for f in checkpoint_files]
    latest_checkpoint = f"model-{max(checkpoint_numbers)}"
    
    return os.path.join(checkpoint_dir, latest_checkpoint)

def load_tf_weights_in_gpt2(model, checkpoint_path, config):
    reader = tf.train.load_checkpoint(checkpoint_path)
    var_list = reader.get_variable_to_shape_map()

    # Mapping TensorFlow variable names to PyTorch attribute names
    name_map = {
        'model/wte': 'transformer.wte.weight',
        'model/wpe': 'transformer.wpe.weight',
        'model/ln_f/g': 'transformer.ln_f.weight',
        'model/ln_f/b': 'transformer.ln_f.bias',
    }

    # Layers mapping
    for layer in range(config.n_layer):
        name_map[f'model/h{layer}/ln_1/g'] = f'transformer.h.{layer}.ln_1.weight'
        name_map[f'model/h{layer}/ln_1/b'] = f'transformer.h.{layer}.ln_1.bias'
        name_map[f'model/h{layer}/ln_2/g'] = f'transformer.h.{layer}.ln_2.weight'
        name_map[f'model/h{layer}/ln_2/b'] = f'transformer.h.{layer}.ln_2.bias'
        name_map[f'model/h{layer}/attn/c_attn/w'] = f'transformer.h.{layer}.attn.c_attn.weight'
        name_map[f'model/h{layer}/attn/c_attn/b'] = f'transformer.h.{layer}.attn.c_attn.bias'
        name_map[f'model/h{layer}/attn/c_proj/w'] = f'transformer.h.{layer}.attn.c_proj.weight'
        name_map[f'model/h{layer}/attn/c_proj/b'] = f'transformer.h.{layer}.attn.c_proj.bias'
        name_map[f'model/h{layer}/mlp/c_fc/w'] = f'transformer.h.{layer}.mlp.c_fc.weight'
        name_map[f'model/h{layer}/mlp/c_fc/b'] = f'transformer.h.{layer}.mlp.c_fc.bias'
        name_map[f'model/h{layer}/mlp/c_proj/w'] = f'transformer.h.{layer}.mlp.c_proj.weight'
        name_map[f'model/h{layer}/mlp/c_proj/b'] = f'transformer.h.{layer}.mlp.c_proj.bias'

    for name in var_list:
        tensor = reader.get_tensor(name)
        if name in name_map:
            mapped_name = name_map[name]
        else:
            print(f"Skipping {name}: no mapping found")
            continue

        # Special handling for kernels/weights
        if 'kernel' in mapped_name or 'weight' in mapped_name:
            tensor = np.transpose(tensor)

        # Now set the tensor in the PyTorch model
        try:
            pointer = model
            for m_name in mapped_name.split('.'):
                pointer = getattr(pointer, m_name)

            # Ensure tensor is compatible with PyTorch by using reshape
            tensor = torch.tensor(tensor, dtype=pointer.data.dtype).reshape(pointer.shape)
            pointer.data = tensor.contiguous()

            print(f"Successfully converted: {mapped_name}")
        except AttributeError as e:
            print(f"Skipping {mapped_name}: {e}")

def convert_bpe_to_vocab(bpe_path, vocab_path, merges_path):
    with open(bpe_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Filter out comment lines
    lines = [line for line in lines if not line.startswith("#")]

    # Create vocab.json
    vocab = {}
    index = 0
    for line in lines:
        token = line.split()[0]
        if token not in vocab:
            vocab[token] = index
            index += 1

    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # Create merges.txt
    with open(merges_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def copy_tokenizer_files(checkpoint_dir, dest_dir):
    bpe_path = os.path.join(checkpoint_dir, 'vocab.bpe')
    vocab_path = os.path.join(dest_dir, 'vocab.json')
    merges_path = os.path.join(dest_dir, 'merges.txt')
    
    convert_bpe_to_vocab(bpe_path, vocab_path, merges_path)
    
    tokenizer_config = {
        "model_max_length": 1024,
        "padding_side": "right",
        "special_tokens_map_file": None,
        "tokenizer_class": "GPT2Tokenizer",
        "use_fast": False
    }
    with open(os.path.join(dest_dir, 'tokenizer_config.json'), 'w') as f:
        json.dump(tokenizer_config, f)

def main(checkpoint_dir="."):
    # Construct the absolute path to hparams.json
    hparams_path = os.path.join(checkpoint_dir, 'hparams.json')

    # Load configuration
    with open(hparams_path) as f:
        hparams = json.load(f)

    config = GPT2Config(
        vocab_size=hparams['n_vocab'],
        n_positions=hparams['n_ctx'],
        n_ctx=hparams['n_ctx'],
        n_embd=hparams['n_embd'],
        n_layer=hparams['n_layer'],
        n_head=hparams['n_head']
    )

    # Initialize PyTorch model
    model = GPT2LMHeadModel(config)

    # Get the latest TensorFlow checkpoint
    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    print(f"Using checkpoint: {checkpoint_path}")

    # Load TensorFlow checkpoint
    load_tf_weights_in_gpt2(model, checkpoint_path, config)

    # Save PyTorch model
    dest_dir = "./converted_model"
    model.save_pretrained(dest_dir)

    # Copy tokenizer files
    copy_tokenizer_files(checkpoint_dir, dest_dir)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
    else:
        checkpoint_dir = "."
    main(checkpoint_dir)
