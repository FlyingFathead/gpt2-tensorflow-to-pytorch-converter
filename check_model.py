#!/usr/bin/env python3
# check_model.py

import os
import re
import sys
import tensorflow as tf

def get_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.match(r'model-\d+\.index', f)]
    if not checkpoint_files:
        raise ValueError("No checkpoint files found in the directory.")
    
    checkpoint_numbers = [int(re.search(r'model-(\d+)\.index', f).group(1)) for f in checkpoint_files]
    latest_checkpoint = f"model-{max(checkpoint_numbers)}"
    
    return os.path.join(checkpoint_dir, latest_checkpoint)

def main(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint(".")
    
    try:
        reader = tf.train.load_checkpoint(checkpoint_path)
        print(reader.get_variable_to_shape_map())
    except Exception as e:
        print(f"Error reading checkpoint: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = None
    main(checkpoint_path)
    