import tensorflow as tf

checkpoint_path = "model-154257500"

try:
    reader = tf.train.load_checkpoint(checkpoint_path)
    print(reader.get_variable_to_shape_map())
except Exception as e:
    print(f"Error reading checkpoint: {e}")
