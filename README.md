# gpt2-tensorflow-to-pytorch-converter

This repository contains a quick-to-use script to convert GPT-2 models from TensorFlow to PyTorch model format.

## Usage

1. Collect all your TensorFlow model files into a singular directory, i.e. these files:

    ```
    model-<number>.meta
    vocab.bpe
    model-<number>.data-00000-of-00001
    model-<number>.index
    checkpoint
    counter
    encoder.json
    hparams.json
    ```

2. Run the script:
    ```bash
    python convert_model.py /path/to/your/model/files
    ```
3. The converted PyTorch model will be saved in the `./converted_model` directory.

## Notes

Have fun, I probably won't be updating this one much.

## License

This project is licensed under the MIT License.

## Contribute

All code improvements are welcome. This should at least work on all TF1.x-based GPT-2 architecture models.

## About
- Flying from the mind of [FlyingFathead](https://github.com/FlyingFathead/)
- Digital ghost code by ChaosWhisperer
