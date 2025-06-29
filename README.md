# Char-level LSTM Memorizer PoC

[LSTM or Transformer as "malware packer"](https://bednarskiwsieci.pl/en/blog/lstm-or-transformer-as-malware-packer/).

A simple proof-of-concept demonstrating how to **embed any text file** (e.g., source code) into the weights of a character-level LSTM neural network model and then **accurately reconstruct** its contents during inference.

## Features

* Builds a character-level vocabulary + a special Beginning-of-Sequence (BOS) token.
* Trains an LSTM on a single file until overfitting (memorization).
* Generates the entire character sequence starting from the BOS token.
* Compares SHA-256 checksums of the original and generated files.

## Requirements

* Python 3.12+
* PyTorch 2.7.1+
* NumPy 2.3.1+

Use `uv` to manage dependencies. If you don't have `uv` installed, you can install it via pip:

```bash
pip install uv
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/piotrmaciejbednarski/lstm-memorizer.git
   cd lstm-memorizer
   ```
2. Synchronize using `uv`:

   ```bash
   uv sync
   ```

## Example

One of the generated examples can be found in the `example` directory. The model weights `model.pt` were trained on `bubble_sort.py`.

You can run the entire process in encoding mode and then decoding mode to verify that the generated file is identical to the original.

### Training

```bash
uv run main.py train ./examples/bubble_sort.py \
    --epochs 4000 \
    --hidden 32 \
    --layers 2 \
    --lr 1e-3 \
    --weights ./output/model.pt
```

```
Using device: mps
Epoch 1/4000 loss=4.0145
Epoch 500/4000 loss=0.6894
Epoch 1000/4000 loss=0.1471
Epoch 1500/4000 loss=0.0501
Epoch 2000/4000 loss=0.0251
Epoch 2500/4000 loss=0.0140
Epoch 3000/4000 loss=0.0088
Epoch 3500/4000 loss=0.0108
Epoch 4000/4000 loss=0.0053
Model saved to ./output/model.pt
```

### Generating

```bash
uv run main.py generate ./examples/bubble_sort.py \
    --weights ./output/model.pt \
    --output ./output/generated.py \
    --hidden 32 \
    --layers 2
```

```
Generated and saved to ./output/generated.py
Input SHA256:  95a82277b56978eeb1de2dd3d3c0e46f1b7ebd96bfe6a4dbedcd18bb40be1b25
Output SHA256: 95a82277b56978eeb1de2dd3d3c0e46f1b7ebd96bfe6a4dbedcd18bb40be1b25
Hashes match.
```

## Extensions and Ideas

* Swap out LSTM for a Transformer decoder-only model.
* Experiment with different network depths and widths.
* Use mixed precision training.
* Checkpointing and resuming training.
* Scheduler for learning rate.
* Early stopping.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.