# Toy ReLU Project

This project replicating [Superposition, Memorization, and Double Descent](https://transformer-circuits.pub/2023/toy-double-descent/index.html) paper from Anthropic.

## Project Structure

- `train.py` - Main training script
- `models/` - Directory for saved model weights

## Requirements

- Python 3.x
- PyTorch
- NumPy
- tqdm

## Usage

To train the model, run:
```bash
python train.py --T 5
```

## License

MIT 