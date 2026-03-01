# Models Directory

This directory is reserved for storing trained models and custom model architectures.

## Usage

- **Trained Models**: Model checkpoints are automatically saved to `../checkpoints/mha/` during training
- **Custom Architectures**: You can place alternative model implementations here for experimentation

## Current Setup

The main GPTNeo model implementation is located in `../mha/transformer.py`. Training checkpoints are saved to the `checkpoints/` directory by default.

To change the checkpoint save location, modify the `checkpoint_dir` parameter in your training configuration.
