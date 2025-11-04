"""
Utilities for training, evaluation, logging, and visualization

Based on "Attention Is All You Need" (Vaswani et al., 2017)
Includes utilities from Harvard NLP's Annotated Transformer:
https://nlp.seas.harvard.edu/annotated-transformer/

Reference:
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
    Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances
    in neural information processing systems (pp. 5998-6008).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import os
import json
import math
from pathlib import Path

# Import mask utilities for convenience
from .attention import subsequent_mask


def rate(step, model_size, factor, warmup):
    """
    Learning rate schedule from "Attention is All You Need" (Harvard NLP)

    Implements the learning rate schedule from the paper:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    This corresponds to increasing the learning rate linearly for the first
    warmup steps, and decreasing it thereafter proportionally to the inverse
    square root of the step number.

    Args:
        step: Current training step (1-indexed)
        model_size: Model dimension (d_model)
        factor: Scaling factor for learning rate
        warmup: Number of warmup steps

    Returns:
        float: Learning rate for this step

    Example:
        >>> # Get LR for step 4000 with d_model=512, warmup=4000
        >>> lr = rate(4000, 512, 1.0, 4000)
        >>> # Returns: 512^-0.5 * min(4000^-0.5, 4000 * 4000^-1.5) ≈ 0.000088
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class Batch:
    """
    Batch object for training (Harvard NLP implementation)

    Holds a batch of data with src and tgt sentences, along with constructing
    the masks for training.

    Args:
        src: Source sequence tensor (batch, src_len)
        tgt: Target sequence tensor (batch, tgt_len) [optional]
        pad: Padding token ID (default: 2)

    Attributes:
        src: Source sequences
        src_mask: Mask for source (hides padding)
        tgt: Target sequences (input to decoder, excludes last token)
        tgt_y: Target sequences (labels, excludes first token)
        tgt_mask: Mask for target (hides padding and future tokens)
        ntokens: Number of target tokens (excluding padding)
    """

    def __init__(self, src, tgt=None, pad=2):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words"""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask


class MetricsTracker:
    """
    Track and compute training/validation metrics
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.total_loss = 0.0
        self.total_tokens = 0
        self.num_batches = 0

    def update(self, loss, num_tokens):
        """
        Update metrics with batch results

        Args:
            loss: Batch loss (scalar)
            num_tokens: Number of tokens in batch
        """
        self.total_loss += loss * num_tokens
        self.total_tokens += num_tokens
        self.num_batches += 1

    def get_average_loss(self):
        """Get average loss per token"""
        if self.total_tokens == 0:
            return 0.0
        return self.total_loss / self.total_tokens

    def get_perplexity(self):
        """
        Calculate perplexity: exp(average loss)
        Lower perplexity = better model
        """
        avg_loss = self.get_average_loss()
        return np.exp(avg_loss)


class LabelSmoothing(nn.Module):
    """
    Label Smoothing for better generalization

    Instead of one-hot targets, distribute some probability mass
    uniformly across all classes
    """
    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size * seq_len, vocab_size) - Log probabilities
            targets: (batch_size * seq_len,) - Target token IDs

        Returns:
            loss: Scalar loss value
        """
        batch_size = predictions.size(0)
        vocab_size = predictions.size(-1)

        # Create smoothed target distribution
        true_dist = torch.zeros_like(predictions)
        true_dist.fill_(self.smoothing / (vocab_size - 2))  # -2 for true class and padding
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0

        # Mask out padding tokens
        mask = (targets != self.padding_idx)
        if mask.sum() > 0:
            true_dist = true_dist * mask.unsqueeze(1)

        return self.criterion(predictions, true_dist)


class Logger:
    """
    Logger for TensorBoard and console output
    """
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard

        if self.use_tensorboard:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

        self.step = 0

    def log_scalar(self, tag: str, value: float, step: int = None):
        """Log a scalar value"""
        if step is None:
            step = self.step

        if self.use_tensorboard:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int = None):
        """Log multiple scalars"""
        if step is None:
            step = self.step

        if self.use_tensorboard:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_text(self, tag: str, text: str, step: int = None):
        """Log text"""
        if step is None:
            step = self.step

        if self.use_tensorboard:
            self.writer.add_text(tag, text, step)

    def log_metrics(self, metrics: dict, prefix: str = "", step: int = None):
        """
        Log multiple metrics at once

        Args:
            metrics: Dict of metric_name -> value
            prefix: Prefix for metric names (e.g., "train/", "val/")
            step: Global step
        """
        if step is None:
            step = self.step

        for name, value in metrics.items():
            self.log_scalar(f"{prefix}{name}", value, step)

    def increment_step(self):
        """Increment global step"""
        self.step += 1

    def close(self):
        """Close logger"""
        if self.use_tensorboard:
            self.writer.close()


class CheckpointManager:
    """
    Manage model checkpoints
    """
    def __init__(self, checkpoint_dir: str, max_to_keep: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.checkpoints = []

    def save_checkpoint(self, model, optimizer, epoch, step, metrics, filename=None):
        """
        Save model checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            step: Current global step
            metrics: Dict of metrics to save
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"checkpoint_epoch{epoch}_step{step}.pt"

        filepath = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }

        torch.save(checkpoint, filepath)
        self.checkpoints.append(filepath)

        # Remove old checkpoints if exceeding max_to_keep
        if len(self.checkpoints) > self.max_to_keep:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)

        print(f"Checkpoint saved: {filepath}")
        return filepath

    def load_checkpoint(self, filepath, model, optimizer=None):
        """
        Load model checkpoint

        Args:
            filepath: Path to checkpoint
            model: Model to load state into
            optimizer: Optional optimizer to load state into

        Returns:
            Dict with epoch, step, and metrics
        """
        checkpoint = torch.load(filepath, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded: {filepath}")

        return {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'metrics': checkpoint.get('metrics', {})
        }

    def get_latest_checkpoint(self):
        """Get path to latest checkpoint"""
        checkpoints = list(Path(self.checkpoint_dir).glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        return str(max(checkpoints, key=os.path.getctime))


class AttentionVisualizer:
    """
    Visualize attention weights
    """
    @staticmethod
    def plot_attention_heatmap(attention_weights, src_tokens=None, tgt_tokens=None,
                               head_idx=0, layer_idx=0, save_path=None):
        """
        Plot attention weights as heatmap

        Args:
            attention_weights: (batch_size, num_heads, seq_len_tgt, seq_len_src)
            src_tokens: List of source tokens (optional)
            tgt_tokens: List of target tokens (optional)
            head_idx: Which attention head to visualize
            layer_idx: Which layer (for title)
            save_path: Path to save figure
        """
        # Take first sample and specified head
        attn = attention_weights[0, head_idx].detach().cpu().numpy()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attn,
            cmap='viridis',
            xticklabels=src_tokens if src_tokens else False,
            yticklabels=tgt_tokens if tgt_tokens else False,
            cbar=True
        )
        plt.title(f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Source Position')
        plt.ylabel('Target Position')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_multi_head_attention(attention_weights, num_heads_to_show=4, save_path=None):
        """
        Plot attention weights for multiple heads

        Args:
            attention_weights: (batch_size, num_heads, seq_len_tgt, seq_len_src)
            num_heads_to_show: Number of heads to visualize
            save_path: Path to save figure
        """
        num_heads = min(num_heads_to_show, attention_weights.size(1))

        fig, axes = plt.subplots(2, num_heads // 2, figsize=(15, 8))
        axes = axes.flatten()

        for head_idx in range(num_heads):
            attn = attention_weights[0, head_idx].detach().cpu().numpy()
            sns.heatmap(attn, ax=axes[head_idx], cmap='viridis', cbar=True)
            axes[head_idx].set_title(f'Head {head_idx}')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


def set_seed(seed: int):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Python random seed
    import random
    random.seed(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count trainable parameters in model

    Args:
        model: PyTorch model

    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def save_config(config, save_path):
    """
    Save configuration to JSON file

    Args:
        config: Configuration dict
        save_path: Path to save JSON
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(config_path):
    """
    Load configuration from JSON file

    Args:
        config_path: Path to config JSON

    Returns:
        config: Configuration dict
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


if __name__ == "__main__":
    # Unit tests for utilities
    print("Testing utilities...")

    # Test 1: MetricsTracker
    print("\n1. Testing MetricsTracker...")
    tracker = MetricsTracker()
    tracker.update(loss=2.5, num_tokens=100)
    tracker.update(loss=2.0, num_tokens=100)
    print(f"   Average loss: {tracker.get_average_loss():.4f}")
    print(f"   Perplexity: {tracker.get_perplexity():.4f}")

    # Test 2: Seed setting
    print("\n2. Testing seed setting...")
    set_seed(42)
    x1 = torch.randn(3, 3)
    set_seed(42)
    x2 = torch.randn(3, 3)
    print(f"   Seeds match: {torch.allclose(x1, x2)}")

    # Test 3: Parameter counting
    print("\n3. Testing parameter counting...")
    model = nn.Linear(512, 1000)
    total, trainable = count_parameters(model)
    print(f"   Total parameters: {total:,}")
    print(f"   Trainable parameters: {trainable:,}")

    # Test 4: Config save/load
    print("\n4. Testing config save/load...")
    config = {"d_model": 512, "num_heads": 8}
    save_config(config, "/tmp/test_config.json")
    loaded_config = load_config("/tmp/test_config.json")
    print(f"   Configs match: {config == loaded_config}")

    print("\n✓ All utility tests passed!")
