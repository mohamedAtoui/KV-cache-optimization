"""
Training Script for GPTNeo Decoder-Only Transformer

A100-optimized training with BFloat16 mixed precision for TinyStories dataset.
Step-based training with cosine learning rate schedule and warmup.

Based on TinyStories paper:
    Eldan, R., & Li, Y. (2023). TinyStories: How Small Can Language Models Be
    and Still Speak Coherent English? arXiv preprint arXiv:2305.07759.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
import sys
import argparse
from tqdm import tqdm
import math
import json

# Import autocast with version compatibility
try:
    # PyTorch >= 1.10
    from torch.cuda.amp import autocast
except ImportError:
    # Older PyTorch versions
    from torch.cuda.amp import autocast as old_autocast
    autocast = old_autocast

# Import local modules
from .transformer import GPTNeoForCausalLM, create_gptneo_model
from .data_loader import TinyStoriesDataModule, load_config
from .utils import (
    MetricsTracker, Logger, CheckpointManager,
    set_seed, count_parameters
)


class GPTNeoTrainer:
    """
    Trainer for GPTNeo Decoder-Only Transformer

    Features:
        - Step-based training (not epoch-based)
        - BFloat16 mixed precision for A100
        - Cosine learning rate schedule with warmup
        - Gradient accumulation support
        - TinyStories dataset
        - Text generation evaluation
    """

    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Check for BFloat16 support
        self.use_bf16 = (
            config['training']['use_bf16'] and
            torch.cuda.is_available() and
            torch.cuda.is_bf16_supported()
        )

        if config['training']['use_bf16'] and not self.use_bf16:
            print("Warning: BFloat16 requested but not supported. Using FP32.")

        print(f"Using device: {self.device}")
        print(f"Mixed precision: {'BFloat16' if self.use_bf16 else 'FP32'}")

        # Set random seed
        set_seed(config['random_seed'])

        # Initialize model
        self.model = self._build_model()
        self.model.to(self.device)

        # Count parameters
        total_params = self.model.get_num_params()
        non_embed_params = self.model.get_num_params(non_embedding=True)
        print(f"\nModel Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Non-embedding: {non_embed_params:,}")
        print(f"  Embedding: {total_params - non_embed_params:,}")

        # Initialize data
        self.data_module = self._build_data_module()
        self.data_module.setup()

        # Initialize optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Initialize GradScaler for mixed precision (only needed for FP16, not BF16)
        # BFloat16 doesn't need gradient scaling
        self.scaler = None  # We're using BF16, not FP16

        # Initialize logger and checkpoint manager
        self.logger = Logger(
            log_dir=config['logging']['log_dir'],
            use_tensorboard=config['logging']['use_tensorboard']
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config['logging']['checkpoint_dir'],
            max_to_keep=config['checkpointing']['save_total_limit']
        )

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.tokens_seen = 0

    def _build_model(self):
        """Build GPTNeo model"""
        model_cfg = self.config['model']
        model = create_gptneo_model(model_cfg)
        return model

    def _build_data_module(self):
        """Build data module for TinyStories"""
        data_cfg = self.config['data']
        train_cfg = self.config['training']

        combined_config = {
            'dataset_name': data_cfg['dataset_name'],
            'tokenizer': data_cfg['tokenizer'],
            'train_samples': train_cfg['train_samples'],
            'val_samples': train_cfg['val_samples'],
            'batch_size': train_cfg['batch_size'],
            'max_seq_length': train_cfg['max_seq_length'],
            'num_workers': data_cfg['num_workers'],
            'pin_memory': data_cfg['pin_memory']
        }
        return TinyStoriesDataModule(combined_config)

    def _build_optimizer(self):
        """Build AdamW optimizer with weight decay"""
        train_cfg = self.config['training']

        # Separate parameters: no weight decay for biases and layer norms
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # No weight decay for biases and layer norm parameters
            if 'bias' in name or 'ln' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = optim.AdamW([
            {'params': decay_params, 'weight_decay': train_cfg['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ],
            lr=train_cfg['learning_rate'],
            betas=(train_cfg['adam_beta1'], train_cfg['adam_beta2']),
            eps=train_cfg['adam_epsilon']
        )

        return optimizer

    def _build_scheduler(self):
        """
        Build learning rate scheduler: Linear warmup + Cosine decay
        """
        train_cfg = self.config['training']
        warmup_steps = train_cfg['warmup_steps']
        max_steps = train_cfg['max_steps']
        min_lr = train_cfg['min_learning_rate']
        max_lr = train_cfg['learning_rate']

        # Linear warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        # Cosine annealing after warmup
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps - warmup_steps,
            eta_min=min_lr
        )

        # Combine warmup and cosine
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        return scheduler

    def train_step(self, batch):
        """Single training step"""
        self.model.train()

        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)

        # Forward pass with mixed precision
        with autocast(enabled=self.use_bf16):
            # Model computes loss internally when labels are provided
            loss, logits = self.model(input_ids, labels=input_ids)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config['training']['gradient_clip']
        )

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, max_steps=None):
        """Evaluate on validation set"""
        self.model.eval()
        tracker = MetricsTracker()

        val_loader = self.data_module.val_dataloader()
        eval_cfg = self.config.get('evaluation', {})

        if max_steps is None:
            max_steps = eval_cfg.get('eval_max_steps', 100)

        pbar = tqdm(val_loader, desc="Evaluating", total=max_steps)

        for step, batch in enumerate(pbar):
            if step >= max_steps:
                break

            input_ids = batch['input_ids'].to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_bf16):
                loss, _ = self.model(input_ids, labels=input_ids)

            # Calculate number of tokens (excluding padding)
            num_tokens = (input_ids != self.data_module.tokenizer.pad_token_id).sum().item()
            tracker.update(loss.item(), num_tokens)

            pbar.set_postfix({'loss': f'{tracker.get_average_loss():.4f}'})

        avg_loss = tracker.get_average_loss()
        perplexity = tracker.get_perplexity()

        return avg_loss, perplexity

    @torch.no_grad()
    def generate_samples(self):
        """Generate text samples for evaluation"""
        self.model.eval()

        eval_cfg = self.config.get('evaluation', {})
        prompts = eval_cfg.get('generation_prompts', ["Once upon a time"])
        max_length = eval_cfg.get('generation_max_length', 100)
        temperature = eval_cfg.get('generation_temperature', 0.8)
        top_k = eval_cfg.get('generation_top_k', 50)
        top_p = eval_cfg.get('generation_top_p', 0.95)

        generated_texts = []
        tokenizer = self.data_module.tokenizer

        for prompt in prompts:
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)

            # Generate
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # Decode
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_texts.append({
                'prompt': prompt,
                'generated': generated_text
            })

        return generated_texts

    def train(self):
        """Main training loop (step-based)"""
        train_cfg = self.config['training']
        log_cfg = self.config['logging']

        max_steps = train_cfg['max_steps']
        log_every = log_cfg['log_every_steps']
        eval_every = log_cfg['eval_every_steps']
        save_every = log_cfg['save_every_steps']

        print(f"\nStarting training for {max_steps} steps...")
        print(f"Effective batch size: {train_cfg['effective_batch_size']}")
        print(f"Gradient accumulation steps: {train_cfg['gradient_accumulation_steps']}")

        train_loader = self.data_module.train_dataloader()
        tracker = MetricsTracker()

        # Create infinite data iterator
        train_iter = iter(train_loader)

        pbar = tqdm(range(max_steps), desc="Training")

        for step in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                # Restart iterator if we run out of data
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Training step
            loss = self.train_step(batch)

            # Update metrics
            batch_tokens = (batch['input_ids'] != self.data_module.tokenizer.pad_token_id).sum().item()
            tracker.update(loss, batch_tokens)
            self.tokens_seen += batch_tokens
            self.global_step += 1

            # Logging
            if (step + 1) % log_every == 0:
                avg_loss = tracker.get_average_loss()
                perplexity = tracker.get_perplexity()
                lr = self.scheduler.get_last_lr()[0]

                self.logger.log_metrics({
                    'loss': avg_loss,
                    'perplexity': perplexity,
                    'learning_rate': lr,
                    'tokens_seen': self.tokens_seen
                }, prefix='train/', step=self.global_step)

                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{perplexity:.2f}',
                    'lr': f'{lr:.2e}'
                })

                # Reset tracker after logging
                tracker = MetricsTracker()

            # Evaluation
            if (step + 1) % eval_every == 0:
                print(f"\n\nEvaluation at step {step + 1}...")
                val_loss, val_ppl = self.evaluate()

                self.logger.log_metrics({
                    'loss': val_loss,
                    'perplexity': val_ppl
                }, prefix='val/', step=self.global_step)

                print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")

                # Generate samples
                print("\nGenerating samples...")
                samples = self.generate_samples()
                for i, sample in enumerate(samples):
                    print(f"\nPrompt: {sample['prompt']}")
                    print(f"Generated: {sample['generated'][:200]}...")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt', {
                        'val_loss': val_loss,
                        'val_ppl': val_ppl
                    })
                    print(f"\n✓ New best model saved! (Val Loss: {val_loss:.4f})")

                print()  # Empty line before resuming training

            # Checkpointing
            if (step + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_step_{step + 1}.pt', {
                    'step': step + 1,
                    'loss': tracker.get_average_loss()
                })

        # Final evaluation
        print("\n\nFinal evaluation...")
        val_loss, val_ppl = self.evaluate()
        print(f"Final Val Loss: {val_loss:.4f}, Final Val PPL: {val_ppl:.2f}")

        # Save final model
        self.save_checkpoint('final_model.pt', {
            'val_loss': val_loss,
            'val_ppl': val_ppl
        })

        print("\n✓ Training completed!")
        self.logger.close()

    def save_checkpoint(self, filename, metrics=None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'tokens_seen': self.tokens_seen,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        if metrics:
            checkpoint['metrics'] = metrics

        save_path = os.path.join(
            self.config['logging']['checkpoint_dir'],
            filename
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.tokens_seen = checkpoint['tokens_seen']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from step {self.global_step}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train GPTNeo on TinyStories')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    config = load_config(config_path)

    # Create trainer
    trainer = GPTNeoTrainer(config)

    # Resume from checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
