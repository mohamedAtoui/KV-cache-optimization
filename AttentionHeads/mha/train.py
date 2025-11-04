"""
Training script for MHA Transformer
Includes training loop, validation, checkpointing, and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import os
import sys
import argparse
from tqdm import tqdm
import math

# Import local modules (using explicit relative imports)
from .transformer import Transformer
from .data_loader import WikiTextDataModule, load_config
from .utils import (
    MetricsTracker, Logger, CheckpointManager, LabelSmoothing,
    set_seed, count_parameters
)
from .attention import create_combined_mask, create_padding_mask


class Trainer:
    """
    Trainer class for MHA Transformer
    """
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Set random seed
        set_seed(config['random_seed'])

        # Initialize model
        self.model = self._build_model()
        self.model.to(self.device)

        # Count parameters
        total_params, trainable_params = count_parameters(self.model)
        print(f"\nModel Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")

        # Initialize data
        self.data_module = self._build_data_module()
        self.data_module.setup()

        # Initialize optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Initialize loss function
        self.criterion = self._build_criterion()

        # Initialize logger and checkpoint manager
        self.logger = Logger(
            log_dir=config['logging_config']['log_dir'],
            use_tensorboard=config['logging_config']['use_tensorboard']
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config['logging_config']['checkpoint_dir'],
            max_to_keep=5
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _build_model(self):
        """Build transformer model"""
        model_cfg = self.config['model_config']
        pe_cfg = self.config['positional_encoding']

        model = Transformer(
            vocab_size=model_cfg['vocab_size'],
            d_model=model_cfg['d_model'],
            num_heads=model_cfg['num_heads'],
            num_encoder_layers=model_cfg['num_encoder_layers'],
            num_decoder_layers=model_cfg['num_decoder_layers'],
            d_ff=model_cfg['d_ff'],
            max_seq_length=model_cfg['max_seq_length'],
            dropout=model_cfg['dropout'],
            pe_type=pe_cfg['type']
        )
        return model

    def _build_data_module(self):
        """Build data module"""
        data_cfg = self.config['data_config']
        train_cfg = self.config['training_config']
        model_cfg = self.config['model_config']

        combined_config = {
            'train_path': data_cfg['train_path'],
            'val_path': data_cfg['val_path'],
            'batch_size': train_cfg['batch_size'],
            'max_seq_length': model_cfg['max_seq_length'],
            'tokenizer': data_cfg['tokenizer']
        }
        return WikiTextDataModule(combined_config)

    def _build_optimizer(self):
        """Build optimizer"""
        train_cfg = self.config['training_config']

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_cfg['learning_rate'],
            betas=train_cfg['adam_betas'],
            eps=train_cfg['adam_eps']
        )
        return optimizer

    def _build_scheduler(self):
        """
        Build learning rate scheduler with warmup

        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
        """
        train_cfg = self.config['training_config']
        warmup_steps = train_cfg['warmup_steps']
        d_model = self.config['model_config']['d_model']

        def lr_lambda(step):
            if step == 0:
                return 0
            return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

        scheduler = LambdaLR(self.optimizer, lr_lambda)
        return scheduler

    def _build_criterion(self):
        """Build loss criterion with label smoothing"""
        train_cfg = self.config['training_config']
        pad_token_id = self.data_module.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.data_module.tokenizer.eos_token_id

        criterion = LabelSmoothing(
            vocab_size=self.config['model_config']['vocab_size'],
            padding_idx=pad_token_id,
            smoothing=train_cfg['label_smoothing']
        )
        return criterion

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        tracker = MetricsTracker()

        train_loader = self.data_module.train_dataloader()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # For language modeling: use input as both source and target (shifted)
            # Source: tokens 0 to N-1
            # Target: tokens 1 to N (shifted by 1)
            src = input_ids[:, :-1]
            tgt_input = input_ids[:, :-1]
            tgt_output = labels[:, 1:]

            # Create masks
            src_mask = create_padding_mask(src, pad_token_id=0)
            tgt_mask = create_combined_mask(tgt_input, pad_token_id=0, causal=True)

            # Forward pass
            output = self.model(src, tgt_input, src_mask, tgt_mask)

            # Compute loss
            # output: (batch_size, seq_len, vocab_size)
            # tgt_output: (batch_size, seq_len)
            output = output.contiguous().view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)

            # Log probabilities for KL divergence
            log_probs = torch.log_softmax(output, dim=-1)
            loss = self.criterion(log_probs, tgt_output)

            # Count non-padding tokens
            num_tokens = (tgt_output != 0).sum().item()
            loss = loss / num_tokens

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training_config']['gradient_clip']
            )

            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            tracker.update(loss.item(), num_tokens)

            # Logging
            self.global_step += 1
            if self.global_step % self.config['logging_config']['log_every'] == 0:
                avg_loss = tracker.get_average_loss()
                perplexity = tracker.get_perplexity()
                lr = self.scheduler.get_last_lr()[0]

                self.logger.log_metrics({
                    'loss': avg_loss,
                    'perplexity': perplexity,
                    'learning_rate': lr
                }, prefix='train/', step=self.global_step)

                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{perplexity:.2f}',
                    'lr': f'{lr:.2e}'
                })

            # Checkpointing
            if self.global_step % self.config['logging_config']['save_every'] == 0:
                self.save_checkpoint(epoch, tracker.get_average_loss())

        return tracker.get_average_loss(), tracker.get_perplexity()

    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        tracker = MetricsTracker()

        val_loader = self.data_module.val_dataloader()

        for batch in tqdm(val_loader, desc="Validation"):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Prepare source and target
            src = input_ids[:, :-1]
            tgt_input = input_ids[:, :-1]
            tgt_output = labels[:, 1:]

            # Create masks
            src_mask = create_padding_mask(src, pad_token_id=0)
            tgt_mask = create_combined_mask(tgt_input, pad_token_id=0, causal=True)

            # Forward pass
            output = self.model(src, tgt_input, src_mask, tgt_mask)

            # Compute loss
            output = output.contiguous().view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)

            log_probs = torch.log_softmax(output, dim=-1)
            loss = self.criterion(log_probs, tgt_output)

            num_tokens = (tgt_output != 0).sum().item()
            loss = loss / num_tokens

            tracker.update(loss.item(), num_tokens)

        avg_loss = tracker.get_average_loss()
        perplexity = tracker.get_perplexity()

        # Log validation metrics
        self.logger.log_metrics({
            'loss': avg_loss,
            'perplexity': perplexity
        }, prefix='val/', step=self.global_step)

        return avg_loss, perplexity

    def train(self, num_epochs=None):
        """
        Main training loop

        Args:
            num_epochs: Number of epochs to train (overrides config if provided)
        """
        if num_epochs is None:
            num_epochs = self.config['training_config']['num_epochs']

        print(f"\nStarting training for {num_epochs} epochs...")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_ppl = self.train_epoch(epoch)
            print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")

            # Validate
            val_loss, val_ppl = self.validate()
            print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.global_step,
                    {'val_loss': val_loss, 'val_ppl': val_ppl},
                    filename='best_model.pt'
                )
                print(f"✓ New best model saved! (Val Loss: {val_loss:.4f})")

        print("\n✓ Training completed!")
        self.logger.close()

    def save_checkpoint(self, epoch, loss):
        """Save checkpoint"""
        self.checkpoint_manager.save_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            self.global_step,
            {'loss': loss}
        )

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        info = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer
        )
        self.current_epoch = info['epoch']
        self.global_step = info['step']


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train MHA Transformer')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create trainer
    trainer = Trainer(config)

    # Resume from checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
