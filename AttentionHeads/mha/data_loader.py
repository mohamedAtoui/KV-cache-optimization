"""
Data loading utilities for WikiText dataset (MHA Transformer)
Loads pre-processed tokenized data from HuggingFace Datasets format
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from transformers import GPT2Tokenizer
import json
import os


class WikiTextDataset(Dataset):
    """
    WikiText Dataset wrapper for PyTorch

    Loads pre-tokenized data with input_ids and attention_mask
    Data shape: input_ids - List[int], attention_mask - List[int]
    """
    def __init__(self, data_path: str, max_seq_length: int = 512):
        """
        Args:
            data_path: Path to the processed dataset directory
            max_seq_length: Maximum sequence length (for truncation/padding)
        """
        self.dataset = load_from_disk(data_path)
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - input_ids: (seq_len,) - Token IDs
                - attention_mask: (seq_len,) - Attention mask (1 for real tokens, 0 for padding)
        """
        item = self.dataset[idx]

        input_ids = item['input_ids']
        attention_mask = item['attention_mask']

        # Truncate if longer than max_seq_length
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]

        # Convert to tensors
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


class WikiTextDataModule:
    """
    Data module for managing WikiText train/val/test dataloaders

    Handles loading, batching, and preparing data for training
    """
    def __init__(self, config):
        """
        Args:
            config: dict or namespace with data configuration
                - train_path: Path to training data
                - val_path: Path to validation data
                - batch_size: Batch size for training
                - max_seq_length: Maximum sequence length
                - tokenizer: Tokenizer name (e.g., "gpt2")
        """
        # Extract config parameters
        if isinstance(config, dict):
            self.train_path = config.get('train_path', '../data_processed/wikitext2_processed/train')
            self.val_path = config.get('val_path', '../data_processed/wikitext2_processed/validation')
            self.batch_size = config.get('batch_size', 32)
            self.max_seq_length = config.get('max_seq_length', 512)
            self.tokenizer_name = config.get('tokenizer', 'gpt2')
        else:
            self.train_path = getattr(config, 'train_path')
            self.val_path = getattr(config, 'val_path')
            self.batch_size = getattr(config, 'batch_size')
            self.max_seq_length = getattr(config, 'max_seq_length')
            self.tokenizer_name = getattr(config, 'tokenizer')

        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer_name)

        # Create datasets
        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        """Load the datasets"""
        print(f"Loading training data from: {self.train_path}")
        self.train_dataset = WikiTextDataset(self.train_path, self.max_seq_length)

        print(f"Loading validation data from: {self.val_path}")
        self.val_dataset = WikiTextDataset(self.val_path, self.max_seq_length)

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Val dataset size: {len(self.val_dataset)}")

    def collate_fn(self, batch):
        """
        Custom collate function for batching sequences of variable length

        Args:
            batch: List of dicts from __getitem__

        Returns:
            dict with batched tensors:
                - input_ids: (batch_size, max_len_in_batch)
                - attention_mask: (batch_size, max_len_in_batch)
                - labels: (batch_size, max_len_in_batch) - Same as input_ids for language modeling
        """
        # Find max length in this batch
        max_len = max(item['input_ids'].size(0) for item in batch)

        # Pad all sequences to max_len
        input_ids_batch = []
        attention_mask_batch = []

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        for item in batch:
            seq_len = item['input_ids'].size(0)
            padding_len = max_len - seq_len

            # Pad input_ids
            padded_input_ids = torch.cat([
                item['input_ids'],
                torch.full((padding_len,), pad_token_id, dtype=torch.long)
            ])

            # Pad attention_mask
            padded_attention_mask = torch.cat([
                item['attention_mask'],
                torch.zeros(padding_len, dtype=torch.long)
            ])

            input_ids_batch.append(padded_input_ids)
            attention_mask_batch.append(padded_attention_mask)

        # Stack into tensors
        input_ids = torch.stack(input_ids_batch)  # (batch_size, max_len)
        attention_mask = torch.stack(attention_mask_batch)  # (batch_size, max_len)

        # For language modeling, labels are the same as input_ids
        # (shifted by 1 position during loss calculation)
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def train_dataloader(self, shuffle=True, num_workers=0):
        """Get training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )

    def val_dataloader(self, num_workers=0):
        """Get validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )


def load_config(config_path):
    """
    Load configuration from JSON file

    Args:
        config_path: Path to config.json

    Returns:
        dict with configuration
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


if __name__ == "__main__":
    # Unit tests for data loading
    print("Testing data loader...")

    # Load config
    config_path = "config.json"
    if os.path.exists(config_path):
        config = load_config(config_path)

        # Combine configs
        data_config = {
            'train_path': config['data_config']['train_path'],
            'val_path': config['data_config']['val_path'],
            'batch_size': config['training_config']['batch_size'],
            'max_seq_length': config['model_config']['max_seq_length'],
            'tokenizer': config['data_config']['tokenizer']
        }

        # Create data module
        print("\n1. Creating WikiTextDataModule...")
        data_module = WikiTextDataModule(data_config)
        data_module.setup()

        # Test train dataloader
        print("\n2. Testing train dataloader...")
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        print(f"   Batch keys: {batch.keys()}")
        print(f"   input_ids shape: {batch['input_ids'].shape}")
        print(f"   attention_mask shape: {batch['attention_mask'].shape}")
        print(f"   labels shape: {batch['labels'].shape}")

        # Test val dataloader
        print("\n3. Testing val dataloader...")
        val_loader = data_module.val_dataloader()
        batch = next(iter(val_loader))
        print(f"   input_ids shape: {batch['input_ids'].shape}")
        print(f"   attention_mask shape: {batch['attention_mask'].shape}")

        # Test tokenizer
        print("\n4. Testing tokenizer...")
        sample_text = "Hello, world!"
        tokens = data_module.tokenizer.encode(sample_text)
        decoded = data_module.tokenizer.decode(tokens)
        print(f"   Original: {sample_text}")
        print(f"   Tokens: {tokens}")
        print(f"   Decoded: {decoded}")

        print("\n✓ All data loader tests passed!")
    else:
        print(f"Config file not found: {config_path}")
        print("Run this test from the mha/ directory")
