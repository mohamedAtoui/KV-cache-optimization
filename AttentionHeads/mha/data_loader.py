"""
Data Loading for TinyStories Dataset (GPTNeo Transformer)

Loads TinyStories dataset from HuggingFace with sampling support for faster training.
Optimized for decoder-only causal language modeling.

Dataset:
    Eldan, R., & Li, Y. (2023). TinyStories: How Small Can Language Models Be
    and Still Speak Coherent English? arXiv preprint arXiv:2305.07759.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from transformers import GPT2Tokenizer
import json
import os
import random


class TinyStoriesDataset(Dataset):
    """
    TinyStories Dataset for causal language modeling

    Loads stories from HuggingFace and tokenizes them on-the-fly.
    Supports sampling for faster training iterations.
    """

    def __init__(
        self,
        split='train',
        tokenizer_name='gpt2',
        max_seq_length=512,
        num_samples=None,
        dataset_name='roneneldan/TinyStories'
    ):
        """
        Args:
            split: 'train' or 'validation'
            tokenizer_name: HuggingFace tokenizer name
            max_seq_length: Maximum sequence length
            num_samples: Number of samples to use (None = use all)
            dataset_name: HuggingFace dataset name
        """
        self.max_seq_length = max_seq_length
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)

        # Set pad token to eos token (GPT-2 doesn't have pad token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset from HuggingFace
        print(f"Loading TinyStories dataset (split={split})...")
        self.dataset = load_dataset(dataset_name, split=split)

        # Sample if requested
        if num_samples is not None and num_samples < len(self.dataset):
            print(f"Sampling {num_samples} from {len(self.dataset)} examples...")
            indices = random.sample(range(len(self.dataset)), num_samples)
            self.dataset = self.dataset.select(indices)

        print(f"Dataset size: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - input_ids: (seq_len,) - Token IDs
                - attention_mask: (seq_len,) - Attention mask
        """
        # Get story text
        story = self.dataset[idx]['text']

        # Tokenize with truncation
        encoding = self.tokenizer(
            story,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt',
            padding=False
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


class TinyStoriesDataModule:
    """
    Data module for managing TinyStories train/val dataloaders

    Features:
        - Automatic dataset loading from HuggingFace
        - Sampling support (e.g., 100K train, 5K val)
        - Dynamic padding for efficient batching
        - Decoder-only causal language modeling setup
    """

    def __init__(self, config):
        """
        Args:
            config: dict with data configuration
                - dataset_name: HuggingFace dataset name
                - tokenizer: Tokenizer name
                - train_samples: Number of training samples (None = all)
                - val_samples: Number of validation samples (None = all)
                - batch_size: Batch size
                - max_seq_length: Maximum sequence length
                - num_workers: Number of data loading workers
                - pin_memory: Pin memory for faster GPU transfer
        """
        self.dataset_name = config.get('dataset_name', 'roneneldan/TinyStories')
        self.tokenizer_name = config.get('tokenizer', 'gpt2')
        self.train_samples = config.get('train_samples', None)
        self.val_samples = config.get('val_samples', None)
        self.batch_size = config.get('batch_size', 32)
        self.max_seq_length = config.get('max_seq_length', 512)
        self.num_workers = config.get('num_workers', 4)
        self.pin_memory = config.get('pin_memory', True)

        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Datasets will be created in setup()
        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        """Load and prepare datasets"""
        print("\n" + "=" * 70)
        print("Setting up TinyStories datasets...")
        print("=" * 70)

        # Create train dataset
        self.train_dataset = TinyStoriesDataset(
            split='train',
            tokenizer_name=self.tokenizer_name,
            max_seq_length=self.max_seq_length,
            num_samples=self.train_samples,
            dataset_name=self.dataset_name
        )

        # Create validation dataset
        self.val_dataset = TinyStoriesDataset(
            split='validation',
            tokenizer_name=self.tokenizer_name,
            max_seq_length=self.max_seq_length,
            num_samples=self.val_samples,
            dataset_name=self.dataset_name
        )

        print(f"\nDataset Summary:")
        print(f"  Train samples: {len(self.train_dataset):,}")
        print(f"  Val samples: {len(self.val_dataset):,}")
        print(f"  Max sequence length: {self.max_seq_length}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Vocab size: {len(self.tokenizer)}")
        print("=" * 70 + "\n")

    def collate_fn(self, batch):
        """
        Dynamic padding collate function

        Pads sequences to the maximum length in the current batch
        (more efficient than padding to max_seq_length).

        Args:
            batch: List of dicts from __getitem__

        Returns:
            dict with batched tensors:
                - input_ids: (batch_size, max_len_in_batch)
                - attention_mask: (batch_size, max_len_in_batch)
        """
        # Find max length in this batch
        max_len = max(item['input_ids'].size(0) for item in batch)

        # Pad all sequences to max_len
        input_ids_batch = []
        attention_mask_batch = []

        pad_token_id = self.tokenizer.pad_token_id

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
        input_ids = torch.stack(input_ids_batch)
        attention_mask = torch.stack(attention_mask_batch)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def train_dataloader(self, shuffle=True):
        """Get training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        """Get validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
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
    # Unit tests for TinyStories data loading
    print("Testing TinyStories Data Loader...")
    print("=" * 70)

    # Test configuration
    test_config = {
        'dataset_name': 'roneneldan/TinyStories',
        'tokenizer': 'gpt2',
        'train_samples': 1000,  # Small sample for testing
        'val_samples': 100,
        'batch_size': 4,
        'max_seq_length': 256,
        'num_workers': 0,  # Single process for testing
        'pin_memory': False
    }

    # Test 1: Create data module
    print("\n1. Creating TinyStoriesDataModule...")
    data_module = TinyStoriesDataModule(test_config)
    data_module.setup()
    print("✓ Data module created")

    # Test 2: Test train dataloader
    print("\n2. Testing train dataloader...")
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   attention_mask shape: {batch['attention_mask'].shape}")
    print(f"   input_ids dtype: {batch['input_ids'].dtype}")
    print(f"   Sample input_ids (first 10): {batch['input_ids'][0, :10].tolist()}")
    print("✓ Train dataloader working")

    # Test 3: Test val dataloader
    print("\n3. Testing val dataloader...")
    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   attention_mask shape: {batch['attention_mask'].shape}")
    print("✓ Val dataloader working")

    # Test 4: Test tokenizer
    print("\n4. Testing tokenizer...")
    sample_story = "Once upon a time, there was a little girl who loved to play."
    tokens = data_module.tokenizer.encode(sample_story)
    decoded = data_module.tokenizer.decode(tokens)
    print(f"   Original: {sample_story}")
    print(f"   Tokens: {tokens}")
    print(f"   Decoded: {decoded}")
    print(f"   Vocab size: {len(data_module.tokenizer)}")
    print(f"   Pad token ID: {data_module.tokenizer.pad_token_id}")
    print("✓ Tokenizer working")

    # Test 5: Verify padding
    print("\n5. Testing dynamic padding...")
    # Get a few batches to see varying lengths
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        print(f"   Batch {i+1} - Max length in batch: {batch['input_ids'].shape[1]}")
    print("✓ Dynamic padding working")

    # Test 6: Decode a sample story
    print("\n6. Decoding a sample story...")
    batch = next(iter(train_loader))
    sample_ids = batch['input_ids'][0]
    # Remove padding
    mask = batch['attention_mask'][0]
    sample_ids = sample_ids[mask == 1]
    decoded_story = data_module.tokenizer.decode(sample_ids)
    print(f"   Story length: {len(sample_ids)} tokens")
    print(f"   Story: {decoded_story[:200]}...")
    print("✓ Story decoding working")

    # Test 7: Check data statistics
    print("\n7. Dataset statistics...")
    lengths = []
    for i, item in enumerate(data_module.train_dataset):
        if i >= 100:  # Check first 100 samples
            break
        lengths.append(len(item['input_ids']))

    print(f"   Samples checked: {len(lengths)}")
    print(f"   Min length: {min(lengths)} tokens")
    print(f"   Max length: {max(lengths)} tokens")
    print(f"   Avg length: {sum(lengths) / len(lengths):.1f} tokens")
    print("✓ Statistics computed")

    print("\n" + "=" * 70)
    print("✓ All TinyStories data loader tests passed!")
    print("=" * 70)
