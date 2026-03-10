"""Perplexity evaluation on WikiText-2 and other datasets."""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_samples: Optional[int] = None,
    max_length: int = 2048,
    stride: int = 512,
    batch_size: int = 1,
    device: str = "cuda",
) -> dict:
    """Evaluate perplexity using sliding window approach.

    Uses the standard sliding window method from HuggingFace's perplexity docs:
    processes overlapping windows of `max_length` with `stride` step size,
    only counting loss on non-overlapping tokens.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset config/subset.
        split: Dataset split to evaluate.
        max_samples: Maximum number of text samples to use (None = all).
        max_length: Context window size for each evaluation chunk.
        stride: Step size for sliding window. Smaller = more accurate but slower.
        batch_size: Batch size (typically 1 for perplexity evaluation).
        device: Device to run on.

    Returns:
        Dict with 'perplexity', 'avg_loss', 'num_tokens', 'dataset'.
    """
    from datasets import load_dataset

    model.eval()

    # Load and concatenate dataset text
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Concatenate all text with double newlines as separator
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    logger.info(f"Evaluating perplexity on {dataset_name}/{dataset_config} ({split})")
    logger.info(f"Total tokens: {seq_len}, max_length: {max_length}, stride: {stride}")

    nlls = []
    num_tokens = 0
    prev_end = 0

    pbar = tqdm(
        range(0, seq_len, stride),
        desc=f"Perplexity ({dataset_name})",
    )

    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end  # Only count loss on new tokens

        input_chunk = input_ids[:, begin_loc:end_loc]

        # Create target labels: -100 for context tokens (overlap), real ids for new tokens
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_chunk, labels=target_ids, use_cache=False)

        # Reset recurrent state between sliding windows to prevent state leakage
        if hasattr(model, '_kv2state_cache'):
            model._kv2state_cache.reset()

        neg_log_likelihood = outputs.loss * trg_len  # Un-average the loss

        nlls.append(neg_log_likelihood.item())
        num_tokens += trg_len

        prev_end = end_loc

        if end_loc >= seq_len:
            break

        # Update progress bar with running perplexity
        running_ppl = torch.exp(torch.tensor(sum(nlls) / num_tokens)).item()
        pbar.set_postfix({"ppl": f"{running_ppl:.2f}", "tokens": num_tokens})

    avg_loss = sum(nlls) / num_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    result = {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "num_tokens": num_tokens,
        "dataset": f"{dataset_name}/{dataset_config}",
        "split": split,
        "max_length": max_length,
        "stride": stride,
    }

    logger.info(f"Perplexity: {perplexity:.2f} (avg loss: {avg_loss:.4f}, tokens: {num_tokens})")
    return result
