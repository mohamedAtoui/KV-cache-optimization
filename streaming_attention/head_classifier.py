"""Head classification: load DuoAttention patterns and compute attention entropy."""

import os
import json
import logging
from typing import Optional
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class HeadClassification:
    """Binary mask indicating which heads are retrieval (True) vs streaming (False).

    Attributes:
        mask: Boolean tensor of shape [num_layers, num_kv_heads].
              True = retrieval head (keep full KV cache).
              False = streaming head (convert to recurrent state).
        num_retrieval: Total number of retrieval heads.
        num_streaming: Total number of streaming heads.
        raw_scores: Optional continuous scores before thresholding.
    """
    mask: torch.Tensor  # [num_layers, num_kv_heads], True=retrieval
    num_retrieval: int
    num_streaming: int
    raw_scores: Optional[torch.Tensor] = None

    @property
    def streaming_fraction(self) -> float:
        return self.num_streaming / (self.num_retrieval + self.num_streaming)

    def get_streaming_heads(self) -> list[tuple[int, int]]:
        """Return list of (layer_idx, head_idx) for streaming heads."""
        indices = (~self.mask).nonzero(as_tuple=False)
        return [(int(r[0]), int(r[1])) for r in indices]

    def get_retrieval_heads(self) -> list[tuple[int, int]]:
        """Return list of (layer_idx, head_idx) for retrieval heads."""
        indices = self.mask.nonzero(as_tuple=False)
        return [(int(r[0]), int(r[1])) for r in indices]


def load_duo_attention_patterns(
    pattern_dir: str,
    sparsity: Optional[float] = None,
    threshold: Optional[float] = 0.5,
) -> HeadClassification:
    """Load pre-computed DuoAttention head classification patterns.

    DuoAttention stores patterns as a TSV file `full_attention_heads.tsv` with
    continuous gate values in [0, 1]. Shape: [num_layers, num_kv_heads].
    Higher values indicate retrieval heads that need full KV cache.

    Args:
        pattern_dir: Path to directory containing full_attention_heads.tsv and config.json.
        sparsity: If set, use quantile-based threshold so this fraction of heads are streaming.
                  Overrides `threshold`.
        threshold: Gate value threshold. Heads with score >= threshold are retrieval. Default 0.5.

    Returns:
        HeadClassification with binary mask and metadata.
    """
    tsv_path = os.path.join(pattern_dir, "full_attention_heads.tsv")
    if not os.path.exists(tsv_path):
        # DuoAttention stores patterns in subdirectories like:
        # attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=...-multi_passkey10/
        # Try to find TSV in first subdirectory
        for subdir in sorted(os.listdir(pattern_dir)):
            candidate = os.path.join(pattern_dir, subdir, "full_attention_heads.tsv")
            if os.path.exists(candidate):
                tsv_path = candidate
                logger.info(f"Found pattern in subdirectory: {subdir}")
                break
        else:
            raise FileNotFoundError(
                f"DuoAttention pattern file not found in {pattern_dir} or subdirectories.\n"
                f"Clone the DuoAttention repo and point to attn_patterns/<model_name>/\n"
                f"Expected file: full_attention_heads.tsv"
            )

    raw_scores = np.loadtxt(tsv_path, dtype=float, delimiter="\t")
    raw_scores = np.clip(raw_scores, 0.0, 1.0)

    # Apply sparsity-based or fixed threshold
    if sparsity is not None:
        # Add tiny noise to break ties (same as DuoAttention)
        noisy = raw_scores + np.random.uniform(0, 1e-6, raw_scores.shape)
        threshold = float(np.quantile(noisy, sparsity))
        logger.info(f"Sparsity={sparsity:.2f} → threshold={threshold:.4f}")

    binary_mask = (raw_scores >= threshold).astype(float)

    mask_tensor = torch.tensor(binary_mask, dtype=torch.bool)
    raw_tensor = torch.tensor(raw_scores, dtype=torch.float32)

    num_retrieval = int(mask_tensor.sum().item())
    num_streaming = int((~mask_tensor).sum().item())

    logger.info(
        f"Loaded DuoAttention patterns: {mask_tensor.shape[0]} layers × "
        f"{mask_tensor.shape[1]} KV heads. "
        f"Retrieval: {num_retrieval}, Streaming: {num_streaming} "
        f"({num_streaming / (num_retrieval + num_streaming):.1%} convertible)"
    )

    # Load config if available
    config_path = os.path.join(pattern_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        logger.info(f"Pattern config: {config}")

    return HeadClassification(
        mask=mask_tensor,
        num_retrieval=num_retrieval,
        num_streaming=num_streaming,
        raw_scores=raw_tensor,
    )


@torch.no_grad()
def compute_attention_entropy(
    model: torch.nn.Module,
    dataloader,
    max_batches: int = 50,
    device: str = "cuda",
) -> HeadClassification:
    """Compute per-head attention entropy to classify retrieval vs streaming heads.

    Retrieval heads have low entropy (peaked, sparse attention patterns).
    Streaming heads have high entropy (diffuse, spread-out attention patterns).

    Args:
        model: HuggingFace causal LM (e.g., LlamaForCausalLM).
        dataloader: Yields dicts with 'input_ids' and 'attention_mask'.
        max_batches: Number of batches to process for entropy estimation.
        device: Device to run on.

    Returns:
        HeadClassification with binary mask derived from entropy bimodality.
    """
    model.eval()

    # Determine architecture params
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)

    # Accumulate per-head entropy: [num_layers, num_heads]
    entropy_sum = torch.zeros(num_layers, num_heads, device=device)
    count = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, total=max_batches, desc="Computing entropy")):
        if batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )

        # outputs.attentions: tuple of [batch, num_heads, seq_len, seq_len]
        for layer_idx, attn_weights in enumerate(outputs.attentions):
            # attn_weights: [batch, num_heads, seq_len, seq_len]
            # Mask out padding positions
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_masked = attn_weights * mask

            # Compute entropy per head: H = -sum(p * log(p))
            # Clamp to avoid log(0)
            attn_clamped = attn_masked.clamp(min=1e-10)
            entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # [batch, num_heads, seq_len]

            # Average over batch, sequence positions
            valid_positions = attention_mask.sum(dim=-1, keepdim=True).unsqueeze(1)  # [batch, 1, 1]
            head_entropy = entropy.sum(dim=(0, 2)) / (valid_positions.sum() + 1e-10)  # [num_heads]

            entropy_sum[layer_idx] += head_entropy

        count += 1

    avg_entropy = entropy_sum / max(count, 1)  # [num_layers, num_heads]

    # For GQA models, average entropy across heads in each KV group
    if num_kv_heads != num_heads:
        heads_per_group = num_heads // num_kv_heads
        avg_entropy = avg_entropy.view(num_layers, num_kv_heads, heads_per_group).mean(dim=-1)

    # Classify using Otsu-like thresholding on the entropy distribution
    flat_entropy = avg_entropy.flatten().cpu().numpy()
    threshold = _otsu_threshold(flat_entropy)

    # High entropy = streaming, Low entropy = retrieval
    mask = avg_entropy.cpu() < threshold  # True = retrieval (low entropy)

    num_retrieval = int(mask.sum().item())
    num_streaming = int((~mask).sum().item())

    logger.info(
        f"Entropy-based classification: threshold={threshold:.4f}. "
        f"Retrieval: {num_retrieval}, Streaming: {num_streaming} "
        f"({num_streaming / (num_retrieval + num_streaming):.1%} convertible)"
    )

    return HeadClassification(
        mask=mask,
        num_retrieval=num_retrieval,
        num_streaming=num_streaming,
        raw_scores=avg_entropy.cpu(),
    )


def _otsu_threshold(values: np.ndarray, num_bins: int = 256) -> float:
    """Compute Otsu's threshold to split a bimodal distribution."""
    hist, bin_edges = np.histogram(values, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    if total == 0:
        return float(np.median(values))

    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg

    mean_bg = np.cumsum(hist * bin_centers)
    mean_bg = np.where(weight_bg > 0, mean_bg / weight_bg, 0)

    mean_fg = np.cumsum(hist[::-1] * bin_centers[::-1])[::-1]
    mean_fg = np.where(weight_fg > 0, mean_fg / weight_fg, 0)

    variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    idx = np.argmax(variance_between)
    return float(bin_centers[idx])
