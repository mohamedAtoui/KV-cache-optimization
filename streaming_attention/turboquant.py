"""TurboQuant: Two-stage online vector quantization for KV cache compression.

Implements the TurboQuant algorithm (Google, ICLR 2026, arxiv:2504.19874):
  Stage 1 — Random rotation + per-vector normalization + optimal scalar quantization
  Stage 2 — QJL residual correction (+1 bit/coord, unbiased inner products)

Key properties:
  - Online / data-oblivious: no calibration, tokens quantized as they arrive
  - Minimal per-vector overhead: one scale scalar + one residual norm per vector
  - 3-bit KV cache with ~0 accuracy loss (paper result)

The rotation makes coordinates approximately i.i.d., enabling a universal
scalar quantizer. Per-vector RMS normalization adapts to data magnitude
before applying the fixed codebook.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


# Module-level cache for Lloyd's-optimized codebooks. Keyed by (bits, seed).
# Lloyd's on 1M samples × 100 iterations takes tens of seconds on shared CI
# runners; caching makes the second caller pay nothing. Entries are small
# CPU float32 tensors (<256 bytes each) so memory cost is negligible.
_CODEBOOK_CACHE: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant quantization."""

    head_dim: int = 128
    bits_stage1: int = 2       # scalar quantizer bits (2 or 3)
    qjl_enabled: bool = True   # +1 bit for QJL residual correction
    seed: int = 42


class TurboQuantState:
    """Pre-computed global matrices and codebooks for TurboQuant.

    These are created once per model and shared across all layers/heads.
    Total memory: ~128 KB for head_dim=128.
    """

    def __init__(
        self,
        rotation_matrix: Tensor,
        qjl_matrix: Tensor,
        codebooks: dict[int, Tensor],
        boundaries: dict[int, Tensor],
    ):
        self.rotation_matrix = rotation_matrix  # [D, D] orthogonal
        self.qjl_matrix = qjl_matrix            # [D, D] random normal
        self.codebooks = codebooks               # {bits: Tensor[2^bits]}
        self.boundaries = boundaries             # {bits: Tensor[2^bits - 1]}


def _compute_codebook(
    bits: int,
    n_samples: int = 100_000,
    max_iter: int = 30,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Compute optimal 1D scalar quantizer via Lloyd's algorithm.

    After random rotation and per-vector RMS normalization, each coordinate
    is approximately N(0, 1). We compute optimal centroids and decision
    boundaries for this standard normal distribution.

    Results are cached at module level keyed by (bits, seed), so repeated
    calls with the same arguments return immediately. Lloyd's on a 1D
    Gaussian with ≤16 levels converges within ~10 iterations, and 100k
    samples gives per-centroid standard error around 3e-3 — well below
    the precision any downstream consumer cares about.

    Args:
        bits: Number of quantization bits (2, 3, or 4).
        n_samples: Monte Carlo samples for Lloyd's algorithm.
        max_iter: Maximum Lloyd iterations.
        seed: Random seed for sample generation.

    Returns:
        (centroids, boundaries): centroids shape [2^bits], boundaries shape [2^bits - 1].
        Both sorted in ascending order, CPU float32.
    """
    cache_key = (bits, seed)
    if cache_key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[cache_key]

    n_levels = 2 ** bits
    rng = torch.Generator().manual_seed(seed)
    samples = torch.randn(n_samples, generator=rng)  # N(0, 1)

    # Initialize centroids uniformly across the data range
    lo, hi = samples.min().item(), samples.max().item()
    centroids = torch.linspace(lo, hi, n_levels)

    for _ in range(max_iter):
        # Assign each sample to nearest centroid
        dists = (samples.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)

        # Update centroids as mean of assigned samples
        new_centroids = torch.zeros_like(centroids)
        for i in range(n_levels):
            mask = assignments == i
            if mask.any():
                new_centroids[i] = samples[mask].mean()
            else:
                new_centroids[i] = centroids[i]

        if (new_centroids - centroids).abs().max() < 1e-7:
            break
        centroids = new_centroids

    centroids, _ = centroids.sort()

    # Decision boundaries: midpoints between consecutive centroids
    boundaries = (centroids[:-1] + centroids[1:]) / 2

    _CODEBOOK_CACHE[cache_key] = (centroids, boundaries)
    return centroids, boundaries


def init_turboquant_state(
    config: TurboQuantConfig,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> TurboQuantState:
    """Initialize TurboQuant global state.

    Creates:
      - Rotation matrix Pi via QR decomposition of random normal matrix
      - QJL projection matrix S (random normal, not orthogonalized)
      - Optimal scalar codebooks for bits 2, 3, 4 (for N(0,1) distribution)

    Args:
        config: TurboQuant configuration.
        device: Target device.
        dtype: Target dtype for matrices.

    Returns:
        TurboQuantState with all pre-computed data.
    """
    d = config.head_dim

    # Rotation matrix Pi: QR decomposition of random normal -> orthogonal
    rng = torch.Generator().manual_seed(config.seed)
    random_matrix = torch.randn(d, d, generator=rng)
    rotation_matrix, _ = torch.linalg.qr(random_matrix)
    rotation_matrix = rotation_matrix.to(device=device, dtype=dtype)

    # QJL projection matrix S: i.i.d. N(0, 1)
    rng_qjl = torch.Generator().manual_seed(config.seed + 1)
    qjl_matrix = torch.randn(d, d, generator=rng_qjl)
    qjl_matrix = qjl_matrix.to(device=device, dtype=dtype)

    # Pre-compute codebooks for bits 2, 3, 4 (standard normal distribution)
    codebooks = {}
    boundaries = {}
    for b in (2, 3, 4):
        cb, bd = _compute_codebook(b, seed=config.seed + 10 + b)
        codebooks[b] = cb.to(device=device, dtype=dtype)
        boundaries[b] = bd.to(device=device, dtype=dtype)

    return TurboQuantState(
        rotation_matrix=rotation_matrix,
        qjl_matrix=qjl_matrix,
        codebooks=codebooks,
        boundaries=boundaries,
    )


def turboquant_quantize(
    x: Tensor,
    state: TurboQuantState,
    config: TurboQuantConfig,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
    """Two-stage TurboQuant quantization.

    Steps:
      1. Rotate: x_rot = x @ Pi^T (makes coordinates independent)
      2. Per-vector RMS normalization (adapts to data scale)
      3. Scalar quantize normalized coordinates using pre-computed codebook
      4. (Optional) QJL on residual for unbiased inner products

    Args:
        x: Input tensor of shape [..., head_dim].
        state: Pre-computed TurboQuant state.
        config: TurboQuant configuration.

    Returns:
        (codes, signs, residual_norms, scales):
          - codes: [..., head_dim] int8 codebook indices
          - signs: [..., head_dim] int8 sign bits (+1/-1), or None if QJL disabled
          - residual_norms: [...] float32 residual L2 norms, or None if QJL disabled
          - scales: [...] float32 per-vector RMS scale factors
    """
    original_dtype = x.dtype
    x_float = x.float()

    # Stage 1: Random rotation
    x_rot = x_float @ state.rotation_matrix.float().T  # [..., D]

    # Per-vector RMS normalization: after rotation, coordinates are ~i.i.d.
    # RMS captures the per-vector scale so the codebook (designed for N(0,1)) matches
    scales = x_rot.pow(2).mean(dim=-1).sqrt().clamp(min=1e-8)  # [...]
    x_norm = x_rot / scales.unsqueeze(-1)  # [..., D], approximately N(0, 1) per coord

    # Scalar quantization on normalized coordinates
    b = config.bits_stage1
    bd = state.boundaries[b].float()
    cb = state.codebooks[b].float()

    codes = torch.bucketize(x_norm, bd)  # [..., D], values in [0, 2^b - 1]

    if not config.qjl_enabled:
        return codes.to(torch.int8), None, None, scales.to(original_dtype)

    # Stage 2: QJL residual correction (in normalized space)
    recon_norm = cb[codes]  # [..., D] in normalized space
    residual = x_norm - recon_norm  # [..., D]
    residual_norms = residual.norm(dim=-1)  # [...]

    # QJL: sign(residual @ S^T)
    projected = residual @ state.qjl_matrix.float().T  # [..., D]
    signs = projected.sign().to(torch.int8)
    signs[signs == 0] = 1  # replace zeros with +1

    return (
        codes.to(torch.int8),
        signs,
        residual_norms.to(original_dtype),
        scales.to(original_dtype),
    )


def turboquant_dequantize(
    codes: Tensor,
    signs: Tensor | None,
    residual_norms: Tensor | None,
    scales: Tensor,
    state: TurboQuantState,
    config: TurboQuantConfig,
) -> Tensor:
    """Dequantize TurboQuant-compressed vectors.

    Args:
        codes: [..., head_dim] int8 codebook indices.
        signs: [..., head_dim] int8 QJL sign bits, or None.
        residual_norms: [...] float32 residual norms, or None.
        scales: [...] float32 per-vector RMS scale factors.
        state: Pre-computed TurboQuant state.
        config: TurboQuant configuration.

    Returns:
        Reconstructed tensor of shape [..., head_dim].
    """
    b = config.bits_stage1
    cb = state.codebooks[b].float()

    # Stage 1: look up centroids (in normalized space)
    recon = cb[codes.long()]  # [..., D]

    # Stage 2: QJL residual correction (in normalized space)
    if config.qjl_enabled and signs is not None and residual_norms is not None:
        d = config.head_dim
        qjl_correction = (
            math.sqrt(math.pi / 2) / d
        ) * (signs.float() @ state.qjl_matrix.float())  # [..., D]
        recon = recon + residual_norms.unsqueeze(-1).float() * qjl_correction

    # Rescale back to original magnitude
    recon = recon * scales.unsqueeze(-1).float()

    # Inverse rotation: result = recon @ Pi
    result = recon @ state.rotation_matrix.float()

    return result


def turboquant_attention_scores(
    q: Tensor,
    codes: Tensor,
    signs: Tensor | None,
    residual_norms: Tensor | None,
    scales: Tensor,
    state: TurboQuantState,
    config: TurboQuantConfig,
) -> Tensor:
    """Compute attention logits q^T k using TurboQuant's unbiased estimator.

    Instead of dequantizing k then computing q @ k^T (high MSE), this
    computes the inner product directly from quantized codes. The QJL
    correction provides an unbiased estimate of each q^T k entry.

    Args:
        q: [B, H, T_q, D] query vectors (full precision, post-RoPE).
        codes: [B, H, T_k, D] int8 codebook indices for keys.
        signs: [B, H, T_k, D] int8 QJL sign bits, or None.
        residual_norms: [B, H, T_k] float residual L2 norms, or None.
        scales: [B, H, T_k] float per-vector RMS scale factors.
        state: Pre-computed TurboQuant state.
        config: TurboQuant configuration.

    Returns:
        [B, H, T_q, T_k] attention logits (unbiased estimate of q^T k).
    """
    b = config.bits_stage1
    cb = state.codebooks[b].float()
    Pi = state.rotation_matrix.float()

    # Pre-rotate queries into the rotated space: q_rot = q @ Pi^T
    q_rot = q.float() @ Pi.T  # [B, H, T_q, D]

    # Stage 1: logits from codebook reconstruction (in rotated space)
    # k_rot_approx = cb[codes] * scale
    recon_rot = cb[codes.long()] * scales.unsqueeze(-1).float()  # [B, H, T_k, D]
    logits = torch.matmul(q_rot, recon_rot.transpose(-2, -1))  # [B, H, T_q, T_k]

    # Stage 2: QJL correction for unbiased inner products
    if config.qjl_enabled and signs is not None and residual_norms is not None:
        d = config.head_dim
        S = state.qjl_matrix.float()

        # Project queries through QJL matrix: [B, H, T_q, D]
        q_proj = q_rot @ S.T

        # signs scaled by (scale * norms): [B, H, T_k, D] * [B, H, T_k, 1]
        scale_norms = (scales * residual_norms).float()  # [B, H, T_k]
        # correction = (sqrt(pi/2)/d) * q_proj @ signs^T * scale_norms
        correction = (math.sqrt(math.pi / 2) / d) * torch.matmul(
            q_proj, signs.float().transpose(-2, -1)
        )  # [B, H, T_q, T_k]
        correction = correction * scale_norms.unsqueeze(-2)  # broadcast over T_q

        logits = logits + correction

    return logits.to(q.dtype)


def simulate_turboquant(
    x: Tensor,
    state: TurboQuantState,
    config: TurboQuantConfig,
) -> Tensor:
    """Round-trip TurboQuant quantization for noise simulation.

    Quantizes and immediately dequantizes, returning a tensor of the
    same shape and dtype as input. Used in forward hooks to inject
    realistic quantization noise.

    Args:
        x: Input tensor of shape [..., head_dim].
        state: Pre-computed TurboQuant state.
        config: TurboQuant configuration.

    Returns:
        Reconstructed tensor, same shape and dtype as x.
    """
    original_dtype = x.dtype
    codes, signs, norms, scales = turboquant_quantize(x, state, config)
    result = turboquant_dequantize(codes, signs, norms, scales, state, config)
    return result.to(original_dtype)
