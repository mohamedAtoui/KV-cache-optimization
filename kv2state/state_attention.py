"""Recurrent state update rules for converting KV cache to fixed-size state.

Implements decayed linear attention state: S_t = λ·S_{t-1} + v_t·k_tᵀ
Supports both recurrent (token-by-token) and parallel (chunk-wise prefill) modes.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecayedLinearState(nn.Module):
    """Decayed linear attention state: S_t = λ·S_{t-1} + v_t·k_tᵀ

    During decoding, maintains a fixed-size state matrix S of shape [d_k, d_v]
    that replaces the growing KV cache for a single attention head.

    The decay factor λ controls forgetting — higher λ retains more history.
    Output is computed as: o_t = S_t · q_t / z_t, where z_t tracks normalization.

    Args:
        head_dim: Dimension of keys/queries (d_k).
        decay_init: Initial decay factor λ. Default 0.99.
        learnable_decay: If True, λ is a learnable parameter (per-head scalar).
    """

    def __init__(
        self,
        head_dim: int,
        decay_init: float = 0.99,
        learnable_decay: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim

        # Store decay in log-space for unconstrained optimization, sigmoid to [0, 1]
        if learnable_decay:
            # inverse sigmoid of decay_init
            log_decay = math.log(decay_init / (1.0 - decay_init))
            self.log_decay = nn.Parameter(torch.tensor(log_decay))
        else:
            self.register_buffer("log_decay", torch.tensor(math.log(decay_init / (1.0 - decay_init))))

    @property
    def decay(self) -> torch.Tensor:
        """Current decay factor λ in [0, 1]."""
        return torch.sigmoid(self.log_decay)

    def initial_state(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """Create zero-initialized state and normalization denominator.

        Returns:
            state: [batch_size, head_dim, head_dim] — the state matrix S_0
            z: [batch_size, head_dim] — normalization denominator z_0
        """
        state = torch.zeros(batch_size, self.head_dim, self.head_dim, dtype=dtype, device=device)
        z = torch.zeros(batch_size, self.head_dim, dtype=dtype, device=device)
        return state, z

    def recurrent_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step recurrent update for autoregressive decoding.

        Args:
            q: Query vector [batch_size, head_dim]
            k: Key vector [batch_size, head_dim]
            v: Value vector [batch_size, head_dim]
            state: Previous state S_{t-1} [batch_size, head_dim, head_dim]
            z: Previous normalization z_{t-1} [batch_size, head_dim]

        Returns:
            output: Attention output [batch_size, head_dim]
            new_state: Updated state S_t [batch_size, head_dim, head_dim]
            new_z: Updated normalization z_t [batch_size, head_dim]
        """
        lam = self.decay

        # S_t = λ · S_{t-1} + v_t ⊗ k_t  (outer product)
        # v: [B, d_v], k: [B, d_k] → outer: [B, d_k, d_v]
        outer = torch.einsum("bi,bj->bij", k, v)  # [B, d_k, d_v]
        new_state = lam * state + outer

        # z_t = λ · z_{t-1} + k_t  (key accumulator for normalization)
        new_z = lam * z + k

        # o_t = (S_t · q_t) / (z_t · q_t + ε)
        # S_t · q_t: [B, d_k, d_v] × [B, d_k] → [B, d_v]
        numerator = torch.einsum("bij,bi->bj", new_state, q)  # [B, d_v]
        denominator = torch.einsum("bi,bi->b", new_z, q).unsqueeze(-1)  # [B, 1]
        denominator = denominator.clamp(min=1.0)  # Prevent division by zero/negative

        output = numerator / denominator

        return output, new_state, new_z

    def parallel_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """Chunk-wise parallel forward for prefill.

        Processes the sequence in chunks, using intra-chunk parallel attention
        and inter-chunk recurrent state propagation.

        Args:
            q: Queries [batch_size, seq_len, head_dim]
            k: Keys [batch_size, seq_len, head_dim]
            v: Values [batch_size, seq_len, head_dim]
            chunk_size: Size of each chunk for parallel processing.

        Returns:
            output: Attention output [batch_size, seq_len, head_dim]
            final_state: Final state after processing all tokens [batch_size, head_dim, head_dim]
            final_z: Final normalization [batch_size, head_dim]
        """
        B, T, D = q.shape
        lam = self.decay

        # Pad sequence to multiple of chunk_size
        pad_len = (chunk_size - T % chunk_size) % chunk_size
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        T_padded = q.shape[1]
        num_chunks = T_padded // chunk_size

        # Reshape into chunks: [B, num_chunks, chunk_size, D]
        q_chunks = q.view(B, num_chunks, chunk_size, D)
        k_chunks = k.view(B, num_chunks, chunk_size, D)
        v_chunks = v.view(B, num_chunks, chunk_size, D)

        outputs = []
        state, z = self.initial_state(B, q.dtype, q.device)

        # Pre-compute decay powers for intra-chunk positions
        # decay_powers[i] = λ^(chunk_size - 1 - i) for computing contribution to state
        positions = torch.arange(chunk_size, device=q.device, dtype=q.dtype)

        for chunk_idx in range(num_chunks):
            qc = q_chunks[:, chunk_idx]  # [B, C, D]
            kc = k_chunks[:, chunk_idx]  # [B, C, D]
            vc = v_chunks[:, chunk_idx]  # [B, C, D]

            # Intra-chunk: causal linear attention
            # A[i,j] = λ^(i-j) * (q_i · k_j) for j <= i
            # positions[i] - positions[j] gives i-j (positive for causal j<=i)
            decay_matrix = lam ** (positions.unsqueeze(1) - positions.unsqueeze(0)).clamp(min=0)
            causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=q.dtype))
            decay_matrix = decay_matrix * causal_mask  # [C, C]

            # QK^T with decay: [B, C, C]
            attn_intra = torch.einsum("bic,bjc->bij", qc, kc) * decay_matrix.unsqueeze(0)

            # Intra-chunk output from within-chunk tokens
            intra_out = torch.einsum("bij,bjd->bid", attn_intra, vc)  # [B, C, D]

            # Intra-chunk normalization
            intra_z = attn_intra.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, C, 1]

            # Cross-chunk: contribution from previous state
            # For position i in chunk: state contributes λ^(i+1) * S · q_i
            # (i+1 because position 0 is one step after the previous chunk's last position)
            decay_for_state = lam ** (positions + 1)  # [C]
            cross_out = torch.einsum("bij,bci->bcj", state, qc) * decay_for_state.unsqueeze(0).unsqueeze(-1)
            cross_z = torch.einsum("bi,bci->bc", z, qc) * decay_for_state.unsqueeze(0)

            # Combine
            total_z = (intra_z.squeeze(-1) + cross_z).clamp(min=1.0).unsqueeze(-1)  # [B, C, 1]
            chunk_out = (intra_out + cross_out) / total_z

            outputs.append(chunk_out)

            # Update state for next chunk: S = λ^C · S + Σ_i λ^(C-1-i) · v_i ⊗ k_i
            state_decay = lam ** chunk_size
            state = state_decay * state

            chunk_decay = lam ** (chunk_size - 1 - positions)  # [C]
            # Weighted outer products: [B, D, D]
            weighted_outer = torch.einsum("bci,bcj,c->bij", kc, vc, chunk_decay)
            state = state + weighted_outer

            # Update z
            z = state_decay * z + torch.einsum("bci,c->bi", kc, chunk_decay)

        # Concatenate and trim padding
        output = torch.cat(outputs, dim=1)[:, :T, :]

        return output, state, z


class StateCache:
    """Manages per-head recurrent states for a full model.

    Stores state matrices and normalization denominators for each
    streaming head across all layers.
    """

    def __init__(self):
        self.states: dict[tuple[int, int], torch.Tensor] = {}   # (layer, head) → state
        self.zs: dict[tuple[int, int], torch.Tensor] = {}       # (layer, head) → z

    def get(self, layer_idx: int, head_idx: int):
        """Get state and z for a specific head. Returns None if not initialized."""
        key = (layer_idx, head_idx)
        if key in self.states:
            return self.states[key], self.zs[key]
        return None, None

    def set(self, layer_idx: int, head_idx: int, state: torch.Tensor, z: torch.Tensor):
        """Set state and z for a specific head."""
        key = (layer_idx, head_idx)
        self.states[key] = state
        self.zs[key] = z

    def reset(self):
        """Clear all states."""
        self.states.clear()
        self.zs.clear()

    @property
    def memory_bytes(self) -> int:
        """Total memory used by all states."""
        total = 0
        for s in self.states.values():
            total += s.nelement() * s.element_size()
        for z in self.zs.values():
            total += z.nelement() * z.element_size()
        return total
