"""
GPTNeo Decoder-Only Transformer with Multi-Head Latent Attention (MLA)

Based on:
- "DeepSeek-V2: A Strong, Economical, and Efficient MoE LM" (DeepSeek-AI, 2024)
- GPT architecture for causal language modeling

Key difference from other variants:
- Uses MultiHeadLatentAttention with low-rank KV compression
- RoPE is built into MLA (no separate position embeddings)
- Smaller KV-cache than MHA while maintaining quality

Architecture:
    - Decoder-only (no encoder)
    - Causal multi-head latent attention throughout
    - Token embeddings only (RoPE inside attention)
    - Stack of N transformer blocks
    - Language modeling head for next-token prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .attention import MultiHeadLatentAttention
from AttentionHeads.mha.attention import create_causal_mask
from AttentionHeads.mha.layers import LayerNorm, PositionwiseFeedForward


class GPTNeoBlock(nn.Module):
    """
    Single GPTNeo Decoder Block with Multi-Head Latent Attention

    Uses pre-normalization (LayerNorm before sub-layers) for better training stability.

    Args:
        hidden_size: Model dimension (d_model)
        num_heads: Number of attention heads
        d_c: Latent compression dimension
        d_rope: RoPE dimension
        intermediate_size: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(self, hidden_size, num_heads, d_c, d_rope, intermediate_size,
                 max_seq_len=256, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Pre-normalization layers
        self.ln_1 = LayerNorm(hidden_size)
        self.ln_2 = LayerNorm(hidden_size)

        # Multi-Head Latent Attention
        self.attn = MultiHeadLatentAttention(
            h=num_heads,
            d_model=hidden_size,
            d_c=d_c,
            d_rope=d_rope,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        # Position-wise feed-forward network
        self.ffn = PositionwiseFeedForward(hidden_size, intermediate_size, dropout)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass through decoder block

        Args:
            x: (batch, seq_len, hidden_size)
            mask: Optional causal attention mask

        Returns:
            output: (batch, seq_len, hidden_size)
        """
        # Self-attention with pre-norm and residual
        attn_input = self.ln_1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, mask)
        x = x + self.dropout(attn_output)

        # Feed-forward with pre-norm and residual
        ffn_input = self.ln_2(x)
        ffn_output = self.ffn(ffn_input)
        x = x + self.dropout(ffn_output)

        return x


class GPTNeoModel(nn.Module):
    """
    GPTNeo Decoder-Only Transformer Model with Multi-Head Latent Attention

    No position embeddings - RoPE is built into MLA.

    Args:
        vocab_size: Size of vocabulary
        hidden_size: Model dimension
        num_layers: Number of decoder blocks
        num_heads: Number of attention heads
        d_c: Latent compression dimension
        d_rope: RoPE dimension
        intermediate_size: Feed-forward hidden dimension
        max_position_embeddings: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        d_c=128,
        d_rope=16,
        intermediate_size=1024,
        max_position_embeddings=256,
        dropout=0.2,
        layer_norm_epsilon=1e-5
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_c = d_c
        self.d_rope = d_rope
        self.max_position_embeddings = max_position_embeddings

        # Token embeddings only (RoPE inside attention - no position embedding)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        # Dropout after embeddings
        self.dropout = nn.Dropout(dropout)

        # Stack of decoder blocks
        self.blocks = nn.ModuleList([
            GPTNeoBlock(
                hidden_size, num_heads, d_c, d_rope,
                intermediate_size, max_position_embeddings, dropout
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.ln_f = LayerNorm(hidden_size, eps=layer_norm_epsilon)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with normal distribution"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.b_2.data.zero_()
            module.a_2.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through GPTNeo model

        Args:
            input_ids: (batch, seq_len) - Input token IDs
            attention_mask: Optional attention mask

        Returns:
            hidden_states: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Token embeddings only (no position embeddings - RoPE is inside attention)
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.dropout(hidden_states)

        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = create_causal_mask(seq_len, device)

        # Pass through all decoder blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        # Final layer normalization
        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class GPTNeoForCausalLM(nn.Module):
    """
    GPTNeo Model with Language Modeling Head and Multi-Head Latent Attention

    Full model for causal language modeling (next-token prediction).

    Args:
        vocab_size: Size of vocabulary
        hidden_size: Model dimension
        num_layers: Number of decoder blocks
        num_heads: Number of attention heads
        d_c: Latent compression dimension
        d_rope: RoPE dimension
        intermediate_size: Feed-forward dimension
        max_position_embeddings: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        d_c=128,
        d_rope=16,
        intermediate_size=1024,
        max_position_embeddings=256,
        dropout=0.2,
        layer_norm_epsilon=1e-5
    ):
        super().__init__()

        # Core GPTNeo model with MLA
        self.transformer = GPTNeoModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            d_c=d_c,
            d_rope=d_rope,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            layer_norm_epsilon=layer_norm_epsilon
        )

        # Language modeling head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.transformer.token_embedding.weight

        # Store config
        self.config = {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'd_c': d_c,
            'd_rope': d_rope,
            'intermediate_size': intermediate_size,
            'max_position_embeddings': max_position_embeddings,
            'dropout': dropout,
            'layer_norm_epsilon': layer_norm_epsilon,
            'position_embedding_type': 'rope'
        }

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for language modeling

        Args:
            input_ids: (batch, seq_len) - Input token IDs
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation

        Returns:
            if labels is None: logits
            else: (loss, logits)
        """
        hidden_states = self.transformer(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            return loss, logits

        return logits

    def generate(
        self,
        input_ids,
        max_length=100,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.0
    ):
        """
        Generate text autoregressively

        Args:
            input_ids: (batch, seq_len) - Input prompt token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Keep only top k tokens for sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens

        Returns:
            generated_ids: (batch, max_length) - Generated token IDs
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        generated = input_ids

        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    for i in range(batch_size):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        next_token_logits[i, indices_to_remove] = float('-inf')

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

                if generated.size(1) >= self.config['max_position_embeddings']:
                    break

        return generated

    def get_num_params(self, non_embedding=False):
        """
        Count number of parameters in the model

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            # MLA has no position_embedding (RoPE is inside attention)
            n_params -= self.transformer.token_embedding.weight.numel()

        return n_params


def create_gptneo_model(config):
    """
    Factory function to create GPTNeo model with MLA from config dict

    Args:
        config: Dictionary with model configuration

    Returns:
        GPTNeoForCausalLM model instance with MLA
    """
    return GPTNeoForCausalLM(
        vocab_size=config['vocab_size'],
        hidden_size=config.get('hidden_size', 256),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 8),
        d_c=config.get('d_c', 128),
        d_rope=config.get('d_rope', 16),
        intermediate_size=config.get('intermediate_size', 1024),
        max_position_embeddings=config.get('max_position_embeddings', 256),
        dropout=config.get('dropout', 0.2),
        layer_norm_epsilon=config.get('layer_norm_epsilon', 1e-5)
    )


# Alias for compatibility
GPTNeo = GPTNeoForCausalLM


if __name__ == "__main__":
    print("Testing GPTNeo Decoder-Only Architecture with MLA...")
    print("=" * 70)

    batch_size = 2
    seq_len = 128
    vocab_size = 50257
    hidden_size = 256
    num_layers = 4
    num_heads = 8
    d_c = 128
    d_rope = 16
    intermediate_size = 1024
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Configuration: batch={batch_size}, seq_len={seq_len}")
    print(f"Model: hidden={hidden_size}, layers={num_layers}, heads={num_heads}")
    print(f"MLA: d_c={d_c}, d_rope={d_rope}\n")

    # Test 1: GPTNeoBlock
    print("1. Testing GPTNeoBlock with MLA...")
    block = GPTNeoBlock(hidden_size, num_heads, d_c, d_rope, intermediate_size,
                        max_seq_len=256, dropout=0.1).to(device)
    x = torch.randn(batch_size, seq_len, hidden_size).to(device)
    mask = create_causal_mask(seq_len, device)
    output = block(x, mask)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == x.shape
    print("   GPTNeoBlock with MLA working correctly")

    # Test 2: Full model
    print("\n2. Testing GPTNeoForCausalLM with MLA...")
    model = GPTNeoForCausalLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        d_c=d_c,
        d_rope=d_rope,
        intermediate_size=intermediate_size,
        max_position_embeddings=256,
        dropout=0.1
    ).to(device)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    logits = model(input_ids)
    print(f"   Input IDs shape: {input_ids.shape}")
    print(f"   Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print("   GPTNeoForCausalLM with MLA working correctly")

    # Test 3: Loss computation
    print("\n3. Testing Loss Computation...")
    labels = input_ids.clone()
    loss, logits = model(input_ids, labels=labels)
    print(f"   Loss: {loss.item():.4f}")
    assert loss.item() > 0
    print("   Loss computation working correctly")

    # Test 4: Parameter counting
    print("\n4. Parameter Counting...")
    total_params = model.get_num_params()
    non_embed_params = model.get_num_params(non_embedding=True)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Non-embedding parameters: {non_embed_params:,}")
    print(f"   Embedding parameters: {total_params - non_embed_params:,}")
    print("   Parameter counting working correctly")

    # Test 5: Text generation
    print("\n5. Testing Text Generation...")
    prompt = torch.randint(0, vocab_size, (1, 10)).to(device)
    generated = model.generate(prompt, max_length=50, temperature=1.0, top_k=50)
    print(f"   Prompt shape: {prompt.shape}")
    print(f"   Generated shape: {generated.shape}")
    assert generated.size(1) <= 50
    print("   Text generation working correctly")

    # Test 6: Factory function
    print("\n6. Testing create_gptneo_model() factory...")
    config = {
        'vocab_size': vocab_size,
        'hidden_size': 256,
        'num_layers': 4,
        'num_heads': 8,
        'd_c': 128,
        'd_rope': 16,
        'intermediate_size': 1024,
        'max_position_embeddings': 256,
        'dropout': 0.1
    }
    factory_model = create_gptneo_model(config).to(device)
    test_input = torch.randint(0, vocab_size, (1, 32)).to(device)
    test_output = factory_model(test_input)
    print(f"   Output shape: {test_output.shape}")
    print(f"   Parameters: {factory_model.get_num_params():,}")
    print("   Factory function working correctly")

    print("\n" + "=" * 70)
    print("All GPTNeo with MLA tests passed!")
    print("MLA architecture ready for training!")
