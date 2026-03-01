"""
GPTNeo Decoder-Only Transformer with Grouped Query Attention (GQA)

Based on:
- "GQA: Training Generalized Multi-Query Transformer Models" (Ainslie et al., 2023)
- GPT architecture for causal language modeling

Key difference from mha/transformer.py and mqa/transformer.py:
- Uses GroupedQueryAttention with configurable num_kv_heads
- When num_kv_heads=1: equivalent to MQA
- When num_kv_heads=num_heads: equivalent to MHA
- When 1 < num_kv_heads < num_heads: true GQA

Architecture:
    - Decoder-only (no encoder)
    - Causal grouped query attention throughout
    - Token + Positional embeddings
    - Stack of N transformer blocks
    - Language modeling head for next-token prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .attention import GroupedQueryAttention
from AttentionHeads.mha.attention import create_causal_mask
from AttentionHeads.mha.layers import LayerNorm, PositionwiseFeedForward


class GPTNeoBlock(nn.Module):
    """
    Single GPTNeo Decoder Block with Grouped Query Attention

    A decoder block consisting of:
        1. Layer Normalization
        2. Causal Grouped Query Self-Attention (GQA)
        3. Residual connection
        4. Layer Normalization
        5. Position-wise Feed-Forward Network
        6. Residual connection

    Uses pre-normalization (LayerNorm before sub-layers) for better training stability.

    Args:
        hidden_size: Model dimension (d_model)
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads (groups of query heads share KV)
        intermediate_size: Feed-forward hidden dimension
        dropout: Dropout probability

    Shape:
        - Input: (batch, seq_len, hidden_size)
        - Output: (batch, seq_len, hidden_size)
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, intermediate_size, dropout=0.1,
                 position_embedding_type="learned", max_seq_len=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        # Pre-normalization layers
        self.ln_1 = LayerNorm(hidden_size)
        self.ln_2 = LayerNorm(hidden_size)

        # Causal grouped query attention (GQA)
        self.attn = GroupedQueryAttention(
            num_heads, hidden_size, num_kv_heads, dropout,
            position_embedding_type=position_embedding_type,
            max_seq_len=max_seq_len
        )

        # Position-wise feed-forward network
        self.ffn = PositionwiseFeedForward(hidden_size, intermediate_size, dropout)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass through decoder block

        Args:
            x: (batch, seq_len, hidden_size) - Input tensor
            mask: (batch, seq_len, seq_len) - Optional causal attention mask

        Returns:
            output: (batch, seq_len, hidden_size) - Block output
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
    GPTNeo Decoder-Only Transformer Model with Grouped Query Attention

    Stack of N decoder blocks with token and positional embeddings.
    Core model without the language modeling head.

    Args:
        vocab_size: Size of vocabulary
        hidden_size: Model dimension
        num_layers: Number of decoder blocks
        num_heads: Number of query heads per block
        num_kv_heads: Number of KV heads per block (groups of query heads share KV)
        intermediate_size: Feed-forward hidden dimension
        max_position_embeddings: Maximum sequence length
        dropout: Dropout probability

    Shape:
        - Input: (batch, seq_len) - Token IDs
        - Output: (batch, seq_len, hidden_size) - Hidden states
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_layers=8,
        num_heads=12,
        num_kv_heads=4,
        intermediate_size=3072,
        max_position_embeddings=512,
        dropout=0.2,
        layer_norm_epsilon=1e-5,
        position_embedding_type="learned"
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        # Learned positional embeddings (like GPT-2) - only when not using RoPE
        if position_embedding_type != "rope":
            self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)

        # Dropout after embeddings
        self.dropout = nn.Dropout(dropout)

        # Stack of decoder blocks
        self.blocks = nn.ModuleList([
            GPTNeoBlock(hidden_size, num_heads, num_kv_heads, intermediate_size, dropout,
                        position_embedding_type=position_embedding_type,
                        max_seq_len=max_position_embeddings)
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
            attention_mask: (batch, seq_len, seq_len) - Optional attention mask

        Returns:
            hidden_states: (batch, seq_len, hidden_size) - Final hidden states
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)

        if self.position_embedding_type == "rope":
            hidden_states = token_embeds
        else:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
            position_embeds = self.position_embedding(position_ids)
            hidden_states = token_embeds + position_embeds

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
    GPTNeo Model with Language Modeling Head and Grouped Query Attention

    Full model for causal language modeling (next-token prediction).
    Adds a linear projection layer to convert hidden states to vocabulary logits.

    Args:
        vocab_size: Size of vocabulary
        hidden_size: Model dimension (default: 768)
        num_layers: Number of decoder blocks (default: 8)
        num_heads: Number of query heads (default: 12)
        num_kv_heads: Number of KV heads (default: 4)
        intermediate_size: Feed-forward dimension (default: 3072)
        max_position_embeddings: Maximum sequence length (default: 512)
        dropout: Dropout probability (default: 0.2)

    Shape:
        - Input: (batch, seq_len) - Token IDs
        - Output: (batch, seq_len, vocab_size) - Logits for next token

    Example:
        >>> model = GPTNeoForCausalLM(vocab_size=50257, num_kv_heads=2)
        >>> input_ids = torch.randint(0, 50257, (2, 128))
        >>> logits = model(input_ids)
        >>> print(logits.shape)  # torch.Size([2, 128, 50257])
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_layers=8,
        num_heads=12,
        num_kv_heads=4,
        intermediate_size=3072,
        max_position_embeddings=512,
        dropout=0.2,
        layer_norm_epsilon=1e-5,
        position_embedding_type="learned"
    ):
        super().__init__()

        # Core GPTNeo model with GQA
        self.transformer = GPTNeoModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            position_embedding_type=position_embedding_type
        )

        # Language modeling head (projects hidden states to vocabulary)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie weights between token embeddings and lm_head (standard practice)
        self.lm_head.weight = self.transformer.token_embedding.weight

        # Store config for easy access
        self.config = {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'num_kv_heads': num_kv_heads,
            'intermediate_size': intermediate_size,
            'max_position_embeddings': max_position_embeddings,
            'dropout': dropout,
            'layer_norm_epsilon': layer_norm_epsilon,
            'position_embedding_type': position_embedding_type
        }

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for language modeling

        Args:
            input_ids: (batch, seq_len) - Input token IDs
            attention_mask: (batch, seq_len, seq_len) - Optional attention mask
            labels: (batch, seq_len) - Optional labels for loss computation

        Returns:
            if labels is None:
                logits: (batch, seq_len, vocab_size) - Next token logits
            else:
                (loss, logits): Loss and logits tuple
        """
        # Get hidden states from transformer
        hidden_states = self.transformer(input_ids, attention_mask)

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross entropy
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
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling (0 = no filtering)
            top_p: Nucleus sampling threshold (1.0 = no filtering)
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)

        Returns:
            generated_ids: (batch, max_length) - Generated token IDs
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        generated = input_ids

        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Get logits for next token
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

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    for i in range(batch_size):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        next_token_logits[i, indices_to_remove] = float('-inf')

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Check if max position embeddings exceeded
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
            if hasattr(self.transformer, 'position_embedding'):
                n_params -= self.transformer.position_embedding.weight.numel()
            n_params -= self.transformer.token_embedding.weight.numel()

        return n_params


def create_gptneo_model(config):
    """
    Factory function to create GPTNeo model with GQA from config dict

    Args:
        config: Dictionary with model configuration
            - vocab_size: Vocabulary size (required)
            - hidden_size: Model dimension (default: 768)
            - num_layers: Number of layers (default: 8)
            - num_heads: Number of query heads (default: 12)
            - num_kv_heads: Number of KV heads (default: 4)
            - intermediate_size: FFN dimension (default: 3072)
            - max_position_embeddings: Max sequence length (default: 512)
            - dropout: Dropout probability (default: 0.2)

    Returns:
        GPTNeoForCausalLM model instance with GQA

    Example:
        >>> config = {'vocab_size': 50257, 'hidden_size': 768, 'num_kv_heads': 2}
        >>> model = create_gptneo_model(config)
    """
    return GPTNeoForCausalLM(
        vocab_size=config['vocab_size'],
        hidden_size=config.get('hidden_size', 768),
        num_layers=config.get('num_layers', 8),
        num_heads=config.get('num_heads', 12),
        num_kv_heads=config.get('num_kv_heads', 4),
        intermediate_size=config.get('intermediate_size', 3072),
        max_position_embeddings=config.get('max_position_embeddings', 512),
        dropout=config.get('dropout', 0.2),
        layer_norm_epsilon=config.get('layer_norm_epsilon', 1e-5),
        position_embedding_type=config.get('position_embedding_type', 'learned')
    )


# Alias for compatibility
GPTNeo = GPTNeoForCausalLM


if __name__ == "__main__":
    # Unit tests for GPTNeo with GQA
    print("Testing GPTNeo Decoder-Only Architecture with GQA...")
    print("=" * 70)

    batch_size = 2
    seq_len = 128
    vocab_size = 50257
    hidden_size = 256
    num_layers = 4
    num_heads = 8
    intermediate_size = 1024
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Configuration: batch={batch_size}, seq_len={seq_len}")
    print(f"Model: hidden={hidden_size}, layers={num_layers}, heads={num_heads}\n")

    # Test 1: GPTNeoBlock with GQA (num_kv_heads=2)
    print("1. Testing GPTNeoBlock with GQA (num_kv_heads=2)...")
    block = GPTNeoBlock(hidden_size, num_heads, num_kv_heads=2, intermediate_size=intermediate_size, dropout=0.1).to(device)
    x = torch.randn(batch_size, seq_len, hidden_size).to(device)
    mask = create_causal_mask(seq_len, device)
    output = block(x, mask)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == x.shape, "GPTNeoBlock output shape mismatch"
    print("   GPTNeoBlock with GQA working correctly")

    # Test 2: GPTNeoModel with num_kv_heads=2
    print("\n2. Testing GPTNeoModel with num_kv_heads=2...")
    model_core = GPTNeoModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=2,
        intermediate_size=intermediate_size,
        max_position_embeddings=512,
        dropout=0.1
    ).to(device)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    hidden_states = model_core(input_ids)
    print(f"   Input IDs shape: {input_ids.shape}")
    print(f"   Hidden states shape: {hidden_states.shape}")
    assert hidden_states.shape == (batch_size, seq_len, hidden_size), "GPTNeoModel output shape mismatch"
    print("   GPTNeoModel with GQA working correctly")

    # Test 3: GPTNeoForCausalLM with num_kv_heads=2
    print("\n3. Testing GPTNeoForCausalLM with num_kv_heads=2...")
    model_2 = GPTNeoForCausalLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=2,
        intermediate_size=intermediate_size,
        max_position_embeddings=512,
        dropout=0.1
    ).to(device)

    logits = model_2(input_ids)
    print(f"   Input IDs shape: {input_ids.shape}")
    print(f"   Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size), "GPTNeoForCausalLM output shape mismatch"
    print("   GPTNeoForCausalLM with GQA-2 working correctly")

    # Test 4: GPTNeoForCausalLM with num_kv_heads=4
    print("\n4. Testing GPTNeoForCausalLM with num_kv_heads=4...")
    model_4 = GPTNeoForCausalLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=4,
        intermediate_size=intermediate_size,
        max_position_embeddings=512,
        dropout=0.1
    ).to(device)

    logits = model_4(input_ids)
    print(f"   Logits shape: {logits.shape}")
    print("   GPTNeoForCausalLM with GQA-4 working correctly")

    # Test 5: Loss computation
    print("\n5. Testing Loss Computation...")
    labels = input_ids.clone()
    loss, logits = model_2(input_ids, labels=labels)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Logits shape: {logits.shape}")
    assert loss.item() > 0, "Loss should be positive"
    print("   Loss computation working correctly")

    # Test 6: Parameter counting comparison
    print("\n6. Comparing Parameter Counts...")
    params_2 = model_2.get_num_params()
    params_4 = model_4.get_num_params()

    # Compare with MHA and MQA models
    from AttentionHeads.mha.transformer import GPTNeoForCausalLM as MHAModel
    from AttentionHeads.mqa.transformer import GPTNeoForCausalLM as MQAModel

    mha_model = MHAModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=512,
        dropout=0.1
    ).to(device)
    mqa_model = MQAModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=512,
        dropout=0.1
    ).to(device)

    mha_params = mha_model.get_num_params()
    mqa_params = mqa_model.get_num_params()

    print(f"   MHA total parameters:   {mha_params:,}")
    print(f"   GQA-4 total parameters: {params_4:,} ({(1 - params_4/mha_params)*100:.1f}% reduction)")
    print(f"   GQA-2 total parameters: {params_2:,} ({(1 - params_2/mha_params)*100:.1f}% reduction)")
    print(f"   MQA total parameters:   {mqa_params:,} ({(1 - mqa_params/mha_params)*100:.1f}% reduction)")
    print("   Parameter counting working correctly")

    # Test 7: Text generation
    print("\n7. Testing Text Generation...")
    prompt = torch.randint(0, vocab_size, (1, 10)).to(device)
    generated = model_2.generate(prompt, max_length=50, temperature=1.0, top_k=50)
    print(f"   Prompt shape: {prompt.shape}")
    print(f"   Generated shape: {generated.shape}")
    assert generated.size(1) <= 50, "Generated sequence too long"
    print("   Text generation working correctly")

    # Test 8: create_gptneo_model factory
    print("\n8. Testing create_gptneo_model() factory...")
    config = {
        'vocab_size': vocab_size,
        'hidden_size': 256,
        'num_layers': 4,
        'num_heads': 8,
        'num_kv_heads': 2,
        'intermediate_size': 1024,
        'max_position_embeddings': 256,
        'dropout': 0.1
    }
    factory_model = create_gptneo_model(config).to(device)
    test_input = torch.randint(0, vocab_size, (1, 32)).to(device)
    test_output = factory_model(test_input)
    print(f"   Model created with config (num_kv_heads=2)")
    print(f"   Output shape: {test_output.shape}")
    print(f"   Parameters: {factory_model.get_num_params():,}")
    print("   Factory function working correctly")

    print("\n" + "=" * 70)
    print("All GPTNeo with GQA tests passed!")
    print("GQA architecture ready for training on TinyStories")
