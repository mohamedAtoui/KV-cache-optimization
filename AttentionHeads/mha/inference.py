"""
Inference and Text Generation Utilities

Based on "Attention Is All You Need" (Vaswani et al., 2017)
Implementation follows Harvard NLP's Annotated Transformer:
https://nlp.seas.harvard.edu/annotated-transformer/

Reference:
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
    Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances
    in neural information processing systems (pp. 5998-6008).
"""

import torch
import torch.nn.functional as F
from .attention import subsequent_mask


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    Greedy decoding for autoregressive generation (Harvard NLP implementation)

    Generate output sequence one token at a time, always choosing the most
    likely next token (argmax). Simple but effective baseline.

    Args:
        model: Trained EncoderDecoder model
        src: Source sequence (batch, src_len)
        src_mask: Source mask (batch, 1, src_len)
        max_len: Maximum length to generate
        start_symbol: Token ID to start generation (e.g., <BOS>)

    Returns:
        ys: Generated sequence (batch, generated_len)

    Example:
        >>> model = make_model(10000, 10000)
        >>> src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        >>> src_mask = torch.ones(1, 1, 10)
        >>> output = greedy_decode(model, src, src_mask, max_len=10, start_symbol=1)
    """
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


class TextGenerator:
    """
    Text generation with multiple decoding strategies

    Provides various decoding methods for transformer models including
    greedy, beam search, sampling, and nucleus sampling.

    Args:
        model: Trained transformer model (EncoderDecoder or Transformer)
        tokenizer: Tokenizer for encoding/decoding text
        device: Device to run on ('cuda' or 'cpu')
    """

    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate_greedy(self, prompt, max_length=50):
        """
        Generate text using greedy decoding

        Args:
            prompt: Input text string
            max_length: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        # For Transformer class (backward compatibility)
        if hasattr(self.model, 'src_embedding'):
            for _ in range(max_length):
                # Forward pass
                output = self.model(input_ids, input_ids)

                # Get next token (greedy)
                next_token_logits = output[0, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Stop if EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        # For EncoderDecoder class (Harvard NLP style)
        else:
            src_mask = torch.ones(1, 1, input_ids.size(1)).to(self.device)
            input_ids = greedy_decode(
                self.model, input_ids, src_mask, max_length,
                start_symbol=self.tokenizer.bos_token_id or 0
            )

        # Decode
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def generate_with_temperature(self, prompt, max_length=50, temperature=1.0):
        """
        Generate text with temperature sampling

        Higher temperature = more random, lower = more conservative

        Args:
            prompt: Input text string
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (default: 1.0)

        Returns:
            str: Generated text
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        for _ in range(max_length):
            # Get logits for last position
            if hasattr(self.model, 'src_embedding'):
                # Transformer class (backward compatibility)
                output = self.model(input_ids, input_ids)
                logits = output[0, -1, :]
            else:
                # EncoderDecoder style (Harvard NLP)
                src_mask = torch.ones(1, 1, input_ids.size(1)).to(self.device)
                memory = self.model.encode(input_ids, src_mask)
                out = self.model.decode(
                    memory, src_mask, input_ids,
                    subsequent_mask(input_ids.size(1)).type_as(input_ids)
                )
                # Generator returns [1, vocab], squeeze to [vocab]
                logits = self.model.generator(out[:, -1]).squeeze(0)

            # Apply temperature
            logits = logits / temperature

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence (reshape to [1, 1])
            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)

            # Stop if EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def generate_top_k(self, prompt, max_length=50, k=50):
        """
        Generate text with top-k sampling

        Only sample from the k most likely tokens at each step.

        Args:
            prompt: Input text string
            max_length: Maximum tokens to generate
            k: Number of top tokens to consider

        Returns:
            str: Generated text
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        for _ in range(max_length):
            # Get logits
            if hasattr(self.model, 'src_embedding'):
                # Transformer class (backward compatibility)
                output = self.model(input_ids, input_ids)
                logits = output[0, -1, :]
            else:
                # EncoderDecoder style (Harvard NLP)
                src_mask = torch.ones(1, 1, input_ids.size(1)).to(self.device)
                memory = self.model.encode(input_ids, src_mask)
                out = self.model.decode(
                    memory, src_mask, input_ids,
                    subsequent_mask(input_ids.size(1)).type_as(input_ids)
                )
                # Generator returns [1, vocab], squeeze to [vocab]
                logits = self.model.generator(out[:, -1]).squeeze(0)

            # Filter to top-k
            top_k_logits, top_k_indices = torch.topk(logits, k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[next_token_idx]

            # Append to sequence (reshape to [1, 1])
            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)

            # Stop if EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def generate_nucleus(self, prompt, max_length=50, p=0.9):
        """
        Generate text with nucleus (top-p) sampling

        Sample from the smallest set of tokens whose cumulative probability
        exceeds p. More dynamic than top-k.

        Args:
            prompt: Input text string
            max_length: Maximum tokens to generate
            p: Cumulative probability threshold (default: 0.9)

        Returns:
            str: Generated text
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        for _ in range(max_length):
            # Get logits
            if hasattr(self.model, 'src_embedding'):
                # Transformer class (backward compatibility)
                output = self.model(input_ids, input_ids)
                logits = output[0, -1, :]
            else:
                # EncoderDecoder style (Harvard NLP)
                src_mask = torch.ones(1, 1, input_ids.size(1)).to(self.device)
                memory = self.model.encode(input_ids, src_mask)
                out = self.model.decode(
                    memory, src_mask, input_ids,
                    subsequent_mask(input_ids.size(1)).type_as(input_ids)
                )
                # Generator returns [1, vocab], squeeze to [vocab]
                logits = self.model.generator(out[:, -1]).squeeze(0)

            # Sort by probability
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Find cutoff (first token that exceeds p)
            sorted_indices_to_remove = cumulative_probs > p
            if sorted_indices_to_remove[..., 1:].any():
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Zero out removed indices
            logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence (reshape to [1, 1])
            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)

            # Stop if EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Example usage and tests
    print("Testing Text Generation Utilities...")
    print("=" * 70)

    print("\nThis module provides:")
    print("  - greedy_decode(): Harvard NLP greedy decoding function")
    print("  - TextGenerator: Class with multiple generation strategies")
    print("    * generate_greedy(): Greedy decoding")
    print("    * generate_with_temperature(): Temperature sampling")
    print("    * generate_top_k(): Top-k sampling")
    print("    * generate_nucleus(): Nucleus (top-p) sampling")

    print("\nUsage example:")
    print("""
    from mha import make_model
    from mha.inference import TextGenerator
    from transformers import GPT2Tokenizer

    # Create model
    model = make_model(50257, 50257, N=6)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Create generator
    generator = TextGenerator(model, tokenizer, device='cuda')

    # Generate text
    text = generator.generate_greedy("The transformer architecture", max_length=50)
    print(text)
    """)

    print("\n" + "=" * 70)
    print("✓ Inference module loaded successfully!")
    print("Implementation includes Harvard NLP's greedy_decode function")
