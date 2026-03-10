"""Benchmark runner: load model once, run strategies, collect metrics."""

import logging
import time
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from kv_bench.strategy import KVCacheStrategy, StrategyResult
from kv_bench.device_config import DeviceConfig, auto_detect

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates benchmark runs across multiple KV cache strategies.

    Loads the model and tokenizer once, then for each strategy:
    1. setup() — patch the model
    2. Evaluate perplexity (sliding window, use_cache=False)
    3. Measure memory and latency
    4. teardown() — restore the model
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        strategies: Optional[list[KVCacheStrategy]] = None,
        device_config: Optional[DeviceConfig] = None,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "test",
        max_samples: Optional[int] = None,
    ):
        self.model_name = model_name
        self.strategies = strategies or []
        self.device_config = device_config or auto_detect()
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.max_samples = max_samples or self.device_config.max_samples

        self.model = None
        self.tokenizer = None
        self._input_ids = None  # cached tokenized dataset

    def _load_model(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {self.model_name}")
        dc = self.device_config

        kwargs = {"torch_dtype": dc.dtype}
        if dc.device == "cuda":
            kwargs["device_map"] = "auto"
        if dc.load_in_8bit:
            kwargs["load_in_8bit"] = True
        elif dc.load_in_4bit:
            kwargs["load_in_4bit"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        if dc.device == "cpu":
            self.model = self.model.to(dc.device)

        self.model.eval()
        logger.info("Model loaded successfully")

    def _load_dataset(self):
        """Load and tokenize the evaluation dataset."""
        from datasets import load_dataset

        logger.info(f"Loading dataset: {self.dataset_name}/{self.dataset_config}")
        dataset = load_dataset(self.dataset_name, self.dataset_config, split=self.split)
        if self.max_samples is not None:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))

        text = "\n\n".join(dataset["text"])
        encodings = self.tokenizer(text, return_tensors="pt")
        self._input_ids = encodings.input_ids.to(self.device_config.device)
        logger.info(f"Dataset tokenized: {self._input_ids.size(1)} tokens")

    @torch.no_grad()
    def _evaluate_perplexity(self, strategy: KVCacheStrategy) -> dict:
        """Evaluate perplexity using sliding window (use_cache=False).

        Returns dict with perplexity, avg_loss, num_tokens, and timing info.
        """
        dc = self.device_config
        input_ids = self._input_ids
        seq_len = input_ids.size(1)
        max_length = dc.max_seq_len
        stride = dc.stride

        nlls = []
        num_tokens = 0
        prev_end = 0
        prefill_times = []
        needs_weights = strategy.needs_attention_weights()

        pbar = tqdm(
            range(0, seq_len, stride),
            desc=f"PPL ({strategy.name})",
        )

        for begin_loc in pbar:
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end

            input_chunk = input_ids[:, begin_loc:end_loc]
            target_ids = input_chunk.clone()
            target_ids[:, :-trg_len] = -100

            t0 = time.perf_counter()

            forward_kwargs = {
                "labels": target_ids,
                "use_cache": False,
            }
            if needs_weights:
                forward_kwargs["output_attentions"] = True

            outputs = self.model(input_chunk, **forward_kwargs)

            prefill_times.append((time.perf_counter() - t0) * 1000)

            # Feed attention weights to strategy if needed
            if needs_weights and outputs.attentions is not None:
                for layer_idx, attn_w in enumerate(outputs.attentions):
                    strategy.on_step(layer_idx=layer_idx, attn_weights=attn_w)

            # Reset strategy state between windows
            strategy.reset()

            # Also reset kv2state cache if present
            if hasattr(self.model, '_kv2state_cache'):
                self.model._kv2state_cache.reset()

            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood.item())
            num_tokens += trg_len
            prev_end = end_loc

            if end_loc >= seq_len:
                break

            running_ppl = torch.exp(torch.tensor(sum(nlls) / num_tokens)).item()
            pbar.set_postfix({"ppl": f"{running_ppl:.2f}", "tokens": num_tokens})

        avg_loss = sum(nlls) / num_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        avg_prefill_ms = sum(prefill_times) / len(prefill_times) if prefill_times else 0

        return {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "num_tokens": num_tokens,
            "prefill_latency_ms": avg_prefill_ms,
        }

    def _measure_memory(self) -> float:
        """Get current peak GPU memory in MB."""
        if self.device_config.device == "cuda":
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0.0

    def run(self) -> list[StrategyResult]:
        """Run all strategies and return results."""
        if self.model is None:
            self._load_model()
        if self._input_ids is None:
            self._load_dataset()

        results = []

        for strategy in self.strategies:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Running strategy: {strategy.name}")
            logger.info(f"{'=' * 60}")

            # Reset peak memory counter
            if self.device_config.device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            # Setup strategy
            t_setup = time.perf_counter()
            self.model = strategy.setup(
                self.model, self.model.config, self.device_config
            )
            setup_time_ms = (time.perf_counter() - t_setup) * 1000
            logger.info(f"Setup took {setup_time_ms:.1f} ms")

            # Evaluate perplexity
            ppl_result = self._evaluate_perplexity(strategy)

            # Measure memory
            peak_mem_mb = self._measure_memory()
            analytical_mem_bytes = strategy.memory_bytes(
                self.device_config.max_seq_len, self.model.config
            )
            analytical_mem_mb = analytical_mem_bytes / (1024 ** 2)

            # Compute compression ratio vs full KV
            model_cfg = self.model.config
            num_layers = model_cfg.num_hidden_layers
            num_kv_heads = getattr(model_cfg, "num_key_value_heads",
                                   model_cfg.num_attention_heads)
            head_dim = model_cfg.hidden_size // model_cfg.num_attention_heads
            full_kv_bytes = (
                num_layers * num_kv_heads * self.device_config.max_seq_len
                * head_dim * 2 * 2  # K + V, 2 bytes per bf16
            )
            compression_ratio = full_kv_bytes / max(analytical_mem_bytes, 1)

            result = StrategyResult(
                name=strategy.name,
                perplexity=ppl_result["perplexity"],
                avg_loss=ppl_result["avg_loss"],
                num_tokens=ppl_result["num_tokens"],
                memory_peak_mb=peak_mem_mb,
                memory_kv_analytical_mb=analytical_mem_mb,
                prefill_latency_ms=ppl_result["prefill_latency_ms"],
                decode_latency_ms_per_token=0.0,  # Not measured in use_cache=False mode
                compression_ratio=compression_ratio,
                extra={"setup_time_ms": setup_time_ms},
            )
            results.append(result)

            logger.info(
                f"Result: PPL={result.perplexity:.2f}, "
                f"Mem={result.memory_kv_analytical_mb:.1f}MB, "
                f"Compression={result.compression_ratio:.2f}x"
            )

            # Teardown
            self.model = strategy.teardown(self.model)
            logger.info(f"Teardown complete for {strategy.name}")

        return results
