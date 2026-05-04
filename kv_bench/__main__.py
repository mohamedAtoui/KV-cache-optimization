"""CLI entry point for kv_bench.

Usage:
    python -m kv_bench \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --pattern-dir attn_patterns/Meta-Llama-3.1-8B-Instruct \
        --strategies baseline streaming_attention h2o snapkv int8 int4 adaptive hybrid \
        --output results.json
"""

import argparse
import logging
import os
import sys

from kv_bench.device_config import auto_detect
from kv_bench.runner import BenchmarkRunner
from kv_bench.report import print_table, save_json, save_markdown
from kv_bench.strategies import (
    FullKVBaseline,
    StreamingAttentionStrategy,
    H2OStrategy,
    SnapKVStrategy,
    UniformQuantStrategy,
    AdaptiveTieredStrategy,
    HybridStrategy,
    StratigraphicStrategy,
    TurboQuantStrategy,
    HybridTQStrategy,
)
from streaming_attention.stratigraphic import StratigraphicConfig


STRATEGY_REGISTRY = {
    "baseline": lambda args: FullKVBaseline(),
    "streaming_attention": lambda args: StreamingAttentionStrategy(
        pattern_dir=args.pattern_dir,
        threshold=args.threshold,
        decay_init=args.decay_init,
    ),
    "h2o": lambda args: H2OStrategy(budget=args.h2o_budget),
    "snapkv": lambda args: SnapKVStrategy(budget=args.snapkv_budget),
    "int8": lambda args: UniformQuantStrategy(bits=8),
    "int4": lambda args: UniformQuantStrategy(bits=4),
    "adaptive": lambda args: AdaptiveTieredStrategy(),
    "stratigraphic": lambda args: StratigraphicStrategy(),
    "stratigraphic-evict": lambda args: StratigraphicStrategy(
        StratigraphicConfig(eviction_only=True)
    ),
    "stratigraphic-quant": lambda args: StratigraphicStrategy(
        StratigraphicConfig(quant_only=True)
    ),
    "hybrid": lambda args: HybridStrategy(
        pattern_dir=args.pattern_dir,
        threshold=args.threshold,
        decay_init=args.decay_init,
    ),
    "tq3": lambda args: TurboQuantStrategy(bits_stage1=2, qjl=True),
    "tq4": lambda args: TurboQuantStrategy(bits_stage1=3, qjl=True),
    "hybrid-tq3": lambda args: HybridTQStrategy(
        pattern_dir=args.pattern_dir,
        threshold=args.threshold,
        decay_init=args.decay_init,
        bits_stage1=2, qjl=True,
    ),
    "hybrid-tq4": lambda args: HybridTQStrategy(
        pattern_dir=args.pattern_dir,
        threshold=args.threshold,
        decay_init=args.decay_init,
        bits_stage1=3, qjl=True,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark KV cache strategies on Llama inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--pattern-dir", default=None,
        help="Path to DuoAttention pattern directory (required for streaming_attention/hybrid)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="DuoAttention threshold for retrieval/streaming classification",
    )
    parser.add_argument(
        "--decay-init", type=float, default=0.99,
        help="Initial decay value for streaming attention",
    )

    # Strategies
    parser.add_argument(
        "--strategies", nargs="+", default=["baseline"],
        choices=list(STRATEGY_REGISTRY.keys()),
        help="Strategies to benchmark",
    )

    # Strategy-specific params
    parser.add_argument("--h2o-budget", type=float, default=0.5)
    parser.add_argument("--snapkv-budget", type=float, default=0.5)

    # Dataset
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)

    # Device overrides
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)

    # Output
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--markdown", default=None, help="Save results as markdown")

    # Logging
    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Device config
    device_config = auto_detect()
    if args.device:
        device_config.device = args.device
    if args.max_seq_len:
        device_config.max_seq_len = args.max_seq_len
    if args.batch_size:
        device_config.batch_size = args.batch_size
    if args.stride:
        device_config.stride = args.stride

    # Build strategies
    strategies = []
    for name in args.strategies:
        try:
            strategy = STRATEGY_REGISTRY[name](args)
            strategies.append(strategy)
        except Exception as e:
            logging.error(f"Failed to create strategy '{name}': {e}")
            sys.exit(1)

    # Validate pattern_dir exists if provided
    if args.pattern_dir and not os.path.exists(args.pattern_dir):
        logging.error(f"--pattern-dir '{args.pattern_dir}' does not exist")
        sys.exit(1)

    # Validate streaming_attention/hybrid require pattern_dir
    for s in strategies:
        if isinstance(s, (StreamingAttentionStrategy, HybridStrategy, HybridTQStrategy)) and args.pattern_dir is None:
            logging.error(
                f"Strategy '{s.name}' requires --pattern-dir. "
                "Provide path to DuoAttention patterns."
            )
            sys.exit(1)

    # Run
    runner = BenchmarkRunner(
        model_name=args.model,
        strategies=strategies,
        device_config=device_config,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        max_samples=args.max_samples,
    )

    results = runner.run()

    # Report
    print_table(results)

    if args.output:
        save_json(results, args.output)
    if args.markdown:
        save_markdown(results, args.markdown)


if __name__ == "__main__":
    main()
