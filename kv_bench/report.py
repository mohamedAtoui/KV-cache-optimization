"""Report generation: print tables, save JSON/markdown."""

import json
import logging
from typing import Optional

from kv_bench.strategy import StrategyResult

logger = logging.getLogger(__name__)


def print_table(results: list[StrategyResult]):
    """Print a formatted comparison table to stdout."""
    if not results:
        print("No results to display.")
        return

    # Find baseline for delta PPL
    baseline_ppl = None
    for r in results:
        if "baseline" in r.name.lower() or "fullkv" in r.name.lower():
            baseline_ppl = r.perplexity
            break
    if baseline_ppl is None:
        baseline_ppl = results[0].perplexity

    header = (
        f"{'Strategy':<25} | {'PPL':>7} | {'ΔPPL':>7} | "
        f"{'Mem MB':>8} | {'Comp.':>7} | {'Prefill ms':>10}"
    )
    sep = "-" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)

    for r in results:
        delta = r.perplexity - baseline_ppl
        delta_str = f"+{delta:.2f}" if delta > 0 else (f"{delta:.2f}" if delta < 0 else "  —")

        print(
            f"{r.name:<25} | {r.perplexity:>7.2f} | {delta_str:>7} | "
            f"{r.memory_kv_analytical_mb:>8.1f} | {r.compression_ratio:>6.1f}x | "
            f"{r.prefill_latency_ms:>10.1f}"
        )

    print(sep)
    print()


def save_json(results: list[StrategyResult], path: str):
    """Save results to a JSON file."""
    data = []
    for r in results:
        data.append({
            "name": r.name,
            "perplexity": r.perplexity,
            "avg_loss": r.avg_loss,
            "num_tokens": r.num_tokens,
            "memory_peak_mb": r.memory_peak_mb,
            "memory_kv_analytical_mb": r.memory_kv_analytical_mb,
            "prefill_latency_ms": r.prefill_latency_ms,
            "decode_latency_ms_per_token": r.decode_latency_ms_per_token,
            "compression_ratio": r.compression_ratio,
            "extra": r.extra,
        })

    with open(path, "w") as f:
        json.dump({"results": data}, f, indent=2)
    logger.info(f"Results saved to {path}")


def save_markdown(results: list[StrategyResult], path: str):
    """Save results as a markdown table."""
    baseline_ppl = None
    for r in results:
        if "baseline" in r.name.lower() or "fullkv" in r.name.lower():
            baseline_ppl = r.perplexity
            break
    if baseline_ppl is None:
        baseline_ppl = results[0].perplexity

    lines = [
        "# KV Cache Strategy Benchmark Results\n",
        "| Strategy | PPL | ΔPPL | Mem MB | Comp. | Prefill ms |",
        "|----------|-----|------|--------|-------|------------|",
    ]

    for r in results:
        delta = r.perplexity - baseline_ppl
        delta_str = f"+{delta:.2f}" if delta > 0 else (f"{delta:.2f}" if delta < 0 else "—")
        lines.append(
            f"| {r.name} | {r.perplexity:.2f} | {delta_str} | "
            f"{r.memory_kv_analytical_mb:.1f} | {r.compression_ratio:.1f}x | "
            f"{r.prefill_latency_ms:.1f} |"
        )

    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Markdown report saved to {path}")
