"""KV Cache Strategy Benchmarking Framework.

Compare different KV-cache strategies on real Llama inference:
perplexity, memory, latency, compression ratio.
"""

from kv_bench.strategy import KVCacheStrategy, StrategyResult
from kv_bench.device_config import DeviceConfig, auto_detect
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
)

__all__ = [
    "KVCacheStrategy",
    "StrategyResult",
    "DeviceConfig",
    "auto_detect",
    "BenchmarkRunner",
    "print_table",
    "save_json",
    "save_markdown",
    "FullKVBaseline",
    "StreamingAttentionStrategy",
    "H2OStrategy",
    "SnapKVStrategy",
    "UniformQuantStrategy",
    "AdaptiveTieredStrategy",
    "HybridStrategy",
    "StratigraphicStrategy",
]
