"""Streaming Attention: Per-Head KV Cache to Recurrent State Conversion for LLMs."""

__version__ = "0.3.0"

from streaming_attention.head_classifier import load_duo_attention_patterns, compute_attention_entropy
from streaming_attention.state_attention import DecayedLinearState
from streaming_attention.hybrid_attention import patch_model_for_streaming_attention
from streaming_attention.eval_perplexity import evaluate_perplexity
from streaming_attention.calibration import calibrate_stage1, calibrate_stage2
from streaming_attention.importance import ImportanceScorer, ImportanceConfig
from streaming_attention.adaptive_cache import TieredKVCache, AdaptiveCacheConfig
from streaming_attention.stratigraphic import StratigraphicConfig, HeadZoneAssigner, AnchorDetector
