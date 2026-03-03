"""KV2State: Per-Head KV Cache to Recurrent State Conversion for LLMs."""

__version__ = "0.2.0"

from kv2state.head_classifier import load_duo_attention_patterns, compute_attention_entropy
from kv2state.state_attention import DecayedLinearState
from kv2state.hybrid_attention import patch_model_for_kv2state
from kv2state.eval_perplexity import evaluate_perplexity
from kv2state.calibration import calibrate_stage1, calibrate_stage2
