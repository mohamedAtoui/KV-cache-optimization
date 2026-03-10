"""GPU detection and adaptive configuration."""

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class DeviceConfig:
    """Device-adaptive benchmark configuration."""
    device: str = "cuda"
    gpu_name: str = "unknown"
    gpu_memory_gb: float = 0.0
    dtype: torch.dtype = torch.bfloat16
    batch_size: int = 1
    max_seq_len: int = 2048
    max_samples: Optional[int] = None
    stride: int = 512
    # Model loading
    load_in_8bit: bool = False
    load_in_4bit: bool = False


def auto_detect() -> DeviceConfig:
    """Auto-detect GPU and return appropriate DeviceConfig."""
    if not torch.cuda.is_available():
        logger.info("No CUDA device found, using CPU config (testing only)")
        return DeviceConfig(
            device="cpu",
            gpu_name="cpu",
            gpu_memory_gb=0,
            dtype=torch.float32,
            batch_size=1,
            max_seq_len=256,
            max_samples=10,
            stride=128,
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_bytes = torch.cuda.get_device_properties(0).total_mem
    gpu_mem_gb = gpu_mem_bytes / (1024 ** 3)

    logger.info(f"Detected GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")

    if gpu_mem_gb >= 70:
        # A100-80GB or H100
        config = DeviceConfig(
            device="cuda",
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_mem_gb,
            batch_size=4,
            max_seq_len=8192,
            stride=2048,
        )
    elif gpu_mem_gb >= 35:
        # A100-40GB
        config = DeviceConfig(
            device="cuda",
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_mem_gb,
            batch_size=2,
            max_seq_len=4096,
            stride=1024,
        )
    else:
        # L4 (24GB) or similar
        config = DeviceConfig(
            device="cuda",
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_mem_gb,
            batch_size=1,
            max_seq_len=2048,
            stride=512,
        )

    logger.info(
        f"Config: batch_size={config.batch_size}, max_seq_len={config.max_seq_len}, "
        f"stride={config.stride}, dtype={config.dtype}"
    )
    return config
