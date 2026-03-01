from setuptools import setup, find_packages

setup(
    name="kv2state",
    version="0.1.0",
    description="Per-Head KV Cache to Recurrent State Conversion for LLMs",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.16.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.60.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "fla": ["flash-linear-attention>=0.1.0"],
        "train": ["peft>=0.10.0", "accelerate>=0.28.0"],
    },
)
