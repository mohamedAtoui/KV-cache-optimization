"""
Setup script for MHA Transformer package
Install with: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mha-transformer",
    version="1.0.0",
    author="LLM-Journey Contributors",
    author_email="",
    description="Multi-Head Attention Transformer Implementation based on Harvard NLP's Annotated Transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/attaimen/LLM-Journey",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "numpy",
        "tqdm",
        "tensorboard",
        "matplotlib",
        "seaborn",
    ],
)
