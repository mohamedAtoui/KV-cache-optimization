"""
Setup script for GPTNeo Decoder-Only Transformer

Installation:
    pip install -e .

This will install the package in editable mode, allowing you to modify
the code without reinstalling.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "GPTNeo Decoder-Only Transformer for TinyStories Dataset"

setup(
    name="gptneo-tinystories",
    version="1.0.0",
    description="GPTNeo Decoder-Only Transformer for TinyStories - A100 Optimized",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Attaimen",
    author_email="wmis066@live.rhul.ac.uk",
    url="https://gitlab.cim.rhul.ac.uk/wmis066/PROJECT",

    # Package discovery
    packages=find_packages(where='.'),
    package_dir={'': '.'},

    # Include non-Python files
    include_package_data=True,
    package_data={
        '': ['*.json', '*.md', '*.txt'],
    },

    # Python version requirement
    python_requires=">=3.8",

    # Dependencies
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "tqdm>=4.65.0",
        "numpy>=1.23.0",
    ],

    # Optional dependencies for development
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ],
        'viz': [
            'matplotlib>=3.5.0',
            'seaborn>=0.12.0',
            'tensorboard>=2.12.0',
        ],
        'notebook': [
            'jupyter>=1.0.0',
            'ipywidgets>=8.0.0',
        ],
    },

    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],

    # Keywords
    keywords="transformer gpt gptneo attention nlp language-model tinystories",

    # Entry points (optional - for command-line scripts)
    entry_points={
        'console_scripts': [
            'train-gptneo=train:main',
        ],
    },

    # Project URLs
    project_urls={
        'Bug Reports': 'https://gitlab.cim.rhul.ac.uk/wmis066/PROJECT/issues',
        'Source': 'https://gitlab.cim.rhul.ac.uk/wmis066/PROJECT',
        'Documentation': 'https://gitlab.cim.rhul.ac.uk/wmis066/PROJECT/blob/main/AttentionHeads/mha/README.md',
    },
)
