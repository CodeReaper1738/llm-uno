#!/usr/bin/env python
from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="llm-uno",   
    version="0.1.0",                      
    description="LLM UNO agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yago Romano",
    author_email="yromanoma42@tntech.edu",
    url="https://github.com/yagoromano/llm-uno",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "openai==1.57.2",
        "huggingface-hub==0.33.0",
        "matplotlib==3.7.3",
        "numpy==1.24.4",
        "pandas==2.1.2",
        "rlcard==1.2.0",
        "scipy==1.11.3",
        "torch==2.1.0",
        "transformers==4.52.4",
    ],
    extras_require={
        "llama70b": ["deepspeed==0.16.7"],
    },
    entry_points={
        "console_scripts": [
            "llm-uno=llm_uno.core:main",
        ],
    },
    license="MIT",
)
