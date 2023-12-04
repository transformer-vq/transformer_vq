from io import open

from setuptools import find_packages
from setuptools import setup


setup(
    name="transformer_vq",
    version="15.0.0",
    url="https://github.com/transformer-vq/transformer_vq/",
    license="MIT",
    author="Anonymous Authors; Paper and Code Under Double-Blind Review at ICLR 2024",
    description="Official Transformer-VQ implementation in Jax.",
    long_description=open("README.md", encoding="utf-8").read(),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    platforms="any",
    python_requires=">=3.8",
    install_requires=[
        "absl-py==2.0.0",
        "etils==1.3.0",
        "ml_collections",
        "chex>=0.1.7",
        "datasets>=2.11.0",
        "flax==0.6.11",
        "ml-dtypes==0.2.0",
        "numpy>=1.22.0",
        "optax==0.1.5",
        "orbax-checkpoint==0.1.7",
        "requests>=2.28.1",
        "sentencepiece==0.1.96",
        "seqio==0.0.16",
        "tensorflow==2.12.1",
        "tensorflow-text==2.12.1",
        "tensorflow-datasets==4.9.1",
        "tensorboard>=2.10.1",
        "tensorstore>=0.1.35",
        "tqdm>=4.65.0",
        "wandb<0.15.0",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "pytest",
            "pytest-cov",
        ],
        "cpu": [
            "jaxlib==0.4.9",
            "jax==0.4.9",
        ],
        "gpu": [
            "jaxlib==0.4.9+cuda11.cudnn86",
            "jax[cuda11_pip]==0.4.9",
            "protobuf<=3.20.1",
        ],
        "tpu": [
            "jaxlib==0.4.9",
            "jax[tpu]==0.4.9",
            "protobuf<=3.20.1",
        ],
        "viz": [
            "matplotlib==3.5.3",
            "pandas==1.4.4",
            "seaborn==0.12.1",
        ],
    },
)
