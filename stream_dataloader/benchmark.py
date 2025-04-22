from __future__ import annotations

from .base import benchmark
from .resnet import resnet_benchmark
from .resnet import resnet_prefetch_benchmark


def benchmark_all_models():
    benchmark(resnet_benchmark, num_iterations=500)
    benchmark(resnet_prefetch_benchmark, num_iterations=500)


if __name__ == "__main__":
    benchmark_all_models()
