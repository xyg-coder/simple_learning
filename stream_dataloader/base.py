from __future__ import annotations

import torch

from .perf_utils import PerfTracker
from .perf_utils import TimeStatKey


class BaseBenchmark:
    def __init__(self, model, dataloader, optimizer, criterion, tag):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.tag = tag

    def get_batch(self):
        for batch in self.dataloader:
            yield batch

    def forward(self, batch):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        return loss

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def benchmark(benchmark_instance: BaseBenchmark, num_iterations=1):
    torch.cuda.empty_cache()
    perf_tracker = PerfTracker()
    perf_tracker.time_start(TimeStatKey.DATA_LOAD)
    perf_tracker.throughput_start(0)
    i = 0
    for batch in benchmark_instance.get_batch():
        perf_tracker.time_end(TimeStatKey.DATA_LOAD)
        perf_tracker.time_start(TimeStatKey.FORWARD)
        loss = benchmark_instance.forward(batch)
        perf_tracker.time_end(TimeStatKey.FORWARD)
        perf_tracker.time_start(TimeStatKey.BACKWARD)
        benchmark_instance.backward(loss)
        perf_tracker.time_end(TimeStatKey.BACKWARD)
        perf_tracker.time_start(TimeStatKey.DATA_LOAD)
        i += 1
        if i >= num_iterations:
            break
        if i % 50 == 0:
            print(f'Iteration {i} completed')
    perf_tracker.throughput_end(num_iterations)
    print('=========================')
    print(f'Benchmarking {benchmark_instance.tag}')
    perf_tracker.print_stats()
    print('=========================')


class ToDeviceDataloader:
    def __init__(self, dataloader, device='cuda'):
        self.dataloader = dataloader
        self.dataloader_iter = iter(self.dataloader)
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataloader_iter)
            inputs, labels = batch
            inputs = inputs.to(device=self.device, non_blocking=True)
            labels = labels.to(device=self.device, non_blocking=True)
            return inputs, labels
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)
            inputs, labels = batch
            inputs = inputs.to(device=self.device, non_blocking=True)
            labels = labels.to(device=self.device, non_blocking=True)
            return inputs, labels
