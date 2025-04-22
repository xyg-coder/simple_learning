from __future__ import annotations

import torch


class StreamPrefetchDataloader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataloader_iter = iter(dataloader)
        self.copy_stream = torch.cuda.Stream(device='cuda')
        self.next_batch = None

    def __iter__(self):
        if self.next_batch is None:
            self._preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        batch = self.next_batch
        for t in batch if isinstance(batch, (list, tuple)) else (batch,):
            if torch.is_tensor(t):
                t.record_stream(torch.cuda.current_stream())
        self._preload()
        return batch

    def _preload(self):
        try:
            host_batch = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            host_batch = next(self.dataloader_iter)
        
        with torch.cuda.stream(self.copy_stream):
            self.next_batch = self._recursive_to(host_batch, device='cuda', non_blocking=True)
        return self.next_batch


    def _recursive_to(self, data, **to_kwargs):
        """
        Recursively call .to(device, **kwargs) on every tensor in
        (nested) tuples / lists / dicts.  Non‑tensors are left untouched.
        """
        if torch.is_tensor(data):
            return data.to(**to_kwargs)
        elif isinstance(data, (list, tuple)):
            return type(data)(self._recursive_to(x, **to_kwargs) for x in data)
        elif isinstance(data, dict):
            return {k: self._recursive_to(v, **to_kwargs) for k, v in data.items()}
        else:
            return data
