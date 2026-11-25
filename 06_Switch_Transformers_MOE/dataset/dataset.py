from typing import Any
from dataset.span_corruption import corrupt_spans
from torch.utils.data import Dataset
import numpy as np
import os
class C4SpanCorruptionDataset(Dataset):
    def __init__(self, path, sentinel_ids, eos_id, seq_len=512):
        self.path = path
        self.sentinel_ids = sentinel_ids
        self.eos_ids = eos_id
        self.seq_len = seq_len
        self.num_tokens = os.path.getsize(path) // 2
        self.data = None

    def __getdata__(self):
        if self.data is None:
            self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        return self.data

    def __getitem__(self, index) -> tuple:
        data = self.__getdata__()
        start = index * self.seq_len
        tokens_ids = data[start : start + self.seq_len].tolist()
        inputs, targets = corrupt_spans(tokens=tokens_ids, sentinel_ids=self.sentinel_ids, eos_id=self.eos_ids, corruption_rate=0.15, mean_span_lenght=3)
        return (inputs, targets)

    
    def __len__(self) -> int:
        return self.num_tokens // self.seq_len


