from typing import Any
from dataset.span_corruption import corrupt_spans
from torch.utils.data import Dataset
import numpy as np
import os
import torch
from torch.nn.utils.rnn import pad_sequence

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
        
        return (torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long))

    
    def __len__(self) -> int:
        return self.num_tokens // self.seq_len


def collate_fn(data, pad_id):
    inputs, targets = zip(*data)
    inputs, targets = list(inputs), list(targets)

    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=pad_id, padding_side="right") 
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_id, padding_side="right")

    
    attention_mask = (inputs_padded != pad_id).int()
    start = torch.full((targets_padded.shape[0],1), pad_id, dtype=targets_padded.dtype) #Make a column of start tokens
    decoder_inputs_ids = torch.cat([start, targets_padded[:, :-1]], dim=1) #Remove the last column
    #Applying so the CrossEntropyIgnore all the data that is not masked    
    targets_padded[targets_padded == pad_id] = -100
    return {
        "inputs_ids": inputs_padded, 
        "attention_mask": attention_mask, 
        "decoder_inputs_ids": decoder_inputs_ids, 
        "labels": targets_padded
        }