import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self._input_ids = []
        self._target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(len(token_ids) - max_length, stride):
            _input_chunk = token_ids[i : i + max_length]
            _target_chunk = token_ids[i + 1 : i + max_length + 1]
            self._input_ids.append(_input_chunk)
            self._target_ids.append(_target_chunk)

    def __len__(self):
        return len(self._input_ids)

    def __getitem__(self, idx):
        return self._input_ids[idx], self._target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
