import json
import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from kobart import get_kobart_tokenizer
from utils import SequentialDistributedSampler


class SummaryDataset(Dataset):
    """Bflysoft Abstractive Summarization Dataset

    :param root_path: Root path of dataset
    :param mode: Choose 'train', 'dev', or 'test'
    :param max_len: Maximum length of BART input
    :param ignore_index: Default ignore index calculating cross entropy loss
    """

    def __init__(
        self, root_path: str, mode: str, max_len: int = 512, ignore_index=-100
    ):
        super(SummaryDataset, self).__init__()
        assert mode in ["train", "dev", "test"]

        self.tokenizer = get_kobart_tokenizer()
        self.max_len = max_len
        self.eos_idx = self.tokenizer.vocab["</s>"]
        self.pad_idx = self.tokenizer.vocab["<pad>"]
        self.ign_idx = ignore_index

        # load cached data
        self.dset = []
        cached_path = os.path.join(root_path, f"cached/cached_{mode}.jsonl")
        with open(cached_path, "r", encoding="utf-8") as f:
            jsonl = list(f)
        for idx, json_str in enumerate(jsonl):
            self.dset.append(json.loads(json_str))
        print(f"Load {len(self.dset)} {mode} sample.")

    def add_eos_pad(self, indice: List[int]) -> List[int]:
        diff = self.max_len - len(indice) - 1
        if diff > 0:
            indice += [self.eos_idx] + [self.pad_idx] * diff
        else:
            indice = indice[: self.max_len - 1] + [self.eos_idx]
        return indice

    def add_ignore_idx(self, indice: List[int]) -> List[int]:
        diff = self.max_len - len(indice) - 1
        if diff > 0:
            indice += [self.eos_idx] + [self.ign_idx] * diff
        else:
            indice = indice[: self.max_len - 1] + [self.eos_idx]
        return indice

    def __len__(self) -> int:
        return len(self.dset)

    def __getitem__(self, idx: int) -> dict:
        """
        For translation and summarization training, decoder_input_ids should be provided.
        If no decoder_input_ids is provided,
            the model will create this tensor by shifting the input_ids to the right
            for denoising pre-training following the paper.
        """
        # load tokenized indice
        src = self.dset[idx]["src"]
        tgt = self.dset[idx]["tgt"]

        # add </s> and <pad> token
        padded_src = torch.tensor(self.add_eos_pad(src))
        padded_tgt = torch.tensor(self.add_ignore_idx(tgt))

        attention_mask = padded_src.ne(self.pad_idx).float()

        assert len(padded_src) == self.max_len
        assert len(padded_tgt) == self.max_len
        assert len(attention_mask) == self.max_len

        return {
            "input_ids": padded_src,
            "attention_mask": attention_mask,
            "labels": padded_tgt,
        }


def get_loader(
    batch_size: int, path: str, workers: int, mode: str, distributed: bool = False
) -> DataLoader:
    """
    :param batch_size: Mini-batch size
    :param path: Root path of dataset
    :param workers: Number of dataloader workers
    :param mode: Choose 'train', 'dev', or 'test'
    :param distributed: Whether to use ddp
    """
    assert mode in ["train", "dev", "test"]

    dset = SummaryDataset(root_path=path, mode=mode)
    shuffle_flag = mode == "train"
    sampler = None
    if distributed:
        sampler = (
            DistributedSampler(dset)
            if mode == "train"
            else SequentialDistributedSampler(dset)
        )
        shuffle_flag = False

    return DataLoader(
        dataset=dset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle_flag,
        num_workers=workers,
        pin_memory=True,
        drop_last=(mode == "train"),
    )
