import json
import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from utils import SequentialDistributedSampler


class SummaryDataset(Dataset):
    """Bflysoft Abstractive Summarization Dataset

    Args:
        root_path (str): Root path of dataset
        mode (str): Choose 'train', 'dev', or 'test'
        tok (BertTokenizer): Kobart tokenizer
        max_len (int): Maximum length of sequence
        ignore_index (int): Default ignore index calculating cross entropy loss
    """

    def __init__(self, root_path, mode, tok, max_len=512, ignore_index=-100):
        super(SummaryDataset, self).__init__()
        assert mode in ["train", "dev", "test"]

        self.tokenizer = tok
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

    def add_pad(self, indice: List[int]) -> List[int]:
        diff = self.max_len - len(indice)
        if diff > 0:
            indice += [self.pad_idx] * diff
        else:
            indice = indice[: self.max_len - 1] + [self.eos_idx]
        return indice

    def add_ignore_idx(self, indice: List[int]) -> List[int]:
        diff = self.max_len - len(indice)
        if diff > 0:
            indice += [self.ign_idx] * diff
        else:
            indice = indice[: self.max_len - 1] + [self.eos_idx]
        return indice

    def __len__(self) -> int:
        return len(self.dset)

    def __getitem__(self, idx: int) -> dict:
        # load tokenized indice
        src = self.dset[idx]["src"]
        tgt = self.dset[idx]["tgt"]
        src.append(self.eos_idx)
        tgt.append(self.eos_idx)

        # add <pad> token to src
        padded_src = torch.tensor(self.add_pad(src))

        # decoder input
        dec_input_ids = [self.pad_idx]
        dec_input_ids += tgt[:-1]
        dec_input_ids = torch.tensor(self.add_pad(dec_input_ids))

        # add ignore token to tgt (default -100 for CrossEntropy loss)
        padded_tgt = torch.tensor(self.add_ignore_idx(tgt))

        assert len(padded_src) == self.max_len
        assert len(padded_tgt) == self.max_len
        assert len(dec_input_ids) == self.max_len

        return {
            "input_ids": padded_src,
            "decoder_input_ids": dec_input_ids,
            "labels": padded_tgt,
        }


def get_loader(tok, batch_size, path, workers, mode, distributed=False) -> DataLoader:
    """
    Args:
        tok (BertTokenizer): Kobart tokenizer
        batch_size (int): Mini-batch size per process
        path (str): Root path of dataset
        workers (int): Number of dataloader workers
        mode (str): Choose 'train', 'dev', or 'test'
        distributed (bool): Whether to use ddp. Defaults=False

    Returns:
        DataLoader
    """
    assert mode in ["train", "dev", "test"]

    dset = SummaryDataset(root_path=path, mode=mode, tok=tok)
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
