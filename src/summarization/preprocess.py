import argparse
import itertools
import json
import os
from typing import List

import jsonlines
from tqdm import tqdm

from kobart import get_kobart_tokenizer


def cache_processed_data(root_pth: str, save_pth: str, mode: str) -> None:
    assert mode in ["train", "dev", "test"]

    # load raw data
    raw_data = []
    with open(os.path.join(root_pth, f"{mode}.jsonl"), "r", encoding="utf-8") as f:
        jsonl = list(f)
    for json_str in tqdm(jsonl):
        raw_data.append(json.loads(json_str))

    # save only source and target data
    with jsonlines.open(save_pth, "w") as f:
        for d in tqdm(raw_data):  # src: List[str], tgt: str
            f.write({"src": d["article_original"], "tgt": d["abstractive"]})


def tokenize_and_cache(
    dset: List[dict], root_pth: str, save_pth: str, mode: str
) -> None:
    assert mode in ["train", "dev", "test"]
    os.makedirs(os.path.join(root_pth, "cached/"), exist_ok=True)

    def tokenize(data, tok) -> dict:
        return {
            "src": [tok.tokenize(d) for d in data["src"]],
            "tgt": tok.tokenize(data["tgt"]),
        }

    def token2id(data, tok) -> dict:
        src = list(itertools.chain.from_iterable(data["src"]))
        src = tok.convert_tokens_to_ids(src)
        tgt = tok.convert_tokens_to_ids(data["tgt"])
        return {"src": src, "tgt": tgt}

    # tokenize text to subword
    tokenizer = get_kobart_tokenizer()
    tokenized = []
    for data in tqdm(dset):
        tokenized.append(tokenize(data, tokenizer))
    assert len(dset) == len(tokenized)

    # subword to indice
    tokenized_indice = []
    for tok_data in tqdm(tokenized):  # src: List[List[str]], tgt: List[str]
        tokenized_indice.append(token2id(tok_data, tokenizer))
    assert len(tokenized) == len(tokenized_indice)

    # cache tokenized indice
    with jsonlines.open(os.path.join(save_pth), "w") as f:
        for indice in tqdm(tokenized_indice):  # src: List[int], tgt: List[int]
            f.write({"src": indice["src"], "tgt": indice["tgt"]})


def main(args):
    root_pth = args.path
    mode = args.mode

    # 1. check and load processed data
    processed_pth = os.path.join(root_pth, f"processed_{mode}.jsonl")
    if not os.path.isfile(processed_pth):
        print("There is no processed data. Create new file...")
        cache_processed_data(root_pth=root_pth, save_pth=processed_pth, mode=mode)
        print("Done!")

    processed_data = []
    with open(processed_pth, "r", encoding="utf-8") as f:
        jsonl = list(f)
    for json_str in jsonl:
        processed_data.append(json.loads(json_str))

    # 2. tokenize and cache indice
    tokenized_pth = os.path.join(root_pth, f"cached/cached_{mode}.jsonl")
    if not os.path.isfile(tokenized_pth):
        print("There is no cached indice. Create new file...")
        tokenize_and_cache(
            dset=processed_data, root_pth=root_pth, save_pth=tokenized_pth, mode=mode
        )
        print("Done!")
    print("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./data/")
    parser.add_argument("--mode", type=str, help="train, dev, test")

    args = parser.parse_args()
    main(args)
