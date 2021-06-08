import argparse
from collections import OrderedDict
from pprint import pprint

import torch
from kobart import get_kobart_tokenizer, get_pytorch_kobart_model
# from distilbart import DistilBART
from transformers import BartForConditionalGeneration
import yaml

MODEL = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model())
MODEL.to("cpu")
TOK = get_kobart_tokenizer()
INITIALIZED_WEIGHT = False
INITIALIZED_MODEL = False


def initialize(ckpt: str) -> None:
    global MODEL
    global INITIALIZED_WEIGHT
    # cpu load
    state_dict = torch.load(ckpt, map_location=torch.device("cpu"))

    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if "module." in k else k  # remove `module.`
        new_state_dict[name] = v

    MODEL.load_state_dict(new_state_dict, strict=True)
    INITIALIZED_WEIGHT = True


@torch.no_grad()
def get_summarized_text(ckpt: str, text: str, n_enc: int = 6, n_dec: int = 6) -> str:
    global MODEL
    global INITIALIZED_MODEL
    global INITIALIZED_WEIGHT

    distilled = (n_enc != 6) or (n_dec != 6)
    if (not INITIALIZED_MODEL) and (distilled):
        MODEL = DistilBART(MODEL, n_enc=n_enc, n_dec=n_dec).to("cpu")
        INITIALIZED_MODEL = True
    if not INITIALIZED_WEIGHT:
        initialize(ckpt)

    MODEL.eval()
    text = text.replace("\n", "")
    input_ids = TOK.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    if distilled:
        output = MODEL.student.generate(
            input_ids, eos_token_id=1, max_length=512, num_beams=5
        )
    else:
        output = MODEL.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
    output = TOK.decode(output[0], skip_special_tokens=True)
    return output


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--ckpt", type=str, help="Checkpoint path to inference", default=None
    # )
    # parser.add_argument(
    #     "--text", type=str, default=None, help="Korean text to summarize"
    # )
    # parser.add_argument("--n-enc", type=int, default=6, help="Number of encoder layer")
    # parser.add_argument("--n-dec", type=int, default=6, help="Number of decoder layer")
    # args = parser.parse_args()

    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    ## 체크포인트 설정
    kobart_ckpt = config['kobart_ckpt']
    input_txt = config['input_txt']
    output_txt = config['output_txt']

    n_enc = config['n_enc']
    n_dec = config['n_dec']

    with open(input_txt, 'r') as f:
        text = f.read()

    summary = get_summarized_text(kobart_ckpt, text, n_enc, n_dec)

    with open(output_txt, 'w') as f:
        f.write(summary)

    # pprint(f"요약문: {summary}")
