"""
Ref: https://dacon.io/competitions/official/235673/talkboard/401911?page=1&dtype=recent
"""
import argparse
import pandas as pd
from tqdm import tqdm
from rouge_metric import Rouge
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('inference.py'))))
from inference import get_summarized_text


class RougeScorer:
    def __init__(self, args):
        self.rouge_evaluator = Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=True,
            length_limit=1000,
            length_limit_type="words",
            use_tokenizer=True,
            apply_avg=True,
            apply_best=False,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
        )
        self.ckpt = args.ckpt
        self.data_type = args.data_type
        self.inference_stop = args.inference_stop

    def compute_rouge(self, ref_df, hyp_df):
        # ref_df = pd.read_csv(ref_path)
        # hyp_df = pd.read_csv(hyp_path)

        #결측치 처리
        hyp_df.iloc[:, 1] = hyp_df.iloc[:, 1].fillna(" ")

        # ref에 있는 article만 남김
        ids = ref_df["id"]
        hyp_df = hyp_df[hyp_df["id"].isin(ids)]
        hyp_df.index = ref_df.index

        # id로 정렬
        ref_df = ref_df.sort_values(by=["id"])
        hyp_df = hyp_df.sort_values(by=["id"])

        # id -> int
        ref_df["id"] = ref_df["id"].astype(int)
        hyp_df["id"] = hyp_df["id"].astype(int)

        # id, summary tuple 만듦
        hyps = [tuple(row) for row in hyp_df.values]
        refs = [tuple(row) for row in ref_df.values]


        # id 같은 article에 대해 정답 요약문과 생성 요약문 rouge score 계산
        reference_summaries = []
        generated_summaries = []

        for ref_tp, hyp_tp in zip(refs, hyps):
            ref_id, ref = ref_tp
            hyp_id, hyp = hyp_tp

            assert ref_id == hyp_id

            reference_summaries.append(ref)
            generated_summaries.append(hyp)

        # rouge score 계산
        scores = self.rouge_evaluator.get_scores(
            generated_summaries, reference_summaries
        )
        str_scores = self.format_rouge_scores(scores)
        print(str_scores)
        self.save_rouge_scores(str_scores)
        return str_scores

    def save_rouge_scores(self, str_scores):
        with open(f"../results/rouge_score/{self.data_type}_{self.ckpt[-36:]}_{self.inference_stop}_rouge_scores.txt", "w") as output:
            output.write(str_scores)

    # def format_rouge_scores(self, scores):
    #     return "{:.3f},{:.3f},{:.3f}".format(
    #         scores["rouge-1"]["f"],
    #         scores["rouge-2"]["f"],
    #         scores["rouge-l"]["f"],
    #     )

    def format_rouge_scores(self, scores):
        return "f1 score: {:.4f},{:.4f},{:.4f} \n recall: {:.4f},{:.4f},{:.4f} \n precision: {:.4f},{:.4f},{:.4f}".format(
            scores["rouge-1"]["f"],
            scores["rouge-2"]["f"],
            scores["rouge-l"]["f"],

            scores["rouge-1"]["r"],
            scores["rouge-2"]["r"],
            scores["rouge-l"]["r"],

            scores["rouge-1"]["p"],
            scores["rouge-2"]["p"],
            scores["rouge-l"]["p"],
        )



# for infernece

class generate_summary:
    def __init__(self, args):
        self.ckpt = args.ckpt
        self.data_path = args.data_path
        self.data_type = args.data_type
        self.generated_summary_path = args.generated_summary_path
        self.inference_stop = args.inference_stop

        self.n_enc = args.n_enc
        self.n_dec = args.n_dec
    # generate sumamary

    def generate(self):

        # 먼저 data load
        ids, article, summary = self.data_load()

        generated = []
        stop = self.inference_stop
        for i, text in enumerate(tqdm(article)):
            if stop == -1:
                stop = len(article)
            elif i >= stop:
                break

            text = ''.join(text)
            summ = get_summarized_text(self.ckpt, text, n_enc=self.n_enc, n_dec=self.n_dec)
            generated.append(summ)
        print(f'{stop} summary generated')
        hyp_df = pd.DataFrame({'id': ids[:stop],'summary': generated})
        ref_df = pd.DataFrame({'id':ids[:stop], 'summary': summary[:stop]})

        hyp_df.to_csv(self.generated_summary_path+f"generated_{self.data_type}_{self.ckpt[-36:]}_{self.inference_stop}.csv")
        ref_df.to_csv(self.generated_summary_path+f"true_{self.data_type}_{self.ckpt[-36:]}_{self.inference_stop}.csv")

        return hyp_df, ref_df

    # data load
    def data_load(self):
        path = self.data_path + self.data_type + '.jsonl'
        with open(path, 'r') as f:
            data = list(f)

        # ids, article, summary 따로 저장
        ids = []
        article = []
        summary = []
        for text in data:
            ids.append(json.loads(text)['id'])
            article.append(json.loads(text)['article_original'])
            summary.append(json.loads(text)['abstractive'])
        return ids, article, summary


    def generated_load(self):
        hyp_df = pd.read_csv(self.generated_summary_path+ f"generated_{self.data_type}_{self.ckpt[-36:]}_{self.inference_stop}.csv")
        ref_df = pd.read_csv(self.generated_summary_path+ f"true_{self.data_type}_{self.ckpt[-36:]}_{self.inference_stop}.csv")

        hyp_df = hyp_df.iloc[:, 1:]
        ref_df = ref_df.iloc[:, 1:]

        return hyp_df, ref_df



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default="./ckpt/best_model_step_20365_loss_1.9004.pt",
        help="type the path for your checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/",
        help="path for data",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="test",
        help="train or dev or test",
    )
    parser.add_argument(
        "--generation",
        type=str2bool,
        default=False,
        help="if summary already generated, type False",
    )
    parser.add_argument(
        "--generated_summary_path",
        type=str,
        default="../results/generated_summary/",
        help="=type the path for generated summary",
    )
    parser.add_argument(
        "--inference_stop",
        type=int,
        default=-1,
        help="=type the number of article to generate summary. if you want to generate all, type '-1' ",
    )
    parser.add_argument('--n_enc', type=int, default=6)
    parser.add_argument('--n_dec', type=int, default=6)

    args = parser.parse_args()

    scorer = RougeScorer(args)
    infernece = generate_summary(args)

    # generation == True
    if args.generation == True:
        print(f'start to generate abstractive summary ({args.inference_stop})')
        hyp_df, ref_df = infernece.generate()

    # else 저장된 hyp_df, ref_df 불러와서 rouge score 계산
    else:
        try:
            hyp_df, ref_df = infernece.generated_load()
            print('summary already generated')
        except:
            print('generated summary not exists')
            hyp_df, ref_df = infernece.generate()

    # rouge score 계산
    rouge_score = scorer.compute_rouge(ref_df, hyp_df)





