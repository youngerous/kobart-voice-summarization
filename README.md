# KoBART Voice Summarization
2021년도 1학기 고려대학교 산업경영공학과 [비정형데이터분석](https://github.com/pilsung-kang/Text-Analytics) 수업 팀프로젝트 repository입니다. <br>


![thumbnail](assets/thumbnail.png)

## 1. Inference Overview
![process](assets/process.png)


본 README에서는 KoBART Fine-tuning을 위한 과정을 안내합니다.

## 2. Environment
먼저 아래의 명령어를 사용하여 KoBART를 다운로드합니다.
```sh
pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
```

그 후 [requirements.txt](https://github.com/youngerous/kobart-voice-summarization/blob/main/requirements.txt)의 패키지를 설치합니다. [Dockerfile](https://github.com/youngerous/kobart-voice-summarization/blob/main/Dockerfile)로도 설치할 수 있습니다.


## 3. Code Structure
```sh
# KoBART 문서요약 관련 코드
src/summarization/
    └─ data/
      └─ cached/ # tokenized indice without special tokens
          ├─ cached_train.jsonl
          ├─ cached_dev.jsonl
          └─ cached_test.jsonl
      ├─ train.jsonl # raw data with many meta-data
      ├─ dev.jsonl
      ├─ test.jsonl
      ├─ processed_train.jsonl # only src, tgt text
      ├─ processed_dev.jsonl
      └─ processed_test.jsonl
    └─ metric/
      ├─ get_rouge.py
      └─ rouge_metric.py
    ├─ 1. data_sample.ipynb
    ├─ 2. tokenizer_sample.ipynb
    ├─ 3. dataset_sample.ipynb
    ├─ 4. inference_sample.ipynb
    ├─ config.py
    ├─ dataset.py
    ├─ distilbart.py
    ├─ inference.py
    ├─ main.py
    ├─ preprocess.py
    ├─ trainer.py
    └─ utils.py
```

## 4. Data and Model

### Text Summarization

- Data: 한국어 생성요약 뉴스 데이터 일부 (Bflysoft - [AI Hub에 공개](https://aihub.or.kr/aidata/8054))
  - Train 260,697개 / Valid 10,000개 / Test 10,000개
- Model: KoBART, DistilKoBART
  - DistilKoBART는 huggingface의 [Pre-trained Summarization Distillation(2020)](https://arxiv.org/abs/2010.13002) 논문에서 제안된 방법론으로, Fine-tuning 시 Pre-trained 레이어의 일부만을 사용하는 모델입니다.

### Voice Synthesis

- Data: 네이버 오디오북, 밀리의서재
  - Train 3,752개
- Model: Tacotron2, Waveglow

## 5. How to 

### Preprocess

```train.jsonl``` / ```dev.jsonl``` / ```test.jsonl```은 미리 다운로드받아 위 구조에 맞게 배치해야 합니다. <br>
```src/summarization/``` 경로에서 아래 명령어를 실행하면 됩니다.

```sh
# train 
python preprocess.py --mode train

# dev 
python preprocess.py --mode dev

# test 
python preprocess.py --mode test
```

### Run
Root 경로에서 아래 명령어를 실행하면 됩니다.
```sh
sh run_kobart.sh
```

### Inference

```python
# 실행 위치에 따른 경로 수정이 필요할 수 있습니다.
from inference import get_summarized_text

ckpt: str = """ 학습된 체크포인트 경로를 입력합니다. """
text: str = """ 요약을 수행할 텍스트를 입력합니다. """
n_enc: int = """ DistilKoBART를 사용하는 경우에만 인코더 레이어 개수를 명시해 주면 됩니다. """
n_dec: int = """ DistilKoBART를 사용하는 경우에만 디코더 레이어 개수를 명시해 주면 됩니다. """
summary = get_summarized_text(ckpt, text, n_enc, n_dec)
print(summary)
```

[여기](https://github.com/youngerous/kobart-voice-summarization/blob/main/src/kobart/4.%20inference_sample.ipynb)에서 몇 개의 예시 데이터를 사용한 결과를 확인하실 수 있습니다.

## 6. Results

- [제안발표영상](https://youtu.be/z6T3j-YmY1w)
- [중간발표영상](https://youtu.be/_BTggEcjYqE)
- [최종발표영상](https://youtu.be/r9pn577B7mI)

10000개의 test data로 성능을 측정하였고, [Dacon 한국어 문서 생성요약 AI 경진대회의 metric](https://dacon.io/competitions/official/235673/talkboard/401911?page=1&dtype=recent)을 사용하였습니다. Mecab 라이브러리 설치가 필요하신 분은 [이 블로그](https://somjang.tistory.com/entry/PythonUbuntu%EC%97%90-mecab-ko-dic-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0)를 참고해주세요.


모델 뒤에 표기된 숫자는 인코더-디코더에 해당하는 레이어의 수입니다.

### KoBART-6-6
- Number of Parameters: 123,859,968
  
|           | Rouge-1 | Rouge-2 | Rouge-L |
| :-------: | :-----: | :-----: | :-----: |
| Precision |  0.467  |  0.314  |  0.373  |
|  Recall   |  0.601  |  0.399  |  0.476  |
| F1-Score  |  0.507  |  0.339  |  0.403  |

### DistilKoBART-3-6
- Number of Parameters: 102,596,352

|           | Rouge-1 | Rouge-2 | Rouge-L |
| :-------: | :-----: | :-----: | :-----: |
| Precision |  0.462  |  0.307  |  0.366  |
|  Recall   |  0.587  |  0.385  |  0.461  |
| F1-Score  |  0.496  |  0.327  |  0.392  |

### DistilKoBART-6-3
- Number of Parameters: 95,504,640

|           |  Rouge-1  |  Rouge-2  |  Rouge-L  |
| :-------: | :-------: | :-------: | :-------: |
| Precision |   0.481   |   0.324   |   0.386   |
|  Recall   |   0.582   |   0.386   |   0.464   |
| F1-Score  | **0.510** | **0.340** | **0.408** |


### DistilKoBART-3-3
- Number of Parameters: 74,241,024

|           | Rouge-1 | Rouge-2 | Rouge-L |
| :-------: | :-----: | :-----: | :-----: |
| Precision |  0.472  |  0.314  |  0.378  |
|  Recall   |  0.575  |  0.377  |  0.457  |
| F1-Score  |  0.501  |  0.331  |  0.399  |



### Demo

문서요약 모델, 음역모듈, 그리고 음성합성 모델의 정성적인 결과는 [데모 페이지](https://youngerous.github.io/kobart-voice-summarization/)에서 확인하실 수 있습니다.
## 7. Members
[Hyeongwon Kang](https://github.com/hwk0702), [Subin Kim](https://github.com/suubkiim), [Jina Kim](https://github.com/jina-kim7), [Takyoung Kim](https://github.com/youngerous)

### Acknowledgement
음성합성 관련 작업에서는 [Jounghee Kim](https://github.com/JoungheeKim)의 도움을 받았습니다.

## 8. Reference

- [[Paper] BART](https://arxiv.org/abs/1910.13461)
- [[Paper] Pre-trained Summarization Distillation](https://arxiv.org/abs/2010.13002)
- [[Repo] KoBART :: SKT-AI](https://github.com/SKT-AI/KoBART)
- [[Repo] KoBART-summarization :: seujung](https://github.com/seujung/KoBART-summarization)
- [[Post] 타코트론2 TTS 시스템 :: joungheekim](https://joungheekim.github.io/2021/04/01/code-review/)
- [[Metric] 한국어 문서 생성요약 AI 경진대회 평가 산식](https://dacon.io/competitions/official/235673/talkboard/401911?page=1&dtype=recent)
