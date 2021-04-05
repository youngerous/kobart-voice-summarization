# Kobart Voice Summarization
2021년도 1학기 고려대학교 산업경영공학과 [비정형데이터분석](https://github.com/pilsung-kang/Text-Analytics) 수업 팀프로젝트 repository입니다. <br>


![thumbnail](assets/thumbnail.png)

## 1. Overview
TBD

본 repo에서는 KoBART 학습코드(PyTorch)를 공개합니다.

## 2. Environment
먼저 아래의 명령어를 사용하여 KoBART를 다운로드합니다.
```sh
pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
```

그 후 [requirements.txt](https://github.com/youngerous/kobart-voice-summarization/blob/main/requirements.txt)의 패키지를 설치합니다. [Dockerfile](https://github.com/youngerous/kobart-voice-summarization/blob/main/Dockerfile)로도 설치할 수 있습니다.


학습은 Titan RTX 4장으로 수행하였습니다.

## 3. Code Structure
```sh
# KoBART 문서요약 관련 코드
src/kobart/
    └─ data/
      └─ cached/ # tokenized indice with no special tokens
          ├─ cached_train.jsonl
          ├─ cached_dev.jsonl
          └─ cached_test.jsonl
      ├─ train.jsonl # raw data with many meta-data
      ├─ dev.jsonl
      ├─ test.jsonl
      ├─ processed_train.jsonl # only src, tgt text
      ├─ processed_dev.jsonl
      └─ processed_test.jsonl
    ├─ config.py
    ├─ dataset.py
    ├─ main.py
    ├─ preprocess.py
    ├─ trainer.py
    └─ utils.py
```

## 4. Data and Model

### Text Summarization

- Data: 한국어 생성요약 뉴스 데이터 일부 (Bflysoft - [AI Hub를 통해 공개 예정](https://aihub.or.kr/aidata/8054))
  - Train 260,697개 / Valid 10,000개 / Test 10,000개
- Model: KoBART 

### Voice Synthesis

- Data: 네이버 오디오북, 밀리의서재
  - Train $$$개
- Model: Tacotron2, Waveglow

## 5. How to 

### Preprocess

```train.jsonl``` / ```dev.jsonl``` / ```test.jsonl```은 미리 다운로드받아 위 구조에 맞게 배치해야 합니다. <br>
```src/kobart/``` 경로에서 아래 명령어를 실행하면 됩니다.

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

ckpt = """ 학습된 체크포인트 경로를 입력합니다. """
text = """ 요약을 수행할 텍스트를 입력합니다. """
summary = get_summarized_text(ckpt, text)
print(summary)
```

[여기](https://github.com/youngerous/kobart-voice-summarization/blob/main/src/kobart/4.%20inference_sample.ipynb)에서 몇 개의 예시 데이터를 사용한 결과를 확인하실 수 있습니다.

## 6. Results

### Video
- [제안발표영상](https://youtu.be/z6T3j-YmY1w)
- [중간발표영상]()
- [최종발표영상]()

### Demo

요약 모델의 정량적인 평가와 정성적인 결과물은 [데모 페이지](https://youngerous.github.io/kobart-voice-summarization/)에서 확인하실 수 있습니다.

## 7. Members
[Hyeongwon Kang](https://github.com/hwk0702), [Subin Kim](https://github.com/suubkiim), [Jina Kim](https://github.com/jina-kim7), [Takyoung Kim](https://github.com/youngerous)

### Acknowledgement
음성합성 관련 작업에서는 [Jounghee Kim](https://github.com/JoungheeKim)의 도움을 받았습니다.

## 8. Reference

- [[Paper] BART](https://arxiv.org/abs/1910.13461)
- [[Repo] KoBART :: SKT-AI](https://github.com/SKT-AI/KoBART)
- [[Repo] KoBART-summarization :: seujung](https://github.com/seujung/KoBART-summarization)
- [[Repo] PyRouge :: pcyin](https://github.com/pcyin/PyRouge)
- [[Post] 타코트론2 TTS 시스템 :: joungheekim](https://joungheekim.github.io/2021/04/01/code-review/)
