# -*- coding: utf-8 -*-
"""
audio_preprocessing.py
Autor: HyeongwonKang, JeoungheeKim

audio clip resampling and Cropping blanks
예시 : python audio_preprocessing.py -r /data/wings -s resamp_data/wings
"""

## 라이브러리 Import
import numpy as np
import os
import argparse
from tqdm.notebook import tqdm
import librosa
from pathlib import Path
import matplotlib.pyplot as plt
import IPython.display as ipd
import glob
import soundfile as sf
import sys


## 함수 설정

def printProgressBar(iteration, total, prefix = 'Progress', suffix = 'Complete',\
                      decimals = 1, length = 70, fill = '█'):
    # 작업의 진행상황을 표시
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    _string_out = '\r%s |%s| %s%% %s' %(prefix, bar, percent, suffix)
    sys.stdout.write(_string_out)
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')

## 파일 읽어오기(지정한 Sampling Rate로)
def load_audio(file_path, sr=22050):
    """
       file_path : 파일위치
       sr : 오디오를 읽을 때 Sampling rate 지정
    """
    ## 확장자 추출
    ext = Path(file_path).suffix

    ## 파일 읽기
    if ext in ['.wav', '.flac']:
        wav, sr = librosa.load(file_path, sr=sr)
    elif ext == '.pcm':
        wav = np.memmap(file_path, dtype='h', mode='r').astype('float32') / 32767
    elif ext in ['.raw', '.RAW']:
        wav, sr = sf.read(file_path, channels=1, samlerate=sr, format='RAW', subtype='PCM_16')
    else:
        raise ValueError("Unsupported preprocess method : {0}".format(ext))

    return wav, sr


## 공백 자르기(패딩 추가)
def trim_audio(wav, top_db=10, pad_len=4000):
    """

    """
    ## 최대 db에 따라 음성의 자를 위치 판별
    non_silence_indices = librosa.effects.split(wav, top_db=top_db)
    start = non_silence_indices[0][0]
    end = non_silence_indices[-1][1]

    ## 음성 자르기
    wav = wav[start:end]

    ## padding 추가
    wav = np.hstack([np.zeros(pad_len), wav, np.zeros(pad_len)])

    return wav


## WAV 그려보기
def plot_wav(wav, sr):
    ## 그려보기
    plt.figure(1)

    plot_a = plt.subplot(211)
    plot_a.plot(wav)
    plot_a.set_xlabel('sample rate * time')
    plot_a.set_ylabel('energy')

    plot_b = plt.subplot(212)
    plot_b.specgram(wav, NFFT=1024, Fs=sr, noverlap=900)
    plot_b.set_xlabel('Time')
    plot_b.set_ylabel('Frequency')

    plt.show()

if __name__ == "__main__":
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)
    # stream_handler = logging.StreamHandler()
    # logger.addHandler(stream_handler)

    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--root_path", required=True, help="audio script csv path")
    ap.add_argument("-s", "--save_path", required=True, help="base audio clip path")

    args = vars(ap.parse_args())

    root_path = args["root_path"]
    save_path = args["save_path"]

    ## 시작하기

    ## 타코트론2는 기본적으로 22050 sampling rate에서 동작
    sampling_rate = 22050
    ## 개인설정에 따라 특정 소리보다 작은 음성을 삭제하도록 설정
    # decibel = 10

    ## Wav 파일 읽어오기  pcm 또는 다른 확장자도 사용 가능.
    file_list = glob.glob(os.path.join(root_path, "*.wav"))
    # file_list = glob.glob(os.path.join(root_path, "*.pcm"))

    ## 저장할 위치 선택
    os.makedirs(save_path, exist_ok=True)

    for i, file_path in enumerate(file_list):
        printProgressBar(i+1, len(file_list))
        ## 파일 불러오기(타코트론2는 기본적으로 22050 sampling rate에서 동작)
        wav, sr = load_audio(file_path, sr=sampling_rate)

        ## 오디오 자르기(패딩 추가)
        # trimed_wav = trim_audio(wav, top_db=decibel)

        filename = Path(file_path).name
        temp_save_path = os.path.join(save_path, filename)

        ## 저장하기
        # sf.write(temp_save_path, trimed_wav, sampling_rate)
        sf.write(temp_save_path, wav, sampling_rate)



