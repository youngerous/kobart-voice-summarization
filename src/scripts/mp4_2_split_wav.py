"""
mp4_2_split_wav.py
Autor: HyeongwonKang

mp4 음성 데이터 json 자막 파일을 통해 문장별로 split 후 wav 파일로 저장
예시 : python mp4_2_split_wav.py -m /wings/mp4_files/ -j /wings/json/ -o /wings/audio_clip/
"""

from moviepy.editor import *
import os
from os.path import isfile, join
import re
import json
import csv
import argparse

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument("-m", "--mp4_path", required=True, help="mp4 path in")
  ap.add_argument("-j", "--json_path", required=True, help="json path in")
  ap.add_argument("-o", "--path_out", required=True, help="path out")

  args = vars(ap.parse_args())

  mp4_path = args["mp4_path"]
  json_path = args["json_path"]
  pathOut = args["path_out"]

  # path out folder 없는 경우 생성
  if not os.path.exists(pathOut):
            os.makedirs(pathOut)
  else:
      print(pathOut + " has been processed!")

  # 모든 mp4, json 파일 리스트
  mp4_files = [f for f in os.listdir(mp4_path) if isfile(join(mp4_path, f))]
  mp4_files.sort()
  json_files = [f for f in os.listdir(json_path) if isfile(join(json_path, f))]
  json_files.sort()
    
  for i, f in enumerate(mp4_files):
    ind = 0
    filename = re.search(r'(\w+).(mp4)', f).group(1)
    mp4_filename = mp4_path + f
    mp4_file = AudioFileClip(mp4_filename)
    json_file = open(json_path+filename+'.json', 'r', encoding='utf-8')
    json_text = json_file.read()
    json_dic = json.loads(json_text)
    segs = json_dic['segments']
    audio_pathOut = pathOut+'{}'.format(filename)
    
    with open('{}/{}_splitted.csv'.format(audio_pathOut, filename), 
              'w', newline='') as c:
        wr = csv.writer(c)
        wr.writerow(['Index', 'File_name', 'Start', 'End', 'Text'])
        
        if not os.path.exists(audio_pathOut):
            os.makedirs(audio_pathOut)
        else:
            print(audio_pathOut + " has been processed!")

        for seg in segs:
            ind += 1
            start = seg['start'] / 1000
            end = seg['end'] / 1000
            text = seg['text']
            audioclip = mp4_file.subclip(start, end)
            audioclip.write_audiofile('{}/{}_{}.wav'.format(
                audio_pathOut, filename, str(ind).zfill(3)))
            wr.writerow([ind, '{}_{}.wav'.format(filename, str(ind).zfill(3)),
                         start, end, text])
            