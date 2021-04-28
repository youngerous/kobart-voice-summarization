# -*- coding: utf-8 -*-
"""
csv2txt.py
Autor: HyeongwonKang

audio clip script csv file to train dataset(txt)
예시 : python csv2txt.py -c audio_script_total.csv -p /resamp_data -s true -o taco_train.txt
    python csv2txt.py -c audio_script_total.csv -p /resamp_data -s false -o waveglow_train.txt
"""

import argparse
import logging
import csv
import re
import sys

def csv2txt(fname, output_name, path, use_script):
    """
    audio clip script csv file to train dataset(txt)

    Arguments
    ---------
    - fname        : audio clip script csv file
    - output_name  : train dataset file name(txt)
    - path         : base audio clip path

    Return
    ------
    None

    """
    try:
        # 오디오 클립 스크립트 정리된 csv 파일 load
        with open(fname, 'r') as c:
            csv_data = csv.reader(c)
            # text 파일로 dataset 생성
            # 예 : /data/wings01/wings01_005.wav|박제가 되어버린 천재를 아시오.
            with open(output_name, 'w') as f:
                for i, line in enumerate(csv_data):
                    if i == 0:
                        pass
                    else:
                        file_dict = re.search(r'(\w+)_(\d+.wav)',line[1]).group(1)
                        file_dict = re.sub(r"[0-9]", "", file_dict)
                        script = line[2]
                        if re.search(r"\w+",script) is not None:
                            print('file : {}, script : {}'.format(line[1], script))
                        if use_script:
                            f.write('{}/{}/{}|{}'.format(path, file_dict, line[1], script))
                        else:
                            f.write('{}/{}/{}'.format(path, file_dict, line[1]))
                        f.write("\n")
        print('convert success!')

    except:
        print('convert failed!')
        logger.exception("error")

def str2bool(v):
    if type(v) is not bool:
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    else:
        return v

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--csv_path", required=True, help="audio script csv path")
    ap.add_argument("-p", "--audio_clip_path", required=True, help="base audio clip path")
    ap.add_argument("-o", "--output_name", required=True, help="txt output name")
    ap.add_argument("-s", "--use_script", required=True, help="Whether to use a script")

    args = vars(ap.parse_args())

    csv_file = args["csv_path"]
    audio_path = args["audio_clip_path"]
    output_name = args["output_name"]
    use_script = str2bool(args["use_script"])

    csv2txt(csv_file, output_name, audio_path, use_script)


