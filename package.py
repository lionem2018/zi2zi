# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import glob
import os
# import cPickle as pickle
import _pickle as pickle
import random


def pickle_examples(paths, train_path, val_path, train_val_split=0.1):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory
    combine 이미지(=examples)의 리스트를 피클 포멧으로 컴파일하면,
    학습 중에 모든 io가 메모리에서 발생할 것임

    """
    # train용 데이터를 저장할 폴더 열기
    with open(train_path, 'wb') as ft:
        # validate용 데이터를 저장할 폴더 열기
        with open(val_path, 'wb') as fv:
            # example 이미지 경로를 하나씩 가져와 처리
            for p in paths:
                # label 정보는 파일 이름에서 얻음
                label = int(os.path.basename(p).split("_")[0])
                # 이미지 하나를 읽기용으로 엶
                with open(p, 'rb') as f:
                    print("img %s" % p, label)
                    # 파일을 바이트 형태로 읽어들임
                    img_bytes = f.read()
                    # 랜덤값 생성(0에서 1사이)
                    r = random.random()
                    # label과 이미지 바이트 데이터를 나란히 example이라는 객체로 생성
                    example = (label, img_bytes)
                    # 데이터의 용도를 나누는 임계값을 기준으로 랜덤값에 따라 데이터 저장 폴더를 달리함
                    if r < train_val_split:
                        pickle.dump(example, fv)
                    else:
                        pickle.dump(example, ft)


parser = argparse.ArgumentParser(description='Compile list of images into a pickled object for training')
parser.add_argument('--dir', dest='dir', required=True, help='path of examples')
parser.add_argument('--save_dir', dest='save_dir', required=True, help='path to save pickled files')
parser.add_argument('--split_ratio', type=float, default=0.1, dest='split_ratio',
                    help='split ratio between train and val')
args = parser.parse_args()

if __name__ == "__main__":
    # 바이트형의 데이터 저장을 위해 obj 파일 생성
    train_path = os.path.join(args.save_dir, "train.obj")
    val_path = os.path.join(args.save_dir, "val.obj")
    pickle_examples(sorted(glob.glob(os.path.join(args.dir, "*.jpg"))), train_path=train_path, val_path=val_path,
                    train_val_split=args.split_ratio)
