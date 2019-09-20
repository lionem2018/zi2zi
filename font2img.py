# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import sys
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import collections
import io
# import importlib
#
# importlib.reload(sys)
# sys.setdefaultencoding("utf-8")

CN_CHARSET = None
CN_T_CHARSET = None
JP_CHARSET = None
KR_CHARSET = None

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHARSET = os.path.join(SCRIPT_PATH, "charset/cjk.json")


def load_global_charset():
    """
    글자셋 정보를 외부 json 파일로부터 읽어들여 가져옴
    명령어 옵션에서 어떤 글자를 지정했느냐에 따라 가져오는 글자셋이 다름(언어별)
    """

    global CN_CHARSET, JP_CHARSET, KR_CHARSET, CN_T_CHARSET
    cjk = json.load(open(DEFAULT_CHARSET))
    CN_CHARSET = cjk["gbk"]
    JP_CHARSET = cjk["jp"]
    KR_CHARSET = cjk["kr"]
    CN_T_CHARSET = cjk["gb2312_t"]


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    """
    하나의 폰트로 해당 글자를 (canvas_size * canvas_size) 크기의 하얀 이미지에 x_offset, y_offset 위치부터 그림
    :param ch: 생성할 글자
    :param font: 생성할 글자의 폰트
    :param canvas_size: 생성할 이미지의 크기
    :param x_offset: 그릴 글자의 x축 상의 위치
    :param y_offset: 그릴 글자의 y축 상의 위치
    :return: 글자가 그려진 이미지
    """

    img = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (255, 255, 255), font=font)
    return img


def draw_example(ch, src_font, dst_font, canvas_size, x_offset, y_offset, filter_hashes, sample_dir, count):
    """
    source 폰트와 destination 폰트 글자 이미지를 나란히 생성한 후 combine 하여 학습을 위한 하나의 이미지 생성
    :param ch: 생성할 글자
    :param src_font: source 폰트 (zi2zi 네트워크 상에서 출발이 되는 폰트 이미지)
    :param dst_font: destination 폰투 (zi2zi 네트워크 상에서 목표가 되는 폰트 이미지)
    :param canvas_size: 생성할 이미지의 크기
    :param x_offset: 그릴 글자의 x축 상의 위치
    :param y_offset: 그릴 글자의 y축 상의 위치
    :param filter_hashes:
    :return: combine된 이미지
    """

    # destination 폰트 글자 이미지 생성
    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    # check the filter example in the hashes or not
    # 이미지의 해시값을 구한 뒤 해시 리스트에 없는 경우 폰트상에 존재하지 않는 글자이므로
    # 생성하지 않고 건너뜀
    dst_hash = hash(dst_img.tobytes())
    if dst_hash in filter_hashes:
        return None
    # source 폰트 글자 이미지 생성
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    # 이미지 combine은 세로로 나란히 이루어지기 때문에 세로 길이가 두배인 이미지 생성
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))

    dst_img.save((os.path.join(sample_dir, "target/%d-targets.png" % count)))
    src_img.save((os.path.join(sample_dir, "input/%d-inputs.png" % count)))
    # 이미지 combine
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img


def filter_recurring_hash(charset, font, canvas_size, x_offset, y_offset):
    """
    Some characters are missing in a given font, filter them by checking the recurring hashes
    주어진 글꼴에 일부 문자가 없으면 반복되는 해시를 검사하여 해당 문자를 필터링
    :param charset: 지정한 언어의 글자셋
    :param font: 문자 존재 여부 확인할 폰트
    :param canvas_size: 이미지 크기
    :param x_offset: 그릴 글자의 x축 상의 위치
    :param y_offset: 그릴 글자의 y축 상이 위치
    :return:
    """

    # 함수 내부에서 사용할 글자셋 생성(복사->셔플->2000개만 추출)
    _charset = charset[:]
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    # 디폴트 값이 int인 딕셔너리(지정하지 않은 키는 그 값이 0으로 지정됨) hash_count 생성
    hash_count = collections.defaultdict(int)
    # 샘플 글자셋에서 글자 하나씩 이미지 생성하여 해시 만든 후 만들어진 해시를 딕셔너리에 카운트
    for c in sample:
        img = draw_single_char(c, font, canvas_size, x_offset, y_offset)
        hash_count[hash(img.tobytes())] += 1
    # 딕셔너리 아이템 중에서 카운트 횟수가 2보다 큰 경우를 뽑아내어 해시 리스트를 만듦
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]


def font2img(src, dst, charset, char_size, canvas_size,
             x_offset, y_offset, sample_count, sample_dir, label=0, filter_by_hash=True):

    # src_font 출발 폰트와 dst_font 목적 폰트를 ImageFont 형태로 불러옴
    src_font = ImageFont.truetype(src, size=char_size)
    dst_font = ImageFont.truetype(dst, size=char_size)

    # filter 옵션이 true라면,
    # 주어진 폰트에 일부 문자가 반복되지 않는지 해시를 검사하여 필터링하여,
    # 글자 해시 리스트를 생성함
    # (폰트 상에서 지원하지 않는 글자를 걸러내기 위함)
    filter_hashes = set()
    if filter_by_hash:
        filter_hashes = set(filter_recurring_hash(charset, dst_font, canvas_size, x_offset, y_offset))
        print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))

    count = 0

    # 각 글자에 대한 폰트 이미지 생성
    for c in charset:
        # 샘플의 개수만큼 생성했다면 멈춤
        if count == sample_count:
            break
        # 
        e = draw_example(c, src_font, dst_font, canvas_size, x_offset, y_offset, filter_hashes, sample_dir, count)
        if e:
            # 생성한 폰트 글자 이미지를 jpg 파일로 저장
            e.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)))
            count += 1
            # 100개마다 진행 상황 출력
            if count % 100 == 0:
                print("processed %d chars" % count)


load_global_charset()
parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--src_font', dest='src_font', required=True, help='path of the source font')
parser.add_argument('--dst_font', dest='dst_font', required=True, help='path of the target font')
parser.add_argument('--filter', dest='filter', type=int, default=0, help='filter recurring characters')
parser.add_argument('--charset', dest='charset', type=str, default='CN',
                    help='charset, can be either: CN, JP, KR or a one line file')
parser.add_argument('--shuffle', dest='shuffle', type=int, default=0, help='shuffle a charset before processings')
parser.add_argument('--char_size', dest='char_size', type=int, default=150, help='character size')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', dest='x_offset', type=int, default=53, help='x offset')
parser.add_argument('--y_offset', dest='y_offset', type=int, default=53, help='y_offset')
parser.add_argument('--sample_count', dest='sample_count', type=int, default=1000, help='number of characters to draw')
parser.add_argument('--sample_dir', dest='sample_dir', help='directory to save examples')
parser.add_argument('--label', dest='label', type=int, default=0, help='label as the prefix of examples')

args = parser.parse_args()

if __name__ == "__main__":
    if args.charset in ['CN', 'JP', 'KR', 'CN_T']:
        charset = locals().get("%s_CHARSET" % args.charset)
    else:
        # charset = [c for c in open(args.charset).readline()[:-1].decode("utf-8")]
        with io.open(args.charset, 'r', encoding='utf-8-sig') as f:
            charset = f.read().splitlines()
            
    # 셔플 옵션이 true 인 경우, 글자 셋을 셔플하여 랜덤으로 글자 이미지를 생성할 수 있도록 함
    if args.shuffle:
        np.random.shuffle(charset)

    font2img(args.src_font, args.dst_font, charset, args.char_size,
             args.canvas_size, args.x_offset, args.y_offset,
             args.sample_count, args.sample_dir, args.label, args.filter)
