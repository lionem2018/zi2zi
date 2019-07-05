# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
# import cPickle as pickle
import _pickle as pickle
import numpy as np
import random
import os
from .utils import pad_seq, bytes_to_file, \
    read_split_image, shift_and_resize_image, normalize_image


# pickled된 obj 파일로부터 이미지를 제공해주는 클래스
class PickledImageProvider(object):
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        """
        object 파일을 열어 pickled 이미지 로드
        :return: object 파일로부터 읽어들인 example 이미지
        """
        # object 파일을 읽기전용으로 엶
        with open(self.obj_path, "rb") as of:
            # example 이미지를 저장하기 위해 빈 리스트 생성
            examples = list()
            # object 파일 내에 더이상 읽을 데이터가 없을 때까지 반복하여 데이터를 읽고 리스트에 추가
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                    if len(examples) % 1000 == 0:
                        print("processed %d examples" % len(examples))
                except EOFError:
                    break
                except Exception:
                    pass
            print("unpickled total %d examples" % len(examples))
            return examples


def get_batch_iter(examples, batch_size, augment):
    # the transpose ops requires deterministic
    # batch size, thus comes the padding
    """
    batch size 만큼의 데이터를 가져오는 iterator
    :param examples: example 이미지 데이터들
    :param batch_size: 한 번 학습에 사용할 데이터의 크기
    :param augment: 이미지 보강
                    1) 이미지 확대
                    2) 무작위로 이미지를 원래 크기로 크롭
                    NOTE: 이미지 A와 B는 얼마만큼 이동해야하는지 동기화되어야함
    :return: iterator
    """
    # examples 리스트의 총 크기가 batch_size의 배수가 되도록 리스트를 패딩
    # 리스트 앞 쪽에서 부족한 만큼 뒤에 복사
    padded = pad_seq(examples, batch_size)

    def process(img):
        img = bytes_to_file(img)
        try:
            img_A, img_B = read_split_image(img)
            if augment:
                # augment the image by:
                # 1) enlarge the image
                # 2) random crop the image back to its original size
                # NOTE: image A and B needs to be in sync as how much
                # to be shifted
                w, h, _ = img_A.shape
                multiplier = random.uniform(1.00, 1.20)
                # add an eps to prevent cropping issue
                nw = int(multiplier * w) + 1
                nh = int(multiplier * h) + 1
                shift_x = int(np.ceil(np.random.uniform(0.01, nw - w)))
                shift_y = int(np.ceil(np.random.uniform(0.01, nh - h)))
                img_A = shift_and_resize_image(img_A, shift_x, shift_y, nw, nh)
                img_B = shift_and_resize_image(img_B, shift_x, shift_y, nw, nh)
            img_A = normalize_image(img_A)
            img_B = normalize_image(img_B)
            return np.concatenate([img_A, img_B], axis=2)
        finally:
            img.close()

    def batch_iter():
        for i in range(0, len(padded), batch_size):
            batch = padded[i: i + batch_size]
            labels = [e[0] for e in batch]
            processed = [process(e[1]) for e in batch]
            # stack into tensor
            yield labels, np.array(processed).astype(np.float32)

    return batch_iter()


class TrainDataProvider(object):
    def __init__(self, data_dir, train_name="train.obj", val_name="val.obj", filter_by=None):
        self.data_dir = data_dir
        self.filter_by = filter_by
        self.train_path = os.path.join(self.data_dir, train_name)
        self.val_path = os.path.join(self.data_dir, val_name)
        self.train = PickledImageProvider(self.train_path)
        self.val = PickledImageProvider(self.val_path)
        if self.filter_by:
            print("filter by label ->", filter_by)
            self.train.examples = filter(lambda e: e[0] in self.filter_by, self.train.examples)
            self.val.examples = filter(lambda e: e[0] in self.filter_by, self.val.examples)
        print("train examples -> %d, val examples -> %d" % (len(self.train.examples), len(self.val.examples)))

    def get_train_iter(self, batch_size, shuffle=True):
        training_examples = self.train.examples[:]
        if shuffle:
            np.random.shuffle(training_examples)
        return get_batch_iter(training_examples, batch_size, augment=True)

    def get_val_iter(self, batch_size, shuffle=True):
        """
        Validation iterator runs forever
        """
        val_examples = self.val.examples[:]
        if shuffle:
            np.random.shuffle(val_examples)
        while True:
            val_batch_iter = get_batch_iter(val_examples, batch_size, augment=False)
            for labels, examples in val_batch_iter:
                yield labels, examples

    def compute_total_batch_num(self, batch_size):
        """Total padded batch num"""
        return int(np.ceil(len(self.train.examples) / float(batch_size)))

    def get_all_labels(self):
        """Get all training labels"""
        return list({e[0] for e in self.train.examples})

    def get_train_val_path(self):
        return self.train_path, self.val_path


class InjectDataProvider(object):
    def __init__(self, obj_path):
        self.data = PickledImageProvider(obj_path)
        print("examples -> %d" % len(self.data.examples))

    def get_single_embedding_iter(self, batch_size, embedding_id):
        examples = self.data.examples[:]
        batch_iter = get_batch_iter(examples, batch_size, augment=False)
        for _, images in batch_iter:
            # inject specific embedding style here
            labels = [embedding_id] * batch_size
            yield labels, images

    def get_random_embedding_iter(self, batch_size, embedding_ids):
        examples = self.data.examples[:]
        batch_iter = get_batch_iter(examples, batch_size, augment=False)
        for _, images in batch_iter:
            # inject specific embedding style here
            labels = [random.choice(embedding_ids) for i in range(batch_size)]
            yield labels, images


class NeverEndingLoopingProvider(InjectDataProvider):
    def __init__(self, obj_path):
        super(NeverEndingLoopingProvider, self).__init__(obj_path)

    def get_random_embedding_iter(self, batch_size, embedding_ids):
        while True:
            # np.random.shuffle(self.data.examples)
            rand_iter = super(NeverEndingLoopingProvider, self) \
                .get_random_embedding_iter(batch_size, embedding_ids)
            for labels, images in rand_iter:
                yield labels, images
