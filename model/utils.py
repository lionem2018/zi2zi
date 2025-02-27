# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import os
import glob

import imageio
import scipy.misc as misc
import numpy as np
# from cStringIO import StringIO
import io
from PIL import Image


def pad_seq(seq, batch_size):
    # pad the sequence to be the multiples of batch_size
    """
    batch_size의 배수가 될 수 있도록 시퀀스를 채움
    :param seq: 시퀀스? 여기서는 examples 리스트
    :param batch_size: 배치 사이즈
    :return: 배치 사이즈의 배수의 크기를 가지도록 크기가 늘어난 시퀀스
    """
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq


def bytes_to_file(bytes_img):
    return io.BytesIO(bytes_img)


def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized


def read_split_image(img):
    # mat = misc.imread(img).astype(np.float)
    mat = imageio.imread(img)
    side = int(mat.shape[1] / 2)
    assert side * 2 == mat.shape[1]
    img_A = mat[:, :side]  # target
    img_B = mat[:, side:]  # source

    return img_A, img_B


def shift_and_resize_image(img, shift_x, shift_y, nw, nh):
    w, h, _ = img.shape
    # enlarged = misc.imresize(img, [nw, nh])
    enlarged = np.array(Image.fromarray(img).resize([nw, nh]))
    return enlarged[shift_x:shift_x + w, shift_y:shift_y + h]


def scale_back(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def save_concat_images(imgs, img_path):
    concated = np.concatenate(imgs, axis=1)
    # misc.imsave(img_path, concated)
    imageio.imsave(img_path, concated)


def compile_frames_to_gif(frame_dir, gif_file):
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    print(frames)
    # images = [misc.imresize(imageio.imread(f), interp='nearest', size=0.33) for f in frames]
    images = [np.array(Image.fromarray(imageio.imread(f)).resize(size=0.33)) for f in frames]
    imageio.mimsave(gif_file, images, duration=0.1)
    return gif_file
