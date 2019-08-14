# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import scipy.misc as misc
import imageio
import os
import time
from collections import namedtuple
from .ops import conv2d, deconv2d, lrelu, fc, batch_norm, init_embedding, conditional_instance_norm
from .dataset import TrainDataProvider, InjectDataProvider, NeverEndingLoopingProvider
from .utils import scale_back, merge, save_concat_images
import matplotlib.pyplot as plt

# Auxiliary wrapper classes
# 보조 래퍼 클래스
# Used to save handles(important nodes in computation graph) for later evaluation
# 나중에 평가할 수 있도록 핸들 (계산 그래프의 중요한 노드)을 저장하는 데 사용
LossHandle = namedtuple("LossHandle", ["d_loss", "g_loss", "const_loss", "l1_loss",
                                       "category_loss", "cheat_loss", "tv_loss"])
InputHandle = namedtuple("InputHandle", ["real_data", "embedding_ids", "no_target_data", "no_target_ids"])
EvalHandle = namedtuple("EvalHandle", ["encoder", "generator", "target", "source", "embedding"])
SummaryHandle = namedtuple("SummaryHandle", ["d_merged", "g_merged"])


class UNet(object):
    def __init__(self, experiment_dir=None, experiment_id=0, batch_size=16, input_width=256, output_width=256,
                 generator_dim=64, discriminator_dim=64, L1_penalty=100, Lconst_penalty=15, Ltv_penalty=0.0,
                 Lcategory_penalty=1.0, embedding_num=40, embedding_dim=128, input_filters=3, output_filters=3):
        # 샘플 생성 이미지, 체크포인트 등 저장하기 위한 디렉토리 경로
        self.experiment_dir = experiment_dir
        # 실행 준비중인 실험의 시퀀스 ID
        self.experiment_id = experiment_id
        # 배치 사이즈(학습이 반영되는 데이터 단위의 크기)
        self.batch_size = batch_size
        # 입력 이미지의 너비(256)
        self.input_width = input_width
        # 출력 이미지의 너비(256)
        self.output_width = output_width
        # 임베딩을 위한 차원
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        # L1 loss를 위한 가중치
        self.L1_penalty = L1_penalty
        # constant loss를 위한 가중치
        self.Lconst_penalty = Lconst_penalty
        # tv loss를 위한 가중치
        self.Ltv_penalty = Ltv_penalty
        # category loss를 위한 가중치
        self.Lcategory_penalty = Lcategory_penalty
        # 별개의 임베딩을 위한 수(?)
        self.embedding_num = embedding_num
        # 임베딩을 위한 차원(?)
        self.embedding_dim = embedding_dim
        
        self.input_filters = input_filters
        self.output_filters = output_filters
        
        # init all the directories
        # 모든 디렉토리를 초기화
        self.sess = None
        # experiment_dir is needed for training
        # experiment_dir가 학습에 요구됨
        # experiment_dir에 필요한 하위 디렉토리 경로 정보 얻음
        if experiment_dir:
            self.data_dir = os.path.join(self.experiment_dir, "data")
            self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoint")
            self.sample_dir = os.path.join(self.experiment_dir, "sample")
            self.log_dir = os.path.join(self.experiment_dir, "logs")

            # data 디렉토리 외에 존재하지 않는 디렉토리 생성
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                print("create checkpoint directory")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                print("create log directory")
            if not os.path.exists(self.sample_dir):
                os.makedirs(self.sample_dir)
                print("create sample directory")

    def encoder(self, images, is_training, reuse=False):
        """
        generator의 encoder 부분 정의
        :param images: 입력이 될 이미지
        :param is_training: 학습상황인지 테스트상황인지 구별
        :param reuse: 재사용 여부
        :return: encoder의 결과값과 encoding 수행 과정상의 결과들
        """
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            # decoder가 각 레이어의 encoding 수행 결과를 참고할 수 있도록 하는 딕셔너리 생성
            # UNet 구조 참고
            encode_layers = dict()

            def encode_layer(x, output_filters, layer):
                """
                encoder의 각 레이어 정의
                :param x: input 데이터
                :param output_filters: output 크기
                :param layer: 레이어 번호
                :return:
                """
                # 데이터가 입력되기 전 lrelu 수행
                act = lrelu(x)
                # convolution 수행
                conv = conv2d(act, output_filters=output_filters, scope="g_e%d_conv" % layer)
                # convolution 수행 후 배치 정규화 수행(학습 시)
                enc = batch_norm(conv, is_training, scope="g_e%d_bn" % layer)
                # 결과를 encode_layers에 저장(decoder가 참고할 수 있도록)
                encode_layers["e%d" % layer] = enc
                return enc

            e1 = conv2d(images, self.generator_dim, scope="g_e1_conv")
            encode_layers["e1"] = e1
            e2 = encode_layer(e1, self.generator_dim * 2, 2)
            e3 = encode_layer(e2, self.generator_dim * 4, 3)
            e4 = encode_layer(e3, self.generator_dim * 8, 4)
            e5 = encode_layer(e4, self.generator_dim * 8, 5)
            e6 = encode_layer(e5, self.generator_dim * 8, 6)
            e7 = encode_layer(e6, self.generator_dim * 8, 7)
            e8 = encode_layer(e7, self.generator_dim * 8, 8)

            return e8, encode_layers

    def decoder(self, encoded, encoding_layers, ids, inst_norm, is_training, reuse=False):
        """
        generator의 decoder 부분 정의
        :param encoded: encoding 한 결과값
        :param encoding_layers: encoding 수행 과정 중 기록한 각 레이어의 결과값
        :param ids:
        :param inst_norm: 조건부 인스턴스 정규화 사용 여부
        :param is_training: 학습상황인지 테스트상황인지 구별
        :param reuse: 재사용 여부
        :return:
        """
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            # 각 레이어의 output 크기 설정
            s = self.output_width
            s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
                s / 64), int(s / 128)

            def decode_layer(x, output_width, output_filters, layer, enc_layer, dropout=False, do_concat=True):
                """
                decoder의 각 레이어 정의
                :param x: 입력
                :param output_width: 출력 크기(최종적으로는 이미지의 크기가 됨)
                :param output_filters: 출력 깊이
                :param layer: 레이어 번호
                :param enc_layer: encoder 레이어 상의 결과값
                :param dropout: 드롭아웃 여부
                :param do_concat: encoder 각 레이어 결과값과의 concat 여부
                :return: 각 레이어의 결과값
                """
                # deconvolution 수행
                dec = deconv2d(tf.nn.relu(x), [self.batch_size, output_width,
                                               output_width, output_filters], scope="g_d%d_deconv" % layer)
                # 마지막 레이어(최종 출력 레이어)가 아니라면,
                if layer != 8:
                    # IMPORTANT: normalization for last layer
                    # Very important, otherwise GAN is unstable
                    # Trying conditional instance normalization to
                    # overcome the fact that batch normalization offers
                    # different train/test statistics
                    # 중요: 마지막 레이어의 정규화
                    # 매우 중요합니다. 그렇지 않으면 GAN이 불안정합니다.
                    # 일괄 표준화가 다른 학습/테스트 통계를 제공한다는 사실을
                    # 극복하기 위해 조건부 인스턴스 정규화 시도
                    # 조건부 인스턴스 정규화를 사용하지 않는 경우 배치 정규화 수행
                    if inst_norm:
                        dec = conditional_instance_norm(dec, ids, self.embedding_num, scope="g_d%d_inst_norm" % layer)
                    else:
                        dec = batch_norm(dec, is_training, scope="g_d%d_bn" % layer)
                # 드롭아웃
                if dropout:
                    dec = tf.nn.dropout(dec, 0.5)
                # 출력값과 encoder 각 레이어 출력값을 concat
                if do_concat:
                    dec = tf.concat([dec, enc_layer], 3)
                return dec

            d1 = decode_layer(encoded, s128, self.generator_dim * 8, layer=1, enc_layer=encoding_layers["e7"], dropout=True)
            d2 = decode_layer(d1, s64, self.generator_dim * 8, layer=2, enc_layer=encoding_layers["e6"], dropout=True)
            d3 = decode_layer(d2, s32, self.generator_dim * 8, layer=3, enc_layer=encoding_layers["e5"], dropout=True)
            d4 = decode_layer(d3, s16, self.generator_dim * 8, layer=4, enc_layer=encoding_layers["e4"])
            d5 = decode_layer(d4, s8, self.generator_dim * 4, layer=5, enc_layer=encoding_layers["e3"])
            d6 = decode_layer(d5, s4, self.generator_dim * 2, layer=6, enc_layer=encoding_layers["e2"])
            d7 = decode_layer(d6, s2, self.generator_dim, layer=7, enc_layer=encoding_layers["e1"])
            d8 = decode_layer(d7, s, self.output_filters, layer=8, enc_layer=None, do_concat=False)

            # 생성 이미지를 그레이스케일 이미지로 만들기 위함
            output = tf.nn.tanh(d8)  # scale to (-1, 1)
            return output

    def generator(self, images, embeddings, embedding_ids, inst_norm, is_training, reuse=False):
        """
        이미지의 스타일을 변환하여 생성하는 generator (UNet 사용)
        :param images: 변환할 이미지
        :param embeddings: 모든 카테고리 임베딩 값
        :param embedding_ids: 카테고리 임베딩 값을 얻을 스타일의 인덱스 번호
        :param inst_norm: 
        :param is_training: 학습 상황인지 테스트 상황인지
        :param reuse: 재사용 여부
        :return: encoder와 decoder 출력 데이터
        """
        
        # encoder 수행
        e8, enc_layers = self.encoder(images, is_training=is_training, reuse=reuse)

        # category 임베딩
        local_embeddings = tf.nn.embedding_lookup(embeddings, ids=embedding_ids)
        local_embeddings = tf.reshape(local_embeddings, [self.batch_size, 1, 1, self.embedding_dim])
        embedded = tf.concat([e8, local_embeddings], 3)

        # decoder 수행
        output = self.decoder(embedded, enc_layers, embedding_ids, inst_norm, is_training=is_training, reuse=reuse)
        return output, e8

    def discriminator(self, image, is_training, reuse=False):
        """
        이미지가 합성(Fake)인지 실제(Real)인지 판별하는 discriminator
        :param image: 판별할 이미지
        :param is_training: 학습 상황인지 테스트 상황인지
        :param reuse: 재사용 여부
        :return: GAN loss와 category loss
        """
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            h0 = lrelu(conv2d(image, self.discriminator_dim, scope="d_h0_conv"))
            h1 = lrelu(batch_norm(conv2d(h0, self.discriminator_dim * 2, scope="d_h1_conv"),
                                  is_training, scope="d_bn_1"))
            h2 = lrelu(batch_norm(conv2d(h1, self.discriminator_dim * 4, scope="d_h2_conv"),
                                  is_training, scope="d_bn_2"))
            h3 = lrelu(batch_norm(conv2d(h2, self.discriminator_dim * 8, sh=1, sw=1, scope="d_h3_conv"),
                                  is_training, scope="d_bn_3"))
            # real or fake binary loss
            # 실제인지 가짜인지 판단
            fc1 = fc(tf.reshape(h3, [self.batch_size, -1]), 1, scope="d_fc1")
            # category loss
            # 어떤 카테고리의 스타일일지 판단
            fc2 = fc(tf.reshape(h3, [self.batch_size, -1]), self.embedding_num, scope="d_fc2")

            return tf.nn.sigmoid(fc1), fc1, fc2

    def build_model(self, is_training=True, inst_norm=False, no_target_source=False):
        """
        학습을 위한 전체적인 모델 구성
        :param is_training: 학습 상황인지 테스트 상황인지
        :param inst_norm: 조건부 인스턴스 정규화 사용 여부
        :param no_target_source: ???
        :return: ???
        """
        # real data를 위한 placeholder와 no target data를 위한 placeholder 생성
        real_data = tf.placeholder(tf.float32,
                                   [self.batch_size, self.input_width, self.input_width,
                                    self.input_filters + self.output_filters],
                                   name='real_A_and_B_images')
        embedding_ids = tf.placeholder(tf.int64, shape=None, name="embedding_ids")
        no_target_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.input_width, self.input_width,
                                         self.input_filters + self.output_filters],
                                        name='no_target_A_and_B_images')
        no_target_ids = tf.placeholder(tf.int64, shape=None, name="no_target_embedding_ids")

        # 학습을 위해 combine 되어 있던 real_data를 타겟과 소스로 나눔
        # target images
        # 타겟 이미지 데이터
        real_B = real_data[:, :, :, :self.input_filters]
        # source images
        # 소스 이미지 데이터
        real_A = real_data[:, :, :, self.input_filters:self.input_filters + self.output_filters]

        embedding = init_embedding(self.embedding_num, self.embedding_dim)
        # generator를 통해 타겟과 유사하도록 하는 fake 이미지, generator의 encoder 부분만을 거친 encoded 이미지를 구함
        fake_B, encoded_real_A = self.generator(real_A, embedding, embedding_ids, is_training=is_training,
                                                inst_norm=inst_norm)
        # discriminator에 입력하기 위해 source와 target, 그리고 source와 fake를 concat
        real_AB = tf.concat([real_A, real_B], 3)
        fake_AB = tf.concat([real_A, fake_B], 3)

        # Note it is not possible to set reuse flag back to False
        # initialize all variables before setting reuse to True
        # 재사용 플래그를 다시 False로 설정할 수는 없습니다.
        # 재사용을 True로 설정하기 전에 모든 변수를 초기화하십시오.
        real_D, real_D_logits, real_category_logits = self.discriminator(real_AB, is_training=is_training, reuse=False)
        fake_D, fake_D_logits, fake_category_logits = self.discriminator(fake_AB, is_training=is_training, reuse=True)

        # encoding constant loss
        # this loss assume that generated imaged and real image
        # should reside in the same space and close to each other
        # 상수 손실 인코딩
        # 이 손실은 생성 된 이미지와 실제 이미지가 같은 공간에 있고 서로 가깝다고 가정합니다
        encoded_fake_B = self.encoder(fake_B, is_training, reuse=True)[0]
        const_loss = (tf.reduce_mean(tf.square(encoded_real_A - encoded_fake_B))) * self.Lconst_penalty

        # category loss
        # 카테고리 손실
        true_labels = tf.reshape(tf.one_hot(indices=embedding_ids, depth=self.embedding_num),
                                 shape=[self.batch_size, self.embedding_num])
        real_category_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_category_logits,
                                                                                    labels=true_labels))
        fake_category_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_category_logits,
                                                                                    labels=true_labels))
        category_loss = self.Lcategory_penalty * (real_category_loss + fake_category_loss)

        # binary real/fake loss
        # 실제 혹은 거짓 판별 loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D_logits,
                                                                             labels=tf.ones_like(real_D)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
                                                                             labels=tf.zeros_like(fake_D)))
        # L1 loss between real and generated images
        # 실제 이미지와 생성된 이미지 간의 L1 loss
        # 순수한 차이
        l1_loss = self.L1_penalty * tf.reduce_mean(tf.abs(fake_B - real_B))
        # total variation loss
        # 변동 loss
        width = self.output_width
        tv_loss = (tf.nn.l2_loss(fake_B[:, 1:, :, :] - fake_B[:, :width - 1, :, :]) / width
                   + tf.nn.l2_loss(fake_B[:, :, 1:, :] - fake_B[:, :, :width - 1, :]) / width) * self.Ltv_penalty

        # maximize the chance generator fool the discriminator
        cheat_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
                                                                            labels=tf.ones_like(fake_D)))

        d_loss = d_loss_real + d_loss_fake + category_loss / 2.0
        g_loss = cheat_loss + l1_loss + self.Lcategory_penalty * fake_category_loss + const_loss + tv_loss

        if no_target_source:
            # no_target source are examples that don't have the corresponding target images
            # however, except L1 loss, we can compute category loss, binary loss and constant losses with those examples
            # it is useful when discriminator get saturated and d_loss drops to near zero
            # those data could be used as additional source of losses to break the saturation
            # no_target 소스는 해당 대상 이미지가없는 예제입니다.
            # 그러나 L1 손실을 제외하고는 범주 손실, 이진 손실 및 그러한 손실을 계산할 수 있습니다
            # discriminator가 포화되고 d_loss가 거의 0에 가까울 때 유용합니다.
            # 이러한 데이터는 포화 상태를 깨뜨리는 추가적인 손실 원으로 사용될 수 있습니다.
            no_target_A = no_target_data[:, :, :, self.input_filters:self.input_filters + self.output_filters]
            no_target_B, encoded_no_target_A = self.generator(no_target_A, embedding, no_target_ids,
                                                              is_training=is_training,
                                                              inst_norm=inst_norm, reuse=True)
            no_target_labels = tf.reshape(tf.one_hot(indices=no_target_ids, depth=self.embedding_num),
                                          shape=[self.batch_size, self.embedding_num])
            no_target_AB = tf.concat([no_target_A, no_target_B], 3)
            no_target_D, no_target_D_logits, no_target_category_logits = self.discriminator(no_target_AB,
                                                                                            is_training=is_training,
                                                                                            reuse=True)
            encoded_no_target_B = self.encoder(no_target_B, is_training, reuse=True)[0]
            no_target_const_loss = tf.reduce_mean(
                tf.square(encoded_no_target_A - encoded_no_target_B)) * self.Lconst_penalty
            no_target_category_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=no_target_category_logits,
                                                        labels=no_target_labels)) * self.Lcategory_penalty

            d_loss_no_target = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=no_target_D_logits,
                                                                                      labels=tf.zeros_like(
                                                                                          no_target_D)))
            cheat_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=no_target_D_logits,
                                                                                 labels=tf.ones_like(no_target_D)))
            d_loss = d_loss_real + d_loss_fake + d_loss_no_target + (category_loss + no_target_category_loss) / 3.0
            g_loss = cheat_loss / 2.0 + l1_loss + \
                     (self.Lcategory_penalty * fake_category_loss + no_target_category_loss) / 2.0 + \
                     (const_loss + no_target_const_loss) / 2.0 + tv_loss

        d_loss_real_summary = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_summary = tf.summary.scalar("d_loss_fake", d_loss_fake)
        category_loss_summary = tf.summary.scalar("category_loss", category_loss)
        cheat_loss_summary = tf.summary.scalar("cheat_loss", cheat_loss)
        l1_loss_summary = tf.summary.scalar("l1_loss", l1_loss)
        fake_category_loss_summary = tf.summary.scalar("fake_category_loss", fake_category_loss)
        const_loss_summary = tf.summary.scalar("const_loss", const_loss)
        d_loss_summary = tf.summary.scalar("d_loss", d_loss)
        g_loss_summary = tf.summary.scalar("g_loss", g_loss)
        tv_loss_summary = tf.summary.scalar("tv_loss", tv_loss)

        d_merged_summary = tf.summary.merge([d_loss_real_summary, d_loss_fake_summary,
                                             category_loss_summary, d_loss_summary])
        g_merged_summary = tf.summary.merge([cheat_loss_summary, l1_loss_summary,
                                             fake_category_loss_summary,
                                             const_loss_summary,
                                             g_loss_summary, tv_loss_summary])

        # expose useful nodes in the graph as handles globally
        # 그래프의 유용한 노드를 핸들로 전역 노출
        input_handle = InputHandle(real_data=real_data,
                                   embedding_ids=embedding_ids,
                                   no_target_data=no_target_data,
                                   no_target_ids=no_target_ids)

        loss_handle = LossHandle(d_loss=d_loss,
                                 g_loss=g_loss,
                                 const_loss=const_loss,
                                 l1_loss=l1_loss,
                                 category_loss=category_loss,
                                 cheat_loss=cheat_loss,
                                 tv_loss=tv_loss)

        eval_handle = EvalHandle(encoder=encoded_real_A,
                                 generator=fake_B,
                                 target=real_B,
                                 source=real_A,
                                 embedding=embedding)

        summary_handle = SummaryHandle(d_merged=d_merged_summary,
                                       g_merged=g_merged_summary)

        # those operations will be shared, so we need
        # to make them visible globally
        # 이러한 작업은 공유되므로 글로벌로 표시해야합니다.
        setattr(self, "input_handle", input_handle)
        setattr(self, "loss_handle", loss_handle)
        setattr(self, "eval_handle", eval_handle)
        setattr(self, "summary_handle", summary_handle)

    def register_session(self, sess):
        self.sess = sess

    def retrieve_trainable_vars(self, freeze_encoder=False):
        t_vars = tf.trainable_variables()

        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        if freeze_encoder:
            # exclude encoder weights
            print("freeze encoder weights")
            g_vars = [var for var in g_vars if not ("g_e" in var.name)]

        return g_vars, d_vars

    def retrieve_generator_vars(self):
        all_vars = tf.global_variables()
        generate_vars = [var for var in all_vars if 'embedding' in var.name or "g_" in var.name]
        return generate_vars

    def retrieve_handles(self):
        input_handle = getattr(self, "input_handle")
        loss_handle = getattr(self, "loss_handle")
        eval_handle = getattr(self, "eval_handle")
        summary_handle = getattr(self, "summary_handle")

        return input_handle, loss_handle, eval_handle, summary_handle

    def get_model_id_and_dir(self):
        """
        학습 모델의 데이터를 저장하기 위한 디렉터리 및 아이디 생성
        :return: 모델 아이디, 저장 디렉터리
        """
        model_id = "experiment_%d_batch_%d" % (self.experiment_id, self.batch_size)
        model_dir = os.path.join(self.checkpoint_dir, model_id)
        return model_id, model_dir

    def checkpoint(self, saver, step):
        """
        학습 데이터 저장
        """
        model_name = "unet.model"
        model_id, model_dir = self.get_model_id_and_dir()

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        saver.save(self.sess, os.path.join(model_dir, model_name), global_step=step)

    def restore_model(self, saver, model_dir):
        """
        저장한 모델 데이터(체크포인트) 불러오기
        """

        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restored model %s" % model_dir)
        else:
            print("fail to restore model %s" % model_dir)

    def generate_fake_samples(self, input_images, embedding_ids):
        """
        fake 이미지를 생성하는 함수
        :param input_images: 입력이 되는 이미지
        :param embedding_ids: 만들고자 하는 스타일에 대한 아이디
        :return: 생성 이미지, 실제 이미지, d_loss, g_loss_ l1_loss
        """
        input_handle, loss_handle, eval_handle, summary_handle = self.retrieve_handles()
        fake_images, real_images, \
        d_loss, g_loss, l1_loss = self.sess.run([eval_handle.generator,
                                                 eval_handle.target,
                                                 loss_handle.d_loss,
                                                 loss_handle.g_loss,
                                                 loss_handle.l1_loss],
                                                feed_dict={
                                                    input_handle.real_data: input_images,
                                                    input_handle.embedding_ids: embedding_ids,
                                                    input_handle.no_target_data: input_images,
                                                    input_handle.no_target_ids: embedding_ids
                                                })
        return fake_images, real_images, d_loss, g_loss, l1_loss

    def validate_model(self, val_iter, epoch, step):
        """
        모델 성능 검증
        이미지를 생성하여 real과 fake를 나란히 둔 이미지 파일 생성
        """
        labels, images = next(val_iter)
        fake_imgs, real_imgs, d_loss, g_loss, l1_loss = self.generate_fake_samples(images, labels)
        print("Sample: d_loss: %.5f, g_loss: %.5f, l1_loss: %.5f" % (d_loss, g_loss, l1_loss))

        # batch_size 만큼의 이미지들을 세로로 배열하여 하나의 긴 이미지로 생성
        merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
        merged_real_images = merge(scale_back(real_imgs), [self.batch_size, 1])
        # fake image와 real image를 합침
        merged_pair = np.concatenate([merged_real_images, merged_fake_images], axis=1)

        model_id, _ = self.get_model_id_and_dir()

        model_sample_dir = os.path.join(self.sample_dir, model_id)
        if not os.path.exists(model_sample_dir):
            os.makedirs(model_sample_dir)

        sample_img_path = os.path.join(model_sample_dir, "sample_%02d_%04d.png" % (epoch, step))
        # misc.imsave(sample_img_path, merged_pair)
        imageio.imsave(sample_img_path, merged_pair)

    def export_generator(self, save_dir, model_dir, model_name="gen_model"):
        """
        체크포인트에서 저장한 generator 불러오기기
        """
        saver = tf.train.Saver()
        self.restore_model(saver, model_dir)

        gen_saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        gen_saver.save(self.sess, os.path.join(save_dir, model_name), global_step=0)

    def infer(self, source_obj, embedding_ids, model_dir, save_dir):
        """
        모델 테스트(추론)
        """
        source_provider = InjectDataProvider(source_obj)

        if isinstance(embedding_ids, int) or len(embedding_ids) == 1:
            embedding_id = embedding_ids if isinstance(embedding_ids, int) else embedding_ids[0]
            source_iter = source_provider.get_single_embedding_iter(self.batch_size, embedding_id)
        else:
            source_iter = source_provider.get_random_embedding_iter(self.batch_size, embedding_ids)

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        self.restore_model(saver, model_dir)

        def save_imgs(imgs, count):
            p = os.path.join(save_dir, "inferred_%04d.png" % count)
            save_concat_images(imgs, img_path=p)
            print("generated images saved at %s" % p)

        count = 0
        batch_buffer = list()
        for labels, source_imgs in source_iter:
            fake_imgs = self.generate_fake_samples(source_imgs, labels)[0]

            for i in range(len(fake_imgs)):
                print(source_imgs.shape, fake_imgs.shape)
                source_img = source_imgs[i, :, :, 3:6]
                print(source_img.shape)
                combined_img = np.concatenate([source_img, fake_imgs[i]])  # source_image의 뒷 이미지만 가져와서 concat하기 (source는 두 이미지가 합쳐진 상태)
                imageio.imsave(os.path.join(save_dir, "test/test_%04d_%04d.png" % (count, i)), combined_img)

            merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
            batch_buffer.append(merged_fake_images)
            if len(batch_buffer) == 10:
                save_imgs(batch_buffer, count)
                batch_buffer = list()
            count += 1
        if batch_buffer:
            # last batch
            save_imgs(batch_buffer, count)

    def interpolate(self, source_obj, between, model_dir, save_dir, steps):
        """
        보간
        소스 이미지에서 타겟 이미지로 fake 이미지가 변화되는 과정을 리스트로 저장
        """
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        self.restore_model(saver, model_dir)
        # new interpolated dimension
        new_x_dim = steps + 1
        alphas = np.linspace(0.0, 1.0, new_x_dim)

        def _interpolate_tensor(_tensor):
            """
            Compute the interpolated tensor here
            """

            x = _tensor[between[0]]
            y = _tensor[between[1]]

            interpolated = list()
            for alpha in alphas:
                interpolated.append(x * (1. - alpha) + alpha * y)

            interpolated = np.asarray(interpolated, dtype=np.float32)
            return interpolated

        def filter_embedding_vars(var):
            var_name = var.name
            if var_name.find("embedding") != -1:
                return True
            if var_name.find("inst_norm/shift") != -1 or var_name.find("inst_norm/scale") != -1:
                return True
            return False

        embedding_vars = filter(filter_embedding_vars, tf.trainable_variables())
        # here comes the hack, we overwrite the original tensor
        # with interpolated ones. Note, the shape might differ

        # this is to restore the embedding at the end
        embedding_snapshot = list()
        for e_var in embedding_vars:
            val = e_var.eval(session=self.sess)
            embedding_snapshot.append((e_var, val))
            t = _interpolate_tensor(val)
            op = tf.assign(e_var, t, validate_shape=False)
            print("overwrite %s tensor" % e_var.name, "old_shape ->", e_var.get_shape(), "new shape ->", t.shape)
            self.sess.run(op)

        source_provider = InjectDataProvider(source_obj)
        input_handle, _, eval_handle, _ = self.retrieve_handles()
        for step_idx in range(len(alphas)):
            alpha = alphas[step_idx]
            print("interpolate %d -> %.4f + %d -> %.4f" % (between[0], 1. - alpha, between[1], alpha))
            source_iter = source_provider.get_single_embedding_iter(self.batch_size, 0)
            batch_buffer = list()
            count = 0
            for _, source_imgs in source_iter:
                count += 1
                labels = [step_idx] * self.batch_size
                generated, = self.sess.run([eval_handle.generator],
                                           feed_dict={
                                               input_handle.real_data: source_imgs,
                                               input_handle.embedding_ids: labels
                                           })
                merged_fake_images = merge(scale_back(generated), [self.batch_size, 1])
                batch_buffer.append(merged_fake_images)
            if len(batch_buffer):
                save_concat_images(batch_buffer,
                                   os.path.join(save_dir, "frame_%02d_%02d_step_%02d.png" % (
                                       between[0], between[1], step_idx)))
        # restore the embedding variables
        print("restore embedding values")
        for var, val in embedding_snapshot:
            op = tf.assign(var, val, validate_shape=False)
            self.sess.run(op)

    def train(self, lr=0.0002, epoch=100, schedule=10, resume=True, flip_labels=False,
              freeze_encoder=False, fine_tune=None, sample_steps=50, checkpoint_steps=500):
        g_vars, d_vars = self.retrieve_trainable_vars(freeze_encoder=freeze_encoder)
        input_handle, loss_handle, _, summary_handle = self.retrieve_handles()

        if not self.sess:
            raise Exception("no session registered")

        learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.d_loss, var_list=d_vars)
        g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.g_loss, var_list=g_vars)
        tf.global_variables_initializer().run()
        real_data = input_handle.real_data
        embedding_ids = input_handle.embedding_ids
        no_target_data = input_handle.no_target_data
        no_target_ids = input_handle.no_target_ids

        # filter by one type of labels
        # 한 가지 유형의 라벨로 필터링
        data_provider = TrainDataProvider(self.data_dir, filter_by=fine_tune)  # tf.dataset의 역할
        total_batches = data_provider.compute_total_batch_num(self.batch_size)  # 몇 개의 batch가 존재하는지 계산
        val_batch_iter = data_provider.get_val_iter(self.batch_size)  # validation을 위한 iterator(영원히 실행)

        saver = tf.train.Saver(max_to_keep=3)
        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        if resume:
            # 이전에 저장한 모델 정보를 가져옴
            _, model_dir = self.get_model_id_and_dir()
            self.restore_model(saver, model_dir)

        current_lr = lr
        counter = 0
        start_time = time.time()

        for ei in range(epoch):
            # training data 얻음
            train_batch_iter = data_provider.get_train_iter(self.batch_size)

            # schedule 횟수만큼 돌 때마다 learning rate 줄여나감(slash)
            if (ei + 1) % schedule == 0:
                update_lr = current_lr / 2.0
                # minimum learning rate guarantee
                # 단, 0.0002보다는 낮아지지 않도록
                update_lr = max(update_lr, 0.0002)
                print("decay learning rate from %.5f to %.5f" % (current_lr, update_lr))
                current_lr = update_lr

            for bid, batch in enumerate(train_batch_iter):
                counter += 1
                labels, batch_images = batch
                shuffled_ids = labels[:]
                if flip_labels:
                    np.random.shuffle(shuffled_ids)
                # Optimize D
                # Discriminator 최적화
                _, batch_d_loss, d_summary = self.sess.run([d_optimizer, loss_handle.d_loss,
                                                            summary_handle.d_merged],
                                                           feed_dict={
                                                               real_data: batch_images,
                                                               embedding_ids: labels,
                                                               learning_rate: current_lr,
                                                               no_target_data: batch_images,
                                                               no_target_ids: shuffled_ids
                                                           })
                # Optimize G
                # Generator 최적화
                _, batch_g_loss = self.sess.run([g_optimizer, loss_handle.g_loss],
                                                feed_dict={
                                                    real_data: batch_images,
                                                    embedding_ids: labels,
                                                    learning_rate: current_lr,
                                                    no_target_data: batch_images,
                                                    no_target_ids: shuffled_ids
                                                })
                # magic move to Optimize G again
                # according to https://github.com/carpedm20/DCGAN-tensorflow
                # collect all the losses along the way
                _, batch_g_loss, category_loss, cheat_loss, \
                const_loss, l1_loss, tv_loss, g_summary = self.sess.run([g_optimizer,
                                                                         loss_handle.g_loss,
                                                                         loss_handle.category_loss,
                                                                         loss_handle.cheat_loss,
                                                                         loss_handle.const_loss,
                                                                         loss_handle.l1_loss,
                                                                         loss_handle.tv_loss,
                                                                         summary_handle.g_merged],
                                                                        feed_dict={
                                                                            real_data: batch_images,
                                                                            embedding_ids: labels,
                                                                            learning_rate: current_lr,
                                                                            no_target_data: batch_images,
                                                                            no_target_ids: shuffled_ids
                                                                        })
                passed = time.time() - start_time
                log_format = "Epoch: [%2d], [%4d/%4d] time: %4.4f, d_loss: %.5f, g_loss: %.5f, " + \
                             "category_loss: %.5f, cheat_loss: %.5f, const_loss: %.5f, l1_loss: %.5f, tv_loss: %.5f"
                print(log_format % (ei, bid, total_batches, passed, batch_d_loss, batch_g_loss,
                                    category_loss, cheat_loss, const_loss, l1_loss, tv_loss))
                summary_writer.add_summary(d_summary, counter)
                summary_writer.add_summary(g_summary, counter)

                if counter % sample_steps == 0:
                    # sample the current model states with val data
                    self.validate_model(val_batch_iter, ei, counter)

                if counter % checkpoint_steps == 0:
                    print("Checkpoint: save checkpoint step %d" % counter)
                    self.checkpoint(saver, counter)
        # save the last checkpoint
        print("Checkpoint: last checkpoint step %d" % counter)
        self.checkpoint(saver, counter)
