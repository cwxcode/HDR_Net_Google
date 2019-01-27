#!/usr/bin/env python
# encoding: utf-8
# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluates a trained network."""

import argparse
import cv2
import logging
import numpy as np
import scipy
import imageio
import os
import re
import setproctitle
import skimage
import skimage.io
import skimage.transform
import sys
sys.path.insert(0, "/home/chenwx/hdrnet-master_v4")
import time
import tensorflow as tf

import hdrnet.models as models
import hdrnet.utils as utils


logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger("train")
log.setLevel(logging.INFO)


def get_input_list(path):
  regex = re.compile(".*.(png|jpeg|jpg|tif|tiff)")  # 支持的输入格式
  if os.path.isdir(path):  # 处理目录下的所有图片
    inputs = os.listdir(path)  # 获取目录下所有文件名
    inputs = [os.path.join(path, f) for f in inputs if regex.match(f)]
    log.info("Directory input {}, with {} images".format(path, len(inputs)))

  elif os.path.splitext(path)[-1] == ".txt":  # 处理文本列表中的图片
    dirname = os.path.dirname(path)  # 获取路径名
    with open(path, 'r') as fid:
      inputs = [l.strip() for l in fid.readlines()]
    inputs = [os.path.join(dirname, 'input', im) for im in inputs]
    log.info("Filelist input {}, with {} images".format(path, len(inputs)))
  elif regex.match(path):  # 处理单张图片
    inputs = [path]
    log.info("Single input {}".format(path))
  return inputs


def main(args):
  setproctitle.setproctitle('hdrnet_run')  # 进程名称

  inputs = get_input_list(args.input)  # 输入图片

  # -------- Load params ----------------------------------------------------
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  # 设置GPU
  with tf.Session(config=config) as sess:
    checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_dir)  # 加载预训练的模型
    if checkpoint_path is None:
      log.error('Could not find a checkpoint in {}'.format(args.checkpoint_dir))
      return

    metapath = ".".join([checkpoint_path, "meta"])
    log.info('Loading graph from {}'.format(metapath))
    tf.train.import_meta_graph(metapath)

    model_params = utils.get_model_params(sess)

  # -------- Setup graph ----------------------------------------------------
  if not hasattr(models, model_params['model_name']):
    log.error("Model {} does not exist".format(params.model_name))
    return
  mdl = getattr(models, model_params['model_name'])

  tf.reset_default_graph()
  net_shape = model_params['net_input_size']
  t_fullres_input = tf.placeholder(tf.float32, (1, None, None, 3))
  t_lowres_input = tf.placeholder(tf.float32, (1, net_shape, net_shape, 3))

  with tf.variable_scope('inference'):
    prediction = mdl.inference(
        t_lowres_input, t_fullres_input, model_params, is_training=False)
  output = tf.cast(255.0*tf.squeeze(tf.clip_by_value(prediction, 0, 1)), tf.uint8)  # 8bit ？？？
  # output = tf.cast(65535.0*tf.squeeze(tf.clip_by_value(prediction, 0, 1)), tf.uint16)  # 改为16bit
  saver = tf.train.Saver()

  if args.debug:  # debug模式
    coeffs = tf.get_collection('bilateral_coefficients')[0]
    if len(coeffs.get_shape().as_list()) == 6:
      bs, gh, gw, gd, no, ni = coeffs.get_shape().as_list()
      coeffs = tf.transpose(coeffs, [0, 3, 1, 4, 5, 2])
      coeffs = tf.reshape(coeffs, [bs, gh*gd, gw*ni*no, 1])
      coeffs = tf.squeeze(coeffs)
      m = tf.reduce_max(tf.abs(coeffs))
      coeffs = tf.clip_by_value((coeffs+m)/(2*m), 0, 1)

    ms = tf.get_collection('multiscale')
    if len(ms) > 0:
      for i, m in enumerate(ms):
        maxi = tf.reduce_max(tf.abs(m))
        m = tf.clip_by_value((m+maxi)/(2*maxi), 0, 1)
        sz = tf.shape(m)
        m = tf.transpose(m, [0, 1, 3, 2])
        m = tf.reshape(m, [sz[0], sz[1], sz[2]*sz[3]])
        ms[i] = tf.squeeze(m)

    fr = tf.get_collection('fullres_features')
    if len(fr) > 0:
      for i, m in enumerate(fr):
        maxi = tf.reduce_max(tf.abs(m))
        m = tf.clip_by_value((m+maxi)/(2*maxi), 0, 1)
        sz = tf.shape(m)
        m = tf.transpose(m, [0, 1, 3, 2])
        m = tf.reshape(m, [sz[0], sz[1], sz[2]*sz[3]])
        fr[i] = tf.squeeze(m)

    guide = tf.get_collection('guide')
    if len(guide) > 0:
      for i, g in enumerate(guide):
        maxi = tf.reduce_max(tf.abs(g))
        g = tf.clip_by_value((g+maxi)/(2*maxi), 0, 1)
        guide[i] = tf.squeeze(g)

  with tf.Session(config=config) as sess:  # 会话
    log.info('Restoring weights from {}'.format(checkpoint_path))
    saver.restore(sess, checkpoint_path)

    for idx, input_path in enumerate(inputs):
      if args.limit is not None and idx >= args.limit:
        log.info("Stopping at limit {}".format(args.limit))
        break

      log.info("Processing {}".format(input_path))
      im_input = cv2.imread(input_path, -1)  # -1 means read as is, no conversions.  使用opencv读取图片
      if im_input.shape[2] == 4:
        log.info("Input {} has 4 channels, dropping alpha".format(input_path))  
        im_input = im_input[:, :, :3]  # 4通道改为3通道

      im_input = np.flip(im_input, 2)  # OpenCV reads BGR, convert back to RGB.
      log.info("input dtype: {}".format(im_input.dtype))

      log.info("Max level: {}".format(np.amax(im_input[:, :, 0])))
      log.info("Max level: {}".format(np.amax(im_input[:, :, 1])))
      log.info("Max level: {}".format(np.amax(im_input[:, :, 2])))

      # HACK for HDR+.
      if im_input.dtype == np.uint16 and args.hdrp:  # 当输入图片是16bit，且开了hdrp
        log.info("Using HDR+ hack for uint16 input. Assuming input white level is 32767.")
        # im_input = im_input / 32767.0  # 2**15-1
        # im_input = im_input / 32767.0 /2
        im_input = im_input / (1.0*2**16)
        # im_input = skimage.img_as_float(im_input)  
      else:
        im_input = skimage.img_as_float(im_input)

      log.info("After Max level: {}".format(np.amax(im_input[:, :, 0])))
      log.info("After Max level: {}".format(np.amax(im_input[:, :, 1])))
      log.info("After Max level: {}".format(np.amax(im_input[:, :, 2])))

      # Make or Load lowres image  低分辨率图片
      if args.lowres_input is None:
        lowres_input = skimage.transform.resize(
            im_input, [net_shape, net_shape], order = 0)
      else:
        raise NotImplemented

      fname = os.path.splitext(os.path.basename(input_path))[0]  # 获取文件名
      output_path = os.path.join(args.output, fname+".png")  # 输出图片保存为png格式
      # output_path = os.path.join(args.output, fname+".tif")  # 输出图片保存为tif格式
      basedir = os.path.dirname(output_path)

      im_input = im_input[np.newaxis, :, :, :]
      lowres_input = lowres_input[np.newaxis, :, :, :]

      feed_dict = {
          t_fullres_input: im_input,
          t_lowres_input: lowres_input
      }

      out_ =  sess.run(output, feed_dict=feed_dict)  # 得到输出图片

      if not os.path.exists(basedir):
        os.makedirs(basedir)

      log.info("output dtype: {}".format(out_.dtype))
      scipy.misc.imsave(output_path, out_)  # 保存输出图片
      # imageio.imwrite(output_path, out_)  # 保存为16bit tif格式
      

      if args.debug:  # debug模式下的输出图片保存方式
        output_path = os.path.join(args.output, fname+"_input.png")
        scipy.misc.imsave(output_path, np.squeeze(im_input))

        coeffs_ = sess.run(coeffs, feed_dict=feed_dict)
        output_path = os.path.join(args.output, fname+"_coeffs.png")
        scipy.misc.imsave(output_path, coeffs_)
        if len(ms) > 0:
          ms_ = sess.run(ms, feed_dict=feed_dict)
          for i, m in enumerate(ms_):
            output_path = os.path.join(args.output, fname+"_ms_{}.png".format(i))
            scipy.misc.imsave(output_path, m)

        if len(fr) > 0:
          fr_ = sess.run(fr, feed_dict=feed_dict)
          for i, m in enumerate(fr_):
            output_path = os.path.join(args.output, fname+"_fr_{}.png".format(i))
            scipy.misc.imsave(output_path, m)

        if len(guide) > 0:
          guide_ = sess.run(guide, feed_dict=feed_dict)
          for i, g in enumerate(guide_):
            output_path = os.path.join(args.output, fname+"_guide_{}.png".format(i))
            scipy.misc.imsave(output_path, g)



if __name__ == '__main__':
  # -----------------------------------------------------------------------------
  # pylint: disable=line-too-long
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_dir', default=None, help='path to the saved model variables')  # 选择模型，hdrp或faces或其它
  parser.add_argument('input', default=None, help='path to the validation data')
  parser.add_argument('output', default=None, help='path to save the processed images')

  # Optional  设置参数
  parser.add_argument('--lowres_input', default=None, help='path to the lowres, TF inputs')  # 不使用低分辨率图片
  parser.add_argument('--hdrp', dest="hdrp", action="store_true", help='special flag for HDR+ to set proper range')  # 设置为hdrp
  parser.add_argument('--nohdrp', dest="hdrp", action="store_false")
  parser.add_argument('--debug', dest="debug", action="store_true", help='If true, dumps debug data on guide and coefficients.')  # 设置为debug
  parser.add_argument('--limit', type=int, help="limit the number of images processed.")  # 限制图片数量
  parser.set_defaults(hdrp=True, debug=False)  # 默认不开hdrp和debug
  # pylint: enable=line-too-long
  # -----------------------------------------------------------------------------

  args = parser.parse_args()
  main(args)
