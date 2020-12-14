#!/usr/bin/python3

import sys;
from os.path import exists;
import numpy as np;
import cv2;
import tensorflow as tf;
from models import YOLOv3, OutputParser;

class Predictor(object):

  anchors = {2: [[10, 13], [16, 30], [33, 23]], 1: [[30, 61], [62, 45], [59, 119]], 0: [[116, 90], [156, 198], [373, 326]]};

  def __init__(self, input_shape = (416,416,3), class_num = 80, yolov3 = None):

    if yolov3 is None:
      self.input_shape = input_shape;
      if exists("yolov3.h5"):
        # restore from serialized file
        self.yolov3 = tf.keras.models.load_model('yolov3.h5', compile = False);
      else:
        # restore from checkpoint
        self.yolov3 = YOLOv3(input_shape, class_num);
        optimizer = tf.keras.optimizers.Adam(1e-4);
        checkpoint = tf.train.Checkpoint(model = self.yolov3, optimizer = optimizer, optimizer_step = optimizer.iterations);
        checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
    else:
      self.input_shape = tuple(yolov3.input.shape[1:]);
      self.yolov3 = yolov3;
    self.parsers = [OutputParser(tuple(self.yolov3.outputs[l].shape[1:]), self.input_shape, self.anchors[l]) for l in range(3)];

  def predict(self, image, conf_thres = 0.5, nms_thres = 0.5):

    images = tf.expand_dims(image, axis = 0);
    resize_images = tf.image.resize(images, self.input_shape[:2], method = tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio = True);
    resize_shape = resize_images.shape[1:3];
    top_pad = (self.input_shape[0] - resize_shape[0]) // 2;
    bottom_pad = self.input_shape[0] - resize_shape[0] - top_pad;
    left_pad = (self.input_shape[1] - resize_shape[1]) // 2;
    right_pad = self.input_shape[1] - resize_shape[1] - left_pad;
    resize_images = tf.pad(resize_images,[[0,0], [top_pad,bottom_pad], [left_pad,right_pad], [0,0]], constant_values = 128);
    deviation = tf.constant([left_pad / self.input_shape[1], top_pad / self.input_shape[0], 0, 0], dtype = tf.float32);
    scale = tf.constant([
      self.input_shape[1] / resize_shape[1], self.input_shape[0] / resize_shape[0],
      self.input_shape[1] / resize_shape[1], self.input_shape[0] / resize_shape[0]
    ], dtype = tf.float32);
    images_data = tf.cast(resize_images, tf.float32) / 255.;
    outputs = self.yolov3(images_data);
    whole_targets = tf.zeros((0,6), dtype = tf.float32);
    for i in range(3):
      pred_xy, pred_wh, pred_box_confidence, pred_class = self.parsers[i](outputs[i]);
      pred_box = tf.keras.layers.Concatenate(axis = -1)([pred_xy, pred_wh]);
      # target_mask.shape = (h, w, anchor num)
      target_mask = tf.greater(pred_box_confidence, conf_thres);
      # pred_box_confidence = (pred target num, 1)
      pred_box_confidence = tf.boolean_mask(pred_box_confidence, target_mask);
      pred_box_confidence = tf.expand_dims(pred_box_confidence, axis = -1);
      # pred_box.shape = (pred target num, 4)
      pred_box = tf.boolean_mask(pred_box, target_mask);
      pred_box = (pred_box - deviation) * scale * [image.shape[1], image.shape[0], image.shape[1], image.shape[0]];
      # pred_class.shape = (pred target num, 1)
      pred_class = tf.boolean_mask(pred_class, target_mask);
      pred_class = tf.math.argmax(pred_class, axis = -1);
      pred_class = tf.cast(tf.expand_dims(pred_class, axis = -1), dtype = tf.float32);
      # targets,sgaoe = (pred target num, 6)
      targets = tf.keras.layers.Concatenate(axis = -1)([pred_box, pred_box_confidence, pred_class]);
      whole_targets = tf.keras.layers.Concatenate(axis = 0)([whole_targets, targets]);
    # nms
    descend_idx = tf.argsort(whole_targets[..., 4], direction = 'DESCENDING');
    i = 0;
    while i < descend_idx.shape[0]:
      idx = descend_idx[i];
      cur_upper_left = whole_targets[idx, 0:2] - whole_targets[idx, 2:4] / 2;
      cur_down_right = cur_upper_left + whole_targets[idx, 2:4];
      wh = whole_targets[idx, 2:4];
      area = wh[..., 0] * wh[..., 1];
      following_idx = descend_idx[i+1:];
      following_targets = tf.gather(whole_targets, following_idx);
      following_upper_left = following_targets[..., 0:2] - following_targets[..., 2:4] / 2;
      following_down_right = following_upper_left + following_targets[..., 2:4];
      following_wh = following_targets[..., 2:4];
      following_area = following_wh[..., 0] * following_wh[..., 1];
      max_upper_left = tf.math.maximum(cur_upper_left, following_upper_left);
      min_down_right = tf.math.minimum(cur_down_right, following_down_right);
      intersect_wh = min_down_right - max_upper_left;
      intersect_wh = tf.where(tf.math.greater(intersect_wh, 0), intersect_wh, tf.zeros_like(intersect_wh));
      intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1];
      overlap = intersect_area / (area + following_area - intersect_area);
      indices = tf.where(tf.less(overlap, nms_thres));
      following_idx = tf.gather_nd(following_idx, indices);
      descend_idx = tf.concat([descend_idx[:i + 1], following_idx], axis = 0);
      i += 1;
    whole_targets = tf.gather(whole_targets, descend_idx);
    upper_left = (whole_targets[..., 0:2] - whole_targets[..., 2:4] / 2)
    down_right = (upper_left + whole_targets[..., 2:4])
    boundings = tf.keras.layers.Concatenate(axis = -1)([upper_left, down_right, whole_targets[..., 4:]]);
    return boundings;

if __name__ == "__main__":

  assert tf.executing_eagerly() == True;
  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <image>");
    exit(0);
  predictor = Predictor();
  img = cv2.imread(sys.argv[1]);
  if img is None:
    print("invalid image!");
    exit(1);
  boundings = predictor.predict(img);
  color_map = dict();
  for bounding in boundings:
    if bounding[5].numpy().astype('int32') in color_map:
      clr = color_map[bounding[5].numpy().astype('int32')];
    else:
      color_map[bounding[5].numpy().astype('int32')] = tuple(np.random.randint(low=0, high=256,size=(3,)).tolist());
      clr = color_map[bounding[5].numpy().astype('int32')];
    cv2.rectangle(img, tuple(bounding[0:2].numpy().astype('int32')), tuple(bounding[2:4].numpy().astype('int32')), clr, 2);
  cv2.imshow('people', img);
  cv2.waitKey();

