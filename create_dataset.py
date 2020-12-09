#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
from shutil import rmtree;
from math import ceil;
from multiprocessing import Process;
from pycocotools.coco import COCO;
import numpy as np;
import cv2;
import tensorflow as tf;

PROCESS_NUM = 80;
label_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, -1, 25, 26, -1, -1, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, -1, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, -1, 61, -1, -1, 62, -1, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, -1, 74, 75, 76, 77, 78, 79, 80];

def bbox_to_tensor(img_shape, num_classes = 80):

  # NOTE: img_shape = (width, height)
  assert len(img_shape) == 2;
  anchors = [[[116,90], [156,198], [373,326]], [[30,61], [62,45], [59,119]], [[10,13], [16,30], [33,23]]]; # anchors.shape = (level num = 3, anchor num = 3, 2)
  bbox = tf.keras.Input((4,)); # bbox.shape = (obj_num, 4)
  labels = tf.keras.Input(()); # labels.shape = (obj_num)
  relative_bbox_center = tf.keras.layers.Lambda(lambda x: tf.reverse((x[..., 0:2] + x[..., 2:4]) / 2, axis = [-1]))(bbox); # relative_bbox_center.shape = (obj_num, 2) in sequence of (center x, center y)
  relative_bbox_wh = tf.keras.layers.Lambda(lambda x: tf.reverse(tf.math.abs(x[..., 2:4] - x[..., 0:2]), axis = [-1]))(bbox); # relative_bbox_wh.shape = (obj_num, 2) in sequence of (w, h)
  relative_bbox = tf.keras.layers.Concatenate(axis = -1)([relative_bbox_center, relative_bbox_wh]); # relative_bbox.shape = (obj_num, 4) in sequence of (center x, center y, w, h)
  valid_bbox = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(x, tf.greater(x[...,2], 0)))(relative_bbox); # valid_bbox.shape = (valid_num, 4) in sequence of (center x, center y, w, h)
  valid_labels = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(x[0], tf.greater(x[1][...,2], 0)))([labels, relative_bbox]); # valid_labels.shape = (valid_num)
  bbox_maxes = tf.keras.layers.Lambda(lambda x, s: x[...,2:4] * tf.expand_dims(s, axis = 0) / 2, arguments = {'s': img_shape})(valid_bbox); # bbox_maxes.shape = (valid_num, 2)
  bbox_mins = tf.keras.layers.Lambda(lambda x: -x)(bbox_maxes); # bbox_mins.shape = (valid_num, 2)
  bbox_wh = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([bbox_maxes, bbox_mins]); # bbox_wh.shape = (valid num, 2)
  bbox_area = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(x[...,0] * x[...,1], axis = 0), axis = 0))(bbox_wh); # bbox_area.shape = (1, 1, valid_num)
  intersect_maxes = tf.keras.layers.Lambda(lambda x, a: tf.math.minimum(tf.expand_dims(a/2, axis = -2), tf.expand_dims(tf.expand_dims(x, axis = 0), axis = 0)), arguments = {'a': anchors})(bbox_maxes); # intersect_maxes.shape = (level num, anchor num, valid num, 2)
  intersect_mins = tf.keras.layers.Lambda(lambda x, a: tf.math.maximum(tf.expand_dims(-a/2, axis = -2), tf.expand_dims(tf.expand_dims(x, axis = 0), axis = 0)), arguments = {'a': anchors})(bbox_mins); # intersect_mins.shape = (level num, anchor num, valid num, 2)
  intersect_wh = tf.keras.layers.Lambda(lambda x: tf.math.maximum(x[0] - x[1], 0.))([intersect_maxes, intersect_mins]); # intersect_wh.shape = (level num, anchor num, valid num, 2)
  intersect_area = tf.keras.layers.Lambda(lambda x: x[...,0] * x[...,1])(intersect_wh); # intersect_area.shape = (level num, anchor num, valid num)
  iou = tf.keras.layers.Lambda(lambda x, a: x[0] / (x[1] + tf.expand_dims(a[...,0] * a[...,1], axis = -1)), arguments = {'a': anchors})([intersect_area, bbox_area]); # iou.shape = (level num, anchor num, valid num)
  best_idx = tf.keras.layers.Lambda(lambda x: tf.math.argmax(tf.reshape(x, (-1, tf.shape(x)[-1])), axis = 0))(iou); # best_idx.shape = (valid num)
  best_levels = tf.keras.layers.Lambda(lambda x: x // 3)(best_idx); # best_levels.shape = (valid_num)
  best_anchors = tf.keras.layers.Lambda(lambda x: x % 3)(best_idx); # best_anchors.shape = (valid_num)
  # generate labels
  level1_mask = tf.keras.layers.Lambda(lambda x: tf.math.equal(x, 0))(best_levels); # level1_mask.shape = (valid_num)
  level1_bbox = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(x[0], x[1]))([valid_bbox, level1_mask]); # level1_bbox.shape = (level1 num, 4) in sequence of (center x, center y, w, h)
  level1_labels = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(x[0], x[1]))([valid_labels, level1_mask]); # level1_labels.shape = (level1 num)
  level1_coords = tf.keras.layers.Lambda(lambda x, h, w: tf.cilp_by_value(tf.cast(tf.reverse(x[..., 0:2], axis = [-1]) * tf.constant([[h // 32, w // 32]], dtype = tf.float32), dtype = tf.int32), clip_value_min = 0, clip_value_max = [[h//32-1, w//32-1]]), arguments = {'h': img_shape[1], 'w': img_shape[0]})(level1_bbox); # level1_coords.shape = (level1_num, 2) in sequence of (h, w)
  level1_outputs = tf.keras.layers.Lambda(lambda x, c: tf.concat([x[0], tf.ones((tf.shape(x[0])[0], 1), dtype = tf.float32), tf.one_hot(x[1], c)], axis = -1), arguments = {'c': num_classes})([level1_bbox, level1_labels]); # level1_outputs.shape = (level1_num, 5 + c)
  level1_gt = tf.keras.layers.Lambda(lambda x, h, w, c: tf.scatter_nd(updates = x[0], indices = x[1], shape = (h // 32, w // 32, 5 + c)), arguments = {'h': img_shape[1], 'w': img_shape[0], 'c': num_classes})([level1_outputs, level1_coords]); # level1_gt.shape = (h//32, w//32, 5+c)
  level2_mask = tf.keras.layers.Lambda(lambda x: tf.math.equal(x, 1))(best_levels); # level2_mask.shape = (valid_num)
  level2_bbox = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(x[0], x[1]))([valid_bbox, level2_mask]); # level2_bbox.shape = (level2 num, 4)
  level2_labels = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(x[0], x[1]))([valid_labels, level2_mask]); # level2_labels.shape = (level2 num)
  level2_coords = tf.keras.layers.Lambda(lambda x, h, w: tf.clip_by_value(tf.cast(tf.reverse(x[..., 0:2], axis = [-1]) * tf.constant([[h // 16, w // 16]], dtype = tf.float32), dtype = tf.int32), clip_value_min = 0, clip_value_max = [[h//16-1. w//16-1]]), arguments = {'h': img_shape[1], 'w': img_shape[0]})(level2_bbox); # level2_outputs.shape = (level2_num, 2) in sequence of (h, w)
  level2_outputs = tf.keras.layers.Lambda(lambda x, c: tf.concat([x[0], tf.ones((tf.shape(x[0])[0], 1), dtype = tf.float32), tf.one_hot(x[1], c)], axis = -1), arguments = {'c': num_classes})([level2_bbox, level2_labels]); # level2_outputs.shape = (level2_num, 5 + c)
  level2_gt = tf.keras.layers.Lambda(lambda x, h, w, c: tf.scatter_nd(updates = x[0], indices = x[1], shape = (h // 16, w // 16, 5 + c)), arguments = {'h': img_shape[1], 'w': img_shape[0], 'c': num_classes})([level2_outputs, level2_coords]); # level2_gt.shape = (h//16, w//16, 5+c)
  level3_mask = tf.keras.layers.Lambda(lambda x: tf.math.equal(x, 2))(best_levels); # level3_mask.shape = (valid_num)
  level3_bbox = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(x[0], x[1]))([valid_bbox, level3_mask]); # level3_bbox.shape = (level3 num, 4)
  level3_labels = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(x[0], x[1]))([valid_labels, level3_mask]); # level3_labels.shape = (level3 num)
  level3_coords = tf.keras.layers.Lambda(lambda x, h, w: tf.clip_by_value(tf.cast(tf.reverse(x[..., 0:2], axis = [-1]) * tf.constant([[h // 8, w // 8]], dtype = tf.float32), dtype = tf.int32), clip_value_min = 0, clip_value_max  = [[h//8-1, w//8-1]]), arguments = {'h': img_shape[1], 'w': img_shape[0]})(level3_bbox); # level3_outputs.shape = (level3_num, 2) in sequence of (h, w)
  level3_outputs = tf.keras.layers.Lambda(lambda x, c: tf.concat([x[0], tf.ones((tf.shape(x[0])[0], 1), dtype = tf.float32), tf.one_hot(x[1], c)], axis = -1), arguments = {'c': num_classes})([level3_bbox, level3_labels]); # level3_outputs.shape = (level3_num, 5 + c)
  level3_gt = tf.keras.layers.Lambda(lambda x, h, w, c: tf.scatter_nd(updates = x[0], indices = x[1], shape = (h // 8, w // 8, 5 + c)), arguments = {'h': img_shape[1], 'w': img_shape[0], 'c': num_classes})([level3_outputs, level3_coords]); # level3_gt.shape = (h//8, w//8, 5+c)
  return tf.keras.Model(inputs = (bbox, labels), outputs = (level1_gt, level2_gt, level3_gt));

def parse_function_generator(num_classes, input_shape = (416,416), random = False, jitter = .3, hue = .1, sat = 1.5, bri = .1):
  assert 0 < jitter < 1;
  assert -1 < hue < 1;
  assert 0 < sat;
  assert 0 < bri < 1;
  def parse_function_noaug(serialized_example):
    feature = tf.io.parse_single_example(
      serialized_example,
      features = {
        'image': tf.io.FixedLenFeature((), dtype = tf.string),
        'bbox': tf.io.VarLenFeature(dtype = tf.float32),
        'label': tf.io.VarLenFeature(dtype = tf.int64),
        'obj_num': tf.io.FixedLenFeature((), dtype = tf.int64)
      });
    obj_num = tf.cast(feature['obj_num'], dtype = tf.int32);
    image = tf.io.decode_jpeg(feature['image']);
    bbox = tf.sparse.to_dense(feature['bbox'], default_value = 0);
    bbox = tf.reshape(bbox, (obj_num, 4));
    label = tf.sharse.to_dense(feature['label'], default_value = 0);
    label = tf.reshape(label, (obj_num));
    # scale the input image to make the wider edge fit the input shape
    # NOTE: I don't use resize_with_pad because it can only stuff zeros, but I want 128
    resize_image = tf.image.resize(image, input_shape, method = tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio = True);
    resize_shape = resize_image.shape[1:3]; #(height, width)
    top_pad = (input_shape[0] - resize_shape[0]) // 2;
    bottom_pad = input_shape[0] - resize_shape[0] - top_pad;
    left_pad = (input_shape[1] - resize_shape[1]) // 2;
    right_pad = input_shape[1] - resize_shape[1] - left_pad;
    resize_image = tf.pad(resize_image, [[0,0], [top_pad,bottom_pad], [left_pad,right_pad], [0,0]], constant_values = 128);
    # cast to float32
    image_data = tf.cast(resize_image, tf.float32) / 255.; # image_data.shape = (h, w, 3)
    # correct boxes
    bbox = bbox * tf.constant([resize_shape[0], resize_shape[1], resize_shape[0], resize_shape[1]], dtype = tf.float32);
    bbox = bbox + tf.constant([top_pad, left_pad, top_pad, left_pad], dtype = tf.float32);
    bbox = bbox / tf.constant([input_shape[0], input_shape[1], input_shape[0], input_shape[1]], dtype = tf.float32); # bbox.shape = (obj_num, 4)
    # return
    return image_data, bbox;
    
    
    image, label1, label2, label3 = tf.py_function(map_function_impl_generator(num_classes), inp = [image, bbox, label], Tout = [tf.float32, tf.float32, tf.float32, tf.float32]);
    return image, (label1, label2, label3);
  def parse_function_aug(serialized_example):
    feature = tf.io.parse_single_example(
      serialized_example,
      features = {
        'image': tf.io.FixedLenFeature((), dtype = tf.string),
        'bbox': tf.io.VarLenFeature(dtype = tf.float32),
        'label': tf.io.VarLenFeature(dtype = tf.int64),
        'obj_num': tf.io.FixedLenFeature((), dtype = tf.int64)
      });
    obj_num = tf.cast(feature['obj_num'], dtype = tf.int32);
    image = tf.io.decode_jpeg(feature['image']);
    bbox = tf.sparse.to_dense(feature['bbox'], default_value = 0);
    bbox = tf.reshape(bbox, (obj_num, 4));
    label = tf.sharse.to_dense(feature['label'], default_value = 0);
    label = tf.reshape(label, (obj_num));
    
  return parse_function_noaug if random == False else parse_function_aug;

def create_dataset(image_dir, label_dir, trainset = True):

  anno = COCO(join(label_dir, 'instances_train2017.json' if trainset else 'instances_val2017.json'));
  if exists('trainset' if trainset else 'testset'): rmtree('trainset' if trainset else 'testset');
  mkdir('trainset' if trainset else 'testset');
  imgs_for_each = ceil(len(anno.getImgIds()) / PROCESS_NUM);
  handlers = list();
  filenames = list();
  for i in range(PROCESS_NUM):
    filename = ('trainset_part_%d' if trainset else 'testset_part_%d') % i;
    filenames.append(join('trainset' if trainset else 'testset', filename));
    handlers.append(Process(target = worker, args = (join('trainset' if trainset else 'testset', filename), anno, image_dir, anno.getImgIds()[i * imgs_for_each:(i+1) * imgs_for_each] if i != PROCESS_NUM - 1 else anno.getImgIds()[i * imgs_for_each:])));
    handlers[-1].start();
  for handler in handlers:
    handler.join();

def worker(filename, anno, image_dir, image_ids):
  writer = tf.io.TFRecordWriter(filename);
  for image in image_ids:
    img_info = anno.loadImgs([image])[0];
    height, width = img_info['height'], img_info['width'];
    img = cv2.imread(join(image_dir, img_info['file_name']));
    if img is None:
      print('can\'t open image %s' % (join(image_dir, img_info['file_name'])));
      continue;
    annIds = anno.getAnnIds(imgIds = image);
    anns = anno.loadAnns(annIds);
    bboxs = list();
    labels = list();
    for ann in anns:
      # bounding box
      bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox'];
      bbox = tf.constant([bbox_y / height, bbox_x / width, (bbox_y + bbox_h) / height, (bbox_x + bbox_w) / width], dtype = tf.float32);
      bboxs.append(bbox);
      # category
      category = label_map[ann['category_id']];
      assert category != -1;
      labels.append(category);
    bboxs = tf.cast(tf.stack(bboxs, axis = 0), dtype = tf.float32); # bboxs.shape = (obj_num, 4)
    labels = tf.cast(tf.stack(labels, axis = 0), dtype = tf.int32); # labels.shape = (obj_num)
    trainsample = tf.train.Example(features = tf.train.Features(
      feature = {
        'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.encode_jpeg(img).numpy()])),
        'bbox': tf.train.Feature(float_list = tf.train.FloatList(value = bbox.reshape(-1))),
        'label': tf.train.Feature(int64_list = tf.train.Int64List(value = label.reshape(-1))),
        'obj_num': tf.train.Feature(int64_list = tf.train.Int64List(value = [bbox.shape[0]]))
      }
    ));
    writer.write(trainsample.SerializeToString());
  writer.close();


def map_function_impl_generator(num_classes):
    def map_function_impl(image, bbox, label):
        image, bbox = preprocess(image, bbox, random = True);
        label1, label2, label3 = bbox_to_tensor(bbox, label, num_classes = num_classes);
        return image, label1, label2, label3;
    return map_function_impl;

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <path/to/annotation>");
        exit(1);
    assert tf.executing_eagerly() == True;
    main(sys.argv[1]);

