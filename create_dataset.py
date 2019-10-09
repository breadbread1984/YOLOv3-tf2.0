#!/usr/bin/python3

import sys;
from re import split;
from random import shuffle;
import enum;
import cv2;
import tensorflow as tf;
from preprocess import preprocess, bbox_to_tensor;

def main(file_path):

    f = open(file_path);
    lines = f.readlines();
    f.close();
    annotations = dict();
    class Status(enum.Enum):
        Num = 1;
        Anno = 2;
    s = Status.Num;
    filepath = None;
    target_num = None;
    img_shape = None;
    for line in lines:
        line = line.strip();
        tokens = split('\t| ', line);
        if s == Status.Num:
            if len(tokens) != 2:
                print("invalid format! line denoting target number should be in format \'<path/to/image> <target num>\'!");
                exit(1);
            filepath = tokens[0];
            target_num = int(tokens[1]);
            img = cv2.imread(filepath);
            if img is None:
                print('invalid image path!');
                exit(1);
            img_shape = img.shape;
            annotations[filepath] = list();
            s = Status.Anno;
        elif s == Status.Anno:
            if len(tokens) != 5:
                print("invalid format! line denoting target annotation should be in format \'<x> <y> <width> <height> <label>\'!");
                exit(1);
            annotations[filepath].append((
                float(tokens[1]) / img_shape[0], # upper left y
                float(tokens[0]) / img_shape[1], # upper let x
                float(tokens[1] + tokens[3]) / img_shape[0], # down right y
                float(tokens[0] + tokens[2]) / img_shape[1], # down right x
                int(tokens[4]), # label
            ));
            target_num -= 1;
            if target_num == 0: s = Status.Num;
        else:
            print('unknown status!');
            exit(1);

    shuffle(annotations);
    trainset_num = int(9 / 10 * len(annotations));

    def write_tfrecord(output, annotations):
        writer = tf.io.TFRecordWriter(output);
        for filepath, labels in annotations.items():
            img = cv2.imread(filepath);
            assert img is not None;
            annotation = np.array(labels, dtype = np.float32);
            bbox = annotation[..., 0:4];
            label = annotation[..., 4].astype('int64');
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
        
    write_tfrecord('trainset.tfrecord', annotations[:trainset_num]);
    write_tfrecord('validationset.tfrecord', annotations[trainset_num:]);

def parse_function_generator(num_classes):
    def parse_function(serialized_example):

        feature = tf.io.parse_single_example(
            serialized_example,
            features = {
                'image': tf.io.FixedLenFeature((), dtype = tf.string),
                'bbox': tf.io.VarLenFeature(dtype = tf.float32),
                'label': tf.io.VarLenFeature(dtype = tf.int64),
                'obj_num': tf.io.FixedLenFeature((), dtype = tf.int64)
            }
        );
        obj_num = tf.cast(feature['obj_num'], dtype = tf.int32);
        image = tf.io.decode_jpeg(feature['image']);
        bbox = tf.sparse.to_dense(feature['bbox'], default_value = 0);
        bbox = tf.reshape(bbox, (obj_num, 4));
        label = tf.sharse.to_dense(feature['label'], default_value = 0);
        label = tf.reshape(label, (obj_num));
        image, label1, label2, label3 = tf.py_function(map_function_impl_generator(num_classes), inp = [image, bbox, label], Tout = [tf.float32, tf.float32, tf.float32, tf.float32]);
        return image, (label1, label2, label3);
    return parse_function;

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

