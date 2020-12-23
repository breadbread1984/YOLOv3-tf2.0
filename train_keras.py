#!/usr/bin/python3

import os;
import numpy as np;
import tensorflow as tf;
from models import YOLOv3, Loss;
from create_dataset import parse_function_generator, parse_function;

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1';
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3';
#os.environ['CUDA_VISIBLE_DEVICES'] = '';
batch_size = 8; # images of different sizes can't be stack into a batch

def main():

  # yolov3 model
  yolov3 = YOLOv3((416,416,3,), 80);
  @tf.function
  def loss(outputs, labels):
    return Loss((416,416,3,),80)([outputs[0], outputs[1], outputs[2], labels[0], labels[1], labels[2]]);
  yolov3.compile(optimizer = tf.keras.optimizers.Adam(1e-4), loss = loss);
  # load downloaded dataset
  trainset_filenames = [join('trainset', filename) for filename in listdir('trainset')];
  testset_filenames = [join('testset', filename) for filename in listdir('testset')];
  trainset = tf.data.TFRecordDataset(trainset_filenames).map(parse_function_generator(80)).repeat(-1).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = tf.data.TFRecordDataset(testset_filenames).map(parse_function_generator(80)).repeat(-1).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  yolov3.fit(trainset, epochs = 100, validation_data = testset);
  yolov3.save('yolov3.h5');

if __name__ == "__main__":
  
  assert tf.executing_eagerly();
  main();

