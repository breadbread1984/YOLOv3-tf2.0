#!/usr/bin/python3

from os import environ, listdir;
from os.path import join;
import numpy as np;
import tensorflow as tf;
from models import YOLOv3, Loss;
from create_dataset import parse_function_generator, parse_function;

environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1';
#environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2'
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3';
#os.environ['CUDA_VISIBLE_DEVICES'] = '';
batch_size = 12; # images of different sizes can't be stack into a batch

def main():

  yolov3 = YOLOv3((416,416,3,), 80);
  loss1 = Loss((416,416,3,), 0, 80);
  loss2 = Loss((416,416,3,), 1, 80);
  loss3 = Loss((416,416,3,), 2, 80);
  yolov3.compile(optimizer = tf.keras.optimizers.Adam(1e-5), loss = {'output1': lambda labels, outputs: loss1([outputs, labels]),
                                                                     'output2': lambda labels, outputs: loss2([outputs, labels]),
                                                                     'output3': lambda labels, outputs: loss3([outputs, labels])});
  # load downloaded dataset
  trainset_filenames = [join('trainset', filename) for filename in listdir('trainset')];
  testset_filenames = [join('testset', filename) for filename in listdir('testset')];
  trainset = tf.data.TFRecordDataset(trainset_filenames).map(parse_function_generator(80)).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = tf.data.TFRecordDataset(testset_filenames).map(parse_function_generator(80)).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  yolov3.fit(trainset, epochs = 100, validation_data = testset);
  yolov3.save('yolov3.h5');

if __name__ == "__main__":
  
  assert tf.executing_eagerly();
  main();

