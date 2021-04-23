#!/usr/bin/python3

from os import environ, listdir;
from os.path import join, exists;
import numpy as np;
import cv2;
import tensorflow as tf;
from models import YOLOv3, Loss;
from Predictor import Predictor;
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
  if exists('./checkpoints/ckpt'): yolov3.load_weights('./checkpoints/ckpt/variables/variables');
  optimizer = tf.keras.optimizers.Adam(1e-5);
  yolov3.compile(optimizer = optimizer, loss = {'output1': lambda labels, outputs: loss1([outputs, labels]),
                                                                     'output2': lambda labels, outputs: loss2([outputs, labels]),
                                                                     'output3': lambda labels, outputs: loss3([outputs, labels])});

  class SummaryCallback(tf.keras.callbacks.Callback):
    def __init__(self, eval_freq = 100):
      self.eval_freq = eval_freq;
      testset = tf.data.TFRecordDataset(testset_filenames).map(parse_function).repeat(-1);
      self.iter = iter(testset);
      self.train_loss = tf.keras.metrics.Mean(name = 'train loss', dtype = tf.float32);
      self.log = tf.summary.create_file_writer('./checkpoints');
    def on_batch_begin(self, batch, logs = None):
      pass;
    def on_batch_end(self, batch, logs = None):
      self.train_loss.update_state(logs['loss']);
      if batch % self.eval_freq == 0:
        image, bbox, labels = next(self.iter);
        image = image.numpy().astype('uint8');
        predictor = Predictor(yolov3 = yolov3);
        boundings = predictor.predict(image);
        color_map = dict();
        for bounding in boundings:
          if bounding[5].numpy().astype('int32') not in color_map:
            color_map[bounding[5].numpy().astype('int32')] = tuple(np.random.randint(low = 0, high = 256, size = (3,)).tolist());
          clr = color_map[bounding[5].numpy().astype('int32')];
          cv2.rectangle(image, tuple(bounding[0:2].numpy().astype('int32')), tuple(bounding[2:4].numpy().astype('int32')), clr, 1);
          cv2.putText(image, predictor.getClsName(bounding[5].numpy().astype('int32')), tuple(bounding[0:2].numpy().astype('int32')), cv2.FONT_HERSHEY_PLAIN, 1, clr, 2);
        image = tf.expand_dims(image, axis = 0);
        with self.log.as_default():
          tf.summary.scalar('train loss', self.train_loss.result(), step = optimizer.iterations);
          tf.summary.image('detect', image[...,::-1], step = optimizer.iterations);
        self.train_loss.reset_states();
    def on_epoch_begin(self, epoch, logs = None):
      pass;
    def on_epoch_end(self, batch, logs = None):
      pass;

  # load downloaded dataset
  trainset_filenames = [join('trainset', filename) for filename in listdir('trainset')];
  testset_filenames = [join('testset', filename) for filename in listdir('testset')];
  trainset = tf.data.TFRecordDataset(trainset_filenames).map(parse_function_generator(80)).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = tf.data.TFRecordDataset(testset_filenames).map(parse_function_generator(80)).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = './checkpoints'),
    tf.keras.callbacks.ModelCheckpoint(filepath = './checkpoints/ckpt', save_freq = 10000),
    SummaryCallback(),
  ];
  yolov3.fit(trainset, epochs = 100, validation_data = testset, callbacks = callbacks);
  yolov3.save('yolov3.h5');

if __name__ == "__main__":
  
  main();
