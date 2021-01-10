#!/usr/bin/python3

from os import environ, mkdir, listdir;
from os.path import join, exists;
import numpy as np;
import cv2;
import tensorflow as tf;
from models import YOLOv3, Loss;
from Predictor import Predictor;
from create_dataset import parse_function_generator, parse_function;

environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1';
#environ['TF_CPP_MIN_VLOG_LEVEL'] = '3';
#environ['CUDA_VISIBLE_DEVICES'] = '';
batch_size = 12; # images of different sizes can't be stack into a batch

def main():

  # yolov3 model
  yolov3 = YOLOv3((416,416,3), 80);
  yolov3_loss = Loss((416,416,3), 80);
  #optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.PiecewiseConstantDecay([400, 1000], [1e-11, 1e-4, 1e-3]));
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = 110000, decay_rate = 0.99));
  checkpoint = tf.train.Checkpoint(model = yolov3, optimizer = optimizer);
  train_loss = tf.keras.metrics.Mean(name = 'train loss', dtype = tf.float32);
  test_loss = tf.keras.metrics.Mean(name = 'test loss', dtype = tf.float32);
  # load downloaded dataset
  trainset_filenames = [join('trainset', filename) for filename in listdir('trainset')];
  testset_filenames = [join('testset', filename) for filename in listdir('testset')];
  trainset = tf.data.TFRecordDataset(trainset_filenames).map(parse_function_generator(80)).repeat(-1).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = tf.data.TFRecordDataset(testset_filenames).map(parse_function_generator(80)).repeat(-1).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  validationset = tf.data.TFRecordDataset(testset_filenames).map(parse_function).repeat(-1);
  trainset_iter = iter(trainset);
  testset_iter = iter(testset);
  validationset_iter = iter(validationset);
  # restore from existing checkpoint
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # tensorboard summary
  log = tf.summary.create_file_writer('checkpoints');
  # train model
  while True:
    images, labels = next(trainset_iter);
    with tf.GradientTape() as tape:
      outputs = yolov3(images);
      loss = yolov3_loss([*outputs, *labels]);
    # check whether the loss numberic is correct
    if tf.math.reduce_any(tf.math.is_nan(loss)) == True:
      print("NaN was detected in loss, skip the following steps!");
      continue;
    grads = tape.gradient(loss, yolov3.trainable_variables);
    # check whether the grad numerics is correct
    if tf.math.reduce_any([tf.math.reduce_any(tf.math.is_nan(grad)) for grad in grads]) == True:
      print("NaN was detected in gradients, skip gradient apply!");
      continue;
    optimizer.apply_gradients(zip(grads, yolov3.trainable_variables));
    train_loss.update_state(loss);
    # save model
    if tf.equal(optimizer.iterations % 10000, 0):
      # save checkpoint every 1000 steps
      checkpoint.save(join('checkpoints','ckpt'));
      yolov3.save('yolov3.h5');
    if tf.equal(optimizer.iterations % 100, 0):
      # evaluate
      for i in range(10):
        images, labels = next(testset_iter); # images.shape = (b, h, w, 3)
        outputs = yolov3(images);
        loss = yolov3_loss([*outputs, *labels]);
        test_loss.update_state(loss);
      # visualize
      image, bbox, labels = next(validationset_iter); # image.shape = (h, w, 3)
      image = image.numpy().astype('uint8');
      predictor = Predictor(yolov3 = yolov3);
      boundings = predictor.predict(image);
      color_map = dict();
      for bounding in boundings:
        if bounding[5].numpy().astype('int32') in color_map:
          clr = color_map[bounding[5].numpy().astype('int32')];
        else:
          color_map[bounding[5].numpy().astype('int32')] = tuple(np.random.randint(low = 0, high = 256, size = (3,)).tolist());
          clr = color_map[bounding[5].numpy().astype('int32')];
        cv2.rectangle(image, tuple(bounding[0:2].numpy().astype('int32')), tuple(bounding[2:4].numpy().astype('int32')), clr, 5);
      image = tf.expand_dims(image, axis = 0);
      # write log
      with log.as_default():
        tf.summary.scalar('train loss', train_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('test loss', test_loss.result(), step = optimizer.iterations);
        tf.summary.image('detect', image[...,::-1], step = optimizer.iterations);
      print('Step #%d Train Loss: %.6f Test Loss: %.6f' % (optimizer.iterations, train_loss.result(), test_loss.result()));
      # break condition
      #if train_loss.result() < 0.001: break;
      # reset
      train_loss.reset_states();
      test_loss.reset_states();
  yolov3.save('yolov3.h5');

if __name__ == "__main__":
  
  assert tf.executing_eagerly();
  main();
