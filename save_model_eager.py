#!/usr/bin/python3

import tensorflow as tf;
from models import YOLOv3;

def main():

  yolov3 = YOLOv3((416, 416, 3), 80);
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps = 110000, decay_rate = 0.99));
  checkpoint = tf.train.Checkpoint(model = yolov3, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  yolov3.save('yolov3.h5');
  yolov3.save_weights('yolov3_weights.h5');

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();

