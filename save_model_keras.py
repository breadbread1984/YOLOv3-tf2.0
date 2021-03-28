#!/usr/bin/python3

import tensorflow as tf;
from models import YOLOv3;

def main():

  yolov3 = YOLOv3((416, 416, 3), 80);
  yolov3.load_weights('./checkpoints/ckpt/variables/variables');
  yolov3.save('yolov3.h5');
  yolov3.save_weights('yolov3_weights.h5');

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();

