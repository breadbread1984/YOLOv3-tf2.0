#!/usr/bin/python3

import os;
import numpy as np;
import tensorflow as tf;
import tensorflow_datasets as tfds;
from YOLOv3 import YOLOv3, Loss;
from preprocess import map_function;

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
    trainset = tfds.load(name = "coco2014", split = tfds.Split.TRAIN, download = False);
    trainset = trainset.map(map_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
    validationset = tfds.load(name = "coco2014", split = tfds.Split.VALIDATION, download = False);
    validationset = validationset.map(map_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
    yolov3.fit(trainset, epochs = 100, validation_data = validationset);
    yolov3.save('yolov3.h5');

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    main();

