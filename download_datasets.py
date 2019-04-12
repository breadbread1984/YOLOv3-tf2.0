#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_datasets as tfds;

def main():
    
    # load dataset
    coco2014_builder = tfds.builder("coco2014");
    coco2014_builder.download_and_prepare();
    # try to load the dataset once
    coco2014_train = tfds.load(name = "coco2014", split = tfds.Split.TRAIN, download = False);
    coco2014_test = tfds.load(name = "coco2014", split = tfds.Split.TEST, download = False);

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    main();
