#!/usr/bin/python3

import numpy as np;
import cv2;
import tensorflow as tf;
import tensorflow_datasets as tfds;
from YOLOv3 import YOLOv3;
from Predictor import Predictor;

def main():

    testset = tfds.load(name = "coco2014", split = tfds.Split.TEST, download = False);
    testset = testset.repeat(1);
    predictor = Predictor();
    for features in testset:
        img = features["image"].numpy().astype("uint8");
        boundings = predictor.predict(img);
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR);
        color_map = dict();
        for bounding in boundings:
            if bounding[5].numpy().astype('int32') in color_map:
                clr = color_map[bounding[5].numpy().astype('int32')];
            else:
                color_map[bounding[5].numpy().astype('int32')] = tuple(np.random.randint(low = 0, high = 256, size=(3,)).tolist());
                clr = color_map[bounding[5].numpy().astype('int32')];
            cv2.rectangle(img, tuple(bounding[0:2].numpy().astype('int32')), tuple(bounding[2:4].numpy().astype('int32')), clr, 5);
        cv2.imshow('detection result', img);
        k = cv2.waitKey();
        if k == 'q': break;

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
