#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
from YOLOv3 import YOLOv3, OutputParser;

class Predictor(object):

    anchors = {2: [[10, 13], [16, 30], [33, 23]], 1: [[30, 61], [62, 45], [59, 119]], 0: [[116, 90], [156, 198], [373, 326]]};

    def __init__(self, img_shape = (416,416,3), class_num = 80):

        self.img_shape = img_shape;
        self.yolov3 = tf.keras.models.load_model('yolov3.h5', compile = False);
        output_shapes = [
            (img_shape[0] // 32, img_shape[1] // 32, 3, 5 + class_num),
            (img_shape[0] // 16, img_shape[1] // 16, 3, 5 + class_num),
            (img_shape[0] // 8, img_shape[1] // 8, 3, 5 + class_num)
        ];
        self.parsers = [OutputParser(output_shapes[l], img_shape, self.anchors[l]) for l in range(3)];

    def predict(self, image):

        images = tf.expand_dims(image, axis = 0);
        resize_images = tf.image.resize(images, self.img_shape[:2], method = tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio = True);
        resize_shape = resize_images.shape[1:3];
        top_pad = (self.img_shape[0] - resize_shape[0]) // 2;
        bottom_pad = self.img_shape[0] - resize_shape[0] - top_pad;
        left_pad = (self.img_shape[1] - resize_shape[1]) // 2;
        right_pad = self.img_shape[1] - resize_shape[1] - left_pad;
        resize_images = tf.pad(resize_images,[top_pad,bottom_pad],[left_pad,right_pad],[0,0]], constant_values = 128);
        images_data = tf.cast(resize_images, tf.float32) / 255.;
        outputs = self.yolov3(images_data);
        for i in range(3):
            pred_xy, pred_wh, pred_box_confidence, pred_class = self.parsers[i](outputs[i]);
        # TODO

if __name__ == "__main__":

    assert tf.executing_eagerly() == True;
    predictor = Predictor();

