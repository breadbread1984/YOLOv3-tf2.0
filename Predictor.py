#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
from YOLOv3 import YOLOv3, OutputParser;

class Predictor(object):

    anchors = {2: [[10, 13], [16, 30], [33, 23]], 1: [[30, 61], [62, 45], [59, 119]], 0: [[116, 90], [156, 198], [373, 326]]};

    def __init__(self, input_shape = (416,416,3), class_num = 80):

        self.input_shape = input_shape;
        self.yolov3 = tf.keras.models.load_model('yolov3.h5', compile = False);
        output_shapes = [
            (input_shape[0] // 32, input_shape[1] // 32, 3, 5 + class_num),
            (input_shape[0] // 16, input_shape[1] // 16, 3, 5 + class_num),
            (input_shape[0] // 8, input_shape[1] // 8, 3, 5 + class_num)
        ];
        self.parsers = [OutputParser(output_shapes[l], input_shape, self.anchors[l]) for l in range(3)];

    def predict(self, image, thres = 0.5):

        images = tf.expand_dims(image, axis = 0);
        resize_images = tf.image.resize(images, self.input_shape[:2], method = tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio = True);
        resize_shape = resize_images.shape[1:3];
        top_pad = (self.input_shape[0] - resize_shape[0]) // 2;
        bottom_pad = self.input_shape[0] - resize_shape[0] - top_pad;
        left_pad = (self.input_shape[1] - resize_shape[1]) // 2;
        right_pad = self.input_shape[1] - resize_shape[1] - left_pad;
        resize_images = tf.pad(resize_images,[top_pad,bottom_pad],[left_pad,right_pad],[0,0]], constant_values = 128);
        deviation = tf.constant([left_pad / self.input_shape[1], top_pad / self.input_shape[0], 0, 0], dtype = tf.float32);
        scale = tf.constant([
            self.input_shape[1] / resize_images.shape[2], self.input_shape[0] / resize_images.shape[1],
            self.input_shape[1] / resize_images.shape[2], self.input_shape[0] / resize_images.shape[1]
        ], dtype = tf.float32);
        images_data = tf.cast(resize_images, tf.float32) / 255.;
        outputs = self.yolov3(images_data);
        whole_targets = tf.constant([0,6], dtype = tf.float32);
        for i in range(3):
            pred_xy, pred_wh, pred_box_confidence, pred_class = self.parsers[i](outputs[i]);
            pred_box = tf.keras.layers.Concatenate(axis = -1)([pred_xy, pred_wh]);
            # target_mask.shape = (h, w, anchor num)
            target_mask = tf.greater(pred_box_confidence, thres);
            # pred_box_confidence = (pred target num, 1)
            pred_box_confidence = tf.boolean_mask(pred_box_confidencen, target_mask);
            pred_box_confidence = tf.expand_dims(pred_box_confidence, axis = -1);
            # pred_box.shape = (pred target num, 4)
            pred_box = tf.boolean_mask(pred_box, target_mask);
            pred_box = (pred_box - deviation) * scale * [image.shape[1], image.shape[0], image.shape[1], image.shape[0]];
            # pred_class.shape = (pred target num, 1)
            pred_class = tf.boolean_mask(pred_class, target_mask);
            pred_class = tf.math.argmax(pred_class, axis = -1);
            pred_class = tf.expand_dims(pred_class, axis = -1);
            # targets,sgaoe = (pred target num, 6)
            targets = tf.keras.layers.Concatenate(axis = -1)([pred_box, pred_box_confidence, pred_class]);
            whole_targets = tf.keras.layers.Concatenate(axis = 0)([whole_targets, targets]);

        return whole_targets;

if __name__ == "__main__":

    assert tf.executing_eagerly() == True;
    predictor = Predictor();

