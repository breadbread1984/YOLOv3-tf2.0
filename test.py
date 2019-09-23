#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_datasets as tfds;
from YOLOv3 import YOLOv3, OutputParser;
from preprocess import map_function;

batch_size = 8;

def main():
   
    img_shape = (416,416,3);
    class_num = 80;
    # create model objects
    yolov3 = YOLOv3(img_shape, class_num);
    anchors = {2: [[10, 13], [16, 30], [33, 23]], 1: [[30, 61], [62, 45], [59, 119]], 0: [[116, 90], [156, 198], [373, 326]]};
    input_shapes = [
        (img_shape[0] // 32, img_shape[1] // 32, 3, 5 + class_num),
        (img_shape[0] // 16, img_shape[1] // 16, 3, 5 + class_num),
        (img_shape[0] // 8, img_shape[1] // 8, 3, 5 + class_num)
    ];
    parsers = [OutputParser(input_shapes[l], img_shape, anchors[l]) for l in range(3)];
    # load downloaded dataset
    testset = tfds.load(name = "coco2014", split = tfds.Split.TEST, download = False);
    testset = testset.map(map_function).repeat(100).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
    count = 1;
    while True:
        optimizer = tf.keras.optimizers.Adam(1e-4);
        checkpoint = tf.train.Checkpoint(model = yolov3, optimizer = optimizer, optimizer_step = optimizer.iterations);
        checkpoint.restore("checkpoints/ckpt-" + str(count));
        for images, labels in testset:
            outputs = yolov3(images);
            for l in range(3):
                input_shape_of_this_layer = input_shapes[l];
                anchors_of_this_layer = anchors[l];
                input_of_this_layer = outputs[l];
                label_of_this_layer = labels[l];
                pred_xy, pred_wh, pred_box_confidence, pred_class = OutputParser(input_shape_of_this_layer, img_shape, anchors_of_this_layer)(input_of_this_layer);
                pred_box = tf.keras.layers.Concatenate()([pred_xy,pred_wh]);
                def calc_iou(x):
                    pred = x[0];
                    label = x[1];
                    # pred_box.shape = (pred target number, 4)
                    pred_box = tf.boolean_mask(pred[..., 0:4], tf.cast(pred[..., 4], dtype = tf.bool));
                    # true_box.shape = (true taget number, 4)
                    true_box = tf.boolean_mask(label[...., 0:4], tf.cast(label[..., 4], dtype = tf.bool));
                    # iou
                    # pred_box.shape = (pred target_num, 1, 4)
                    pred_box = tf.expand_dims(pred_box, axis = -2);
                    pred_box_xy = pred_box[..., 0:2];
                    pred_box_wh = pred_box[..., 2:4];
                    pred_box_wh_half = pred_box_wh / 2.;
                    pred_box_mins = pred_box_xy - pred_box_wh_half;
                    pred_box_maxs = pred_box_mins + pred_box_wh;
                    # true_box.shape = (1, true target num, 4)
                    true_box = tf.expand_dims(true_box, axis = 0);
                    true_box_xy = true_box[..., 0:2];
                    true_box_wh = true_box[..., 2:4];
                    true_box_wh_half = true_box_wh / 2.;
                    true_box_mins = true_box_xy - true_box_wh_half;
                    true_box_maxs = true_box_mins + true_box_wh;
                    # intersection.shape = (pred target num, true target_num, 2)
                    intersect_mins = tf.math.maximum(pred_box_mins, true_box_mins);
                    intersect_maxs = tf.math.minimum(pred_box_maxs, true_box_maxs);
                    intersect_wh = tf.math.maximum(intersect_maxs - intersect_mins, 0.);
                    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1];
                    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1];
                    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1];
                    # iou.shape = (pred target num, true target_num)
                    iou = intersect_area / (pred_box_area + true_box_area - intersect_area);
                    # IOU of detected target with the best overlapped labeled box
                    # best_iou.shape = (h, w, anchor_num)
                    best_iou = tf.math.reduce_max(iou, axis = -1);
                    return best_iou;
                #TODO
        count += 1;

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
