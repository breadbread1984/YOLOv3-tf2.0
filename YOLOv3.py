#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_addons as tfa;

# NOTE: using functional API can save a lot of gmem

def ConvBlock(input_shape, filters, kernel_size, strides = (1,1), padding = None):
  # 3 layers in total

  padding = 'valid' if strides == (2,2) else 'same';
  
  inputs = tf.keras.Input(shape = input_shape);
  conv = tf.keras.layers.Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_regularizer = tf.keras.regularizers.l2(l = 5e-4))(inputs);
  bn = tf.keras.layers.BatchNormalization()(conv);
  relu = tf.keras.layers.LeakyReLU(alpha = 0.1)(bn);
  return tf.keras.Model(inputs = inputs, outputs = relu);

def ResBlock(input_shape, filters, blocks):
  # 4 + 7 * blocks layers in total
  
  inputs = tf.keras.Input(shape = input_shape);
  pad = tf.keras.layers.ZeroPadding2D(padding = ((1,0),(1,0)))(inputs);
  results = ConvBlock(pad.shape[1:], filters = filters, kernel_size = (3,3), strides = (2,2))(pad);
  for i in range(blocks):
    results_conv = ConvBlock(results.shape[1:], filters = filters // 2, kernel_size = (1,1))(results);
    results_conv = ConvBlock(results_conv.shape[1:], filters = filters, kernel_size = (3,3))(results_conv);
    results = tf.keras.layers.Add()([results_conv, results]);
  return tf.keras.Model(inputs = inputs, outputs = results);

def Body(input_shape):
  # 3 + (4 + 7 * 1) + (4 + 7 * 2) + (4 + 7 * 8) + (4 + 7 * 8) + (4 + 7 * 4) = 184 layers in total
  
  inputs = tf.keras.Input(shape = input_shape);
  cb = ConvBlock(inputs.shape[1:], filters = 32, kernel_size = (3,3))(inputs); # (batch, 416, 416, 32)
  rb1 = ResBlock(cb.shape[1:], filters = 64, blocks = 1)(cb); # (batch, 208, 208, 64)
  rb2 = ResBlock(rb1.shape[1:], filters = 128, blocks = 2)(rb1); # (batch, 104, 104, 128)
  rb3 = ResBlock(rb2.shape[1:], filters = 256, blocks = 8)(rb2); # (batch, 52, 52, 256)
  rb4 = ResBlock(rb3.shape[1:], filters = 512, blocks = 8)(rb3); # (batch, 26, 26, 512)
  rb5 = ResBlock(rb4.shape[1:], filters = 1024, blocks = 4)(rb4); # (batch, 13, 13, 1024)
  return tf.keras.Model(inputs = inputs, outputs = (rb5, rb4 ,rb3));

def Output(input_shape, input_filters, output_filters):
  # 3 * 7 = 21 layer in total
  
  inputs = tf.keras.Input(shape = input_shape);
  cb1 = ConvBlock(inputs.shape[1:], filters = input_filters, kernel_size = (1,1))(inputs);
  cb2 = ConvBlock(cb1.shape[1:], filters = input_filters * 2, kernel_size = (3,3))(cb1);
  cb3 = ConvBlock(cb2.shape[1:], filters = input_filters, kernel_size = (1,1))(cb2);
  cb4 = ConvBlock(cb3.shape[1:], filters = input_filters * 2, kernel_size = (3,3))(cb3);
  cb5 = ConvBlock(cb4.shape[1:], filters = input_filters, kernel_size = (1,1))(cb4);
  cb6 = ConvBlock(cb5.shape[1:], filters = input_filters * 2, kernel_size = (3,3))(cb5);
  cb7 = ConvBlock(cb6.shape[1:], filters = output_filters, kernel_size = (1,1))(cb6);
  return tf.keras.Model(inputs = inputs, outputs = (cb5,cb7));

def YOLOv3(input_shape, class_num = 80):

  anchor_num = 3;
  inputs = tf.keras.Input(shape = input_shape);
  large,middle,small = Body(inputs.shape[1:])(inputs);
  # feature for detecting large scale objects
  x1,y1 = Output(large.shape[1:], 512, anchor_num * (class_num + 5))(large);
  y1 = tf.keras.layers.Reshape((input_shape[0] // 32, input_shape[1] // 32, 3, 5 + class_num))(y1);
  # feature for detecting middle scale objects
  cb1 = ConvBlock(x1.shape[1:], filters = 256, kernel_size = (1,1))(x1);
  us1 = tf.keras.layers.UpSampling2D(2)(cb1);
  cat1 = tf.keras.layers.Concatenate()([us1, middle]);
  x2,y2 = Output(cat1.shape[1:], 256, anchor_num * (class_num + 5))(cat1);
  y2 = tf.keras.layers.Reshape((input_shape[0] // 16, input_shape[1] // 16, 3, 5 + class_num))(y2);
  # feature for detecting small scale objects
  cb2 = ConvBlock(x2.shape[1:], filters = 128, kernel_size = (1,1))(x2);
  us2 = tf.keras.layers.UpSampling2D(2)(cb2);
  cat2 = tf.keras.layers.Concatenate()([us2, small]);
  x3,y3 = Output(cat2.shape[1:], 128, anchor_num * (class_num + 5))(cat2);
  y3 = tf.keras.layers.Reshape((input_shape[0] // 8, input_shape[1] // 8, 3, 5 + class_num))(y3);
  return tf.keras.Model(inputs = inputs, outputs = (y1,y2,y3));

def OutputParser(input_shape, img_shape, anchors):

  # feats.shape = batch x grid h x grid w x anchor_num x (1(delta x) + 1(delta y) + 1(width scale) + 1(height scale) + 1(object mask) + class_num(class probability))
  # NOTE: box center absolute x = delta x + prior box upper left x, box center absolute y = delta y + prior box upper left y
  # NOTE: width scale = box width / anchor width, height scale = box height / anchor height
  tf.debugging.Assert(tf.math.logical_and(tf.equal(tf.shape(input_shape)[0],4), tf.equal(input_shape[2], 3)), [input_shape]);
  tf.debugging.Assert(tf.equal(tf.shape(img_shape)[0],3), [img_shape]);
  # anchors.shape = (3,2)
  tf.debugging.Assert(tf.math.logical_and(tf.equal(tf.shape(anchors)[0], 3), tf.equal(tf.shape(anchors)[1], 2)), [anchors]);
  feats = tf.keras.Input(input_shape);
  # [x,y] = meshgrid(x,y) get the upper left positions of prior boxes
  # grid.shape = (grid h, grid w, 1, 2)
  grid_y = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(tf.range(tf.cast(tf.shape(x)[1], dtype = tf.float32), dtype = tf.float32), (-1, 1, 1, 1)), (1, tf.shape(x)[2], 1, 1)))(feats);
  grid_x = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(tf.range(tf.cast(tf.shape(x)[2], dtype = tf.float32), dtype = tf.float32), (1, -1, 1, 1)), (tf.shape(x)[1], 1, 1, 1)))(feats);
  grid = tf.keras.layers.Concatenate(axis = -1)([grid_x, grid_y]);
  # box center proportional position = (delta x, delta y) + (priorbox upper left x,priorbox upper left y) / (feature map.width, feature map.height)
  # box_xy.shape = (batch, grid h, grid w, anchor_num, 2)
  box_xy = tf.keras.layers.Lambda(lambda x: (tf.math.sigmoid(x[0][...,0:2]) + x[1]) / tf.cast([tf.shape(x[1])[1], tf.shape(x[1])[0]], dtype = tf.float32))([feats, grid]);
  # box proportional size = (width scale, height scale) * (anchor width, anchor height) / (image.width, image.height)
  # box_wh.shape = (batch, grid h, grid w, anchor_num, 2)
  box_wh = tf.keras.layers.Lambda(lambda x, y, z: tf.math.exp(x[...,2:4]) * y / tf.cast([z[1], z[0]], dtype = tf.float32), arguments = {'y': anchors, 'z': img_shape})(feats);
  # confidence of being an object
  box_confidence = tf.keras.layers.Lambda(lambda x: tf.math.sigmoid(x[..., 4]))(feats);
  # class confidence
  box_class_probs = tf.keras.layers.Lambda(lambda x: tf.math.sigmoid(x[..., 5:]))(feats);
  return tf.keras.Model(inputs = feats, outputs = (box_xy, box_wh, box_confidence, box_class_probs));

def Loss(img_shape, class_num = 80):

  # outputs is a tuple
  # outputs.shape[layer] = batch x h x w x anchor_num x (1(delta x) + 1(delta y) + 1(width scale) + 1(height scale) + 1(object mask) + class_num(class probability))
  # labels is a tuple
  # labels.shape[layer] = batch x h x w x anchor_num x (1(proportional x) + 1 (proportional y) + 1(proportional width) + 1(proportional height) + 1(object mask) + class_num(class probability))
  # NOTE: the info carried by the output and the label is different.
  tf.debugging.Assert(tf.equal(tf.shape(img_shape)[0], 3), [img_shape]);
  anchors = {2: [[10, 13], [16, 30], [33, 23]], 1: [[30, 61], [62, 45], [59, 119]], 0: [[116, 90], [156, 198], [373, 326]]};
  input_shapes = [
    (img_shape[0] // 32, img_shape[1] // 32, 3, 5 + class_num),
    (img_shape[0] // 16, img_shape[1] // 16, 3, 5 + class_num),
    (img_shape[0] // 8, img_shape[1] // 8, 3, 5 + class_num)
  ];
  inputs = [tf.keras.Input(input_shape) for input_shape in input_shapes]; # inputs[0].shape = (13, 13, 3, 5 + class num), inputs[1].shape = (26, 26, 3, 5 + class num), inputs[2].shape = (52, 52, 3, 5 + class num)
  labels = [tf.keras.Input(input_shape) for input_shape in input_shapes]; # labels[0].shape = (13, 13, 3, 5 + class num), labels[1].shape = (26, 26, 3, 5 + class num), labels[2].shape = (52, 52, 3, 5 + class num)
  losses = list();
  # for each branch (scale)
  for l in range(3):
    input_shape_of_this_layer = input_shapes[l];
    anchors_of_this_layer = anchors[l];
    input_of_this_layer = inputs[l];
    label_of_this_layer = labels[l];
    # bounding info from YOLOv3
    pred_xy, pred_wh, pred_box_confidence, pred_class = OutputParser(input_shape_of_this_layer, img_shape, anchors_of_this_layer)(input_of_this_layer);
    pred_box = tf.keras.layers.Concatenate()([pred_xy, pred_wh]); # pred_box.shape = (batch, grid h, grid w, anchor_num, 4)
    # bounding info from label
    true_box = tf.keras.layers.Lambda(lambda x: x[..., 0:4])(label_of_this_layer); # true_box.shape = (batch, grid h, grid w, anchor_num, 4)
    true_box_confidence = tf.keras.layers.Lambda(lambda x: x[..., 4])(label_of_this_layer); # true_box_confidence.shape = (batch, grid h, grid w, anchor_num)
    true_class = tf.keras.layers.Lambda(lambda x: x[..., 5:])(label_of_this_layer); # true_class.shape = (batch, grid h, grid w, anchor_num, class_num)
    # mask of true positive
    object_mask = tf.keras.layers.Lambda(lambda x: tf.cast(x, dtype = tf.bool))(true_box_confidence);
    # mean square error of bounding location in proportional coordinates
    # pos_loss.shape = ()
    # 1) only supervise boundings of positve examples.
    pos_loss = tf.keras.layers.Lambda(lambda x:
      tf.math.reduce_sum(tf.keras.losses.MSE(
        tf.boolean_mask(x[0], x[2]), # obj_true_box.shape = (object_num, 4)
        tf.boolean_mask(x[1], x[2]) # obj_pred_box.shape = (object_num, 4)
      ))
    )([true_box, pred_box, object_mask]);
    # confidence_loss.shape = ()
    # 2) punish wrongly predicted confidence with focal loss
    confidence_loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits = False)(true_box_confidence, pred_box_confidence);
    # class_loss.shape = ()
    # 3) only supervise classes of positive examples.
    class_loss = tf.keras.layers.Lambda(lambda x:
      tf.keras.losses.BinaryCrossentropy(from_logits = False)(
        tf.boolean_mask(x[0], x[2]), # obj_true_class.shape = (object_num, class_num)
        tf.boolean_mask(x[1], x[2]) # obj_pred_class.shape = (object_num, class_num)
      )
    )([true_class, pred_class, object_mask]);
    loss = tf.keras.layers.Lambda(lambda x: tf.math.add_n(x))([pos_loss, confidence_loss, class_loss]);
    losses.append(loss);
  loss = tf.keras.layers.Lambda(lambda x: tf.math.add_n(x))(losses);
  return tf.keras.Model(inputs = (*inputs, *labels), outputs = loss);

if __name__ == "__main__":
 
  yolov3 = YOLOv3((416,416,3), 80);
  loss = Loss((416,416,3), 80);
  yolov3.save('yolov3.h5');
  loss.save('loss.h5');
