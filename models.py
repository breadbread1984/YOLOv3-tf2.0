#!/usr/bin/python3

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

def Loss(img_shape, class_num = 80, ignore_thresh = 0.5):

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
    # 1) preprocess prediction
    pred_xy, pred_wh, pred_box_confidence, pred_class = OutputParser(input_shape_of_this_layer, img_shape, anchors_of_this_layer)(input_of_this_layer);
    pred_half_wh = tf.keras.layers.Lambda(lambda x: x / 2)(pred_wh);
    pred_upperleft = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([pred_xy, pred_half_wh]); # pred_upperleft.shape = (batch, grid h, grid w, anchor_num, 2) in sequence of (xmin, ymin)
    pred_bottomright = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([pred_xy, pred_half_wh]); # pred_bottomright.shape = (batch, grid h, grid w, anchor_num, 2) in sequence of (xmax, ymax)
    pred_bbox = tf.keras.layers.Lambda(lambda x: tf.concat([tf.reverse(x[0], axis = [-1]), tf.reverse(x[1], axis = [-1])], axis = -1))([pred_upperleft, pred_bottomright]); # pred_bbox.shape = (batch, grid h, grid w, anchor_num, 4) in sequence of (ymin, xmin, ymax, xmax)
    # 2) preprocess label
    true_position = tf.keras.layers.Lambda(lambda x: x[..., 0:4])(label_of_this_layer); # true_box.shape = (batch, grid h, grid w, anchor_num, 4) in sequence of (center x, center y, w, h)
    true_xy = tf.keras.layers.Lambda(lambda x: x[..., 0:2])(true_position); # true_xy.shape = (batch, grid h, grid w, anchor_num, 2)
    true_wh = tf.keras.layers.Lambda(lambda x: x[..., 2:4])(true_position); # true_wh.shape = (batch, grid h, grid w, anchor_num, 2)
    true_half_wh = tf.keras.layers.Lambda(lambda x: x[..., 2:4] / 2)(true_position); # true_half_wh.shape = (batch, grid h, grid w, anchor_num, 2)
    true_upperleft = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([true_xy, true_half_wh]); # true_upperleft.shape = (batch, grid h, grid w, anchor_num, 2) in sequence of (xmin, ymin)
    true_bottomright = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([true_xy, true_half_wh]); # true_bottomright.shape = (batch, grid h, grid w, anchor_num, 2) in sequence of (xmax, ymax)
    object_mask = tf.keras.layers.Lambda(lambda x: x[..., 4])(label_of_this_layer); # object_mask.shape = (batch, grid h, grid w, anchor_num)
    
    object_mask_bool = tf.keras.layers.Lambda(lambda x: tf.cast(x, dtype = tf.bool))(object_mask); # object_mask_bool.shape = (batch, grid h, grid w, anchor_num)
    true_bbox = tf.keras.layers.Lambda(lambda x: tf.concat([tf.reverse(x[0], axis = [-1]), tf.reverse(x[1], axis = [-1])], axis = -1))([true_upperleft, true_bottomright]); # true_bbox.shape = (batch, grid h, grid w, anchor_num, 4) in sequence of (ymin, xmin, ymax, xmax)
    true_class = tf.keras.layers.Lambda(lambda x: x[..., 5:])(label_of_this_layer); # true_class.shape = (batch, grid h, grid w, anchor_num, class_num)
    loss_scale = tf.keras.layers.Lambda(lambda x: 2 - x[..., 2] * x[..., 3])(true_position); # loss_scale.shape = (batch, grid h, grid w, anchor_num) punish harshly for smaller targets
    # 3) ignore mask
    def body(x):
      true_bbox, object_mask_bool, pred_bbox = x;
      # true_bbox.shape = (grid h, grid w, anchor_num, 4)
      # object_mask_bool.shape = (grid h, grid w, anchor_num)
      # pred_bbox.shape = (grid h, grid w, anchor_num, 4)
      true_bbox_list = tf.boolean_mask(true_bbox, object_mask_bool); # true_bbox_list.shape = (obj_num, 4)
      shape = tf.shape(pred_bbox)[:-1];
      pred_bbox_list = tf.reshape(pred_bbox, (-1, 4));
      bbox1_hw = true_bbox_list[..., 2:4] - true_bbox_list[..., 0:2]; # bbox1_hw.shape = (obj_num1, 2)
      bbox1_area = bbox1_hw[..., 0] * bbox1_hw[..., 1]; # bbox1_area.shape = (obj_num1)
      bbox2_hw = pred_bbox_list[..., 2:4] - pred_bbox_list[..., 0:2]; # bbox2_hw.shape = (obj_num2, 2)
      bbox2_area = bbox2_hw[..., 0] * bbox2_hw[..., 1]; # bbox2_area.shape = (obj_num2)
      intersect_min = tf.maximum(tf.expand_dims(true_bbox_list[..., 0:2], axis = 1), tf.expand_dims(pred_bbox_list[..., 0:2], axis = 0)); # intersect_min.shape = (obj_num1, obj_num2, 2)
      intersect_max = tf.minimum(tf.expand_dims(true_bbox_list[..., 2:4], axis = 1), tf.expand_dims(pred_bbox_list[..., 2:4], axis = 0)); # intersect_max.shape = (obj_num1, obj_num2, 2)
      intersect_hw = tf.maximum(intersect_max - intersect_min, 0); # intersect_hw.shape = (obj_num1, obj_num2, 2)
      intersect_area = intersect_hw[..., 0] * intersect_hw[..., 1]; # intersect_area.shape = (obj_num1, obj_num2)
      iou = intersect_area / tf.maximum(tf.expand_dims(bbox1_area, axis = 1) + tf.expand_dims(bbox2_area, axis = 0) - intersect_area, 1e-5); # iou.shape = (obj_num1, obj_num2)
      iou = tf.reshape(iou, tf.concat([tf.shape(true_bbox_list)[0:1], tf.shape(pred_bbox)[:-1]], axis = 0)); # iou.shape = (obj_num, grid h, grid w, anchor_num)
      best_iou = tf.math.reduce_max(iou, axis = 0); # iou.shape = (grid h, grid w, anchor_num)
      ignore_mask = tf.where(tf.math.less(best_iou, ignore_thresh), tf.ones_like(best_iou), tf.zeros_like(best_iou)); # ignore_mask.shape = (grid h, grid w, anchor_num)
      return ignore_mask;
    ignore_mask = tf.keras.layers.Lambda(lambda x, s: tf.map_fn(body, x, fn_output_signature = tf.TensorSpec(shape = s[:3])), arguments = {'s': input_shape_of_this_layer})([true_bbox, object_mask_bool, pred_bbox]); # ignore_mask.shape = (batch, grid h, grid w, anchor_num)
    # 4) position loss
    # NOTE: only punish foreground area
    # NOTE: punish smaller foreground targets more harshly
    xy_loss = tf.keras.layers.Lambda(lambda x:
      x[0] * x[1] * tf.keras.losses.BinaryCrossentropy(from_logits = False, reduction = tf.keras.losses.Reduction.NONE)(x[2], x[3])
    )([object_mask, loss_scale, true_xy, pred_xy]); # xy_loss.shape = (batch, grid h, grid w, anchor_num)
    wh_loss = tf.keras.layers.Lambda(lambda x: x[0] * x[1] * 0.5 * tf.math.reduce_sum(tf.math.square(x[2] - x[3]), axis = -1))([object_mask, loss_scale, true_wh, pred_wh]); # wh_loss.shape = (batch, grid h, grid w, anchor_num)
    # 5) confidence loss
    # NOTE: punish foreground area which is miss classified
    # NOTE: and punish background area which is far from foreground area and miss classified
    confidence_loss = tf.keras.layers.Lambda(lambda x: 
      x[0] * tf.keras.losses.BinaryCrossentropy(from_logits = False, reduction = tf.keras.losses.Reduction.NONE)(
        tf.expand_dims(x[0], axis = -1), tf.expand_dims(x[1], axis = -1)
      ) + \
      (1. - x[0]) * x[2] * tf.keras.losses.BinaryCrossentropy(from_logits = False, reduction = tf.keras.losses.Reduction.NONE)(
        tf.expand_dims(x[0], axis = -1), tf.expand_dims(x[1], axis = -1)
      )
    )([object_mask, pred_box_confidence, ignore_mask]); # confidence_loss.shape = (batch, grid h, grid w, anchor_num)
    # 6) class loss
    # NOTE: only punish foreground area
    class_loss = tf.keras.layers.Lambda(lambda x: 
      x[0] * tf.keras.losses.BinaryCrossentropy(from_logits = False, reduction = tf.keras.losses.Reduction.NONE)(x[1], x[2])
    )([object_mask, true_class, pred_class]); # class_loss.shape = (batch, grid h, grid w, anchor_num)
    # 7) total
    loss = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(tf.math.reduce_sum(tf.math.add_n(x), axis = [1,2,3]), axis = [0]))([xy_loss, wh_loss, confidence_loss, class_loss]); # loss.shape = ()
    losses.append(loss);
  loss = tf.keras.layers.Lambda(lambda x: tf.math.add_n(x))(losses);
  return tf.keras.Model(inputs = (*inputs, *labels), outputs = loss);

if __name__ == "__main__":
 
  yolov3 = YOLOv3((416,416,3), 80);
  loss = Loss((416,416,3), 80);
  yolov3.save('yolov3.h5');
  loss.save('loss.h5');
