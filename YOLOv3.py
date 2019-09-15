#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

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

def OutputParser(input_shape, img_shape, anchors, calc_loss = False):

    # feats.shape = batch x grid h x grid w x anchor_num x (1(delta x) + 1(delta y) + 1(width scale) + 1(height scale) + 1(object mask) + class_num(class probability))
    # NOTE: box center absolute x = delta x + prior box upper left x, box center absolute y = delta y + prior box upper left y
    # NOTE: width scale = box width / anchor width,  height scale = box height / anchor height
    tf.debugging.Assert(tf.math.logical_and(tf.equal(tf.shape(input_shape)[0],4), tf.equal(input_shape[2], 3)), [input_shape]);
    tf.debugging.Assert(tf.equal(tf.shape(img_shape)[0],3), [img_shape]);
    # anchors.shape = (3,2)
    tf.debugging.Assert(tf.math.logical_and(tf.equal(anchors.shape[0], 3), tf.equal(anchors.shape[1], 2)), [anchors]);
    feats = tf.keras.Input(input_shape);
    # [x,y] = meshgrid(x,y) get the upper left positions of prior boxes
    # grid.shape = (grid h, grid w, 1, 2)
    grid_y = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(tf.range(x.shape[1], dtype = tf.float32), (-1, 1, 1, 1)), (1, x.shape[2], 1, 1)))(feats);
    grid_x = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(tf.range(x.shape[2], dtype = tf.float32), (1, -1, 1, 1)), (x.shape[1], 1, 1, 1)))(feats);
    grid = tf.keras.layers.Concatenate(axis = -1)([grid_x, grid_y]);
    # box center proportional position = (delta x, delta y) + (priorbox upper left x,priorbox upper left y) / (feature map.width, feature map.height)
    # box_xy.shape = (batch, grid h, grid w, anchor_num, 2)
    box_xy = tf.keras.layers.Lambda(lambda x: (tf.math.sigmoid(x[0][...,0:2]) + x[1]) / tf.cast([x[1].shape[1], x[1].shape[0]], dtype = tf.float32))([feats, grid]);
    # box proportional size = (width scale, height scale) * (anchor width, anchor height) / (image.width, image.height)
    # box_wh.shape = (batch, grid h, grid w, anchor_num, 2)
    box_wh = tf.keras.layers.Lambda(lambda x, anchors, img_shape: tf.math.exp(feats[...,2:4]) * anchors / tf.cast([img_shape[1], img_shape[0]], dtype = tf.float32), arguments = {'anchors': anchors, 'img_shape': img_shape})(feats);
    # confidence of being an object
    box_confidence = tf.keras.layers.Lambda(lambda x: tf.math.sigmoid(x[..., 4:5]))(feats);
    # class confidence
    box_class_probs = tf.keras.layers.Lambda(lambda x: tf.math.sigmoid(x[..., 5:]))(feats);
    if calc_loss == True:
        return tf.keras.Model(inputs = feats, outputs = (grid, box_xy, box_wh));
    else:
        return tf.keras.Model(inputs = feats, outputs = (box_xy, box_wh, box_confidence, box_class_probs));

def Loss(img_shape, class_num = 80, ignore_thresh = .5):

    # outputs is a tuple
    # outputs.shape[layer] = batch x h x w x anchor_num x (1(delta x) + 1(delta y) + 1(width scale) + 1(height scale) + 1(object mask) + class_num(class probability))
    # labels is a tuple
    # labels.shape[layer] = batch x h x w x anchor_num x (1(proportional x) + 1 (proportional y) + 1(proportional width) + 1(proportional height) + 1(object mask) + class_num(class probability))
    # NOTE: the info carried by the output and the label is different.
    tf.debugging.Assert(tf.equal(tf.shape(img_shape)[0], 3), [img_shape]);
    anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]], dtype = np.int32);
    input_shapes = [
        (img_shape[0] // 32, img_shape[1] // 32, 3, 5 + class_num),
        (img_shape[0] // 16, img_shape[1] // 16, 3, 5 + class_num),
        (img_shape[0] // 8, img_shape[1] // 8, 3, 5 + class_num)
    ];
    inputs = [tf.keras.Input(input_shape) for input_shape in input_shapes];
    labels = [tf.keras.Input(input_shape) for input_shape in input_shapes];
    losses = list();
    for l in range(len(labels)):
        # 1) ignore masks
        anchors_of_this_layer = anchors[{0:[6,7,8],1:[3,4,5],2:[0,1,2]}[l]];
        grid, pred_xy, pred_wh = OutputParser(input_shapes[l], img_shape, anchors_of_this_layer, True)(inputs[l]);
        # box proportional coordinates: pred_box.shape = (batch,h,w,anchor_num,4)
        pred_box = tf.keras.layers.Concatenate()([pred_xy, pred_wh]);
        def ignore_mask(x):
            pred_box = x[0];
            label = x[1];
            # true_box.shape = (labeled target num, 4)
            true_box = tf.boolean_mask(label[..., 0:4], tf.cast(label[..., 4], dtype = tf.bool));
            # calculate IOU
            # pred_box.shape = (h, w, anchor_num, 1, 4)
            pred_box = tf.expand_dims(pred_box, axis = -2);
            pred_box_xy = pred_box[..., 0:2];
            pred_box_wh = pred_box[..., 2:4];
            pred_box_wh_half = pred_box_wh / 2.;
            pred_box_mins = pred_box_xy - pred_box_wh_half;
            pred_box_maxs = pred_box_mins + pred_box_wh;
            # true_box.shape = (1, target num, 4)
            true_box = tf.expand_dims(true_box, axis = 0);
            true_box_xy = true_box[..., 0:2];
            true_box_wh = true_box[..., 2:4];
            true_box_wh_half = true_box_wh / 2.;
            true_box_mins = true_box_xy - true_box_wh_half;
            true_box_maxs = true_box_mins + true_box_wh;
            # intersection.shape = (h, w, anchor_num, target_num, 2)
            intersect_mins = tf.math.maximum(pred_box_mins, true_box_mins);
            intersect_maxs = tf.math.minimum(pred_box_maxs, true_box_maxs);
            intersect_wh = tf.math.maximum(intersect_maxs - intersect_mins, 0.);
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1];
            pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1];
            true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1];
            # iou.shape = (h, w, anchor_num, labeled target_num)
            iou = intersect_area / (pred_box_area + true_box_area - intersect_area);
            # IOU of detected target with the best overlapped labeled box
            # best_iou.shape = (h, w, anchor_num)
            best_iou = tf.math.reduce_max(iou, axis = -1);
            # ignore_mask.shape = (h, w, anchor_num)
            ignore_mask = tf.where(tf.less(best_iou, ignore_thresh), tf.ones_like(best_iou), tf.zeros_like(best_iou));
            return ignore_mask;
        # ignore_masks.shape = (b, h, w, anchor_num)
        ignore_masks = tf.keras.layers.Lambda(lambda x: tf.map_fn(ignore_mask, x, dtype = tf.float32))((pred_box, labels[l]));
        # 2) loss
        # raw_true_xy.shape = (b, h, w, anchor_num, 2)
        raw_true_xy = tf.keras.layers.Lambda(lambda x, input_shape: x[0][..., 0:2] * tf.cast([input_shape[1], input_shape[0]], dtype = tf.float32) - x[1], arguments = {'input_shape': input_shapes[l]})([labels[l], grid]);
        # raw_true_wh.shape = (b, h, w, anchor_snum, 2)
        raw_true_wh = tf.keras.layers.Lambda(lambda x, img_shape, anchors: tf.math.log(x[..., 2:4] * tf.cast([img_shape[1], img_shape[0]], dtype = tf.float32) / tf.cast(anchors, dtype = tf.float32)), arguments = {'img_shape': img_shape, 'anchors': anchors_of_this_layer})(labels[l]);
        raw_true_wh = tf.keras.layers.Lambda(lambda x: tf.where(tf.cast(tf.concat([x[0][..., 4:5], x[0][..., 4:5]], axis = -1), dtype = tf.bool), x[1], tf.zeros_like(x[1])))([labels[l], raw_true_wh]);
        # box_loss_scale.shape = (b, h, w, anchor_num, 1)
        # box area is larger, loss is smaller.
        box_loss_scale = tf.keras.layers.Lambda(lambda x: 2 - x[...,2:3] * x[...,3:4])(labels[l]);
        # xy_loss.shape = (b, h, w, anchor_num, 2)
        xy_loss = tf.keras.layers.Lambda(lambda x: x[0][..., 4:5] * x[1] * tf.keras.losses.BinaryCrossentropy(from_logits = True)(x[2], x[3][..., 0:2]))([labels[l], box_loss_scale, raw_true_xy, inputs[l]]);
        xy_loss = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.math.reduce_mean(x, 0)))(xy_loss);
        # wh_loss.shape = (b, h, w, anchor_num, 2)
        wh_loss = tf.keras.layers.Lambda(lambda x: x[0][..., 4:5] * x[1] * 0.5 * tf.math.square(x[2] - x[3][..., 2:4]))([labels[l], box_loss_scale, raw_true_wh, inputs[l]]);
        wh_loss = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.math.reduce_mean(x, 0)))(wh_loss);
        # confidence_loss.shape = (b, h, w, anchor_num, 1)
        confidence_loss = tf.keras.layers.Lambda(
            lambda x:
                x[0][..., 4] * tf.keras.losses.BinaryCrossentropy(from_logits = True)(x[0][..., 4], x[1][..., 4]) +
                (1 - x[0][..., 4]) * tf.keras.losses.BinaryCrossentropy(from_logits = True)(x[0][..., 4], x[1][..., 4]) * x[2]
        )([labels[l], inputs[l], ignore_masks]);
        confidence_loss = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.math.reduce_mean(x, 0)))(confidence_loss);
        # class_loss.shape = ()
        class_loss = tf.keras.layers.Lambda(
            lambda x:
                x[0][..., 4:5] * tf.keras.losses.BinaryCrossentropy(from_logits = True)(x[0][...,5:], x[1][...,5:])
        )([labels[l], inputs[l]]);
        class_loss = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.math.reduce_mean(x, 0)))(class_loss);
        loss = tf.keras.layers.Lambda(lambda x: tf.math.add_n(x))([xy_loss, wh_loss, confidence_loss, class_loss]);
        losses.append(loss);
    loss = tf.keras.layers.Lambda(lambda x: tf.math.add_n(x))(losses);
    return tf.keras.Model(inputs = inputs + labels, outputs = loss);

if __name__ == "__main__":
    
    yolov3 = YOLOv3((416,416,3), 80);
    yolov3loss = Loss((416,416,3), 80);
