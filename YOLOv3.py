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

def YOLOv3(input_shape, anchor_num, class_num):
    
    inputs = tf.keras.Input(shape = input_shape);
    body = Body(inputs.shape[1:]);
    large,middle,small = body(inputs);
    assert len(body.layers) == 7;
    # feature for detecting large scale objects
    x1,y1 = Output(large.shape[1:], 512, anchor_num * (class_num + 5))(large);
    # feature for detecting middle scale objects
    cb1 = ConvBlock(x1.shape[1:], filters = 256, kernel_size = (1,1))(x1);
    us1 = tf.keras.layers.UpSampling2D(2)(cb1);
    cat1 = tf.keras.layers.Concatenate()([us1, middle]);
    x2,y2 = Output(cat1.shape[1:], 256, anchor_num * (class_num + 5))(cat1);
    # feature for detecting small scale objects
    cb2 = ConvBlock(x2.shape[1:], filters = 128, kernel_size = (1,1))(x2);
    us2 = tf.keras.layers.UpSampling2D(2)(cb2);
    cat2 = tf.keras.layers.Concatenate()([us2, small]);
    x3,y3 = Output(cat2.shape[1:], 128, anchor_num * (class_num + 5))(cat2);
    return tf.keras.Model(inputs = inputs, outputs = (y1,y2,y3));

class OutputParser(tf.keras.Model):
    
    def __init__(self, anchors, class_num, input_shape):

        # NOTE: input_shape is given in (input height, input width) order
        # anchors: the sizes of all anchor boxes
        # class_num: the class num of objects
        # input_shape: the size of the input image
        super(OutputParser, self).__init__();
        self.anchors_tensor = tf.constant(anchors, dtype = tf.float32);
        self.anchor_num = anchors.shape[0];
        self.class_num = class_num;
        self.input_shape = input_shape;

    def call(self, feats, calc_loss = False):
        
        # feats.shape = batch x h x w x anchor_num x (1(delta x) + 1(delta y) + 1(width scale) + 1(height scale) + 1(object mask) + class_num(class probability))
        # NOTE: box center absolute x = delta x + prior box upper left x, box center absolute y = delta y + prior box upper left y
        # NOTE: width scale = box width / anchor width,  height scale = box height / anchor height
        grid_shape = tf.shape(feats)[1:3]; #(height, width)
        # [x,y] = meshgrid(x,y) get the upper left positions of prior boxes
        grid_y = tf.tile(tf.reshape(tf.convert_to_tensor(range(grid_shape[0]), dtype = tf.float32),[-1, 1, 1, 1]),[1, grid_shape[1], 1, 1]);
        grid_x = tf.tile(tf.reshape(tf.convert_to_tensor(range(grid_shape[1]), dtype = tf.float32),[1, -1, 1, 1]),[grid_shape[0], 1, 1, 1]);
        grid = tf.concat([grid_x, grid_y], axis = -1);
        # reshape features
        feats = tf.reshape(feats, (-1, grid_shape[0], grid_shape[1], self.anchor_num, self.class_num + 5));
        # box center proportional position = (delta x, delta y) + (priorbox upper left x,priorbox upper left y) / (feature map.width, feature map.height)
        # box_xy.shape = (batch,h,w,anchor_num,2)
        box_xy = (tf.math.sigmoid(feats[..., 0:2]) + grid) / tf.cast(tf.reverse(grid_shape, axis = [0]),dtype = tf.float32);
        # box proportional size = (width scale, height scale) * (anchor width, anchor height) / (image.width, image.height)
        # box_wh.shape = (batch,h,w,anchor_num,2)
        box_wh = tf.exp(feats[..., 2:4]) * self.anchors_tensor / tf.cast(tf.reverse(self.input_shape, axis = [0]),dtype = tf.float32);
        # confidence of being an object
        box_confidence = tf.math.sigmoid(feats[..., 4:5]);
        # class confidence
        box_class_probs = tf.math.sigmoid(feats[..., 5:]);
        # return
        # 1) prior box upper left coordinates
        # 2) reshaped output feature
        # 3) box center proportional positions
        # 4) box proportional sizes
        return grid, feats, box_xy, box_wh if calc_loss == True else box_xy, box_wh, box_confidence, box_class_probs;

class YOLOv3Loss(tf.keras.Model):
    
    PRESET_ANCHORS = np.array([[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]], dtype = np.int32);
    
    def __init__(self, anchors = PRESET_ANCHORS, class_num = None, ignore_thresh = .5):
        
        super(YOLOv3Loss,self).__init__();
        self.num_layers = anchors.shape[0] // 3;
        self.anchors = anchors;
        # which anchor ratios are used for each layer of output
        self.anchor_mask = [[6,7,8],[3,4,5],[0,1,2]] if self.num_layers == 3 else [[3,4,5],[0,1,2]];
        self.class_num = class_num;
        self.ignore_thresh = ignore_thresh;

    def call(self, outputs, labels):

        # outputs is a tuple
        # outputs.shape[layer] = batch x h x w x anchor_num x (1(delta x) + 1(delta y) + 1(width scale) + 1(height scale) + 1(object mask) + class_num(class probability))
        # labels is a tuple
        # labels.shape[layer] = batch x h x w x anchor_num x (1(proportional x) + 1 (proportional y) + 1(proportional width) + 1(proportional height) + 1(object mask) + class_num(class probability))
        # NOTE: the info carried by the output and the label is different.
        assert type(outputs) is tuple;
        assert len(outputs) == self.num_layers;
        # get the input image size according to first output layer
        input_shape = tf.shape(outputs[0])[1:3] * 32;
        # grid size of all layers
        grid_shapes = [tf.shape(outputs[l])[1:3] for l in range(self.num_layers)];
        # get batch size
        batch_size = tf.shape(outputs[0])[0];
        batch_size_float = tf.cast(batch_size, dtype = tf.float32);
        loss = 0;
        for l in range(self.num_layers):
            # objectness: object_mask.shape = (batch,h,w,anchor_num,1)
            object_mask = labels[l][..., 4:5];
            object_mask_bool = tf.cast(object_mask, dtype = tf.bool);
            # 1) ignore masks
            # class type: true_class_probs.shape = (batch,h,w,anchor_num,class_num)
            true_class_probs = labels[l][...,5:];
            grid, raw_pred, pred_xy, pred_wh = OutputParser(self.anchors[self.anchor_mask[l]], self.class_num, input_shape)(outputs[l], calc_loss = True);
            # box proportional coordinates: pred_box.shape = (batch,h,w,anchor_num,4)
            pred_box = tf.concat([pred_xy, pred_wh], axis = -1);
            # boolean_mask doesnt support batch, so have to iterate over each of batch
            ignore_masks = list();
            for b in range(batch_size):
                # true boxes proportional coordinates of this layer and current batch
                true_box = tf.boolean_mask(labels[l][b,...,0:4], object_mask_bool[b,...,0]);
                # iou.shape = (h,w,anchor_num,true_box_num)
                iou = self.box_iou(pred_box[b], true_box);
                # select a true box having the maximum iou for each anchor box: best_iou.shape = (h,w,anchor_num)
                best_iou = tf.math.reduce_max(iou, axis = -1);
                # ignore anchor box with iou below given threshold
                ignore_mask = tf.where(tf.less(best_iou,self.ignore_thresh),tf.ones_like(best_iou),tf.zeros_like(best_iou));
                ignore_masks.append(ignore_mask);
            ignore_masks = tf.stack(ignore_masks);
            # ignore_masks.shape = (batch, h, w, anchor_num, 1)
            ignore_masks = tf.expand_dims(ignore_masks, axis = -1);
            # 2) loss
            # anchor box sizes: anchors_tensor.shape = (anchor_num, 2)
            anchors_tensor = tf.constant(self.anchors[self.anchor_mask[l]], dtype = tf.float32);
            # (delta x, delta y) = (proportional x, proportional y) * (feature map.width, feature map.height) - (priorbox upper left x,priorbox upper left y)
            # true box's delta x, delta y: raw_true_xy.shape = (batch,h,w,anchor_num,2)
            raw_true_xy = labels[l][...,:2] * tf.reverse(grid_shapes[l], axis = [0]) - grid;
            # (width scale, height scale) = (proportional width, proportional height) * (image.width, image.height) / (anchor width, anchor height)
            # true box's width scale,height scale: raw_true_wh.shape = (batch,h,w,anchor_num,2)
            raw_true_wh = tf.math.log(labels[l][..., 2:4] * tf.cast(tf.reverse(input_shape, axis = [0]),dtype = tf.float32) / anchors_tensor);
            # filter out none object anchor boxes
            raw_true_wh = tf.where(tf.stack([object_mask_bool,object_mask_bool], axis = -1), raw_true_wh, tf.zeros_like(raw_true_wh));
            # 2 - proportional area = 2 - proportional width * proportional height
            # box area is larger, loss is smaller
            box_loss_scale = 2 - labels[l][...,2:3] * labels[l][...,3:4];
            # xy_loss = -raw_true_xy * log(sigmoid(raw_xy)) - (1-raw_true_xy) * log(1 - sigmoid(raw_xy))
            xy_loss = object_mask * box_loss_scale * tf.keras.losses.BinaryCrossentropy(from_logits = True)(raw_true_xy, raw_pred[...,0:2]);
            # wh_loss = (raw_true_wh - raw_wh)^2
            wh_loss = object_mask * box_loss_scale * 0.5 * tf.math.square(raw_true_wh - raw_pred[...,2:4]);
            # confidence_loss = true_mask*(-true_mask*log(sigmoid(raw_mask))-(1-true_mask)*log(1-sigmoid(raw_mask))) + 
            # (1-true_mask)*(-true_mask*log(sigmoid(raw_mask))-(1-true_mask)*log(1-sigmoid(raw_mask)))*ignore_mask
            confidence_loss = object_mask * tf.keras.losses.BinaryCrossentropy(from_logits = True)(object_mask, raw_pred[...,4:5]) + \
                (1 - object_mask) * tf.keras.losses.BinaryCrossentropy(from_logits = True)(object_mask, raw_pred[...,4:5]) * ignore_masks;
            class_loss = object_mask * tf.keras.losses.BinaryCrossentropy(from_logits = True)(true_class_probs, raw_pred[...,5:]);
            
            xy_loss = tf.math.reduce_sum(xy_loss) / batch_size_float;
            wh_loss = tf.math.reduce_sum(wh_loss) / batch_size_float;
            confidence_loss = tf.math.reduce_sum(confidence_loss) / batch_size_float;
            class_loss = tf.math.reduce_sum(class_loss) / batch_size_float;
            loss += xy_loss + wh_loss + confidence_loss + class_loss;

        return loss;

    def box_iou(self, b1, b2):
        
        # calculate ious of given boxes' proportional coordinates
        assert len(b1.shape) == 4 and b1.shape[-1] == 4;
        assert len(b2.shape) == 2 and b2.shape[-1] == 4;
        # b1.shape = (h,w,anchor_num,1,4)
        b1 = tf.expand_dims(b1, axis = -2);
        b1_xy = b1[...,:2];
        b1_wh = b1[...,2:4];
        b1_wh_half = b1_wh / 2.;
        b1_mins = b1_xy - b1_wh_half; #(left, top).shape = (h,w,anchor_num,1,2)
        b1_maxes = b1_xy + b1_wh_half; #(right, bottom).shape = (h,w,anchor_num,1,2)
        # b2.shape = (1,true_box_num,4)
        b2 = tf.expand_dims(b2, axis = 0);
        b2_xy = b2[...,:2];
        b2_wh = b2[...,2:4];
        b2_wh_half = b2_wh / 2.;
        b2_mins = b2_xy - b2_wh_half; #(left, top).shape = (1,true_box_num,2)
        b2_maxes = b2_xy + b2_wh_half; #(right, bottom).shape = (1,true_box_num,2)
        # the following operations are done between each anchor box and each true box by broadcasting
        # intersect_mins.shape = (h,w,anchor_num,true_box_num,2)
        intersect_mins = tf.math.maximum(b1_mins, b2_mins);
        # intersect_maxes.shape = (h,w,anchor_num,true_box_num,2)
        intersect_maxes = tf.math.maximum(b1_maxes, b2_maxes);
        intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins, 0.); #(intersect w, intersect h).shape = (h,w,anchor_nu,true_box_num,2)
        # intersect_area.shape = (h,w,anchor_num,true_box_num)
        intersect_area = intersect_wh[...,0] * intersect_wh[...,1];
        b1_area = b1_wh[...,0] * b1_wh[...,1];
        b2_area = b2_wh[...,0] * b2_wh[...,1];
        # iou between each anchor box and each true box
        iou = intersect_area / (b1_area + b2_area - intersect_area);
        # iou.shape = (h,w,anchor_num,true_box_num)
        return iou;

if __name__ == "__main__":
    
    anchors = np.array([[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]], dtype = np.int32);
    yolov3 = YOLOv3(anchors.shape[0] // 3, 80);
    yolov3_loss = YOLOv3Loss(anchors, 80);
