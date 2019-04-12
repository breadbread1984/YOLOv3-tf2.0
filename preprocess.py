#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_datasets as tfds;

# sample example
# is_crowd means whether one bbox bounds several objects
# {
# 'bbox': <tf.Tensor: id=538, shape=(1, 3, 4), dtype=float32, numpy=
# array([[[0.38278195, 0.34665626, 0.8226504 , 0.7704375 ],
#         [0.40229324, 0.65940624, 0.6712218 , 0.7636719 ],
#         [0.1137594 , 0.38051564, 0.33261278, 0.5337656 ]]], dtype=float32)>,
# 'label': <tf.Tensor: id=540, shape=(1, 3), dtype=int64, numpy=array([[9, 0, 9]])>, 
# 'is_crowd': <tf.Tensor: id=539, shape=(1, 3), dtype=bool, numpy=array([[False, False, False]])>
# }

# anchor boxes are given in (width, height) order
PRESET_ANCHORS = np.array([[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]], dtype = np.int32);

def parse_function(serialized_example):
    
    feature = tfds.features.FeaturesDict({
        # Images can have variable shape
        "image": tfds.features.Image(encoding_format="jpeg"),
        "image/filename": tfds.features.Text(),
        "objects": tfds.features.SequenceDict({
            "bbox": tfds.features.BBoxFeature(),
            # Coco has 91 categories but only 80 are present in the dataset
            "label": tfds.features.ClassLabel(num_classes=80),
            "is_crowd": tf.bool,
        }),
    });
    image, bbox = preprocess(feature["image"], feature["objects"]["bbox"], random = True);
    label = bbox_to_tensor(bbox, feature["objects"]["label"]);
    
    return image, label;

def preprocess(image, bbox, input_shape = (416,416), random = False, jitter = .3, hue = .1, sat = 1.5, bri = .1):
    
    assert 4 == len(image.shape) and 3 == image.shape[-1];
    assert 0 < jitter < 1;
    assert -1 < hue < 1;
    assert 0 < sat;
    assert 0 < bri < 1;
    img_shape = image.shape[1:3]; #(height, width)
    
    if False == random:
        # scale the input image to make the wider edge fit the input shape
        # NOTE: I don't use resize_with_pad because it can only stuff zeros, but I want 128
        resize_image = tf.image.resize(image, input_shape, method = tf.image.ResizeMethod.mitchellcubic, preserve_aspect_ratio = True);
        resize_shape = resize_image.shape[1:3]; #(height, width)
        top_pad = (input_shape[0] - resize_shape[0]) // 2;
        bottom_pad = input_shape[0] - resize_shape[0] - top_pad;
        left_pad = (input_shape[1] - resize_shape[1]) // 2;
        right_pad = input_shape[1] - resize_shape[1] - bottom_pad;
        resize_image = tf.pad(resize_image,[[0,0],[top_pad,bottom_pad],[left_pad,right_pad],[0,0]], constant_values = 128);
        # cast to float32
        image_data = tf.cast(resize_image, tf.float32) / 255.;
        # correct boxes
        bbox = bbox * tf.convert_to_tensor([resize_shape[0], resize_shape[1], resize_shape[0], resize_shape[1]], dtype = tf.float32);
        bbox = bbox + tf.convert_to_tensor([top_pad,left_pad,top_pad,left_pad], dtype = tf.float32);
        bbox = bbox / tf.convert_to_tensor([input_shape[0],input_shape[1],input_shape[0],input_shape[1]], dtype = tf.float32);
        # return
        return image_data, bbox;
    else:
        # randomly sample aspect ratio to input shape
        # resize image to the randomly sampled input shape
        aspect_ratio_jitter = tf.random.uniform(shape = [2], minval = 1-jitter, maxval = 1+jitter, dtype = tf.float32);
        resize_input_shape = tf.convert_to_tensor(input_shape, dtype = tf.float32) * aspect_ratio_jitter;
        scale = tf.random.uniform(shape=[1], minval = .25, maxval = 2, dtype = tf.float32);
        resize_shape = tf.cond(tf.greater(resize_input_shape[0],resize_input_shape[1]),true_fn = lambda: scale * resize_input_shape / aspect_ratio_jitter[0], false_fn = lambda: scale * resize_input_shape / aspect_ratio_jitter[1]);
        resize_image = tf.image.resize(image, resize_shape, method = tf.image.ResizeMethod.mitchellcubic);
        if input_shape[0] > resize_shape[0]:
            pad = input_shape[0] - resize_shape[0];
            resize_image = tf.pad(resize_image,[[0,0],[pad,pad],[0,0],[0,0]], constant_values = 128);
            # sample crop offset_height
            offset_height = int(np.random.rand() * pad);
            # correct boxes
            bbox = bbox * tf.convert_to_tensor([resize_shape[0], resize_shape[1], resize_shape[0], resize_shape[1]], dtype = tf.float32);
            bbox = bbox + tf.convert_to_tensor([pad, 0, pad, 0], dtype = tf.float32);
            resize_shape = resize_shape + tf.convert_to_tensor([2 * pad,0], dtype = tf.float32);
            bbox = bbox / tf.convert_to_tensor([resize_shape[0], resize_shape[1], resize_shape[0], resize_shape[1]], dtype = tf.float32);
        else:
            crop = resize_shape[0] - input_shape[0];
            # sample crop offset_height
            offset_height = int(np.random.rand() * crop);
        if input_shape[1] > resize_shape[1]:
            pad = input_shape[1] - resize_shape[1];
            resize_image = tf.pad(resize_image,[[0,0],[0,0],[pad,pad],[0,0]], constant_values = 128);
            # sample crop offset_width
            offset_width = int(np.random.rand() * pad);
            # correct boxes
            bbox = bbox * tf.convert_to_tensor([resize_shape[0], resize_shape[1], resize_shape[0], resize_shape[1]], dtype = tf.float32);
            bbox = bbox + tf.convert_to_tensor([0, pad, 0, pad], dtype = tf.float32);
            resize_shape = resize_shape + tf.convert_to_tensor([0, 2 * pad], dtype = tf.float32);
            bbox = bbox / tf.convert_to_tensor([resize_shape[0], resize_shape[1], resize_shape[0], resize_shape[1]], dtype = tf.float32);
        else:
            crop = resize_shape[1] - input_shape[1];
            # sample crop offset_width
            offset_width = int(np.random.rand() * crop);
        # crop
        resize_image = tf.image.crop_to_bounding_box(resize_image, offset_height, offset_width, input_shape[0], input_shape[1]);
        # correct boxes
        bbox = bbox * tf.convert_to_tensor([resize_shape[0], resize_shape[1], resize_shape[0], resize_shape[1]], dtype = tf.float32);
        bbox = bbox + tf.convert_to_tensor([-offset_height, -offset_width, -offset_height, -offset_width], dtype = tf.float32);
        bbox = bbox / tf.convert_to_tensor([input_shape[0], input_shape[1], input_shape[0], input_shape[1]], dtype = tf.float32);
        # randomly flip image
        if np.random.rand() < .5:
            resize_image = tf.image.flip_left_right(resize_image);
            # correct boxes(y remains while x = 1 - x)
            bbox = tf.convert_to_tensor([0, 1, 0, 1], dtype = tf.float32) + tf.convert_to_tensor([1,-1,1,-1], dtype = tf.float32) * bbox;
        # distort image in HSV color space
        image_data = tf.cast(resize_image, tf.float32) / 255.;
        image_data = tf.image.random_hue(image_data, hue);
        image_data = tf.image.random_saturation(image_data, lower = 1./sat, upper = sat);
        image_data = tf.image.random_brightness(image_data, bri);
        # discard invalid boxes (small box or box having negative width or height)
        bbox_hw = bbox[...,2:4] - bbox[...,0:2] # bbox_hw.shape = (1,bbox_num,2)
        bbox_hw = bbox_hw * tf.convert_to_tensor(input_shape, dtype = tf.float32);
        valid = tf.math.logical_and(bbox_hw[...,0] > 1,bbox_hw[...,1] > 1); # valid.shape = (1,bbox_num)
        valid_bbox = tf.boolean_mask(bbox, valid); # valid_bbox.shape = (valid box num, 4)
        bbox = tf.expand_dims(valid_bbox, axis = 0); # bbox.shape = (1,valid box num, 4)
        # return
        return image_data, bbox;

def bbox_to_tensor(bbox, label, input_shape = (416,416), anchors = PRESET_ANCHORS, num_classes = 80):
    
    # bbox.shape = (1, box num, 4) which represents (ymin,xmin,ymax,xmax)
    # label.shape = (1, box num)
    # anchors = (9,2)
    assert tf.math.reduce_sum(tf.cast(label >= num_classes, dtype = tf.int32)).numpy() == 0;
    num_layers = len(anchors) // 3;
    anchor_mask = [[6,7,8],[3,4,5],[0,1,2]] if num_layers == 3 else [[3,4,5],[1,2,3]];
    
    true_boxes = tf.zeros_like(bbox);
    true_boxes[..., 0:2] = tf.reverse(bbox[...,0:2] + bbox[...,2:4] // 2, axis = [-1]); # box center proportional position
    true_boxes[..., 2:4] = tf.reverse(bbox[...,2:4] - bbox[...,0:2], axis = [-1]); # box proportional size
    input_shape_tensor = tf.reverse(tf.convert_to_tensor(input_shape, dtype = tf.float32), axis = [0]);
    boxes_xy = true_boxes[..., 0:2] * input_shape_tensor; # box center absolute position
    boxes_wh = true_boxes[..., 2:4] * input_shape_tensor; # box absolute size
    
    # create tensor for label
    y_true = tuple((tf.zeros(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l],input_shape[1] // {0:32, 1:16, 2:8}[l],anchor_mask[l], 5 + num_classes), dtype = tf.float32) for l in range(num_layers)));
    
    # center the anchor boxes at the origin, get the max and min of corners' (x,y)
    anchors = tf.expand_dims(tf.convert_to_tensor(anchors, dtype = tf.float32), 0); # anchors.shape = (1, 9, 2)
    anchors_maxes = anchors / 2.; # max of width, height, anchors_maxes.shape = (1, 9, 2)
    anchors_mins = -anchors_maxes; # min of width, height, anchors_mins.shape = (1, 9, 2)
    
    # center the bbox at the origin, get the max and min of corners' (x,y)
    valid_mask = boxes_wh[...,0] > 0; # valid box should have width > 0: valid_mask.shape = (1, box_num)
    wh = tf.boolean_mask(boxes_wh, valid_mask); # absolute size: wh.shape = (valid box num, 2)
    valid_true_boxes = tf.boolean_mask(true_boxes, valid_mask); # box proportional position: valid_true_boxes.shape = (valid box num, 4)
    valid_label = tf.boolean_mask(label, valid_mask); # valid_label.shape = (valid box num)
    # if there is any valid bbox, get anchor box which has the maximum iou with current bbox.
    if wh.shape[0] > 0:
        wh = tf.expand_dims(wh, -2); # wh.shape = (valid box num, 1, 2)
        box_maxes = wh / 2; # max of width, height, box_maxes.shape = (valid box num, 1, 2)
        box_mins = -box_maxes; # min of width, height, box_mins.shape = (valid box num, 1, 2)
        intersect_mins = tf.math.maximum(box_mins, anchor_mins); # intersect_mins.shape = (valid box num, anchor num(9), 2)
        intersect_maxes = tf.math.minimum(box_maxes, anchor_maxes); # intersect_maxes.shape = (valid box num, anchor num(9), 2)
        intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins, 0.); # intersect_wh.shape = (valid box num, anchor num(9), 2)
        intersect_area = intersect_wh[...,0] * intersect_wh[...,1]; # intersect_area.shape = (valid box num, anchor num(9))
        box_area = wh[...,0] * wh[...,1]; # box_area.shape = (valid box_num, 1)
        anchor_area = anchors[...,0] * anchors[...,1]; # anchor_area.shape = (1, anchor num(9))
        iou = intersect_area / (box_area + anchor_area - intersect_area); # iou.shape = (valid box num, anchor num(9))
        # get the anchor box having maximum iou with each true bbbox
        best_anchor = tf.math.argmax(iou, axis = -1); # best_anchor.shape = (valid box num)
        # fill in label tensor
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = int(valid_true_boxes[t,1] * y_true[l].shape[0]) # absolute center y = proportional y * grid_shape.height
                    j = int(valid_true_boxes[t,0] * y_true[l].shape[1]) # absolute center x = proportional x * grid_shape.width
                    k = anchor_mask[l][n]; # best anchor box id
                    c = valid_label[t]; # class
                    y_true[l][i,j,k,0:4] = valid_true_boxes[t,0:4]; # box proportional position (w,y,width,height)
                    y_true[l][i,j,k,4] = 1; # object mask
                    y_true[l][i,j,k,5 + c] = 1; # class mask

    return y_true;