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
    image, label = preprocess(feature["image"], feature["objects"], random = True);
    
    return image, label;

def preprocess(image, objects, input_shape = (416,416), random = False, jitter = .3, hue = .1, sat = 1.5, bri = .1):
    
    assert 4 == len(image.shape) and 3 == image.shape[-1];
    assert 0 < jitter < 1;
    assert -1 < hue < 1;
    assert 0 < sat;
    assert 0 < bri < 1;
    bbox = objects["bbox"]; # bbox.shape = (batch, object_num,4), 4-tuple is (ymin,xmin,ymax,xmax)
    label = objects["label"]; # label.shape = (batch, object_num)
    is_crowd = objects["is_crowd"]; # is_crowd.shape = (batch, object_num)
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
