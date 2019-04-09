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

def preprocess(image, objects, input_shape = (416,416), random = False, jitter = .3):
    
    assert 4 == len(image.shape) and 3 == image.shape[-1];
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
        # randomly resize image
        new_ar = 
    # TODO
