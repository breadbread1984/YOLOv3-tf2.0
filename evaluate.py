#!/usr/bin/python3

from absl import app, flags;
from os.path import join;
from pycocotools.coco import COCO;

FLAGS = flags.FLAGS;
flags.DEFINE_string('model', 'yolov3.h5', 'path to model file to evaluate');
flags.DEFINE_string('coco_eval_dir', None, 'path to coco evaluate directory');
flags.DEFINE_string('annotation_dir', None, 'path to annotation directory');

label_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, -1, 25, 26, -1, -1, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, -1, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, -1, 61, -1, -1, 62, -1, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, -1, 74, 75, 76, 77, 78, 79, 80];

def main(argv):

  yolov3 = tf.keras.models.load_model(FLAGS.model, compile = False);
  anno = COCO(join(FLAGS.annotation_dir, 'instances_val2017.json'));
  # TODO
  
def preprocess(image, input_shape = (416,416,3), conf_thres = 0.5, nms_thres = 0.5):

  images = tf.expand_dims(image, axis = 0);
  resize_images = tf.image.resize(images, input_shape[:2], method = tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio = True);
  resize_shape = resize_images.shape[1:3]
  top_pad = (input_shape[0] - resize_shape[0]) // 2;
  bottom_pad = input_shape[0] - resize_shape[0] - top_pad;
  left_pad = (input_shape[1] - resize_shape[1]) // 2;
  right_pad = input_shape[1] - resize_shape[1] - left_pad;
  resize_images = tf.pad(resize_images,[[0,0], [top_pad,bottom_pad], [left_pad,right_pad], [0,0]], constant_values = 128);
  deviation = tf.constant([left_pad / input_shape[1], top_pad / input_shape[0], 0, 0], dtype = tf.float32);
  scale = tf.constant([
    input_shape[1] / resize_shape[1], input_shape[0] / resize_shape[0],
    input_shape[1] / resize_shape[1], input_shape[0] / resize_shape[0]
  ], dtype = tf.float32);
  images_data = tf.cast(resize_images, tf.float32) / 255.;
  return images_data;

if __name__ == "__main__":

  app.run(main);

