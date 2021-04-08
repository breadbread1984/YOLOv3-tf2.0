#!/usr/bin/python3

from absl import app, flags;
from os.path import join;
from pycocotools.coco import COCO;
from pycocotools.cocoeval import COCOeval;
import numpy as np;
import cv2;
import tensorflow as tf;
from Predictor import Predictor;

FLAGS = flags.FLAGS;
flags.DEFINE_string('model', 'yolov3.h5', 'path to model file to evaluate');
flags.DEFINE_string('coco_eval_dir', None, 'path to coco evaluate directory');
flags.DEFINE_string('annotation_dir', None, 'path to annotation directory');

label_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, -1, 25, 26, -1, -1, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, -1, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, -1, 61, -1, -1, 62, -1, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, -1, 74, 75, 76, 77, 78, 79, 80];

def main(argv):

  yolov3 = tf.keras.models.load_model(FLAGS.model, compile = False);
  predictor = Predictor(yolov3 = yolov3);
  anno = COCO(join(FLAGS.annotation_dir, 'instances_val2017.json'));
  count = 0;
  for imgid in anno.getImgIds():
    print("processing (%d/%d)" % (count, len(anno.getImgIds())));
    detections = list();
    # predict
    img_info = anno.loadImgs([imgid])[0];
    img = cv2.imread(join(FLAGS.coco_eval_dir, img_info['file_name']));
    boundings = predictor.predict(img).numpy();
    # collect results
    for bounding in boundings:
      detections.append([imgid, bounding[0], bounding[1], bounding[2] - bounding[0], bounding[3] - bounding[1], bounding[4], label_map.index(int(bounding[5]) + 1)]);
    count += 1;
  cocoDt = anno.loadRes(np.array(detections));
  cocoEval = COCOeval(anno, cocoDt, iouType = 'bbox');
  cocoEval.params.imgIds = anno.getImgIds();
  cocoEval.evaluate();
  cocoEval.accumulate();
  cocoEval.summarize();

if __name__ == "__main__":

  app.run(main);

