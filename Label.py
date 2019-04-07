#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_datasets as tfds;

def objects2labels(objects):
    
    batch_size = objects.shape[0];
    for b in range(batch_size):
        for obj in objects[b]:
            bbox = objects["bbox"]; #relative bbox = (ymin,xmin,ymax,xmax)
            label = objects["label"];
            # TODO
