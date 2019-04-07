#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_datasets as tfds;

def objects2labels(objects):
    bbox = objects["bbox"];
    label = objects["label"];
    # TODO
