#!/usr/bin/python3

import os;
import numpy as np;
import tensorflow as tf;
import tensorflow_datasets as tfds;
from YOLOv3 import YOLOv3, YOLOv3Loss;
from preprocess import map_function_impl;

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1';
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3';
#os.environ['CUDA_VISIBLE_DEVICES'] = '';
batch_size = 14; # images of different sizes can't be stack into a batch

def main():

    # yolov3 model
    yolov3 = YOLOv3((416,416,3), 80);
    yolov3_loss = YOLOv3Loss((416,416,3), 80);
    # load downloaded dataset
    trainset = tfds.load(name = "coco2014", split = tfds.Split.TRAIN, download = False);
    trainset = trainset.repeat().batch(1).prefetch(tf.data.experimental.AUTOTUNE);
    # restore from existing checkpoint
    optimizer = tf.keras.optimizers.Adam(1e-3);
    if False == os.path.exists('checkpoints'): os.mkdir('checkpoints');
    checkpoint = tf.train.Checkpoint(model = yolov3, optimizer = optimizer, optimizer_step = optimizer.iterations);
    checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
    # tensorboard summary
    log = tf.summary.create_file_writer('checkpoints');
    # train model
    print("training...");
    avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
    count = 0;
    batch = {'images': list(), 'labels1': list(), 'labels2': list(), 'labels3': list()};
    for feature in trainset:
        image, label1, label2, label3 = map_function_impl(tf.squeeze(feature["image"],[0]), tf.squeeze(feature["objects"]["bbox"],[0]), tf.squeeze(feature["objects"]["label"],[0]));
        batch['images'].append(image);
        batch['labels1'].append(label1);
        batch['labels2'].append(label2);
        batch['labels3'].append(label3);
        count = count + 1;
        if count == batch_size:
            images = tf.stack(batch['images']);
            labels1 = tf.stack(batch['labels1']);
            labels2 = tf.stack(batch['labels2']);
            labels3 = tf.stack(batch['labels3']);
            # reset buffer
            count = 0;
            batch = {'images': list(), 'labels1': list(), 'labels2': list(), 'labels3': list()};
            with tf.GradientTape() as tape:
                outputs = yolov3(images);
                loss = yolov3_loss(outputs,(labels1, labels2, labels3));
            avg_loss.update_state(loss);
            print('Step #%d Loss: %.6f' % (optimizer.iterations, loss));
            # write log
            if tf.equal(optimizer.iterations % 10, 0):
                with log.as_default():
                    tf.summary.scalar('loss',avg_loss.result(), step = optimizer.iterations);
                avg_loss.reset_states();
            grads = tape.gradient(loss, yolov3.trainable_variables);
            optimizer.apply_gradients(zip(grads, yolov3.trainable_variables));
            # save model
            if tf.equal(optimizer.iterations % 10, 0):
                checkpoint.save(os.path.join('checkpoints','ckpt'));

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    main();
