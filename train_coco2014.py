#!/usr/bin/python3

import os;
import numpy as np;
import tensorflow as tf;
import tensorflow_datasets as tfds;
from YOLOv3 import YOLOv3, YOLOv3Loss;
from preprocess import map_function;

batch_size = 32; # images of different sizes can't be stack into a batch

def main():

    # yolov3 model
    anchors = np.array([[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]], dtype = np.int32);
    yolov3 = YOLOv3(anchors.shape[0] // 3, 80);
    yolov3_loss = YOLOv3Loss(anchors, 80);
    # load downloaded dataset
    trainset = tfds.load(name = "coco2014", split = tfds.Split.TRAIN, download = False);
    trainset = trainset.map(map_function).repeat().shuffle(1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
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
    for images, labels in trainset:
        with tf.GradientTape() as tape:
            outputs = yolov3(images);
            loss = yolov3_loss(images,outputs,labels);
            avg_loss.update_state(loss);
        # write log
        if tf.equal(optimizer.iterations % 100, 0):
            with log.as_default():
                tf.summary.scalar('loss',avg_loss.result(), step = optimizer.iterations);
            print('Step #%d Loss: %.6f' % (optimizer.iterations, avg_loss.result()));
            avg_loss.reset_states();
        grads = tape.gradient(loss, model.trainable_variables);
        optimizer.apply_gradients(zip(grads, model.trainable_variables));
        # save model
        if tf.equal(optimizer.iterations % 1000, 0):
            checkpoint.save(os.path.join('checkpoints','ckpt'));
        if loss < 0.01: break;

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    main();
