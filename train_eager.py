#!/usr/bin/python3

import os;
import numpy as np;
import tensorflow as tf;
import tensorflow_datasets as tfds;
from YOLOv3 import YOLOv3, Loss;
from preprocess import map_function;

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1';
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3';
#os.environ['CUDA_VISIBLE_DEVICES'] = '';
batch_size = 8; # images of different sizes can't be stack into a batch

def main():

    # yolov3 model
    yolov3 = YOLOv3((416,416,3), 80);
    yolov3_loss = Loss((416,416,3), 80);
    # load downloaded dataset
    trainset = tfds.load(name = "coco2014", split = tfds.Split.TRAIN, download = False);
    trainset = trainset.map(map_function).repeat(100).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
    # restore from existing checkpoint
    optimizer = tf.keras.optimizers.Adam(1e-4);
    if False == os.path.exists('checkpoints'): os.mkdir('checkpoints');
    checkpoint = tf.train.Checkpoint(model = yolov3, optimizer = optimizer, optimizer_step = optimizer.iterations);
    checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
    # tensorboard summary
    log = tf.summary.create_file_writer('checkpoints');
    # train model
    print("training...");
    avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
    for images,labels in trainset:
        with tf.GradientTape() as tape:
            outputs = yolov3(images);
            loss = yolov3_loss([*outputs, *labels]);
        # never update model with nan loss
        if tf.math.is_inf(loss) or tf.math.is_nan(loss):
            continue;
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
        if tf.equal(optimizer.iterations % 1000, 0):
            checkpoint.save(os.path.join('checkpoints','ckpt'));
            yolov3.save_weights('yolov3.h5');
    yolov3.save('yolov3.h5');

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    main();
