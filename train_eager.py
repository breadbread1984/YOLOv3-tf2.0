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
    testset = tfds.load(name = "coco2014", split = tfds.Split.TEST, download = False);
    testset = testset.map(map_function).repeat(100).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
    testset_iter = testset.__iter__();
    # restore from existing checkpoint
    optimizer = tf.keras.optimizers.Adam(1e-4);
    if False == os.path.exists('checkpoints'): os.mkdir('checkpoints');
    checkpoint = tf.train.Checkpoint(model = yolov3, optimizer = optimizer, optimizer_step = optimizer.iterations);
    checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
    # tensorboard summary
    log = tf.summary.create_file_writer('checkpoints');
    # train model
    print("training...");
    train_loss = tf.keras.metrics.Mean(name = 'train loss', dtype = tf.float32);
    eval_loss = tf.keras.metrics.Mean(name = 'eval_loss', dtype = tf.float32);
    for images, labels in trainset:
        with tf.GradientTape() as tape:
            outputs = yolov3(images);
            loss = yolov3_loss([*outputs, *labels]);
        # check whether the loss numberic is correct
        try:
            loss_check = tf.debugging.check_numerics(loss, 'the loss is not correct! cancel train_loss update!');
            with tf.control_dependencies([loss_check]):
                train_loss.update_state(loss);
        except tf.errors.OpError as e:
            continue;
        print('Step #%d Loss: %.6f' % (optimizer.iterations, loss));
        # write log
        if tf.equal(optimizer.iterations % 10, 0):
            with log.as_default():
                tf.summary.scalar('train loss',train_loss.result(), step = optimizer.iterations);
            train_loss.reset_states();
        grads = tape.gradient(loss, yolov3.trainable_variables);
        # check whether the grad numerics is correct
        try:
            grads_check = [tf.debugging.check_numerics(grad, 'the grad is not correct! cancel gradient apply!') for grad in grads];
            with tf.control_dependencies(grads_check):
                optimizer.apply_gradients(zip(grads, yolov3.trainable_variables));
        except tf.errors.OpError as e:
            continue;
        # save model
        if tf.equal(optimizer.iterations % 1000, 0):
            # evaluate evey 1000 steps
            for i in range(10):
                images, labels = next(testset_iter);
                outputs = yolov3(images);
                loss = yolov3_loss([*outputs, * labels]);
                try:
                    loss_check = tf.debugging.check_numerics(loss, 'the loss is not correct! cancel eval_loss update!');
                    with tf.control_dependencies([loss]):
                        eval_loss.update_state(loss);
                except tf.errors.OpError as e:
                    continue;
            print('Step #%d Eval Loss: %.6f' % (optimizer.iterations, eval_loss.result()));
            with log.as_default():
                tf.summary.scalar('eval loss', eval_loss.result(), step = optimizer.iterations);
            eval_loss.reset_states();
            # save checkpoint every 1000 steps
            checkpoint.save(os.path.join('checkpoints','ckpt'));
            yolov3.save('yolov3.h5');
    yolov3.save('yolov3.h5');

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    main();
