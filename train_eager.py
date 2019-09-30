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

def safe_execution(values, func):
    assert type(values) is list;
    try:
        values_check = [tf.debugging.check_numerics(value, 'the value is not currect! cancel func execution!') for value in values];
        with tf.control_dependencies(values_check):
            func(values);
        return True;
    except:
        print('invalid numeric detected!');
        return False;

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
        if False == safe_execution([loss], lambda x: train_loss.update_state(x[0])): continue;
        print('Step #%d Loss: %.6f' % (optimizer.iterations, loss));
        # write log
        if tf.equal(optimizer.iterations % 10, 0):
            with log.as_default():
                tf.summary.scalar('train loss',train_loss.result(), step = optimizer.iterations);
            train_loss.reset_states();
        grads = tape.gradient(loss, yolov3.trainable_variables);
        # check whether the grad numerics is correct
        if False == safe_execution(grads, lambda x: optimizer.apply_gradients(zip(x, yolov3.trainable_variables))): continue;
        # save model
        if tf.equal(optimizer.iterations % 1000, 0):
            # evaluate evey 1000 steps
            for i in range(10):
                images, labels = next(testset_iter);
                outputs = yolov3(images);
                loss = yolov3_loss([*outputs, * labels]);
                if False == safe_execution([loss], lambda x: eval_loss.update_state(x[0])): continue;
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
