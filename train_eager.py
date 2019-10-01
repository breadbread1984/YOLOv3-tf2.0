#!/usr/bin/python3

import os;
import numpy as np;
import tensorflow as tf;
import tensorflow_datasets as tfds;
from YOLOv3 import YOLOv3, Loss;
from Predictor import Predictor;
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
    testset = tfds.load(name = "coco2014", split = tfds.Split.TEST, download = False); # without label
    testset = testset.repeat(100).prefetch(tf.data.experimental.AUTOTUNE);
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
        # save model and eval on test set
        if tf.equal(optimizer.iterations % 1000, 0):
            # save checkpoint every 1000 steps
            checkpoint.save(os.path.join('checkpoints','ckpt'));
            yolov3.save('yolov3.h5');
            # evaluate evey 1000 steps
            features = next(testset_iter);
            img = features["image"].numpy().astype('uint8');
            predictor = Predictor(yolov3 = yolov3);
            boundings = predictor.predict(img);
            color_map = dict();
            for bounding in boundings:
                if bounding[5].numpy().astype('int32') in color_map:
                    clr = color_map[bounding[5].numpy().astype('int32')];
                else:
                    color_map[bounding[5].numpy().astype('int32')] = tuple(np.random.randint(low=0, high=256,size=(3)).tolist());
                    clr = color_map[bounding[5].numpy().astype('int32')];
                cv2.rectangle(img, tuple(bounding[0:2].numpy().astype('int32')), tuple(bounding[2:4].numpy().astype('int32')), clr, 2);
            img = tf.expand_dims(img, axis = 0);
            with log.as_default():
                tf.summary.image('detect', img, step = optimizer.iterations);
    yolov3.save('yolov3.h5');

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    main();
