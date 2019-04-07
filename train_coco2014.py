#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_datasets as tfds;
from YOLOv3 import YOLOv3, YOLOv3Loss;
from Label import objects2labels;

batch_size = 32;

def main():

    # yolov3 model
    anchors = np.array([[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]], dtype = np.int32);
    yolov3 = YOLOv3(anchors.shape[0] // 3, 80);
    yolov3_loss = YOLOv3Loss(anchors, 80);
    # load dataset
    coco2014_builder = tfds.builder("coco2014");
    coco2014_builder.download_and_prepare();
    trainset = coco2014_builder.as_dataset(split = tfds.Split.TRAIN);
    testset = coco2014_builder.as_dataset(split = tfds.Split.TEST);
    trainset = trainset.repeat().shuffle(1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
    testset = testset.repeat().shuffle(1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
    # restore from existing checkpoint
    optimizer = tf.keras.optimizers.Adam(1e-3);
    if False == os.path.exists('checkpoints'): os.mkdir('checkpoints');
    checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer, optimizer_step = optimizer.iterations);
    checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
    # tensorboard summary
    log = tf.summary.create_file_writer('checkpoints');
    # train model
    print("training...");
    avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
    for features in trainset:
        images = features["image"];
        labels = objects2labels(features["objects"]);
        with tf.GradientTape() as tape:
            outputs = yolov3(images);
            loss = yolov3_loss(outputs,labels);
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
