# YOLOv3-tf2.0

YOLOv3 implemented with tensorflow 2.0

### how to train on MS COCO 2014

download datasets by executing the following command

```Bash
python3 download_datasets.py
```

make sure no errors occur during the execution.

then train the model by executing the following command

```Bash
python3 train_eager.py
```
or
```Bash
python3 train_keras.py
```

here are some results of my model which can be downloaded [here](#). I trained YOLO v3 from scratch for 500k iterations with batch size 8 on a single gtx 1080 ti. which is far from the quality of the official darknet model which was trained on four gtx1080 for 500k iterations. I show my result here anyway.

<p align="center">
  <img src="pics/train_loss.png" alt="train loss" width="200" />
  <img src="pics/validation_loss.png" alt="validation loss" width="200" />
</p>
<p>
  <img src="pics/detection1.png" width="200" />
  <img src="pics/detection2.png" width="200" />
  <img src="pics/detection3.png" width="200" />
  <img src="pics/detection4.png" width="200" />
</p>

### how to predict with the trained model

detect objects in an image by executing the following command

```bash
python3 Predictor.py <path/to/image>
```

### how to train YOLOv3 on your own data

compose label file in the following format.

```text
<path/to/image1> <target num>
<x> <y> <width> <height> <label>
<x> <y> <width> <height> <label>
...
<x> <y> <width> <height> <label>
<path/to/image2> <target num>
...
```

generate tfrecord file by executing the following command.

```bash
python3 create_dataset.py <path/to/annotation>
```

the script will generate trainset.tfrecord and validationset.tfrecord.

read the tfrecord with following code.

```python
from create_dataset import parse_function_generator;
trainset = tf.data.TFRecordDataset('trainset.tfrecord').map(parse_function_generator(num_classes = num_classes)).repeat(100).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
```

