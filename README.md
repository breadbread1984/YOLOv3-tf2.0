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

currently, only eager training is workable. I am still debugging keras's fit training.

### how to test

after training, you can find yolov3.h5 under current directory. test the model with following command

```bash
python3 test.py
```

### how to predict with the trained model

