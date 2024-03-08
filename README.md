# driver_monitoring_system

![](media/test1.gif)

![](media/test2.gif)

## Objective ##
This Repo consist of Driver Monitoring task(Alert/Not-Alert) on Beaglebone Black and Arduino Nano 33 BLE using the tensorflow framework.

## Dataset ##
The dataset is taken from Kagle's [Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection). Check it out.

## Neural Network Architecture ##
### MobileNetv1 ##
- We performed transfer-learning using keras's [MobileNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/MobileNet) architecture with *0.25%*  downsizing.
- Fed images are RGB(224, 224, 3) with data Augmentation.
- The total model size **before int8 quantization** is 863.73 KB.

## ShuffleNetv2 ##
- We used [ShuffleNetV2](https://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html) for training with the same architecture of *0.5%* downsizing.
- Fed images are grayscale(96, 96, 1) with data augmentation.
- The total model size **before int8 quantization** is 599.51 KB.

## Usage ##
```bash
# Clone the repo
git clone https://github.com/rohanNkhaire/driver_monitoring_system.git

# cd into the dir
cd driver_monitoring_system

# Run the training script
# Using MobileNetV1
python3 MobileNetV1.py

# Using ShuffleNetV2
python3 shufflenetv2.py
```

## Embedded Devices ##
- The above training scripts outputs *int8* quantized models to reduce the memory.
- The memory requirements is significantly reduced after *int8 quantization*.
- Quantization is necessary to run the CNN models on **Arduino Nano 33 BLE**.
- For **Beaglebone Black**, [OpenCV's](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html) Deep Neural Network module is used for inference.

## Note ##
The Convolutional Neural Network architecture is implemented using **Tensorflow**.

The *beagle_bone* repo provides script to host a local server using socket to send images to and recieve images from the *BeagleBone Black*.

*utils* folder provides a script to save data from the dataset.

