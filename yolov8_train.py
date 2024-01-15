# coding = utf-8

# IMPORTS---------------------------------------------------------------------------------------------------------------
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.yaml')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='/home/rohan/Projects/EmbeddedMachineLearning/train_split_gray', epochs=50, imgsz=96, optimizer='Adam', batch=32,
                                 pretrained=False, dropout=0.5)

#Inference
# Define path to the image file
#source = 'resized_imgs/test/img_2.jpg'

# Run inference on the source
#results = model(source)  # list of Results objects

# exporting a model
model.export(format='tflite', imgsz=96, int8=True)

