import pickle
import socket
import struct
import numpy as np
import cv2
import time
import csv
from tempfile import mkdtemp
class_names = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
# load the DNN model
model = cv2.dnn.readNetFromTensorflow('MobileNet_NW_rgb.pb')
#model = cv2.dnn.readNet(model='MobileNet_rgb.pb',                config='MobileNet_rgb.pbtxt',framework='TensorFlow')
# setting up server to recieve images from client
HOST = '192.168.7.2'
PORT = 8089

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn, addr = s.accept()

data = b'' ### CHANGED
payload_size = struct.calcsize("L") ### CHANGED

while True:

    # Retrieve message size
    while len(data) < payload_size:
        data += conn.recv(4096)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0] ### CHANGED

    # Retrieve all data based on message size
    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Extract frame

    frame = pickle.loads(frame_data) 
    #cv2.imshow('frame', frame)
    #cv2.waitKey(1)

    image = frame

       # create blob from image
    blob = cv2.dnn.blobFromImage(image=image, size=(96, 96), mean=(100, 100, 100), swapRB=False, ddepth=cv2.CV_32F)
    #blob = np.asarray(image).reshape(1, 1, 96, 96)
    #image = cv2.resize(image,None,fx=0.5,fy=0.5)
    model.setInput(blob)
    output = model.forward() 
    #print(output)
    #print(blob.shape)
   
    x = np.argmax(output)
    #print(x)
    if x==0:
        print("Alert")
    else:
        print("Not Alert")
  
    cv2.imshow('frame', image)
    cv2.waitKey(1)

