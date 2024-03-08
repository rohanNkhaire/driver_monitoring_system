import cv2
import numpy as np
import socket
import sys
import pickle
import struct
# setting up OpenCV to get video from camera
cap=cv2.VideoCapture(0)
frame,image = cap.read()
# using socket to send video as bytes
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('192.68.7.2',8089))

while True:
    ret,frame=cap.read()
    resize = cv2.resize(frame,(224,224))
    #resize = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    # Serialize frame
    data = pickle.dumps(resize)

    # Send message length first
    message_size = struct.pack("L", len(data)) ### CHANGED

    # Then data
    clientsocket.sendall(message_size + data)
    cv2.imshow('frame', resize)
    cv2.waitKey(1)
    
