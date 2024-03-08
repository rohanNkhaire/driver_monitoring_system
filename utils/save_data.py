# -*- coding: utf-8 -*-

# IMPORTS
import cv2
import os
import splitfolders

# CREATE TEST AND TRAIN DATA -------------------------------------------------------------------------------------------
# TRAIN DATA
def save_training_data(directory, classes):
    training_data = []

    # OS MODULE TO JOIN AND SEARCH FOR IMAGE FOLDER LOCATION
    for category in classes:
        save_path = os.path.join(save_train_directory, category)
        path = os.path.join(directory, category)
        class_num = classes.index(category)

        # IMAGE SIZE DESCALING(using resize()) TO 240*240 AND APPENDING DATA TO training_data
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_img = cv2.resize(img_array, (160, 120))
            cv2.imwrite(save_path + '/' + str(img), new_img)

# TESTING DATA - Around 79k test images for 10 classes------------------------------------------------------------------
# IMAGES WILL BE APPENDED TO testing_data LIST
def save_testing_data(test_directory):
    testing_data = []

    for img in os.listdir(test_directory):
        img_array = cv2.imread(os.path.join(test_directory, img), cv2.IMREAD_GRAYSCALE)
        new_img = cv2.resize(img_array, (96, 96))
        cv2.imwrite(save_test_directory + '/' + str(img), new_img)
# eof

# LOAD TRAIN AND TEST DATA----------------------------------------------------------------------------------------------
# IMAGE PATH ON DISK FOR TRAIN AND TEST IMAGES
save_train_directory = '/home/rohan/Projects/EmbeddedMachineLearning/resized_imgs_qqvga/train'
save_test_directory = '/home/rohan/Projects/EmbeddedMachineLearning/resized_imgs_gray/test'
directory = '/home/rohan/Projects/EmbeddedMachineLearning/imgs/train'
test_directory = '/home/rohan/Projects/EmbeddedMachineLearning/imgs/test'
classes = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

# CALL LOAD FUNCTIONS FOR TRAINING IMAGES
#save_training_data(directory, classes)
#splitfolders.ratio(save_train_directory, output="train_split_qqvga", ratio=(0.7, 0.3, 0.0))

# CALL LOAD FUNCTIONS FOR TEST IMAGES
save_testing_data(test_directory)

# eof
