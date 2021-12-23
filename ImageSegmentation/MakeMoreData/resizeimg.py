# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 17:21:35 2021

@author: Arya
"""
# imports
import cv2
import glob
import os

# loading images
directory = r'\Dataset\Br35H-Mask-RCNN\Train_New'
images_path = "/Dataset/Br35H-Mask-RCNN/Train/"
train_images = glob.glob(images_path + "*.jpg")


for img in train_images:
   image = cv2.imread(img)
   file_name = str(img).replace("/Dataset/Br35H-Mask-RCNN/Train\\", "")
   file_name = file_name.replace(".jpg", "")
   tmp = cv2.resize(image, (256, 256))
   os.chdir(directory)
   cv2.imwrite("{}.png".format(file_name), tmp.astype('uint8'))
