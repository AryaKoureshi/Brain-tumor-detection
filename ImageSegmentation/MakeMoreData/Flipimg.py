# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 00:37:44 2021

@author: Arya
"""

import cv2
import glob
import os

# loading images
directory = r'\Dataset\Br35H-Mask-RCNN\Annotation_Train_New'
images_path = "/Dataset/Brain_tumor/Br35H-Mask-RCNN/Annotation_Train_New/"
train_images = glob.glob(images_path + "*.png")


for img in train_images:
   image = cv2.imread(img)
   file_name = str(img).replace("/Dataset/Br35H-Mask-RCNN/Annotation_Train_New\\", "")
   file_name = file_name.replace(".png", "")
   image = cv2.flip(image, 1)
   tmp = cv2.resize(image, (256, 256))
   os.chdir(directory)
   cv2.imwrite("f{}.png".format(file_name), tmp.astype('uint8'))
