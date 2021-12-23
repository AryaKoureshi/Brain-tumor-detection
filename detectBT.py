# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 01:19:31 2021

@author: Arya
"""
import cv2
import numpy as np
from tensorflow.keras.optimizers import Adam
import skimage.transform as trans
import sys
sys.path.insert(1, 'D:/BTproject/CNN/')
import model
sys.path.insert(1, 'D:/BTproject/ImageSegmentation/')
import model_bt

image_path = 'D:/BTproject/imgForTest/t (5).jpg'
cnn_weight = 'D:/BTproject/CNN/Model/weights.hdf5'
unet_weight = 'D:/BTproject/ImageSegmentation/Model/weights.hdf5'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = image / 255
image = trans.resize(image,(256,256))

x = []
x.append(image)
x = np.array(x)
x = np.reshape(x, (1, 256, 256, 1))

cnn = model.cnn_bt(pretrained_weights=cnn_weight)
pd_cnn = cnn.predict(x)

if pd_cnn[0][0] > 0.5:
    print('Tumor detected.')
    org = cv2.imread(image_path)
    rows, cols, channels = org.shape     
    unet = model_bt.unet_bt(pretrained_weights=unet_weight)
    unet.compile(optimizer = Adam(), loss = 'binary_crossentropy')
    predicted = unet.predict(np.reshape(image, (1, 256, 256, 1)))
    predicted = predicted.astype(np.float64) * 255
    predicted = np.reshape(predicted, (256, 256))
    predicted = trans.resize(predicted, (rows,cols))
    predicted = predicted.astype(np.uint8)
    predicted = cv2.cvtColor(predicted, cv2.COLOR_GRAY2BGR)
   
    ret, mask = cv2.threshold(predicted, 120, 255, cv2.THRESH_BINARY)
    white_pixels = np.where((mask[:, :, 0] == 255) & 
                            (mask[:, :, 1] == 255) & 
                            (mask[:, :, 2] == 255))
    mask[white_pixels] = [0, 0, 255]
    add = cv2.addWeighted(org, 0.9, mask, 0.7, 0)   
    
    add = cv2.putText(add,
                     'Tumor detected.',
                     (int(rows/20),int(cols/15)),
                     cv2.FONT_HERSHEY_SIMPLEX,
                     cols/(rows+cols),
                     (0,0,255),
                     1,
                     cv2.LINE_AA)
    cv2.imshow('image', add)
    
    file_name = image_path.replace("D:/BTproject/imgForTest/", "")
    file_name = file_name.replace(".jpg", "")
    cv2.imwrite('D:/BTproject/imgForTest/{}predicted.png'.format(file_name),
                add)   
else:
    print('no tumor detected.')
    org = cv2.imread(image_path)
    rows, cols, channels = org.shape     
    org = cv2.putText(org,
                     'No tumor detected.',
                     (int(rows/20),int(cols/15)),
                     cv2.FONT_HERSHEY_SIMPLEX,
                     cols/(rows+cols),
                     (0,255,0),
                     1,
                     cv2.LINE_AA)
    cv2.imshow('image', org)
    file_name = image_path.replace("D:/BTproject/imgForTest/", "")
    file_name = file_name.replace(".jpg", "")
    cv2.imwrite('D:/BTproject/imgForTest/{}predicted.png'.format(file_name),
                org)
cv2.waitKey(0)
cv2.destroyAllWindows()