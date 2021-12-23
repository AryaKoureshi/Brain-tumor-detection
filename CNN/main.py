# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 19:28:35 2021

@author: Arya
"""

# imports
import cv2
import glob
import skimage.transform as trans
import numpy as np
import model
from tensorflow.keras.models import save_model

# loading data
x_train = []
y_train = []
x_test = []
y_test = []

path = "D:/BTproject/Dataset/no/"
images = glob.glob(path + "*.jpg")
images.sort()
for x in images:
   image = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
   image = image / 255
   image = trans.resize(image,(256,256))
   y_train.append(0)
   x_train.append(image)

path = "D:/BTproject/Dataset/yes/"
images = glob.glob(path + "*.jpg")
images.sort()
for x in images:
   image = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
   image = image / 255
   image = trans.resize(image,(256,256))
   y_train.append(1)
   x_train.append(image)

path = "D:/BTproject/Dataset/test/no/"
images = glob.glob(path + "*.jpg")
images.sort()
for x in images:
   image = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
   image = image / 255
   image = trans.resize(image,(256,256))
   y_test.append(0)
   x_test.append(image)

path = "D:/BTproject/Dataset/test/yes/"
images = glob.glob(path + "*.jpg")
images.sort()
for x in images:
   image = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
   image = image / 255
   image = trans.resize(image,(256,256))
   y_test.append(1)
   x_test.append(image)
   
del(x, image, images, path)

# preparing data
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_train = np.reshape(x_train, (len(x_train), 256, 256, 1))
x_test = np.reshape(x_test, (len(x_test), 256, 256, 1))

# train model
BTcnn = model.cnn_bt()
BTcnn.summary()

BTcnn.fit(x_train,
          y_train,
          epochs=15,
          batch_size=4,
          verbose=1)
save_model(BTcnn, 'BTcnnModel.h5')

# Evaluate the model on test set
score = BTcnn.evaluate(x_test,
                       y_test,
                       verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])
