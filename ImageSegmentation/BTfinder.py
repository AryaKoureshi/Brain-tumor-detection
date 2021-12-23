from tensorflow.keras.models import load_model, Model
from tensorflow.keras import optimizers
import model_bt
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import skimage.transform as trans

BTfinder = model_bt.unet_bt(pretrained_weights='/BTproject/Model/weights.hdf5')
BTfinder.compile(optimizer = optimizers.Adam(), loss = 'binary_crossentropy')
BTfinder.summary()

t_path = "/BTproject/Test/"
t_images = glob.glob(t_path + "*.jpg")
t_images.sort()

for img in t_images:
   image = cv2.imread(img)
   rows, cols, channels = image.shape   
   
   tmp = cv2.imread(img, cv2.IMREAD_GRAYSCALE)    
   tmp = tmp / 255
   tmp = trans.resize(tmp,(256,256))
   predicted = BTfinder.predict(np.reshape(tmp, (1, 256, 256, 1)))
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
   add = cv2.addWeighted(image, 0.9, mask, 0.7, 0)   
   
   file_name = str(img).replace("/BTproject/Test\\", "")
   file_name = file_name.replace(".jpg", "")
   cv2.imwrite('/BTproject/Test/{}predicted.png'.format(file_name),
               add)
