# imports
import model_bt
from tensorflow.keras.models import save_model, Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
import cv2
import glob
import skimage.transform as trans
import numpy as np

# load data
xt_path = "/Dataset/Br35H-Mask-RCNN/Train_New/"
xt_images = glob.glob(xt_path + "*.png")
xt_images.sort()
x_t = []
for img in xt_images:
   image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
   image = image / 255
   image = trans.resize(image,(256,256))
   x_t.append(image)

xa_path = "/Dataset/Br35H-Mask-RCNN/Annotation_Train_New/"
xa_images = glob.glob(xa_path + "*.png")
xa_images.sort()
x_a = []
for msk in xa_images:
   mask = cv2.imread(msk, cv2.IMREAD_GRAYSCALE)
   mask = mask /255
   mask = trans.resize(mask,(256,256))
   mask[mask != 0] = 1
   x_a.append(mask)
'''
yt_path = "/Dataset/Br35H-Mask-RCNN/Test_New/"
yt_images = glob.glob(yt_path + "*.png")
yt_images.sort()
y_t = []
for img in yt_images:
   image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
   image = image / 255
   image = trans.resize(image,(256,256))
   y_t.append(image)

ya_path = "/Dataset/Br35H-Mask-RCNN/Annotation_Test_New/"
ya_images = glob.glob(ya_path + "*.png")
ya_images.sort()
y_a = []
for msk in ya_images:
   mask = cv2.imread(msk, cv2.IMREAD_GRAYSCALE)
   mask = mask /255
   mask = trans.resize(mask,(256,256))
   mask[mask != 0] = 1
   y_a.append(mask)

del(mask, msk, image, img, ya_path, ya_images, xa_path, xa_images, yt_path, yt_images, xt_path, xt_images)
'''

# prepare data
x_t = np.array(x_t)
x_a = np.array(x_a)
#y_t = np.array(y_t)
#y_a = np.array(y_a)
x_t = np.reshape(x_t, (len(x_t), 256, 256, 1))
x_a = np.reshape(x_a, (len(x_a), 256, 256, 1))
#y_t = np.reshape(y_t, (len(y_t), 256, 256, 1))
#y_a = np.reshape(y_a, (len(y_a), 256, 256, 1))

# train model
BTfinder = model_bt.unet_bt(pretrained_weights='ptrtrained weights path')
BTfinder.layers.pop()
BTfinder.outputs = [BTfinder.layers[-1].output]
BTfinder.layers[-1].outbound_nodes = []
x = Conv2D(1, 1, activation = 'sigmoid')(BTfinder.output)
BTfinder = Model(BTfinder.input, x)
for layer in BTfinder.layers[:8]:
   layer.trainable = False
BTfinder.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

BTfinder.summary()
BTfinder.fit(x_t, x_a, epochs=10, batch_size=4)
save_model(BTfinder, '/BTproject/Model/BTfinderModel.h5')

cv2.waitKey(0)
cv2.destroyAllWindows()

'''
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
BTfinder = load_model('/content/drive/MyDrive/Python/Brain_Tumor/BTfinder.h5')
BTfinder.compile(optimizer = optimizers.Adam(), loss = 'binary_crossentropy')
for i in range(len(y_t)):
  predicted = BTfinder.predict(np.reshape(y_t[i], (1, 256, 256, 1)))
  predicted = np.reshape(predicted, (256, 256))
  predicted = predicted.astype(np.float32) * 255
  cv2.imwrite('/content/drive/MyDrive/Python/Datasets/Brain_Tumor/Br35H-Mask-RCNN/Predicted/y{}.png'.format(701+i), predicted)
'''

