# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 20:52:37 2021

@author: Arya
"""
# imports
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def cnn_bt(pretrained_weights=None, input_shape=(256,256,1)):
    model = Sequential()
    model.add(Conv2D(32, 4, activation = 'relu', padding = 'same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, 4, activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, 4, activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model
