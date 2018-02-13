#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:16:58 2018

@author: ruusiala
"""

from matplotlib.image import imread
import numpy as np
import os as os
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical

images = []
lbp = []
x = []
y = []
path = 'GTSRB_subset_2/class1/'

for file in os.listdir(path):
    
    if file == '.DS_Store':
        continue
    
    img = imread(''.join([path, file]))
    images.append(img)
    x.append(img)
    y.append(1)
    
path = 'GTSRB_subset_2/class2/'

for file in os.listdir(path):
    
    if file == '.DS_Store':
        continue
    
    img = imread(''.join([path, file]))
    images.append(img)
    x.append(img)
    y.append(2)
    
x = np.array(x)
y = np.array(y)

x = (x - np.min(x)) / np.max(x)
y = to_categorical(y-1)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

N = 32 # Number of feature maps
w, h = 5, 5 # Conv. window size

model = Sequential()

model.add(Conv2D(N, (w, h),
          input_shape=(64, 64, 3),
          activation = 'relu',
          padding = 'same'))

model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Conv2D(N, (w, h),
          activation = 'relu',
          padding = 'same'))

model.add(MaxPooling2D((4,4)))

model.add(Flatten())

model.add(Dense(100, activation = 'sigmoid'))
model.add(Dense(2, activation = 'sigmoid'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data = [X_test, y_test])
