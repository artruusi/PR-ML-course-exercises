# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from matplotlib.image import imread
import numpy as np
import os as os
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16

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

base_model = VGG16(include_top = False, weights = "imagenet",
                   input_shape = (64, 64, 3))

w = base_model.output

w = Flatten()(w)

w = Dense(100, activation = "relu")(w)

output = Dense(2, activation = "sigmoid")(w)

model = Model(input = [base_model.input], outputs = [output])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data = [X_test, y_test])