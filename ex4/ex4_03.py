#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:21:02 2018

@author: ruusiala
"""

from matplotlib.image import imread
from skimage.feature import local_binary_pattern
import numpy as np
import os as os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


images = []
lbp = []
x = []
y = []
path = 'GTSRB_subset/class1/'

for file in os.listdir(path):
    
    if file == '.DS_Store':
        continue
    
    img = imread(''.join([path, file]))
    images.append(img)
    lbp_img = local_binary_pattern(img, 8, 5)
    lbp.append(lbp_img)
    x.append(np.histogram(lbp_img, bins=range(256))[0])
    y.append(1)
    
path = 'GTSRB_subset/class2/'

for file in os.listdir(path):
    
    if file == '.DS_Store':
        continue
    
    img = imread(''.join([path, file]))
    images.append(img)
    lbp_img = local_binary_pattern(img, 8, 5)
    lbp.append(lbp_img)
    x.append(np.histogram(lbp_img, bins=range(256))[0])
    y.append(2)
    
x = np.array(x)
y = np.array(y)

classifiers = []

classifiers.append(KNeighborsClassifier())
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(SVC())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

scores = []

for classifier in classifiers:
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    score =  accuracy_score(y_test, y_pred)
    scores.append(score)
    print(score)


    
