#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:16:58 2018

@author: ruusiala
"""

from matplotlib.image import imread
from skimage.feature import local_binary_pattern
import numpy as np
import os as os
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
AdaBoostClassifier, GradientBoostingClassifier)

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


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

clf_list = [RandomForestClassifier(100), ExtraTreesClassifier(100),
            AdaBoostClassifier(n_estimators=100), GradientBoostingClassifier()]
clf_name = ['RandomForest', 'ExtraTrees', 'AdaBoost', 'GradientBoost']


for clf,name in zip(clf_list, clf_name):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("%s : %f" % (name, score))
