#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:54:00 2018

@author: ruusiala
"""
from matplotlib.image import imread
from skimage.feature import local_binary_pattern
import numpy as np
import os as os
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer, scale
from sklearn.linear_model import LogisticRegression


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

nrmz = Normalizer()
X = nrmz.fit_transform(x)
X = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

C_range = 10.0 ** np.arange(-5,0)

clf_list = [LogisticRegression(), SVC()]
clf_name = ['LR', 'SVC']

for clf,name in zip(clf_list, clf_name):
    for C in C_range:
        for penalty in ["l1", "l2"]:
            clf.C = C
            clf.penalty = penalty
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            print("%s : %f : %s : %f" % (name, C, penalty, score))

