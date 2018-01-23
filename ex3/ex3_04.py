#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:05:20 2018

@author: ruusiala
"""

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    
    digits = load_digits()
    
    print(digits.keys())
    
    plt.gray()
    plt.imshow(digits.images[0])
    plt.show()
    
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.20)
    
    clf = KNeighborsClassifier()
    
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    print(accuracy_score(y_test, y_pred))
    
    