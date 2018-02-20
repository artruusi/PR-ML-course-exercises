# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:25:28 2018

@author: ruusiala
"""

from scipy.io import loadmat
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

arcene = loadmat('arcene.mat')

X_train = arcene['X_train']
X_test = arcene['X_test']
y_train = arcene['y_train']
y_test = arcene['y_test']

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

clf = LogisticRegression(penalty='l1')
C_range = 10.0 ** np.arange(0,5)

scores = []
for C in C_range:
    clf.C = C
    
    #selector = RFECV(clf, step=50 , verbose=1)
   # selector.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)

clf.C = C_range[np.argmax(scores)]
clf.fit(X_train, y_train)
    
#selector = RFECV(clf, step=50 , verbose=1)
#selector.fit(X_train, y_train)

'''count = 0
for x in clf.coef_:
    if x:
        count += 1'''

y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)