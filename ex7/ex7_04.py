# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:52:21 2018

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

model = LogisticRegression()

selector = RFECV(model, step=50 , verbose=1)
selector.fit(X_train, y_train)

count = 0
for x in selector.support_:
    if x:
        count += 1
        
plt.plot(range(0,10001,50), selector.grid_scores_)

y_pred = selector.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)