#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:35:11 2018

@author: ruusiala
"""

import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu, sigma):
    res = 1 / np.sqrt(2 * np.pi * np.power(sigma, 2)) * np.exp(-1/(2*np.power(sigma, 2)) * np.power((x - mu), 2))
    return res

def log_gaussian(x, mu, sigma):
    res = -np.log(np.sqrt(2 * np.pi * np.power(sigma, 2))) - 1/(2*np.power(sigma, 2)) * np.power((x - mu), 2)
    return res

mu = 0
sigma = 1
x = np.linspace(-5,5)

p = gaussian(x,mu,sigma)
p2 = np.log(p)
p3 = log_gaussian(x,mu,sigma)

plt.figure()
plt.plot(p, 'b')
plt.figure()
plt.plot(p2, 'r')
plt.figure()
plt.plot(p3, 'g')