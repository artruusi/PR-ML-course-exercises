#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:21:06 2018

@author: ruusiala
"""
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    f0 = 0.017
    
    w = np.sqrt(0.25) + np.random.randn(100)
    n = np.arange(100)
    
    x = np.sin(2 * np.pi * f0 * n) + w
    
    plt.plot(x)
    
    scores = []
    frequencies = []
    
    for f in np.linspace(0, 0.5, 1000):
        
        # Create vector e. Assume data is in x.
        
        n = np.arange(100)
        z = -2 * np.pi * 1j * f * n # <compute -2*pi*i*f*n. Imaginary unit is 1j>
        e = np.exp(z)
        
        score = np.abs(np.dot(x,e)) # <compute abs of dot product of x and e>
        scores.append(score)
        frequencies.append(f)
        
    fHat = frequencies[np.argmax(scores)]