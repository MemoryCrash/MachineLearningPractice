#!/usr/bin/env python
# -*-coding:UTF-8 -*-

#用来临时绘制一些数学图形

import numpy as np
import matplotlib.pyplot as plt
from pylab import annotate

def f(z):
	return (1/(1+np.exp(-z)))

X = np.linspace(-10, 10, 256, endpoint=True)
Y = f(X)

plt.plot(X, Y)
plt.text(3, 0.5, r'$\sigma\left(z\right)=\frac{1}{1+e^{-z}}$', fontsize = 15)

plt.show()





