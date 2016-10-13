"""Softmax."""


import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)




x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
print (scores)
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()




# Caveat 2, multiply and Divide scores by 10
# 1. sofmax funxtion on normal scores
scores = [3.0, 1.0, 0.2]
print (softmax(scores))
# 1. Multiply the scores by 10
scores = scores*10
