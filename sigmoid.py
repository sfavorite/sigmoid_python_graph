import matplotlib.pyplot as plt
from math import exp
import numpy as np
from scipy.stats import logistic

# This is the Sigmoid (logisitic) function from scratch
# you can also compute this using built in python
# functions. Example below.

def LogisticFromScratch(x):
    if x >= 0:
        z = exp(-x)
        return 1 / (1 + z)
    else:
        z = exp(x)
        return z / (1 + z)


def sigmoid(X):
    return 1 / (1 + np.exp(- X))


x_values = [x / 10.0 for x in range(-60, 60, 5)]
y_values = []

x_linear =[ -1.5, 1.5]
y_linear = [0, 1]
for num in x_values:
    y_values.append(LogisticFromScratch(num))


fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x_values, y_values, label='Logistic', color='r', linestyle='-')
ax.plot(x_linear, y_linear, label='Slope', color='b', linestyle='--')
plt.title('Sigmoid Function')
ax.legend(loc='upper left')
plt.show()

# Compate the from 'scratch' logisitc function with precision 13
for num in x_values:
    print(round(logistic.cdf(num), 13), ' = ', round(LogisticFromScratch(num), 13), round(logistic.cdf(num), 13) == round(LogisticFromScratch(num), 13))
