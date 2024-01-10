from concurrent.futures import thread
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
CSCI 635: Introduction to Machine Learning
Problem 1: Univariate Regression

@author/lecturer - Alexander G. Ororbia II

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
alpha = 0.1  # step size coefficient
eps = 0.01  # controls convergence criterion
n_epoch = 10  # number of epochs (full passes through the dataset)


# begin simulation
def compute_grad_weight(mu, y, X):
    grad_w = np.average(np.multiply((mu - y.T), X.T))
    return grad_w


def compute_grad_bias(mu, y):
    grad_b = np.average((mu - y.T))
    return grad_b


def regress(X, theta):
    mu = X * theta[1] + theta[0]
    return mu


############################################################################

def gaussian_log_likelihood(mu, y):
    mse = np.average((mu - y.T) ** 2)/2
    return mse

############################################################################
# implement equation 2 on data
def computeCost(X, y, theta):  # loss is now Bernoulli cross-entropy/log likelihood
    w = theta[1]
    b = theta[0]
    mu = np.dot(w.T, X.T) + b

    cost = gaussian_log_likelihood(mu, y)

    return cost


############################################################################

def computeGrad(X, y, theta):
    ############################################################################
    # NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)

    w = theta[1]
    b = theta[0]
    mu = np.dot(w.T, X.T) + b
    dL_dfy = None  # derivative w.r.t. to model output units (fy)
    dL_dw = compute_grad_weight(mu, y, X)  # derivative w.r.t. model weights w
    dL_db = compute_grad_bias(mu, y)
    nabla = (dL_db, dL_dw)  # nabla represents the full gradient
    return nabla


############################################################################

path = os.getcwd() + '/data/prob1.dat'
data = pd.read_csv(path, header=None, names=['X', 'Y'])

# display some information about the dataset itself here
print('lenght: ', len(data))
print('shape: ', data.shape)
print('head: ', data.head())
############################################################################
# WRITEME: write your code here to print out information/statistics about the
#          data-set "data" using Pandas (consult the Pandas documentation to learn how)
# WRITEME: write your code here to create a simple scatterplot of the dataset
#          itself and print/save to disk the result
############################################################################

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)

# convert to numpy arrays and initalize the parameter array theta
w = np.zeros((1, X.shape[1]))
b = np.array([0])
theta = [b, w]

L = computeCost(X, y, theta)
print("-1 L = {0}".format(L))
L_best = L
halt = 0  # halting variable (you can use these to terminate the loop if you have converged)
i = 0
cost = []  # you can use this list variable to help you create the loss versus epoch plot at the end (if you want)
while (i < n_epoch and halt == 0):
    dL_db, dL_dw = computeGrad(X, y, theta)
    b = theta[0]
    w = theta[1]
    ############################################################################
    # Update Rules
    b = b - (dL_db * alpha)
    w = w - (dL_dw * alpha)
    theta[0] = b
    theta[1] = w

    ############################################################################

    # (note: don't forget to override the theta variable...)
    L = computeCost(X, y, theta)  # track our loss after performing a single step
    cost.append(L)
    print(" {0} L = {1}".format(i, L))

    i = i + 1
# print parameter values found after the search
print("weight = ", w)
print("bias = ", b)

kludge = 0.25  # helps with printing the plots (you can tweak this value if you like)
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_test = np.expand_dims(X_test, axis=1)

plt.plot(X_test, regress(X_test, theta), label="Model")
plt.scatter(X[:, 0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")

############################################################################
# WRITEME: write your code here to save plot to disk (look up documentation or
#          the inter-webs for matplotlib)
############################################################################
plt.savefig('out/prob1.png')
plt.show()
############################################################################
# visualize the loss as a function of passes through the dataset
# WRITEME: write your code here create and save a plot of loss versus epoch
############################################################################
plt.plot(range(len(cost)), cost, 'b')
plt.legend(['Loss Graph'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('out/prob1_loss.png')
plt.show()  # convenience command to force plots to pop up on desktop
