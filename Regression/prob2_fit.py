import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
CSCI 635: Introduction to Machine Learning
Problem 2: Polynomial Regression &

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

# NOTE: you will need to tinker with the meta-parameters below yourself
#       (do not think of them as defaults by any means)
# meta-parameters for program
trial_name = 'p2_fit'  # will add a unique sub-string to output of this program
degree = 15  # p, order of model
beta = 1  # regularization coefficient
alpha = 0.5  # step size coefficient
eps = 0.0  # controls convergence criterion
n_epoch = 5000  # number of epochs (full passes through the dataset)


# begin simulation
def compute_grad_weight(mu, y, X, beta, w):
    grad_w = np.average(np.multiply((mu - y), X), axis=1) + beta * np.average(w, axis=1)
    return grad_w


def compute_grad_bias(mu, y):
    grad_b = np.average(mu - y, axis=1)
    return grad_b


def mapping(X, degree):
    mapped = []

    for i in X:
        j = 0
        while j < abs(degree):
            mapped = np.append(mapped, np.power(i, j + 1))
            j = j + 1
    map_array = np.array(mapped)
    shaped_array = map_array.reshape(-1, degree)
    # print(shapedarray)
    return shaped_array


def regress(X, theta):
    ############################################################################
    # WRITEME: write your code here to complete the routine
    theta_b = theta[0]
    theta_w = theta[1]
    mu = np.dot(X, theta_w.T) + theta_b
    return mu


############################################################################

def gaussian_log_likelihood(mu, y):
    ############################################################################
    m = mu.size
    cost = 0.0
    for i in range(m):
        cost += (mu[i][0] - y[i][0]) ** 2
    return cost / (2 * m)


############################################################################

def computeCost(X, y, theta, beta):  ## loss is now Bernoulli cross-entropy/log likelihood
    ############################################################################
    # WRITEME: write your code here to complete the routine
    b, w = theta
    p = 0.0

    mu = np.dot(X, w.T) + b
    m = mu.size
    cost = gaussian_log_likelihood(mu, y)
    p = np.sum(np.power(w,2))
    p = (p * beta) / (2 * m)
    cost = cost + p
    return cost





############################################################################


def computeGrad(X, y, theta, beta):
    ############################################################################
    # WRITEME: write your code here to complete the routine (
    # NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
    dL_dfy = None  # derivative w.r.t. to model output units (fy)
    X = X.T
    y = y.T

    w = theta[1]
    b = theta[0]

    mu = np.dot(w, X) + b

    dL_dw = compute_grad_weight(mu, y, X, beta, w)  # derivative w.r.t. model weights w
    dL_db = compute_grad_bias(mu, y)

    nabla = (dL_db, dL_dw)  # nabla represents the full gradient
    return nabla


############################################################################

path = os.getcwd() + '/data/prob2.dat'
data = pd.read_csv(path, header=None, names=['X', 'Y'])

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)

############################################################################
# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you
#          could use a loop and array concatenation)
############################################################################
X = mapping(X, degree)

# convert to numpy arrays and initalize the parameter array theta
w = np.zeros((1, X.shape[1]))
b = np.array([0])
theta = [b, w]

L = computeCost(X, y, theta, beta)
halt = 0  # halting variable (you can use these to terminate the loop if you have converged)
print("-1 L = {0}".format(L))
cost = []
i = 0
while (i < n_epoch and halt == 0):
    dL_db, dL_dw = computeGrad(X, y, theta, beta)
    b = theta[0]
    w = theta[1]
    ############################################################################
    # update rules go here...
    # WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
    ############################################################################
    b = b - (alpha * dL_db)
    w = w - (alpha * dL_dw)
    theta = b, w

    L = computeCost(X, y, theta, beta)
    cost.append(L)
    ############################################################################
    # WRITEME: write code to perform a check for convergence (or simply to halt early)
    ############################################################################

    print(" {0} L = {1}".format(i, L))
    i = i + 1
# print parameter values found after the search
print("w = ", w)
print("b = ", b)

kludge = 0.25
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
# we need this otherwise, the dimension is missing (turns shape(value,) to shape(value,value))
X_feat = np.expand_dims(X_test, axis=1)

############################################################################
# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you
#          could use a loop and array concatenation)
############################################################################
X_feat = mapping(X_feat, degree)

plt.plot(X_test, regress(X_feat, theta), label="Model")
plt.scatter(X[:, 0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")
plt.savefig('out/prob2_resultb1.png')
plt.show()
############################################################################
# WRITEME: write your code here to save plot to disk (look up documentation or
#          the inter-webs for matplotlib)
############################################################################

plt.show()
