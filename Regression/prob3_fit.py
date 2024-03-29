import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
CSCI 635: Introduction to Machine Learning
Problem 3: Multivariate Regression & Classification

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
trial_name = 'p6_reg0'  # will add a unique sub-string to output of this program
degree = 6  # p, degree of model (PLEASE LEAVE THIS FIXED TO p = 6 FOR THIS PROBLEM)
beta = 100  # regularization coefficient
alpha = 0.2  # step size coefficient
n_epoch = 1000  # number of epochs (full passes through the dataset)
eps = 0.0  # controls convergence criterion


# begin simulation

def compute_grad_weight(X, y, theta, beta):
    m = X.shape[0]
    mu = regress(X, theta)
    var1 = (mu - y.T) * mu * (1 - mu)
    var2 = (beta / m) * np.sum(theta[1])
    grad_w = (np.dot(var1, X) / m) + var2
    return grad_w


def compute_grad_bias(X, y, theta):
    mu = regress(X, theta)
    grad_b = np.average((mu - y.T) * mu * (1 - mu))
    return grad_b


def sigmoid(z):
    ############################################################################
    # WRITEME: write your code here to complete the routine
    sig = 1 / (1 + np.exp(-z))
    return sig


############################################################################

def predict(X, theta):
    prob = []
    probd = regress(X, theta)
    for i in probd[0]:
        if i < 0.5:
            prob.append(0)
        else:
            prob.append(1)
    return prob


############################################################################

def regress(X, theta):
    ############################################################################
    # WRITEME: write your code here to complete the routine
    thetab = theta[0]
    thetaw = theta[1]

    mu = sigmoid(thetab + np.dot(thetaw, X.T))
    return mu


############################################################################

def bernoulli_log_likelihood(p, y):
    ############################################################################
    # WRITEME: write your code here to complete the routine
    loga = np.multiply(-y, np.log(p)) - np.multiply((1 - y), np.log(1 - p))
    log = np.average(loga)
    return log


############################################################################

def computeCost(X, y, theta, beta):  ## loss is now Bernoulli cross-entropy/log likelihood
    ############################################################################
    # WRITEME: write your code here to complete the routine
    weight = theta[1]

    mu = regress(X, theta)
    pen = np.average(np.square(weight), axis=1)
    penf = bernoulli_log_likelihood(mu, y) + pen * (beta / 2)
    return penf


############################################################################

def computeGrad(X, y, theta, beta):
    ############################################################################
    # WRITEME: write your code here to complete the routine (
    # NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
    dL_dfy = None  # derivative w.r.t. to model output units (fy)

    dL_db = compute_grad_bias(X, y, theta)  # derivative w.r.t. model weights w
    dL_dw = compute_grad_weight(X, y, theta, beta)  # derivative w.r.t model bias b
    nabla = (dL_db, dL_dw)  # nabla represents the full gradient
    return nabla


############################################################################

path = os.getcwd() + '/data/prob3.dat'
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

x1 = data2['Test 1']
x2 = data2['Test 2']

# apply feature map to input features x1 and x2
cnt = 0
for i in range(1, degree + 1):
    for j in range(0, i + 1):
        data2['F' + str(i) + str(j)] = np.power(x1, i - j) * np.power(x2, j)
        cnt += 1

data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)

# set X and y
cols = data2.shape[1]
X2 = data2.iloc[:, 1:cols]
y2 = data2.iloc[:, 0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
w = np.zeros((1, X2.shape[1]))
b = np.float64([0])
theta2 = (b, w)

L = computeCost(X2, y2, theta2, beta)
halt = 0  # halting variable (you can use these to terminate the loop if you have converged)
print("-1 L = {0}".format(L))
i = 0
while (i < n_epoch and halt == 0):
    dL_db, dL_dw = computeGrad(X2, y2, theta2, beta)

    b = theta2[0]
    w = theta2[1]
    ############################################################################
    # update rules go here...
    # WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
    ############################################################################
    w = (w - (alpha * dL_dw))
    b = b - (alpha * dL_db)
    theta2 = b, w
    L = computeCost(X2, y2, theta2, beta)

    ############################################################################
    # WRITEME: write code to perform a check for convergence (or simply to halt early)
    ############################################################################

    print(" {0} L = {1}".format(i, L))
    i += 1
# print parameter values found after the search
print("w = ", w)
print("b = ", b)

############################################################################
predictions = predict(X2, theta2)
# compute error (100 - accuracy)
err = 0.0
# WRITEME: write your code here calculate your actual classification error (using the "predictions" variable)
n = len(predictions)
count = 0
i = 0
while i < n:
    if predictions[i] != y2[i]:
        count += 1
    i = i + 1
err = count / n
print('Error = {0}%'.format(err * 100.))
############################################################################

## make contour plot input data
xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
xx1 = xx.ravel()
yy1 = yy.ravel()
grid = np.c_[xx1, yy1]
grid_nl = []
# re-apply feature map to inputs x1 & x2
for i in range(1, degree + 1):
    for j in range(0, i + 1):
        feat = np.power(xx1, i - j) * np.power(yy1, j)
        if (len(grid_nl) > 0):
            grid_nl = np.c_[grid_nl, feat]
        else:
            grid_nl = feat
probs = regress(grid_nl, theta2).reshape(xx.shape)

## create contour plot to visualize decision boundaries of model above
f, ax = plt.subplots(figsize=(8, 6))
ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)
ax.scatter(x1, x2, s=50, c=np.squeeze(y2),
           cmap="RdBu",
           vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)
ax.set(aspect="equal",
       xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
       xlabel="$X_1$", ylabel="$X_2$")
## plot done...ready for using/saving

############################################################################
# WRITEME: write your code here to model to save this plot to disk 
#          (look up documentation or the inter-webs for matplotlib)
############################################################################

plt.show()
