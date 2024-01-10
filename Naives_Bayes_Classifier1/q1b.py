import os
import numpy as np
import matplotlib.pyplot as plt

"""
This code was submitted by Anirudh Narayanan.
Class : CSCI 635 (Introduction to Machine Learning)
"""

l = 0.3
alpha = 0.001
n_epoch = 10000
epsilon = 0.001



# Performing hot-encoding
def hot_encoding(y, numClasses):
    return np.squeeze(np.eye(numClasses)[y.reshape(-1)])

# defining the softmax function
def softmax(x):
    z = x - np.max(x)
    sf = np.exp(z) / np.sum(np.exp(z), axis=0)
    return sf


# f function to calculate softmax value
def f_function(theta, X):
    b, W = theta
    f = np.dot(X, W) + b
    #print(f)
    return softmax(f)


# cost helper to help calculate cost
def cost_helper(f):
    p = np.divide(np.exp(f), np.sum(np.exp(f)))
    return p


# main function to calculate cost
def cost_func(theta, y, X):
    b, w = theta
    f = f_function(theta, X)
    N = y.shape[0]
    p = cost_helper(f)
    cost_1 = -(np.sum(np.multiply(y, np.log(p)))) / N
    cost_2 = np.sum(np.square(w)) * l / 2
    cost_final = cost_1 + cost_2
    return cost_final


# gradient function for weights
def compute_grad_weight(theta, X, y):
    b, w = theta
    f = f_function(theta, X)
    N = y.shape[0]
    p = cost_helper(f)
    grad_w = np.dot(X.T, (p - y) / N) + np.dot(l, w)
    # print(grad_w)
    return grad_w

# gradient function for bias
def compute_grad_bias(y, theta):
    f = f_function(theta, X)
    N = y.shape[0]
    p = cost_helper(f)
    grad_b = np.sum(p - y) / N
    return grad_b


# Main function that returns updated gradients
def computeGrad(theta, X, y):
    dL_dw = compute_grad_weight(theta, X, y)  # derivative w.r.t. model weights w
    dL_db = compute_grad_bias(y, theta)
    nabla = (dL_db, dL_dw)  # nabla represents the full gradient
    # print(nabla)
    return nabla


# Predict function
def predict(X, theta):
    predicted = f_function(theta, X)
    return np.argmax(predicted, axis=1)


# Performing gradient check as given in the question, return 0 or 1 depending on whether they match
def gradient_check(X, y, theta):
    b, w = theta
    w_p = w
    b_p = b
    i = j = 0
    while i < (b.shape[0]):
        while j < b.shape[1]:
            b[i][j] = b[i][j] + epsilon
            cost_a = cost_func(theta, y, X)
            b[i][j] = b[i][j] - epsilon
            b[i][j] = b[i][j] - epsilon
            cost_s = cost_func(theta, y, X)
            sec_1 = (cost_a - cost_s)
            sec_final = sec_1 / 2 * epsilon
            b_p[i][j] = sec_final
            b[i][j] = b[i][j] + epsilon
            j = j + 1
        i = i + 1

    m = k = 0
    while m < (w.shape[0]):
        while k < w.shape[1]:
            w[m][k] = w[m][k] + epsilon
            cost_a = cost_func(theta, y, X)
            w[m][k] = w[m][k] - epsilon
            w[m][k] = w[m][k] - epsilon
            cost_s = cost_func(theta, y, X)
            sec_1 = (cost_a - cost_s)
            sec_final = sec_1 / 2 * epsilon
            w_p[m][k] = sec_final
            w[m][k] = w[m][k] + epsilon
            k = k + 1
        m = m + 1



    result = False
    b, w = theta
    b_flat = b.ravel()
    for m in w:
        b_flat = np.append(b_flat, m)
    bp_flat = b_p.ravel()
    for n in w_p:
        bp_flat = np.append(bp_flat, n)
    i = 0
    diff = b_flat - bp_flat
    while i < len(diff):
        if diff[i] < 1e-4:
            result = True
        i = i + 1
    return result


# Get path, calculate values from data
path = os.getcwd() + '/data/spiral_train.dat'
data = np.loadtxt(path, delimiter=',')
X = np.array(data[:, :2])
y_y = np.array(data[:, 2:], dtype=int)


# get types of unique lasses
typeclasses = np.unique(y_y)
numClasses = len(typeclasses)
y = hot_encoding(y_y, numClasses)

# initialize w and b
w = np.ones([X.shape[1], numClasses])
b = np.ones((1, numClasses))
theta = [b, w]

if gradient_check(X, y, theta) == 1:
    print("True, Gradient checking works")
else:
    print('False, Error in Gradient Calculation')
loss_list = []
L = cost_func(theta, y, X)
print("-1 L = {0}".format(L))
i = 0
halt = 0
while i < n_epoch and halt == 0:
    dL_db, dL_dw = computeGrad(theta, X, y)
    b = b - alpha * dL_db
    w = w - alpha * dL_dw

    theta = b, w

    L = cost_func(theta, y, X)
    loss_list.append(L)
    print(" {0} L = {1}".format(i, L))
    i += 1
# print parameter values found after the search
print("w = ", w)
print("b = ", b)

predictions = predict(X, theta).T
# predictions = hot_encoding(predictions, numClasses)

err = 0.0
n = len(predictions)
count = 0
index = 0
i = 0
#print(y_y[0])
while i < n:
    if predictions[i] != y_y[i]:
        count += 1
    i = i + 1
err = count / n
print('Error = {0}%'.format(err * 100.))

# plot the graph for loss over epochs
plt.plot(range(len(loss_list)), loss_list, 'b--')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('out/q1b.png')
plt.show()
