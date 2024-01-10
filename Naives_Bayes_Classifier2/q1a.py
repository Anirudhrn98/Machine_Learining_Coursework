import os
import numpy as np
import matplotlib.pyplot as plt

"""
This code was submitted by Anirudh Narayanan.
Class : CSCI 635 (Introduction to Machine Learning)
"""


alpha = 0.005
n_epoch = 10000
epsilon = 0.01


# Performing hot-encoding
def hot_encoding(y, numClasses):
    return np.squeeze(np.eye(numClasses)[y.reshape(-1)])


# defining the softmax function
def softmax(x):
    z = x - np.max(x)
    sf = np.exp(z) / np.sum(np.exp(z), axis=0)
    return sf


# f function to calculate softmax value
def f_function(X, theta):
    b = theta[0]
    W = theta[1]
    f = np.dot(X, W) + b
    return f


# implement hidden layer with forward propagation
def forward_propagation(X, theta):
    b = theta[0]
    w = theta[1]
    b2 = theta[2]
    w2 = theta[3]
    theta1 = (b, w)
    theta2 = (b2, w2)
    f1 = f_function(X, theta1)
    a1 = np.tanh(f1)
    f2 = f_function(a1, theta2)
    a2 = softmax(f2)
    cache = (f1, a1, f2, a2)
    return cache


# calculate cost
def cost_func(theta, y, X):
    cache = forward_propagation(X, theta)
    predict = cache[3]
    cost = np.sum(np.square(predict - y)) / (2 * X.shape[0])
    return cost


# gradient function for weights
def compute_grad_weight(X, err_1, err_2, a1):
    dl_dw1 = (np.dot(X.T, err_1)) * (1 / X.shape[0])
    dl_dw2 = np.dot(a1.T, err_2)
    dl_dw = (dl_dw1, dl_dw2)
    return dl_dw


# gradient function for bias
def compute_grad_bias(X, err_1, err_2):
    dl_db1 = np.sum(err_1, axis=1, keepdims=True) * (1 / X.shape[0])
    dl_db2 = np.sum(err_2, axis=1, keepdims=True) * (1 / X.shape[0])
    dl_db = (dl_db1, dl_db2)
    return dl_db


# Main function that returns updated gradients
def computeGrad(theta, X, y):
    cache = forward_propagation(X, theta)
    a1 = cache[1]
    a2 = cache[3]
    w2 = theta[3]
    err_2 = a2 - y
    errx = np.dot(err_2, w2.T)
    err_1 = np.multiply(errx, (1 - np.square(a1)))
    dL_dw = compute_grad_weight(X, err_1, err_2, a1)  # derivative w.r.t. model weights w
    dL_db = compute_grad_bias(X, err_1, err_2)
    dl_dw1 = dL_dw[0]
    dl_dw2 = dL_dw[1]
    dl_db1 = dL_db[0]
    dl_db2 = dL_db[1]
    nabla = (dl_db1, dl_dw1, dl_db2, dl_dw2)
    return nabla


# Predict function
def predict(X, theta):
    cache = forward_propagation(X, theta)
    predicted = cache[3]
    return np.argmax(predicted, axis=1)


# Performing gradient check as given in the question, return 0 or 1 depending on whether they match
def gradient_check(X, y, theta):
    b = theta[0]
    w = theta[1]
    b2 = theta[2]
    w2 = theta[3]
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

    i2 = j2 = 0
    while i2 < (b2.shape[0]):
        while j2 < b2.shape[1]:
            b2[i2][j2] = b2[i2][j2] + epsilon
            cost_a = cost_func(theta, y, X)
            b2[i2][j2] = b2[i2][j2] - epsilon
            b2[i2][j2] = b2[i2][j2] - epsilon
            cost_s = cost_func(theta, y, X)
            sec_1 = (cost_a - cost_s)
            sec_final = sec_1 / 2 * epsilon
            b_p[i2][j2] = sec_final
            b2[i2][j2] = b2[i2][j2] + epsilon
            j2 = j2 + 1
        i2 = i2 + 1

    m2 = k2 = 0
    while m2 < (w2.shape[0]):
        while k2 < w2.shape[1]:
            w2[m2][k2] = w2[m2][k2] + epsilon
            cost_a = cost_func(theta, y, X)
            w2[m2][k2] = w2[m2][k2] - epsilon
            w2[m2][k2] = w2[m2][k2] - epsilon
            cost_s = cost_func(theta, y, X)
            sec_1 = (cost_a - cost_s)
            sec_final = sec_1 / 2 * epsilon
            w_p[m2][k2] = sec_final
            w2[m2][k2] = w2[m2][k2] + epsilon
            k2 = k + 1
        m2 = m + 1
    result = False
    b_c = theta[0]
    w_c = theta[1]
    b_flat = b_c.ravel()
    for m in w_c:
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
# for part 2
    b2_c = theta[0]
    w2_c = theta[1]
    b_flat = b2_c.ravel()
    for m in w2_c:
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


# Get path, calculate values from training data
path = os.getcwd() + '/data/xor.dat'
data = np.loadtxt(path, delimiter=',')
X = np.array(data[:, :2])
y_y = np.array(data[:, 2:], dtype=int)

# get types of unique lasses
typeclasses = np.unique(y_y)
numClasses = len(typeclasses)
y = hot_encoding(y_y, numClasses)

# initialize w and b
w = np.random.normal(loc=0, scale=1, size=(X.shape[1], numClasses))
b = np.zeros((1, numClasses))
w2 = np.random.normal(loc=0, scale=1, size=(X.shape[1], numClasses))
b2 = np.zeros((1, numClasses))
theta = (b, w, b2, w2)

# Performing gradient checking
if gradient_check(X, y, theta) == 1:
    print("True, Gradient checking works")
else:
    print('False, Error in Gradient Calculation')

L = cost_func(theta, y, X)
print("-1 L = {0}".format(L))
i = 0
halt = 0
loss_list = []

# Run the model for number of epochs
while i < n_epoch and halt == 0:
    nabla = computeGrad(theta, X, y)
    b = theta[0]
    w = theta[1]
    b2 = theta[2]
    w2 = theta[3]

    b = b - alpha * nabla[0]
    w = w - alpha * nabla[1]
    b2 = b2 - alpha * nabla[2]
    w2 = w2 - alpha * nabla[3]
    theta = (b, w, b2, w2)
    L = cost_func(theta, y, X)
    loss_list.append(L)

    print(" {0} L = {1}".format(i, L))
    i += 1


# Calculate predictions and plot the loss
predictions = predict(X, theta)


err = 0.0
n = len(predictions)
count = 0
index = 0
i = 0
# print(y_y[0])
while i < n:
    if predictions[i] != y_y[i]:
        count += 1
    i = i + 1
err = count / n
print('Error = {0}%'.format(err * 100.))
plt.plot(range(len(loss_list)), loss_list, 'b-')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.savefig('out/q1a.png')
plt.show()
