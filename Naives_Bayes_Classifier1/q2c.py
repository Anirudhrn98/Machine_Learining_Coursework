import numpy as np
import pandas as pd

"""
This code was submitted by Anirudh Narayanan.
Class : CSCI 635 (Introduction to Machine Learning)
"""


# Calculate and return the priors as a list
def priors(df, Y):
    prior_list = []
    cs = sorted(df[Y].unique())
    for i in range(len(cs)):
        prior_list.append(len(df[df[Y] == i]) / len(df))
    return prior_list


# Calculate mean and standard deviation of the dataFrame features and compute the gaussian likelihood
def getGauss(df, feat, feat_value, label, Y):
    df = df[df[Y] == label]
    mean = df[feat].mean()
    std = df[feat].std()
    exp1 = (1 / np.sqrt(2 * np.pi) * std)
    exp2 = np.exp(-(feat_value - mean) ** 2) / (2 * (std ** 2))
    return exp1 * exp2


# Calculate probability if data is discrete
def getCategorical(df: pd.DataFrame, feat, feat_value, label, Y):
    df = df[df[Y] == label]
    return len(df[df[feat] == feat_value]) / len(df)


# Helper function for Bayesian Classifier
def _bayesClassifier_helper(df, x, y, labels, feats):
    likelihood = [1] * len(labels)
    for i in range(len(labels)):
        for j in range(len(feats) - 1):
            feat = feats[j]
            feat_Value = x[j]
            label = labels[i]
            if feat in c:
                likelihood[i] *= getGauss(df, feat, feat_Value, label, y)
            else:
                likelihood[i] *= getCategorical(df, feat, feat_Value, label, y)
    return likelihood


# Main Naive Bayes Classifier Function
def bayesClassifier(df, X, y):
    feats = list(df.columns)
    predictedList = []
    prior = priors(df, y)
    labels = sorted(list(df[y].unique()))
    for x in X:
        likelihood = _bayesClassifier_helper(df, x, y, labels, feats)
        post_probability = [1] * len(labels)
        for m in range(len(labels)):
            post_probability[m] = likelihood[m] * prior[m]
        predictedList.append(np.argmax(post_probability))
    return np.array(predictedList)


# Calculate accuracy of classifier
def calc_accuracy(yTest, pred):
    acc = np.sum(yTest == pred) / len(yTest)
    return acc


# Input data processing
def type_conversion(df):
    all_columns = df.columns
    for i in all_columns:
        df[i] = df[i].astype(int)
    return df


data_training = pd.read_csv("/data/q3.csv")
trainData = pd.DataFrame(data_training)
trainData.drop([149, 199], axis=0, inplace=True)
trainData = type_conversion(trainData)

data_testing = pd.read_csv("/data/q3b.csv")
testData = pd.DataFrame(data_testing)

testData = type_conversion(testData)


# split data into Xtest and yTest
xTest = testData.iloc[:, :8].values
yTest = testData.iloc[:, 8].values

# continuous data types
c = [' # sentences', '# words']

# Run classifier and print accuracy
pred = bayesClassifier(trainData, xTest, " is spam")
print("Accuracy is:", calc_accuracy(yTest, pred))
