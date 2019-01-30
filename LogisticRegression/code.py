import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import pylab

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))

posTrue = 1
posFalse = 0

mc_l2 = np.zeros((10, 15));
train_conditional_log_likelihood = np.zeros((10, 15))
test_conditional_log_likelihood = np.zeros((10, 15))
zero_weights_l2 = np.zeros((10, 15));
zero_weights_l1 = np.zeros((10, 15));

depths = range(15)
c_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]


def model_complexity(w0, Ws):
    complexity = np.square(w0) + np.sum(np.square(Ws))
    return complexity


for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(Xs[i], ys[i], test_size=1. / 3, random_state=4435)
    for d in depths:
        clf = LogisticRegression(penalty='l2', C=c_values[d], random_state=42)
        clf.fit(X_train, y_train)
        w0 = clf.intercept_
        Ws = clf.coef_
        mc_l2[i][d] = model_complexity(w0, Ws)

        zero_weights_l2[i][d] = len(Ws[0]) - np.count_nonzero(Ws)
        zero_weights_l2[i][d] += len(w0) - np.count_nonzero(w0)

        log_prob_y_x_train = clf.predict_log_proba(X_train)
        total_cll_train = 0.0
        for l in range(len(y_train)):
            if (y_train[l] == False):
                total_cll_train += log_prob_y_x_train[l][posFalse]
            else:
                total_cll_train += log_prob_y_x_train[l][posTrue]
        train_conditional_log_likelihood[i][d] = total_cll_train

        log_prob_y_x_test = clf.predict_log_proba(X_test)
        total_cll_test = 0.0
        for l in range(len(y_test)):
            if (y_test[l] == False):
                total_cll_test += log_prob_y_x_test[l][posFalse]
            else:
                total_cll_test += log_prob_y_x_test[l][posTrue]
        test_conditional_log_likelihood[i][d] = total_cll_test

        clf = LogisticRegression(penalty='l1', C=c_values[d], random_state=42)
        clf.fit(X_train, y_train)
        w0 = clf.intercept_
        Ws = clf.coef_

        zero_weights_l1[i][d] = len(Ws[0]) - np.count_nonzero(Ws)
        zero_weights_l1[i][d] += len(w0) - np.count_nonzero(w0)

print("Model Complexity for l2")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in mc_l2[i]))

print("\nConditional log likelihood l2 - Train")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in train_conditional_log_likelihood[i]))

print("\nConditional log likelihood l2 - Test")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in test_conditional_log_likelihood[i]))

print("\nnZero Weights in l1")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in zero_weights_l1[i]))

print("\nZero Weights in l2")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in zero_weights_l2[i]))

pickle.dump(
    (mc_l2, train_conditional_log_likelihood, test_conditional_log_likelihood, zero_weights_l2, zero_weights_l1),
    open('results.pkl', 'wb'))

c_exp_values = [-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7]
for i in range(10):
    plt.title("Dataset "+str(i))
    plt.xlabel("Model Complexity")
    plt.ylabel("Conditional Log Likelihood")
    _,ax = plt.subplots()
    ax.plot(mc_l2[i], train_conditional_log_likelihood[i], label='Train CLL')
    ax.plot(mc_l2[i], test_conditional_log_likelihood[i], label='Test CLL')
    ax.legend()

for i in range(10):
    plt.title("Dataset "+str(i))
    plt.xlabel("C")
    plt.ylabel("Zero Weights")
    _,ax = plt.subplots()
    ax.plot(c_exp_values, zero_weights_l2[i], label='L2')
    ax.plot(c_exp_values, zero_weights_l1[i], label='L1')
    ax.legend()