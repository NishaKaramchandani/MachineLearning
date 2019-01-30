import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

print("Loading datasets...")
Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))
print("Done.")

train_joint_likelihood = np.zeros((10, 15))
test_joint_likelihood = np.zeros((10, 15))

distribution = []
for i in range(-7, 8):
    distribution.append(10 ** i)

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(Xs[i], ys[i], test_size=1. / 3, random_state=4435)

    for j in range(15):
        classifier = BernoulliNB(alpha=distribution[j])
        classifier.fit(X_train, y_train)

        train_Y_score = classifier._joint_log_likelihood(X_train)
        individual_joint_likelihood = 0.0
        for k in range(0, len(y_train)):
            if y_train[k] == True:
                individual_joint_likelihood += train_Y_score[k][1]
            else:
                individual_joint_likelihood += train_Y_score[k][0]

        train_joint_likelihood[i][j] = individual_joint_likelihood

        test_Y_score = classifier._joint_log_likelihood(X_test)
        individual_joint_likelihood = 0.0
        for k in range(0, len(y_test)):
            if y_test[k] == True:
                individual_joint_likelihood += test_Y_score[k][1]
            else:
                individual_joint_likelihood += test_Y_score[k][0]

        test_joint_likelihood[i][j] = individual_joint_likelihood

print("Joint Log likelihood - Train")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in train_joint_likelihood[i]))

print("\nJoint Log likelihood - Test")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in test_joint_likelihood[i]))

x = np.arange(0, 15)
for i in range(10):
    plt.plot(x, train_joint_likelihood[i], x, test_joint_likelihood[i])
    title = "graph " + str(i)
    plt.title(title)
    plt.show()

pickle.dump((train_joint_likelihood, test_joint_likelihood), open('results.pkl', 'wb'))
