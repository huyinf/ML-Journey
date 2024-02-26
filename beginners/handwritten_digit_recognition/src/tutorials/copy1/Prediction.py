"""
	perform forward propagation to predict the digit
"""

import numpy as np


def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def predict(Theta1, Theta2, X):
	m = X.shape[0]
	one_matrix = np.ones((m, 1))
	# add bias unit to first layer
	X = np.append(one_matrix, X, axis=1)
	z2 = np.dot(X, Theta1.transpose())
	# activation for second layer
	a2 = sigmoid(z2)
	# add bias unit to second layer
	a2 = np.append(one_matrix, a2, axis=1)
	z3 = np.dot(a2, Theta2.transpose())
	# activation for third layer
	a3 = sigmoid(z3)
	# predict the class on the basis of max value of hypothesis
	p = (np.argmax(a3, axis=1))

	return p
