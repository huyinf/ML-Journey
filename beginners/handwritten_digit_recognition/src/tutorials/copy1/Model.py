"""
	perform feed-forward and backpropagation
		feed-forward:
			activation function: sigmoid function
"""
import numpy as np


def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda):
	"""
	:param nn_params: a flattened array containing all the weights and biases for the neural nerwork.
						These are initially set by the function "initialize" and are updated during the training process.
	:param input_layer_size: the number of input features
	:param hidden_layer_size: the number of neurons in the hidden layer
	:param num_labels: the number of output classes
	:param X: the input features matrix
	:param y: the vector of labels
	:param _lambda: the regularization parameter
	:return:
	"""
	# weights are split back to Theta1, Theta2
	Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
						(hidden_layer_size, input_layer_size + 1))
	Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))

	# Forward propagation
	m = X.shape[0]
	one_matrix = np.ones((m, 1))
	# Add bias unit to first layer
	X = np.append(one_matrix, X, axis=1)
	a1 = X
	z2 = np.dot(X, Theta1.transpose())
	# Activattion for second layer
	a2 = sigmoid(z2)
	# Add bias unit to hidden layer
	a2 = np.append(one_matrix, a2, axis=1)
	z3 = np.dot(a2, Theta2.transpose())
	# Activation for third layer
	a3 = sigmoid(z3)

	# Change the y labels into vectors of boolean values
	# For each layer between 0 and 9, there will be a vector of length 10
	# where the ith element will be 1 if the label equals i
	y_vect = np.zeros((m, 10))
	for i in range(m):
		y_vect[i, int(y[i])] = 1

	# Calculate cost function
	J = (1 / m) * (np.sum(np.sum(-y_vect * np.log(a3) - (1 - y_vect) * np.log(1 - a3)))) + (_lambda / (2 * m)) * (
			sum(sum(pow(Theta1[:,1:], 2))) + sum(sum(pow(Theta2[:, 1:], 2))))

	# backpropagation
	Delta3 = a3 - y_vect
	Delta2 = np.dot(Delta3, Theta2) * a2 * (1 - a2)
	Delta2 = Delta2[:, 1:]

	# gradient
	Theta1[:, 0] = 0
	Theta1_grad = (1 / m) * np.dot(Delta2.transpose(), a1) + (_lambda / m) * Theta1
	Theta2[:, 0] = 0
	Theta2_grad = (1 / m) * np.dot(Delta3.transpose(), a2) + (_lambda / m) * Theta2
	grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))

	return J, grad
