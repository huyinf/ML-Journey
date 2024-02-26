import os
from scipy.io import loadmat
from scipy.optimize import minimize
import numpy as np
from Model import neural_network
from Prediction import predict
from RandInitialize import initialize

# directory dependency & load data
data_file = 'mnist-original.mat'

# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the data file
# Go up two levels to reach the root directory, then navigate to the data directory
data_file_path = os.path.join(current_script_dir, '../..', '..', 'data', data_file)
# print(data_file_path)
data = loadmat(data_file_path)
# print(data)


# extract features from mat file
X = data['data']
X = X.transpose()
# print(X.shape)

# Normalize the data
X = X / 255
# print(X)

# Extract labels from mat file
y = data['label']
# print(y)
y = y.flatten()
# print(y)


# Split data into training set with 60,000 examples
train_size = 60000
X_train = X[:train_size, :]
y_train = y[:train_size]
# print(X_train.shape)
# print(y_train.shape)

# split data into testing set with 10,000 examples := test_size = X.shape[1] - train_size
X_test = X[train_size:, :]
y_test = y[train_size:]
# print(X_test.shape)
# print(y_test.shape)

# initialize layers' size
m = X.shape[0]
input_layer_size = 784  # Images are of (28 x 28) px so there will be 784 features
hidden_layer_size = 100
num_labels = 10  # There are 10 classes [0, 9]

# randomly initialize thetas
init_Theta1 = initialize(hidden_layer_size, input_layer_size)
init_Theta2 = initialize(num_labels, hidden_layer_size)

# Unrolling parameters into a single column vector
init_nn_params = np.concatenate((init_Theta1.flatten(), init_Theta2.flatten()))
maxiter = 100
# avoid overfitting
lambda_reg = 0.1
myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg)

# call minimize function to minimize cost function and to train weights
results = minimize(neural_network, x0=init_nn_params, args=myargs, options={'disp': True, 'maxiter': maxiter},
				   method="L-BFGS-B", jac=True)

# trained Theta is extracted
nn_params = results["x"]

# weights are splited back to Theta1, Theta2
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
					(hidden_layer_size, input_layer_size + 1))  # shape = (100, 785)
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
					(num_labels, hidden_layer_size + 1))  # shape = (10, 101)

# check test set accuracy of our model
test_pred = predict(Theta1, Theta2, X_test)
print('Test Set Accuracy: {:f}'.format((np.mean(test_pred == y_test) * 100)))

# check train set accuracy of our model
train_pred = predict(Theta1, Theta2, X_train)
print('Train Set Accuracy: {:f}'.format((np.mean(train_pred == y_train) * 100)))

# Evaluate precision of our model
true_positive = 0
for i in range(len(train_pred)):
	if train_pred[i] == y_train[i]:
		true_positive += 1
false_positive = len(y_train) - true_positive

print('Precision = ', true_positive / (true_positive + false_positive))

# save Theta in .txt file
np.savetxt('Theta1.txt', Theta1, delimiter=' ')
np.savetxt('Theta2.txt', Theta2, delimiter=' ')
