"""
	suppress the warning:
	do not plan to upgrade to pandas 3.0 and do not need pyarrow
"""
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# import dependencies
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split


class _LinearRegression():

	def __init__(self, w_in, b_in, learning_rate, iterations):
		self.w = w_in
		self.b = b_in
		self.lr = learning_rate
		self.num_iters = iterations

		self.J_hist = []
		self.w_hist = [w_in]
		self.b_hist = [b_in]

	def compute_cost(self, x, y):
		# the number of samples
		m = x.shape[0]

		cost = 0.

		for i in range(m):
			# wx = x[i]*self.w
			# wx_dot = np.dot(x[i],self.w)
			# print(f"wx = {wx}")
			# print(f"wx_dot = {wx_dot}")
			# print(type(wx_dot))
			# print(type(self.b))
			# print(type(y[i]))
			cost += (np.dot(x[i], self.w) + self.b - y[i]) ** 2
		# print(f"cost = {cost}")
		cost /= 2 * m

		return cost

	def compute_gradient(self, x, y):

		m = x.shape[0]

		dj_dw = np.zeros_like(self.w)
		dj_db = 0

		for i in range(m):
			dj_dw += (np.dot(x[i], self.w) + self.b - y[i]) * x[i]
			dj_db += np.dot(x[i], self.w) + self.b - y[i]

		dj_dw /= m
		dj_db /= m

		return dj_dw, dj_db

	def fit(self, x, y):

		m = x.shape
		for i in range(self.num_iters):
			dj_dw, dj_db = self.compute_gradient(x, y)

			self.w -= self.lr * dj_dw
			self.b -= self.lr * dj_db

			if i < 10000:
				cost = self.compute_cost(x, y)
				self.J_hist.append(cost)
			# print(cost)

			# Print cost every at intervals 10 times or as many iterations if < 10
			if i % math.ceil(self.num_iters / 10) == 0:
				self.w_hist.append(self.w)
				print(f"Iteration {i:4}: Cost {float(self.J_hist[-1]):8.2f}   ")

	def predict(self, x):
		return np.dot(x, self.w) + self.b


data = pd.read_csv("C:/Users/HuyTP/PycharmProjects/ML/beginners/linear_regression/data/advertising.csv")
# print(data.info())

# data preparation
x = np.array(data.drop(["Sales"], axis=1))
y = np.array(data["Sales"])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
# prediction
# features = [[TV,Radio,Newspaper]]
features = np.array([[230.4, 37.9, 69.2]])

# initialize parameters
w = np.zeros(x.shape[1])
b = 0.
alpha = 0.01
iterations = 1000

model = _LinearRegression(w, b, alpha, iterations)
model.fit(xtrain, ytrain)
print(model.predict(features))
