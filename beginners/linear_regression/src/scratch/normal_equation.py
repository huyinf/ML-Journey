"""
	suppress the warning:
	do not plan to upgrade to pandas 3.0 and do not need pyarrow
"""
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd

data = pd.read_csv("C:/Users/HuyTP/PycharmProjects/ML/beginners/linear_regression/data/advertising.csv")

X = data[["TV", "Newspaper", "Radio"]].values
y = data["Sales"].values

X_transpose = np.transpose(X)
X_transpose_dot_X = np.dot(X_transpose, X)
inverse_X_transpose_dot_X = np.linalg.inv(X_transpose_dot_X)
X_transpose_dot_y = np.dot(X_transpose, y)
beta = np.dot(inverse_X_transpose_dot_X, X_transpose_dot_y)

new_data = np.array([[230.4, 37.9, 69.2]])
prediction = np.dot(new_data, beta)
print("Predicted Sales:", prediction)
