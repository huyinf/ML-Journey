"""
	suppress the warning:
	do not plan to upgrade to pandas 3.0 and do not need pyarrow
"""
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# import dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv("C:/Users/HuyTP/PycharmProjects/ML/beginners/linear_regression/data/advertising.csv")
# print(data.info())
# # print(data.info())
# #
# # # check null value
# # print(data.isnull().sum())

# visualize the relationship between the amount spent on advertising on [TV,Newspaper,Radio] and units sold
def visualize(data):

	cols = list(data.drop(["Sales"],axis=1))
	for col in cols:
		figure = px.scatter(data_frame=data, x="Sales", y=col, size=col, trendline="ols")
		figure.show()


# # the correlation of all the columns with the sales column
# correlation = data.corr()
# print(correlation["TV"].sort_values(ascending=False))


# data preparation
x = np.array(data.drop(["Sales"], axis=1))
y = np.array(data["Sales"])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
# prediction
# features = [[TV,Radio,Newspaper]]
features = np.array([[230.4, 37.9, 69.2]])


# # using libraries
model = LinearRegression()
model.fit(xtrain,ytrain)
print(model.predict(features))
