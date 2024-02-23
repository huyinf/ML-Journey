"""
	suppress the warning:
	do not plan to upgrade to pandas 3.0 and do not need pyarrow
"""
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# import librabies
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# load data
train_data = pd.read_csv("C:/Users/HuyTP/PycharmProjects/ML/beginners/disease_prediction/data/Training.csv")
test_data = pd.read_csv("C:/Users/HuyTP/PycharmProjects/ML/beginners/disease_prediction/data/Testing.csv")

# data cleaning

# drop the last columnn of train_data because of error reading method --> unnamed column
# train_data = train_data.drop(train_data.columns[-1],axis=1)

train_data = train_data.dropna(axis=1)
data = pd.DataFrame(train_data)

# print(len(train_data.keys()))
# print(train_data.keys()[-1])

# # check whether the dataset is balanced or not
# disease_counts = train_data["prognosis"].value_counts()
# tmp_df = pd.DataFrame({
# 	"Disease":disease_counts.index,
# 	"Counts":disease_counts.values,
# })
#
# # Plot
# plt.figure(figsize=(10,7))
# sns.barplot(x="Disease",y="Counts",data=tmp_df)
# plt.xticks(rotation=90)
# plt.show()

# label encoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# data splitting
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# print(f"Train: {X_train.shape}, {y_train.shape}")
# print(f"Test: {X_test.shape}, {y_test.shape}")


# model building

# K-Fold cross-validation

# define scoring metric for k-fold cross validation
def cv_scoring(estimator, x, y):
	return accuracy_score(y, estimator.predict(x))


# initialize models
models = {
	"SVC": SVC(),
	"Gaussian NB": GaussianNB(),
	"Random Forest": RandomForestClassifier(random_state=18),
}

# produce cross validation score for the models

for model_name in models:
	model = models[model_name]
	scores = cross_val_score(model, X, y, cv=10, n_jobs=-1, scoring=cv_scoring)

	print("==" * 30)
	print(model_name)
	print(f"Scores: {scores}")
	print(f"Mean score: {np.mean(scores)}")


# create a figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 10))

# Train and Test Support Vector Classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

# print(f"Accuracy on train data by SVM Classifier: {accuracy_score(y_train, svm_model.predict(X_train)) * 100}")
# print(f"Accuracy on test data by SVM Classifier: {accuracy_score(y_test, svm_preds) * 100}")

svm_cf_matrix = confusion_matrix(y_test, svm_preds)
# plt.figure(figsize=(10,6))
sns.heatmap(svm_cf_matrix, annot=True, ax=axs[0][0])
# plt.title("Confusion Matrix for SVM Classifier on Test Data")
# plt.show()

axs[0][0].set_title("Confusion Matrix \nfor SVM Classifier on Test Data")

# Guassian Navie Bayes Classifier

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

# print(f"Accuracy on train data by Gaussian Navie Bayes Classifier: {accuracy_score(y_train, nb_model.predict(X_train)) * 100}")
# print(f"Accuracy on test data by Gaussian Navie Bayer Classifier: {accuracy_score(y_test, nb_preds) * 100}")

nb_cf_matrix = confusion_matrix(y_test, nb_preds)
# plt.figure(figsize=(10,6))
sns.heatmap(nb_cf_matrix, annot=True, ax=axs[0][1])
# plt.title("Confusion Matrix for Gaussian Navie Bayes Classifier on Test Data")
# plt.show()

axs[0][1].set_title("Confusion Matrix \nfor Gaussian Navie Bayes Classifier on Test Data")

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# print(f"Accuracy on train data by Random Forest Classifier: {accuracy_score(y_train, rf_model.predict(X_train)) * 100}")
# print(f"Accuracy on test data by Random Forest Classifier: {accuracy_score(y_test, rf_preds) * 100}")

rf_cf_matrix = confusion_matrix(y_test, rf_preds)
# plt.figure(figsize=(10,6))
sns.heatmap(rf_cf_matrix, annot=True, ax=axs[1][0])
# plt.title("Confusion Matrix for Random Forest Classifier on Test Data")
# plt.show()

axs[1][0].set_title("Confusion Matrix \nfor Random Forest Classifier on Test Data")

# Fitting the model on whole data and validating on the Test dataset

# Train the models on whole data
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)

final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# clean and split test data
test_data = test_data.dropna(axis=1)

test_X = test_data.iloc[:, :-1]
test_y = encoder.transform(test_data.iloc[:, -1])

# make prediction by take mode of predictions made by all the classifiers
final_svm_preds = final_svm_model.predict(test_X)
final_nb_preds = final_nb_model.predict(test_X)
final_rf_preds = final_rf_model.predict(test_X)

# print(f"final_svm_preds: {final_svm_preds}")
# print(f"final_nb_preds: {final_nb_preds}")
# print(f"final_rf_preds: {final_rf_preds}")

# The mode function is used to find the most common value in a list
final_preds = [mode([i, j, k])[0] for i, j, k in zip(final_svm_preds, final_nb_preds, final_rf_preds)]

# print(f"final_preds: {final_preds}")


# print(f"Accuracy on Test dataset by the combined model: {accuracy_score(test_y, final_preds) * 100}")

cf_matrix = confusion_matrix(test_y, final_preds)
sns.heatmap(cf_matrix, annot=True, ax=axs[1][1])
axs[1][1].set_title("Confusion Matrix \nfor Combined Model on Test Dataset")

# adjust layout to prevent overlapping
plt.tight_layout()
plt.show()


# create function that can take symptoms as input and generate for disease

symptoms = X.columns.values

# print(symptoms)

symptom_index = {}

for index, value in enumerate(symptoms):
	symptom = " ".join([i.capitalize() for i in value.split("_")])
	symptom_index[symptom] = index

# print(symptom_index)


data_dict = {
	"symptom_index": symptom_index,
	"predictions_classes": encoder.classes_,
}


def predictDisease(symptoms):
	"""

	:param symptoms: string contains symptoms seperated by commas
	:return: Generated Predictions by models
	"""

	symptoms = symptoms.split(",")

	# create input data for the models
	input_data = [0] * len(data_dict["symptom_index"])

	for symptom in symptoms:
		index = data_dict["symptom_index"][symptom]
		input_data[index] = 1

	# reshape the input data and convert it into suitable format for model predictions
	input_data = np.array(input_data).reshape(1, -1)


	# # generate individual outpus
	# svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
	# nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
	# rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
	#
	# # mode function requires numeric input not string value --> can not use
	# final_prediction = mode([svm_prediction,nb_prediction,rf_prediction])
	# print(final_prediction)
	#
	# predictions = {
	# 	"svm_model_prediction":svm_prediction,
	# 	"nb_model_prediction":nb_prediction,
	# 	"rf_model_prediction":rf_prediction,
	# 	"final_prediction":final_prediction,
	# }

	svm_pred_class = final_svm_model.predict(input_data)[0]
	nb_pred_class = final_nb_model.predict(input_data)[0]
	rf_pred_class = final_rf_model.predict(input_data)[0]

	final_pred_class = mode([svm_pred_class, nb_pred_class, rf_pred_class])[0]

	predictions = {
		"svm_model_prediction": data_dict["predictions_classes"][svm_pred_class],
		"nb_model_prediction": data_dict["predictions_classes"][nb_pred_class],
		"rf_model_prediction": data_dict["predictions_classes"][rf_pred_class],
		"final_prediction": data_dict["predictions_classes"][final_pred_class],
	}

	return predictions


# test the function

print(predictDisease("Itching,Skin Rash,Nodal Skin Eruptions"))
