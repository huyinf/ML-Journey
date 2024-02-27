import warnings

warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


curr_dir = os.getcwd()
file_name = 'SMSSpamCollection'
file_path = os.path.join(curr_dir, '..', '..', 'data', file_name)

data = pd.read_csv(file_path, sep='\t', names=['label', 'message'])

# print(data.info())
# print(data.head())

data['label'] = data.label.map({'ham':0,'spam':1})
# print(data.head())

X_train,X_test,y_train,y_test = train_test_split(data['message'],data['label'],test_size=0.2,random_state=1)

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)


# frequency distribution
count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train).toarray()

testing_data = count_vector.transform(X_test).toarray()

frequency_matrix = pd.DataFrame(training_data,columns = count_vector.get_feature_names_out())

# print(frequency_matrix.head())

clf = LogisticRegression(random_state=0).fit(training_data,y_train)

predictions = clf.predict(testing_data)

# print(predictions)

print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))
print('\nConfusion Matrix :\n', confusion_matrix(y_test, predictions))