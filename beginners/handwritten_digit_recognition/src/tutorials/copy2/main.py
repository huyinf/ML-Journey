import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
import random

# directory dependency
curr_dir = os.getcwd()
model_name = "MNIST_MODEL.h5"
model_path = os.path.join(curr_dir,'..',model_name)

objects = tf.keras.datasets.mnist
(training_images,training_labels),(testing_images,testing_labels) = objects.load_data()


training_images = training_images.reshape(-1,28*28)/255.0
testing_images = testing_images.reshape(-1,28*28)/255.0

# split training set and validation set
validation_size = 10000
validation_images = training_images[-validation_size:]
validation_labels = training_labels[-validation_size:]
training_images = training_images[:-validation_size]
training_labels = training_labels[:-validation_size]
#
# print(validation_images.shape)
# print(training_images.shape)
def create_model():
	model = tf.keras.models.Sequential([tf.keras.layers.Dense(128,activation='relu',input_shape=(784,)),
									   tf.keras.layers.Dense(64,activation='relu'),
										tf.keras.layers.Dense(10,activation=tf.nn.softmax)])

	model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

	return model

model  = create_model()

model.summary()

history = model.fit(training_images,training_labels,batch_size=128,epochs=32,validation_data=(validation_images,validation_labels))

print(model.evaluate(testing_images,testing_labels))

prediction = model.predict(testing_images)
prediction_labels = [np.argmax(i) for i in prediction]

num = 5
lower,upper = 0,testing_images.shape[1]
test = [random.randint(lower,upper) for _ in range(num)]
for i in test:
	plt.imshow(testing_images[i].reshape(28,28))
	plt.savefig(f"{testing_labels[i]}.png")
# print(prediction_lalels[n])
#
cm = tf.math.confusion_matrix(labels=testing_labels,predictions=prediction_labels)
print(cm)

plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel("Predictions")
plt.ylabel("Truth")
plt.savefig("confusion_matrix.png")

report = classification_report(testing_labels,prediction_labels)

with open("report.txt","w") as f:
	f.write(report)

model.save(model_path)
