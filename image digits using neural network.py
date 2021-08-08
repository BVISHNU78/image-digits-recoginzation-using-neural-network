# image-digits-recoginzation-using-neural-network
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
len(x_train)
len(x_test)
x_train[0].shape
x_train[0]
plt.matshow(x_train[2])
y_train[2]
x_train.shape
x_train=x_train/255
x_test=x_test/255
x_train_flattend=x_train.reshape(len(x_train),28*28)
x_train_flattend
x_test_flattend=x_test.reshape(len(x_test),28*28)
x_test_flattend
model=keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')   
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
) 
model.fit(x_train_flattend,y_train,epochs=5)
model.evaluate(x_test_flattend,y_test)
x_test[0]
plt.matshow(x_test[0])
model.predict(x_test_flattend)
y_predicted=model(x_test_flattend)
y_predicted[0]
y_predicted_labels=[np.argmax(j)for j in y_predicted]
y_predicted_labels[:5]
y_predicted=model(x_test_flattend)
y_predicted_labels=[np.argmax(j)for j in y_predicted]
cm=tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('Truth')
