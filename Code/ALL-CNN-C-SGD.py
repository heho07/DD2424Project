import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
import csv

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', strides =(2,2)))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', strides =(2,2)))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =1, padding = "same", activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters =10, kernel_size =1, padding = "same", activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())


model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10,activation ='softmax'))

model.summary()

sgd = optimizers.SGD(lr=0.001)
model.compile(optimizer=sgd,
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


result = model.fit(x_train, y_train, epochs=1 , validation_data = (x_test, y_test))
model.evaluate(x_test, y_test)

# prints the result to csv file
f= open("guru100_sgd.csv","w")
loss = result.history["loss"]
acc = result.history["acc"]
val_loss = result.history["val_loss"]
val_acc = result.history["val_acc"]
f.write("iteration , Loss , acc , valLoss , valAcc\n")
for i in range(len(loss)):
        row = str(i) + " ," + str(loss[i]) + " ,"  + str(acc[i]) + " ," + str(val_loss[i]) + " ," + str(val_acc[i]) + "\n"
        f.write(row)

f.close()
