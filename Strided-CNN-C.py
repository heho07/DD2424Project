import numpy as np
import tensorflow as tf


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


model = tf.keras.Sequential()
#table 2
model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', strides =(2,2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', strides =(2,2)))
model.add(tf.keras.layers.BatchNormalization())

#table 1
model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =1, padding = "same", activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters =10, kernel_size =1, padding = "same", activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())

# flatten to be able to use SoftMax
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10,activation ='softmax'))

model.summary()

model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
			 

model.fit(x_train, y_train, epochs=25, validation_data = (x_test, y_test))
model.evaluate(x_test, y_test)