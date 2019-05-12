import numpy as np
import tensorflow as tf
#from tensorflow import Conv2D
# import numpy.plot as plt


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#print(len(y_train[0]))


#z-score
#mean = np.mean(x_train,axis=(0,1,2,3))
#std = np.std(x_train,axis=(0,1,2,3))
#x_train = (x_train-mean)/(std+1e-7)
#x_test = (x_test-mean)/(std+1e-7)

#num_classes = 10
#y_train = np_utils.to_categorical(y_train,num_classes)
#y_test = np_utils.to_categorical(y_test,num_classes)

# # print(x_train)

# x_trainTensor = tf.convert_to_tensor(x_train)

# model = tf.keras.models.Sequential([
# 	tf.keras.layers.Flatten(input_shape(3, 32, 32)),

# 	])
# print(x_trainTensor)

# import tensorflow as tf
# mnist = tf.keras.datasets.mnist

# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

#model = tf.keras.models.Sequential([
 #  tf.keras.layers.Conv2D(filters = 32,kernel_size = 2, padding='same', activation ='relu',input_shape=(32,32,3),  #[1:]),

  #tf.keras.layers.Dense(512, activation=tf.nn.relu),
  #tf.keras.layers.Dropout(0.2),
  #tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  
  
#  tf.keras.layers.Flatten(input_shape=(32, 32, 3))
#])
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

#model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#model.add(tf.keras.layers.Dropout(0.2))
#model.add(Convolution2D(64, 3, 3, border_mode = 'same', input_shape = (32, 32, 3)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10,activation ='softmax'))

model.summary()

model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
			 

model.fit(x_train, y_train, epochs=25, validation_data = (x_test, y_test))
model.evaluate(x_test, y_test)