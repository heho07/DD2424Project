import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
import csv

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32')/255  # .astype('float32')
x_test = x_test.astype('float32')/255

# checks all the possible epoch states we want to catch
def lr_schedule(epoch):
    lrate = 0.01
    if epoch > 200:
        lrate = 0.001
    if epoch > 250:
        lrate = 0.0001
    if epoch > 300:
        lrate = 0.00001        
    return lrate

def setUpModel():

    model = tf.keras.Sequential()

    # apply weight_decay to all the layers.
    weight_decay = 0.001
    l2_reg = weight_decay / 2 # according to https://bbabenko.github.io/weight-decay/

    # apply 0.2 dropout layer to the input image
    model.add(tf.keras.layers.Dropout(0.2, input_shape=x_train.shape[1:]))
    # Two 3 x 3 conv.96 ReLU
    model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))
    model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))

    # "max-pooling ish" with dropout 0.5 after
    model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', strides =(2,2), kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))
    model.add(tf.keras.layers.Dropout(0.5))

    # Two 3 x 3 conv.96 ReLU
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))

    # "max-pooling" with dropout 0.5 after
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', strides =(2,2), kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))
    model.add(tf.keras.layers.Dropout(0.5))

    # One 3 x 3 conv. 192 ReLU
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))
    # One 1 x 1 conv. 192 ReLU
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =1, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))
    # One 1 x 1 conv. 10 ReLU
    model.add(tf.keras.layers.Conv2D(filters =10, kernel_size =1, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))

    # global averaging over 6 ? 6 spatial dimensions
    model.add(tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last', input_shape=(6, 6, 10)))
    # Full connected layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10,activation ='softmax', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))

    model.summary()

    # SGD optimizer
    # learning rate, one of  [0.25, 0.1, 0.05, 0.01], which one is the best?
    sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9) # decay=0.1 on learning rate, but should only be appled to epochs [200,250, 300]

    model.compile(optimizer=sgd,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def trainModel(model):
    # callback used to save the model during runtime
    checkpoint_path = "../WeightsFromTraining/replicatingStudy/replicatingStudy.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # datagen = tf.keras.preprocessing.image.ImageDataGenerator( ##this is the start of the data augmentation
    #         featurewise_center=False,
    #         featurewise_std_normalization=False,
    #         rotation_range=20,
    #         width_shift_range=0.2,
    #         height_shift_range=0.2,
    #         horizontal_flip=True)

    result = model.fit(x_train, y_train, epochs=350, batch_size = 100, validation_data = (x_test, y_test), callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_schedule), cp_callback]) 


    # result = model.fit_generator(datagen.flow(x_train, y_train, batch_size=100),
    #             epochs=350, validation_data = (x_test,y_test), callbacks = [LearningRateScheduler(lr_schedule), cp_callback])

     # prints the result to csv file
    f= open("../X.csv","w")
    loss = result.history["loss"]
    acc = result.history["acc"]
    val_loss = result.history["val_loss"]
    val_acc = result.history["val_acc"]
    f.write("iteration , Loss , acc , valLoss , valAcc\n")
    for i in range(len(loss)):
        row = str(i+1) + " ," + str(loss[i]) + " ,"  + str(acc[i]) + " ," + str(val_loss[i]) + " ," + str(val_acc[i]) + "\n"
        f.write(row)

    f.close()

    return result


# loading_checkpoint_path = "../WeightsFromTraining/all-cnn-c-dataaugment-dropout-400epochs-startingFrom90percent.ckpt"

model = setUpModel()
trainModel(model)

model.save("../Models/replicatingStudy.h5")
model.evaluate(x_test, y_test)
# model.load_weights(loading_checkpoint_path)
# model.evaluate(x_test, y_test)
# result = trainModel()
