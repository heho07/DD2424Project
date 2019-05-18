import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
import csv

import errno
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32')/255  # .astype('float32')
x_test = x_test.astype('float32')/255

# checks all the possible epoch states we want to catch
def lr_schedule(epoch, base_learning_rate):
    lrate = base_learning_rate
    if epoch > 200:
        lrate = 0.1*base_learning_rate
    if epoch > 250:
        lrate = 0.01*base_learning_rate
    if epoch > 300:
        lrate = 0.001*base_learning_rate        
    return lrate

def check_directory_exists(full_path):
   if not os.path.exists(os.path.dirname(full_path)):
       try:
           os.makedirs(os.path.dirname(full_path))
       except OSError as exc: # Guard against race condition
           if exc.errno != errno.EEXIST:
               raise

def setUpModel(iteration_learning_rate):

    tf.keras.backend.clear_session()
    model = tf.keras.Sequential()
	
    model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', strides =(2,2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', strides =(2,2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =1, padding = "same", activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =10, kernel_size =1, padding = "same", activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10,activation ='softmax'))

    model.summary()

    # SGD optimizer
    # learning rate, one of  [0.25, 0.1, 0.05, 0.01], which one is the best?
    #sgd = tf.keras.optimizers.SGD(lr=iteration_learning_rate, momentum=0.9) # decay=0.1 on learning rate, but should only be appled to epochs [200,250, 300]

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def trainModel(model, iteration_learning_rate, number_of_epochs, folder_name, environment):
    # callback used to save the model during runtime

# for google cloud:
    if environment == "googleCloud":
        checkpoint_path = "./weights"+ ".ckpt"
# for herman PC:
    elif environment == "hermanPC":
        checkpoint_path = "../WeightsFromTraining/"+folder_name+"/learning_rate" + str(iteration_learning_rate) + ".ckpt"
        check_directory_exists(checkpoint_path) 
    else:
        checkpoint_path = "./weights"+ ".ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator( ##this is the start of the data augmentation 
		featurewise_center=False,
		featurewise_std_normalization=False,
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True)

    result = model.fit_generator(datagen.flow(x_train, y_train, batch_size = 32), epochs=number_of_epochs, validation_data = (x_test, y_test), callbacks = [tf.keras.callbacks.LearningRateScheduler(lambda epoch : lr_schedule(epoch, iteration_learning_rate)), cp_callback]) 



    # result = model.fit_generator(datagen.flow(x_train, y_train, batch_size=100),
    #             epochs=350, validation_data = (x_test,y_test), callbacks = [LearningRateScheduler(lr_schedule), cp_callback])

     # prints the result to csv file


# for google cloud: 
    if environment == "googleCloud":
        full_path = "./X"+ ".csv"
#for herman PC:
    elif environment == "hermanPC":
        full_path = "../ResultsFromTraining/"+folder_name+"/learning_rate" + str(iteration_learning_rate) + ".csv"
        check_directory_exists(full_path)
    else:
        full_path = "./X"+ ".csv"

    try:
        f= open(full_path,"w")
        loss = result.history["loss"]
        acc = result.history["acc"]
        val_loss = result.history["val_loss"]
        val_acc = result.history["val_acc"]
        f.write("iteration , Loss , acc , valLoss , valAcc\n")
        for i in range(len(loss)):
            row = str(i+1) + " ," + str(loss[i]) + " ,"  + str(acc[i]) + " ," + str(val_loss[i]) + " ," + str(val_acc[i]) + "\n"
            f.write(row)

        f.close()
    except:
        print("error occured")

    return result

def initializeTraining(iteration_learning_rate = 0.01, folder_name = "foo", epochs = 5, environment = "googleCloud"):
    print("starting with learning rate " + str(iteration_learning_rate))
    model = setUpModel(iteration_learning_rate)
    
    trainModel(model, iteration_learning_rate, epochs, folder_name, environment)
    full_path = ""

# for google cloud:
    if environment == "googleCloud":
        full_path ="./model"+".h5"
        
# for herman PC:
    elif environment == "hermanPC":
        full_path ="../Models/"+folder_name+"/learning_rate" + str(iteration_learning_rate) + ".h5"
        check_directory_exists(full_path) 
    else:
        full_path == "./model.h5"

    model.save(full_path)
    model.evaluate(x_test, y_test)



initializeTraining(0.01, "all-cnn-c-dataaugment90noclr", 350, "hermanPC")

# for loading the model see:
# loading_checkpoint_path = "../Models/all-cnn-c-dataaugment90noclr/learning_rate0.01.h5"
# model = tf.keras.models.load_model(loading_checkpoint_path)
# model.evaluate(x_test, y_test)



