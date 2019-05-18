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
    print("in lr scheduler with lr as : " + str(lrate))
    return lrate

# def check_directory_exists(full_path):
   # if not os.path.exists(os.path.dirname(full_path)):
       # try:
           # os.makedirs(os.path.dirname(full_path))
       # except OSError as exc: # Guard against race condition
           # if exc.errno != errno.EEXIST:
               # raise

def setUpModel(iteration_learning_rate):

    # tf.keras.backend.clear_session()
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
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.5))

    # Two 3 x 3 conv.96 ReLU
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))

    # "max-pooling" with dropout 0.5 after
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
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
    print("Declaring SGD with learning rate: " + str(iteration_learning_rate))
    sgd = tf.keras.optimizers.SGD(lr=iteration_learning_rate, momentum=0.9) # decay=0.1 on learning rate, but should only be appled to epochs [200,250, 300]

    model.compile(optimizer=sgd,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def trainModel(model, iteration_learning_rate, number_of_epochs, folder_name = "foo"):
    # callback used to save the model during runtime
    checkpoint_path = "./"+ ".ckpt"
   # check_directory_exists(checkpoint_path) 
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

    print("Starting to fit the model with epochs: " + str(number_of_epochs))
    result = model.fit(x_train, y_train, epochs=number_of_epochs, batch_size = 100, validation_data = (x_test, y_test), callbacks = [tf.keras.callbacks.LearningRateScheduler(lambda epoch : lr_schedule(epoch, iteration_learning_rate)), cp_callback]) 



    # result = model.fit_generator(datagen.flow(x_train, y_train, batch_size=100),
    #             epochs=350, validation_data = (x_test,y_test), callbacks = [LearningRateScheduler(lr_schedule), cp_callback])

     # prints the result to csv file


    full_path = "X"+".csv"
    #check_directory_exists(full_path)



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

    return result

def initializeTraining(iteration_learning_rate = 0.01, folder_name = "foo", epochs = 5):
    print("starting with learning rate " + str(iteration_learning_rate))
    model = setUpModel(iteration_learning_rate)
    
    trainModel(model, iteration_learning_rate, epochs, folder_name)

    full_path ="./"+".h5"
   # check_directory_exists(full_path) 
    model.save(full_path)
    model.evaluate(x_test, y_test)

learning_rates = [0.25, 0.1, 0.05, 0.01]

# for current_iteration_learning_rate in learning_rates:
#     initializeTraining(current_iteration_learning_rate, "replicating_study", 350)

<<<<<<< HEAD
initializeTraining(0.25, "replicating_study", 5)
=======
initializeTraining(0.01, "replicating_study", 350)
>>>>>>> 9eaf1c8f636cd5c5bab8acd53c7aa379404c01f7

# model.evaluate(x_test, y_test)
# model = tf.keras.models.load_model('../Models/replicating_study/learning_rate0.01.h5')
# model.summary()
# model.evaluate(x_test, y_test)


# loading_checkpoint_path = "../WeightsFromTraining/all-cnn-c-dataaugment-dropout-400epochs-startingFrom90percent.ckpt"

# model.load_weights(loading_checkpoint_path)
# result = trainModel()
