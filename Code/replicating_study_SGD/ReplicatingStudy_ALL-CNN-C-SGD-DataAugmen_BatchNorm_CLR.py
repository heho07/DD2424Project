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

# from https://github.com/bckenstler/CLR
class CyclicLR(tf.keras.callbacks.Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}
        self.learning_rate_decay = 1

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 200 or epoch == 250 or epoch == 300:
            self.learning_rate_decay*= 0.1

        print("clr: " + str(self.clr()))
        print("learning rate decay factor: " + str(self.learning_rate_decay))

def check_directory_exists(full_path):
   if not os.path.exists(os.path.dirname(full_path)):
       try:
           os.makedirs(os.path.dirname(full_path))
       except OSError as exc: # Guard against race condition
           if exc.errno != errno.EEXIST:
               raise

def setUpModel(iteration_learning_rate):

    tf.keras.backend.clear_session()
	
    # tf.keras.backend.clear_session()
    model = tf.keras.Sequential()

    # apply weight_decay to all the layers.
    weight_decay = 0.001
    l2_reg = weight_decay / 2 # according to https://bbabenko.github.io/weight-decay/

    # apply 0.2 dropout layer to the input image
    model.add(tf.keras.layers.Dropout(0.2, input_shape=x_train.shape[1:]))
    # Two 3 x 3 conv.96 ReLU
    model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))
    model.add(tf.keras.layers.BatchNormalization())
    # "max-pooling ish" with dropout 0.5 after
    model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', strides =(2,2), kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))
    model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.5))

    # Two 3 x 3 conv.96 ReLU
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))
    model.add(tf.keras.layers.BatchNormalization())
    # "max-pooling" with dropout 0.5 after
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.5))

    # One 3 x 3 conv. 192 ReLU
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))
    model.add(tf.keras.layers.BatchNormalization())
    # One 1 x 1 conv. 192 ReLU
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =1, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))
    model.add(tf.keras.layers.BatchNormalization())
    # One 1 x 1 conv. 10 ReLU
    model.add(tf.keras.layers.Conv2D(filters =10, kernel_size =1, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))
    model.add(tf.keras.layers.BatchNormalization())
    
    # global averaging over 6 ? 6 spatial dimensions
    model.add(tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last', input_shape=(6, 6, 10)))
    # Full connected layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10,activation ='softmax', kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg)))


    model.summary()

    # SGD optimizer
    # learning rate, one of  [0.25, 0.1, 0.05, 0.01], which one is the best?
    sgd = tf.keras.optimizers.SGD(lr=iteration_learning_rate, momentum=0.9) # decay=0.1 on learning rate, but should only be appled to epochs [200,250, 300]

    model.compile(optimizer=sgd,
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
        checkpoint_path = "../../WeightsFromTraining/"+folder_name+"/learning_rate" + str(iteration_learning_rate) + ".ckpt"
        check_directory_exists(checkpoint_path) 
    else:
        checkpoint_path = "./weights"+ ".ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    clr = CyclicLR(base_lr=0.01, max_lr=0.06, step_size=7815, mode='triangular2')
													 
# for google cloud:
    if environment == "googleCloud":
        model_checkpoint_path = "./model"+ ".h5"
# for herman PC:
    elif environment == "hermanPC":
        model_checkpoint_path = "../../Models/"+folder_name+"/modelCheckpoint_learning_rate" + str(iteration_learning_rate) + ".h5"
        check_directory_exists(model_checkpoint_path) 
    else:
        model_checkpoint_path = "./model"+ ".h5"

    cp_callback_model = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator( ##this is the start of the data augmentation 
		featurewise_center=False,
		featurewise_std_normalization=False,
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True)

    result = model.fit_generator(datagen.flow(x_train, y_train, batch_size = 32), epochs=number_of_epochs, validation_data = (x_test, y_test), callbacks = [clr, cp_callback, cp_callback_model]) 

   # result = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
   #             epochs=350, validation_data = (x_test,y_test), callbacks = [clr, cp_callback,cp_callback_model])

    # result = model.fit(x_train, y_train, epochs=number_of_epochs, batch_size = 32, validation_data = (x_test, y_test), callbacks = [cp_callback,cp_callback_model]) 
	
	# result = model.fit_generator(datagen.flow(x_train, y_train, batch_size=100),
    #             epochs=350, validation_data = (x_test,y_test), callbacks = [LearningRateScheduler(lr_schedule), cp_callback])

     # prints the result to csv file


# for google cloud: 
    if environment == "googleCloud":
        full_path = "./X"+ ".csv"
#for herman PC:
    elif environment == "hermanPC":
        full_path = "../../ResultsFromTraining/"+folder_name+"/learning_rate" + str(iteration_learning_rate) + ".csv"
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

# takes in inital learning rate, what folder to save results in, how many epochs to run for
# as well as what environment it will be run on - either googleCloud or hermanPC
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
        full_path ="../../Models/"+folder_name+"/learning_rate" + str(iteration_learning_rate) + ".h5"
        check_directory_exists(full_path) 
    else:
        full_path == "./model.h5"

    model.save(full_path)
    model.evaluate(x_test, y_test)



initializeTraining(0.01,"replicating_study_SGD_dataaugment_batchnorm_clr", 350, "hermanPC")

# for loading the model see:
# loading_checkpoint_path = "../Models/all-cnn-c-dataaugment90noclr/learning_rate0.01.h5"
# model = tf.keras.models.load_model(loading_checkpoint_path)
# model.evaluate(x_test, y_test)



