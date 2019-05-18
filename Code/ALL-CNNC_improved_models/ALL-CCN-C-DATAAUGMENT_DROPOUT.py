import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
#from tensorflow import Conv2D
# import numpy.plot as plt
import csv

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

learning_rate_history = []
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
        
    # on each epoch end we save the current learning rate
    # this way we can later review which order of magnitude the learning rate had
    def on_epoch_end(self, epoch, logs=None):
        learning_rate_history.append(self.clr())
    

    # def on_epoch_end(self, epoch, logs=None):
    #     print("current lr" + str(self.clr()))
    #     print("base lr" + str(self.base_lr))
    
# ------------------------------------- end of code from https://github.com/bckenstler/CLR


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# trying to use this one now
x_train = x_train.astype('float32')/255  # .astype('float32')
x_test = x_test.astype('float32')/255

# mean = np.mean(x_train,axis=(0,1,2,3))  #(0,1,2,3)
# std = np.std(x_train,axis=(0,1,2,3))
# x_train = (x_train-mean)/(std+1e-7)
# x_test = (x_test-mean)/(std+1e-7) 


datagen = tf.keras.preprocessing.image.ImageDataGenerator( ##this is the start of the data augmentation 
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# using low L2 regularizer of 0.00001, 0.001 last time proved too high.

def setUpModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', input_shape=x_train.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(0.00001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', strides =(2,2), kernel_regularizer=tf.keras.regularizers.l2(0.00001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', strides =(2,2), kernel_regularizer=tf.keras.regularizers.l2(0.00001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =1, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =10, kernel_size =1, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10,activation ='softmax'))

    model.summary()

    model.compile(optimizer='Adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model			 


def trainModel():
    # one update step is 1 batch - 1 cycle is step_size * 2 no. of update steps
    # callback used for cyclical learning

    # if step_size set to 2500 -> 10 epochs per cycle. 350 total epochs -> 35 cycles, each with decreasing LR (1/2^cycle)
    # want to try fewer cycles to avoid getting too small LR.
    # 35 cycles lead to a factor of ~1e-11 multiplied to the LR.
    # so i try 20 cycles, for the final LR to be multiplied by factor of 1e-7  
    clr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=4375, mode='triangular2')
    
    # callback used to save the model during runtime
    checkpoint_path = "../WeightsFromTraining/wednesdaynight/all-ccn-c-datagument-dropout.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)
#result = model.fit(x_train, y_train, epochs=25 , validation_data = (x_test, y_test), callbacks = [clr]) 

    # runs the training procedure
    result = model.fit_generator(datagen.flow(x_train, y_train, batch_size=100),
                epochs=350, validation_data = (x_test,y_test), callbacks = [clr, cp_callback])

    # prints the result to csv file
    f= open("../X.csv","w")
    loss = result.history["loss"]
    acc = result.history["acc"]
    val_loss = result.history["val_loss"]
    val_acc = result.history["val_acc"]
    f.write("iteration , Loss , acc , valLoss , valAcc, learning_rate\n")
    for i in range(len(loss)):
    	row = str(i+1) + " ," + str(loss[i]) + " ,"  + str(acc[i]) + " ," + str(val_loss[i]) + " ," + str(val_acc[i]) + " ," + str(learning_rate_history[i]) +"\n"
    	f.write(row)

    f.close()

    return result

# loading_checkpoint_path = "../WeightsFromTraining/wednesdaynight/all-ccn-c-datagument-dropout.ckpt"

#used to load the model
# model = tf.keras.models.load_model("../Models/test.h5")
# model.summary()

model = setUpModel()

model.evaluate(x_test, y_test)

# model.load_weights(loading_checkpoint_path)
# model.evaluate(x_test, y_test)

result = trainModel()
# print(learning_rate_history)
model.save("../Models/test.h5")

model.evaluate(x_test, y_test) ##this one can be commented out to reduce read time. 

"""
trying to run this with
ADAM
cyclical learning rate triangular with 20 epochs -> final learning rates will be a factor of 1/2^20 (i.e. 1e-11) of the original LR
350 epochs

regularization methods:
3 x dropout 0.2
data augmentation dynamically
pre-training data normalization
LOW L2 regularization on the convnet layers to try and avoid overfitting


model saved to: "../Models/test.h5"
weights saved to: "../WeightsFromTraining/wednesdaynight/all-ccn-c-datagument-dropout.ckpt"
accuracy results save to : "../X.csv" along with information about the learning rate order of magnitude on each epochs end
"""