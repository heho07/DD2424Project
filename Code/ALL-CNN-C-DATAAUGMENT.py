import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
#from tensorflow import Conv2D
# import numpy.plot as plt
import csv

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# from https://github.com/bckenstler/CLR
#print(len(y_train[0]))
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

# ------------------------------------- end of code from https://github.com/bckenstler/CLR


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#print("getShape of x_train")
#print(tf.Tensor.get_shape(x_train))
#print("starting to merge ytrain")

#x_train = x_test
#y_train = y_test

#print(tf.Tensor.get_shape(y_train))
#y_train = np.concatenate((y_train, y_train), axis = 0)
#print(tf.Tensor.get_shape(y_train))
#print("ytrain merged")

#x_trainChanged = []
#print("starting to create changed array")
#for index, image in enumerate(x_train):
#	tensor = (tf.image.random_flip_left_right(image))
#	tensor = tf.image.random_brightness(tensor, 0.5)
#	#x_trainChanged = tensor
#	tensor = tf.keras.backend.eval(tensor)
#	#x_trainChanged += [tensor]
#	x_trainChanged.append(tensor)
#	# x_train = tf.concat([x_train, tensor], 0)
#	if index%500 ==0:
#		#x_trainChanged = np.asarray(x_trainChanged)
#		np.save('text.npy',x_trainChanged) 
#		print("index " +str(index))
#		#print(tf.Tensor.get_shape(x_trainChanged))
#		test = np.load('text.npy')
#		print(x_trainChanged == test)
#x_trainChanged = np.asarray(x_trainChanged)
#print("x_train_changed shape:")
#print(tf.Tensor.get_shape(x_trainChanged))

#print("xtrain shape:")
#print(tf.Tensor.get_shape(x_train))

#x_train = np.concatenate((x_train, x_trainChanged), axis = 0)
#print("changed array created\n starting to merge the two arrays")
# x_train = [x_train, x_trainChanged]
#print("merged arrays")
# sess = tf.Session()
# with sess.as_default():
# 	x_train = x_train.eval() 
# 	y_train = y_train.eval()
#z-score
#print("converted to numpy")
#print(tf.Tensor.get_shape(x_train))
#print(tf.Tensor.get_shape(x_test))
x_train = x_train.astype('float32')/255  # .astype('float32')
x_test = x_test.astype('float32')/255
# mean = np.mean(x_train,axis=(0,1,2,3))  #(0,1,2,3)
# std = np.std(x_train,axis=(0,1,2,3))
# x_train = (x_train-mean)/(std+1e-7)
# x_test = (x_test-mean)/(std+1e-7) 
#"""
#aaa  


datagen = tf.keras.preprocessing.image.ImageDataGenerator( ##this is the start of the data augmentation 
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
	

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

model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
			 




clr = CyclicLR(base_lr=0.001, max_lr=0.1, step_size=2000., mode='triangular2')
#result = model.fit(x_train, y_train, epochs=25 , validation_data = (x_test, y_test), callbacks = [clr]) 

result = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=25, validation_data = (x_test,y_test), callbacks = [clr])

#model.evaluate(x_test, y_test) ##this one can be commented out to reduce read time. 

# weights = model.get_weights()



# prints the result to csv file
f= open("DataAugmentTEST.csv","w")
loss = result.history["loss"]
acc = result.history["acc"]
val_loss = result.history["val_loss"]
val_acc = result.history["val_acc"]
f.write("iteration , Loss , acc , valLoss , valAcc\n")
for i in range(len(loss)):
	row = str(i+1) + " ," + str(loss[i]) + " ,"  + str(acc[i]) + " ," + str(val_loss[i]) + " ," + str(val_acc[i]) + "\n"
	f.write(row)

f.close()
