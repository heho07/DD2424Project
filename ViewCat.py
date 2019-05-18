import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
#from tensorflow import Conv2D
# import numpy.plot as plt
import csv

import matplotlib.pyplot as plt
from matplotlib.image import imread

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

# ------------------------------------- end of code from https://github.com/bckenstler/CLR


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32')/255  # .astype('float32')
x_test = x_test.astype('float32')/255

# mean = np.mean(x_train,axis=(0,1,2,3))  #(0,1,2,3)
# std = np.std(x_train,axis=(0,1,2,3))
# x_train = (x_train-mean)/(std+1e-7)
# x_test = (x_test-mean)/(std+1e-7) 

frog = x_train[1,:]

datagen = tf.keras.preprocessing.image.ImageDataGenerator( ##this is the start of the data augmentation 
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip = True)
	
def setUpModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', input_shape=x_train.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(l = 0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l = 0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', strides =(2,2), kernel_regularizer=tf.keras.regularizers.l2(l = 0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l = 0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l = 0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', strides =(2,2), kernel_regularizer=tf.keras.regularizers.l2(l = 0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l = 0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =1, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l = 0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters =10, kernel_size =1, padding = "same", activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l = 0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10,activation ='softmax', kernel_regularizer=tf.keras.regularizers.l2(l = 0.001)))

    model.summary()

    model.compile(optimizer='Adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model			 


def trainModel():
    # callback used for cyclical learning
    clr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular2')
    
    # callback used to save the model during runtime
    checkpoint_path = "WeightsFromTraining/pictures/frog-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                save_weights_only=True,
                                                 verbose=1,
                                                 period = 5)
#result = model.fit(x_train, y_train, epochs=25 , validation_data = (x_test, y_test), callbacks = [clr]) 

    # runs the training procedure
    result = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                epochs=5, validation_data = (x_test,y_test), callbacks = [clr, cp_callback])

    # prints the result to csv file
    f= open("X.csv","w")
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

# loading_checkpoint_path = "WeightsFromTraining/pictures/frog-0010.ckpt"

# model = setUpModel()

# model.evaluate(x_test, y_test)
# model.load_weights(loading_checkpoint_path)
# model.evaluate(x_test, y_test)
# result = trainModel()


# cat = imread("cat.png")
# cat = np.array(cat)
# plt.figure()

frog2 = np.asarray(frog)
print(frog2.shape)
# plt.figure()
# plt.imshow(frog2)
# plt.show()

loading_checkpoint_path = "./Models/replicating_study/learning_rate0.01.h5"
model = tf.keras.models.load_model(loading_checkpoint_path)
model.evaluate(x_test, y_test)


# with a Sequential model
cat_batch = np.expand_dims(frog2,axis=0)
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([cat_batch])[0]

get_6th_layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].output])
layer_output = get_6th_layer_output([cat_batch])[0]

layer_output = np.squeeze(layer_output, axis = 0)

height, width, depth = layer_output.shape
nb_plot = int(np.rint(np.sqrt(depth)))
fig = plt.figure(figsize=(20, 20))
for i in range(depth):
    plt.subplot(nb_plot, nb_plot, i+1)
    plt.imshow(layer_output[:,:,i], cmap='gray')
    plt.title('feature map {}'.format(i+1))
plt.show()

# datagen = tf.keras.preprocessing.image.ImageDataGenerator( ##this is the start of the data augmentation
#         zca_whitening = True
#         )
# datagen.fit(cat_batch)
# cat = np.squeeze(cat_batch, axis = 0)
# plt.figure()
# cat3 = datagen.flow(cat_batch)
# cat3 = np.squeeze(cat3, axis = 0)
# cat3 = datagen.standardize((cat))
# print(cat3.shape)
# plt.imshow(cat3)
# plt.show()



# model.evaluate(x_test, y_test) ##this one can be commented out to reduce read time. 