import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from random import randint
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

def plot_images(dataset, n_images, samples_per_image):
    output = np.zeros((32 * n_images, 32 * samples_per_image, 3))

    row = 0
    for images in dataset.repeat(samples_per_image).batch(n_images):
        output[:, row*32:(row+1)*32] = np.vstack(images.numpy())
        row += 1

    plt.figure()
    plt.imshow(output)
    plt.show()


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print("getShape of x_train")
print(tf.Tensor.get_shape(x_train))
print("starting to merge ytrain")

x_train = x_test[0:8]
y_train = y_test[0:8]

print(tf.Tensor.get_shape(y_train))
y_train = np.concatenate((y_train, y_train), axis = 0)
print(tf.Tensor.get_shape(y_train))
print("ytrain merged")

x_trainChanged = []
print("starting to create changed array")
for index, image in enumerate(x_train):
	flip_chance = randint(0,9)
	crop_chance = randint(0,5)
	contrast_chance = randint(0,2)
	tensor = image
	if flip_chance == 0:
		tensor = (tf.image.random_flip_left_right(image))
	if crop_chance == 0:
		tensor = tf.image.random_crop(image, size = [32, 32, 3])
	if contrast_chance == 0:
		tensor = tf.image.random_contrast(image, 0.5, 0.7) 
	tensor = tf.image.random_brightness(tensor, 0.5)
	
	# if nothing happened to the image
	if flip_chance != 0 & crop_chance != 0 & contrast_chance != 0:
		tensor = tf.image.random_saturation(image, 0.5, 0.8)

	# numpyTensor = K.eval(tensor)
	x_trainChanged.append(tensor)
	if index%500 == 0:
		print("index" + str(index))
		
	# x_train = tf.concat([x_train, tensor], 0)
# print("x_train_changed shape BEFORE np array:")
# print(tf.Tensor.get_shape(x_trainChanged))
print("finished augmenting data")
x_trainChanged = tf.stack(x_trainChanged)
x_train = tf.stack(x_train)
# print("changed x_trainChanged to tensor")
# x_trainChanged = tf.keras.backend.eval(x_trainChanged)
# x_trainChanged = np.asarray(x_trainChanged)
print("changed x_train_changed to numpy")
# np.save('x_trainChanged.npy', x_trainChanged)
print("x_train_changed shape AFTER np array:")
print(tf.Tensor.get_shape(x_trainChanged))

print("xtrain shape:")
print(tf.Tensor.get_shape(x_train))

x_train = tf.concat((x_train, x_trainChanged), 0)

# x_train = np.concatenate((x_train, x_trainChanged), axis = 0)
print("changed array created\n starting to merge the two arrays")
# x_train = [x_train, x_trainChanged]
print("merged arrays")
# sess = tf.Session()
# with sess.as_default():
# 	x_train = x_train.eval() 
# 	y_train = y_train.eval()
#z-score
print("converted to numpy")
print(tf.Tensor.get_shape(x_train))
print(tf.Tensor.get_shape(x_test))
x_train = x_train/255  # .astype('float32')
x_test = x_test/255
# mean = np.mean(x_train,axis=(0,1,2,3))  #(0,1,2,3)
# std = np.std(x_train,axis=(0,1,2,3))
# x_train = (x_train-mean)/(std+1e-7)
# x_test = (x_test-mean)/(std+1e-7)
print(tf.Tensor.get_shape(x_train)[0])

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
			 


# callback codes below
batch_print_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_begin = lambda batch,logs: tf.print(model.optimizer.lr)
		# 	
	) #				model.set_value(self.mode.optimizer.lr, value)))

batch_set_callback = tf.keras.callbacks.LambdaCallback(
    on_batch_begin = lambda batch,logs: K.set_value(model.optimizer.lr, 0.1/(batch+1))
		# model.optimizer.lr.set_value(0.001)	
	) #				model.set_value(self.mode.optimizer.lr, value)))

batch_setBegin_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_begin = lambda batch,logs: K.set_value(model.optimizer.lr, 0.1)
		# model.optimizer.lr.set_value(0.001)	
	) #				model.set_value(self.mode.optimizer.lr, value)))


batch_learning_callback = tf.keras.callbacks.LearningRateScheduler(
    # on_epoch_begin = lambda batch,logs: 
    schedule = lambda epoch, verbose = 1: 0.999 
		# model.optimizer.lr.set_value(0.001)	
	) #				model.set_value(self.mode.optimizer.lr, value)))

clr = CyclicLR(base_lr=0.001, max_lr=0.1, step_size=2000., mode='triangular2')
result = model.fit(x_train, y_train, epochs=25 , validation_data = (x_test, y_test), callbacks = [clr], steps_per_epoch = tf.Tensor.get_shape(x_train)[0]) 
model.evaluate(x_test, y_test)

# weights = model.get_weights()



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
