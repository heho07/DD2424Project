# Visualizes the layer output from a pre-trained model

# Takes a picture of a truck from the training data set from CIFAR-10
# Loads model
# Creates images from layer output from a given layer

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
import csv
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import sys
import errno
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32')/255 
x_test = x_test.astype('float32')/255

# image of a truck

truck_image = x_train[0,:]
truck_image = np.asarray(truck_image)

#loading the model we want to visualize
loading_checkpoint_path = "./Models/all-cnn-c-dataaugment90noclr/learning_rate0.01.h5"
model = tf.keras.models.load_model(loading_checkpoint_path)
model.summary()
#model.evaluate(x_test, y_test)  #checking that it works

def check_directory_exists(full_path):
   if not os.path.exists(os.path.dirname(full_path)):
       try:
           os.makedirs(os.path.dirname(full_path))
       except OSError as exc: # Guard against race condition
           if exc.errno != errno.EEXIST:
               raise

# get the layer output from a certain layer
def get_nth_layer_output(layer, image):
    get_layer_output = K.function([model.layers[0].input],
                                      [model.layers[layer].output])
    layer_output = get_layer_output([image])[0]
    layer_output = np.squeeze(layer_output, axis = 0)
    return layer_output

# visualize the layer output
def visualize_layer_output(layer_output, model_name, layer, show = True):
    height, width, depth = layer_output.shape
    nb_plot = int(np.rint(np.sqrt(depth)))
    figure_limit = np.ceil(np.sqrt(depth))
    fig = plt.figure(figsize=(figure_limit, figure_limit))
    for i in range(depth):
        #print(nb_plot)
        plt.subplot(nb_plot, nb_plot, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(layer_output[:,:,i], cmap='gray')
    plt.suptitle('layer: {}'.format(layer), size = 22)

    save_path = './images/layer_visualization/' + model_name + '/_layer_' + str(layer) + ".png"
    check_directory_exists(save_path)
    plt.savefig(save_path)
    
    if show:
        plt.show()
    plt.close()
        
# gets input layer and visualizes it
def get_and_visualize_layer_output(layer, image, model_name = "foo", show = True):
    truck_output_layer_n = get_nth_layer_output(layer, image)
    visualize_layer_output(truck_output_layer_n, model_name, layer, show)

# saves the original image to file
def get_original_image(image, image_name):
    plt.figure()
    plt.imshow(image)
    plt.title('original image')
    save_path = './images/layer_visualization/'+image_name+'.png'
    check_directory_exists(save_path)
    plt.savefig(save_path)
    plt.close()



# gets the layer output for a number of images in the range of

lower_image_bound = 1
upper_image_bound = 10

for j in range(lower_image_bound,upper_image_bound):
    truck_image = x_train[j,:]
    truck_image = np.asarray(truck_image)
    get_original_image(truck_image, "all-cnn-c-dataugment90noclr/image"+str(j)+"/original_image")
    truck_image = np.expand_dims(truck_image,axis=0)
    for i in range(len(model.layers)):    
        print("getting visualization of layer " + str(i) + " for image " + str(j))
        try:
            get_and_visualize_layer_output(i, truck_image, "all-cnn-c-dataugment90noclr/image" + str(j), False)
        except KeyboardInterrupt:
            sys.exit("user exited program")
        except:
            print("error on layer " + str(i))



