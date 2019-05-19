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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32')/255 
x_test = x_test.astype('float32')/255

# image of a truck

truck_image = x_train[1,:]
truck_image = np.asarray(truck_image)
truck_image = np.expand_dims(truck_image,axis=0)

#loading the model we want to visualize
loading_checkpoint_path = "./Models/replicating_study/learning_rate0.01.h5"
model = tf.keras.models.load_model(loading_checkpoint_path)
#model.evaluate(x_test, y_test)  #checking that it works


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
    fig = plt.figure(figsize=(20, 20))
    for i in range(depth):
        #print(nb_plot)
        plt.subplot(nb_plot, nb_plot, i+1)
        plt.imshow(layer_output[:,:,i], cmap='gray')
        plt.title('l: {}'.format(i+1))
    plt.savefig('./images/layer_visualization/' + model_name + '_layer_' + str(layer) + ".png")
    if show:
        plt.show()
        

def get_and_visualize_layer_output(layer, image, model_name = "foo", show = True):
    truck_output_layer_n = get_nth_layer_output(layer, image)
    visualize_layer_output(truck_output_layer_n, model_name, layer, show)

for i in range(len(model.layers)):
    #try:
        get_and_visualize_layer_output(i, truck_image, "replicating_study_learning_rate0.01", False)
    #except:
        #print("error on layer" + str(i))
    

