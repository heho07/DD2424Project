import tarfile
import pickle
import random
import numpy as np
import tensorflow as tf
from urllib.request import urlretrieve
from os.path import isfile, isdir
# import numpy.plot as plt
import keras
import matplotlib.pyplot as plt


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
#     with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
#         # note the encoding type is 'latin1'
#         batch = pickle.load(file, encoding='latin1')
        
#     features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
#     labels = batch['labels']
        
#     return features, labels

# def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):
#     features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)
    
#     if not (0 <= sample_id < len(features)):
#         print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
#         return None

#     print('\nStats of batch #{}:'.format(batch_id))
#     print('# of Samples: {}\n'.format(len(features)))
    
#     label_names = load_label_names()
#     label_counts = dict(zip(*np.unique(labels, return_counts=True)))
#     for key, value in label_counts.items():
#         print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))
    
#     sample_image = features[sample_id]
#     sample_label = labels[sample_id]
    
#     print('\nExample of Image {}:'.format(sample_id))
#     print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
#     print('Image - Shape: {}'.format(sample_image.shape))
#     print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))
#     plt.imshow(sample_image)
#     plt.show()

# def load_label_names():
#     return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# # images, labels = (load_cfar10_batch("./cifar-10-batches-py", 1))
# # print(len(images))
# (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()

# xtrain = xtrain.astype('float32')

# print((xtrain[0]))
# newX = xtrain[0]
# testTensor = tf.constant(xtrain)
# print(testTensor)
# print(tf.rank(testTensor))
# # newX = tf.keras.backend.batch_flatten(testTensor)
# # print(newX)
# normXtrain = tf.keras.backend.batch_normalization(
# 	testTensor, 
# 	0.01,
# 	0,
# 	0,
# 	1,
# 	-1,
# 	0
# 	)
