###First test using GPU locally

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu'))
model.add(tf.keras.layers.Conv2D(filters =96, kernel_size =3, padding = "same", activation = 'relu', strides =(2,2)))

model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu'))
model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu'))
model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu', strides =(2,2)))

model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =3, padding = "same", activation = 'relu'))
model.add(tf.keras.layers.Conv2D(filters =192, kernel_size =1, padding = "same", activation = 'relu'))
model.add(tf.keras.layers.Conv2D(filters =10, kernel_size =1, padding = "same", activation = 'relu'))


model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10,activation ='softmax'))

model.summary()

model.compile(optimizer='SGD',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
			 

model.fit(x_train, y_train, epochs=25)
model.evaluate(x_test, y_test)



Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 32, 32, 96)        2688
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 96)        83040
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 96)        83040
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 192)       166080
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 16, 16, 192)       331968
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 192)         331968
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 8, 8, 192)         331968
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 8, 8, 192)         37056
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 8, 8, 10)          1930
_________________________________________________________________
flatten (Flatten)            (None, 640)               0
_________________________________________________________________
dense (Dense)                (None, 10)                6410
=================================================================
Total params: 1,376,148
Trainable params: 1,376,148
Non-trainable params: 0
_________________________________________________________________
Epoch 1/25
50000/50000 [==============================] - 38s 760us/sample - loss: 2.0969 - acc: 0.2353
Epoch 2/25
50000/50000 [==============================] - 36s 715us/sample - loss: 1.6932 - acc: 0.3936
Epoch 3/25
50000/50000 [==============================] - 36s 716us/sample - loss: 1.5284 - acc: 0.4481
Epoch 4/25
50000/50000 [==============================] - 36s 718us/sample - loss: 1.4280 - acc: 0.4897
Epoch 5/25
50000/50000 [==============================] - 36s 723us/sample - loss: 1.3463 - acc: 0.5196
Epoch 6/25
50000/50000 [==============================] - 36s 725us/sample - loss: 1.2687 - acc: 0.5494
Epoch 7/25
50000/50000 [==============================] - 36s 719us/sample - loss: 1.1976 - acc: 0.5758
Epoch 8/25
50000/50000 [==============================] - 36s 721us/sample - loss: 1.1313 - acc: 0.6007
Epoch 9/25
50000/50000 [==============================] - 36s 721us/sample - loss: 1.0682 - acc: 0.6223
Epoch 10/25
50000/50000 [==============================] - 36s 723us/sample - loss: 1.0088 - acc: 0.6455
Epoch 11/25
50000/50000 [==============================] - 36s 720us/sample - loss: 0.9473 - acc: 0.6694
Epoch 12/25
50000/50000 [==============================] - 36s 720us/sample - loss: 0.8884 - acc: 0.6900
Epoch 13/25
50000/50000 [==============================] - 36s 719us/sample - loss: 0.8357 - acc: 0.7068
Epoch 14/25
50000/50000 [==============================] - 36s 720us/sample - loss: 0.7785 - acc: 0.7272
Epoch 15/25
50000/50000 [==============================] - 36s 721us/sample - loss: 0.7170 - acc: 0.7485
Epoch 16/25
50000/50000 [==============================] - 36s 723us/sample - loss: 0.6561 - acc: 0.7704
Epoch 17/25
50000/50000 [==============================] - 36s 719us/sample - loss: 0.6001 - acc: 0.7873
Epoch 18/25
50000/50000 [==============================] - 37s 732us/sample - loss: 0.5382 - acc: 0.8112
Epoch 19/25
50000/50000 [==============================] - 36s 722us/sample - loss: 0.4785 - acc: 0.8312
Epoch 20/25
50000/50000 [==============================] - 37s 738us/sample - loss: 0.4234 - acc: 0.8500
Epoch 21/25
50000/50000 [==============================] - 37s 733us/sample - loss: 0.3696 - acc: 0.8680
Epoch 22/25
50000/50000 [==============================] - 36s 726us/sample - loss: 0.3272 - acc: 0.8831
Epoch 23/25
50000/50000 [==============================] - 36s 723us/sample - loss: 0.2829 - acc: 0.8983
Epoch 24/25
50000/50000 [==============================] - 36s 721us/sample - loss: 0.2405 - acc: 0.9142
Epoch 25/25
50000/50000 [==============================] - 36s 725us/sample - loss: 0.2148 - acc: 0.9245
10000/10000 [==============================] - 3s 294us/sample - loss: 2.0533 - acc: 0.5872










###BATCH NORMALIZATION WITH SGD 25 epochs



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

#model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#model.add(tf.keras.layers.Dropout(0.2))
#model.add(Convolution2D(64, 3, 3, border_mode = 'same', input_shape = (32, 32, 3)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10,activation ='softmax'))

model.summary()

model.compile(optimizer='SGD',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
			 

model.fit(x_train, y_train, epochs=25)
model.evaluate(x_test, y_test)









Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 32, 32, 96)        2688
_________________________________________________________________
batch_normalization_v1 (Batc (None, 32, 32, 96)        384
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 96)        83040
_________________________________________________________________
batch_normalization_v1_1 (Ba (None, 32, 32, 96)        384
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 96)        83040
_________________________________________________________________
batch_normalization_v1_2 (Ba (None, 16, 16, 96)        384
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 192)       166080
_________________________________________________________________
batch_normalization_v1_3 (Ba (None, 16, 16, 192)       768
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 16, 16, 192)       331968
_________________________________________________________________
batch_normalization_v1_4 (Ba (None, 16, 16, 192)       768
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 192)         331968
_________________________________________________________________
batch_normalization_v1_5 (Ba (None, 8, 8, 192)         768
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 8, 8, 192)         331968
_________________________________________________________________
batch_normalization_v1_6 (Ba (None, 8, 8, 192)         768
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 8, 8, 192)         37056
_________________________________________________________________
batch_normalization_v1_7 (Ba (None, 8, 8, 192)         768
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 8, 8, 10)          1930
_________________________________________________________________
batch_normalization_v1_8 (Ba (None, 8, 8, 10)          40
_________________________________________________________________
flatten (Flatten)            (None, 640)               0
_________________________________________________________________
dense (Dense)                (None, 10)                6410
=================================================================
Total params: 1,381,180
Trainable params: 1,378,664
Non-trainable params: 2,516
_________________________________________________________________
Epoch 1/25
50000/50000 [==============================] - 47s 945us/sample - loss: 1.5105 - acc: 0.4587
Epoch 2/25
50000/50000 [==============================] - 45s 894us/sample - loss: 1.1182 - acc: 0.6029
Epoch 3/25
50000/50000 [==============================] - 45s 895us/sample - loss: 0.9066 - acc: 0.6832
Epoch 4/25
50000/50000 [==============================] - 46s 919us/sample - loss: 0.7362 - acc: 0.7439
Epoch 5/25
50000/50000 [==============================] - 46s 914us/sample - loss: 0.5981 - acc: 0.7934
Epoch 6/25
50000/50000 [==============================] - 46s 913us/sample - loss: 0.4710 - acc: 0.8364
Epoch 7/25
50000/50000 [==============================] - 46s 915us/sample - loss: 0.3616 - acc: 0.8772
Epoch 8/25
50000/50000 [==============================] - 46s 916us/sample - loss: 0.2588 - acc: 0.9137
Epoch 9/25
50000/50000 [==============================] - 46s 913us/sample - loss: 0.1916 - acc: 0.9360
Epoch 10/25
50000/50000 [==============================] - 46s 914us/sample - loss: 0.1380 - acc: 0.9542
Epoch 11/25
50000/50000 [==============================] - 47s 931us/sample - loss: 0.0845 - acc: 0.9736
Epoch 12/25
50000/50000 [==============================] - 46s 927us/sample - loss: 0.0497 - acc: 0.9860
Epoch 13/25
50000/50000 [==============================] - 45s 904us/sample - loss: 0.0249 - acc: 0.9944
Epoch 14/25
50000/50000 [==============================] - 61s 1ms/sample - loss: 0.0103 - acc: 0.9988
Epoch 15/25
50000/50000 [==============================] - 50s 997us/sample - loss: 0.0050 - acc: 0.9997
Epoch 16/25
50000/50000 [==============================] - 49s 983us/sample - loss: 0.0028 - acc: 0.9999
Epoch 17/25
50000/50000 [==============================] - 49s 987us/sample - loss: 0.0019 - acc: 1.0000
Epoch 18/25
50000/50000 [==============================] - 49s 985us/sample - loss: 0.0013 - acc: 1.0000
Epoch 19/25
50000/50000 [==============================] - 49s 987us/sample - loss: 0.0011 - acc: 1.0000
Epoch 20/25
50000/50000 [==============================] - 47s 946us/sample - loss: 9.9176e-04 - acc: 1.0000
Epoch 21/25
50000/50000 [==============================] - 47s 945us/sample - loss: 8.9135e-04 - acc: 1.0000
Epoch 22/25
50000/50000 [==============================] - 47s 944us/sample - loss: 8.0287e-04 - acc: 1.0000
Epoch 23/25
50000/50000 [==============================] - 47s 946us/sample - loss: 6.8220e-04 - acc: 1.0000
Epoch 24/25
50000/50000 [==============================] - 47s 944us/sample - loss: 6.4841e-04 - acc: 1.0000
Epoch 25/25
50000/50000 [==============================] - 47s 950us/sample - loss: 7.0925e-04 - acc: 1.0000
10000/10000 [==============================] - 3s 325us/sample - loss: 1.2132 - acc: 0.7508





###Batch normalization with 5 epochs SGD

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 32, 32, 96)        2688
_________________________________________________________________
batch_normalization_v1 (Batc (None, 32, 32, 96)        384
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 96)        83040
_________________________________________________________________
batch_normalization_v1_1 (Ba (None, 32, 32, 96)        384
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 96)        83040
_________________________________________________________________
batch_normalization_v1_2 (Ba (None, 16, 16, 96)        384
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 192)       166080
_________________________________________________________________
batch_normalization_v1_3 (Ba (None, 16, 16, 192)       768
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 16, 16, 192)       331968
_________________________________________________________________
batch_normalization_v1_4 (Ba (None, 16, 16, 192)       768
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 192)         331968
_________________________________________________________________
batch_normalization_v1_5 (Ba (None, 8, 8, 192)         768
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 8, 8, 192)         331968
_________________________________________________________________
batch_normalization_v1_6 (Ba (None, 8, 8, 192)         768
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 8, 8, 192)         37056
_________________________________________________________________
batch_normalization_v1_7 (Ba (None, 8, 8, 192)         768
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 8, 8, 10)          1930
_________________________________________________________________
batch_normalization_v1_8 (Ba (None, 8, 8, 10)          40
_________________________________________________________________
flatten (Flatten)            (None, 640)               0
_________________________________________________________________
dense (Dense)                (None, 10)                6410
=================================================================
Total params: 1,381,180
Trainable params: 1,378,664
Non-trainable params: 2,516
_________________________________________________________________
Epoch 1/5
50000/50000 [==============================] - 56s 1ms/sample - loss: 1.5161 - acc: 0.4574
Epoch 2/5
50000/50000 [==============================] - 50s 1ms/sample - loss: 1.1047 - acc: 0.6077
Epoch 3/5
50000/50000 [==============================] - 50s 1ms/sample - loss: 0.8901 - acc: 0.6847
Epoch 4/5
50000/50000 [==============================] - 50s 997us/sample - loss: 0.7260 - acc: 0.7465
Epoch 5/5
50000/50000 [==============================] - 50s 1ms/sample - loss: 0.5924 - acc: 0.7923
10000/10000 [==============================] - 4s 381us/sample - loss: 0.9265 - acc: 0.6892






###BATCH NORMALIZATION WITH ADAM 25 epochs



Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 32, 32, 96)        2688
_________________________________________________________________
batch_normalization_v1 (Batc (None, 32, 32, 96)        384
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 96)        83040
_________________________________________________________________
batch_normalization_v1_1 (Ba (None, 32, 32, 96)        384
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 96)        83040
_________________________________________________________________
batch_normalization_v1_2 (Ba (None, 16, 16, 96)        384
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 192)       166080
_________________________________________________________________
batch_normalization_v1_3 (Ba (None, 16, 16, 192)       768
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 16, 16, 192)       331968
_________________________________________________________________
batch_normalization_v1_4 (Ba (None, 16, 16, 192)       768
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 192)         331968
_________________________________________________________________
batch_normalization_v1_5 (Ba (None, 8, 8, 192)         768
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 8, 8, 192)         331968
_________________________________________________________________
batch_normalization_v1_6 (Ba (None, 8, 8, 192)         768
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 8, 8, 192)         37056
_________________________________________________________________
batch_normalization_v1_7 (Ba (None, 8, 8, 192)         768
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 8, 8, 10)          1930
_________________________________________________________________
batch_normalization_v1_8 (Ba (None, 8, 8, 10)          40
_________________________________________________________________
flatten (Flatten)            (None, 640)               0
_________________________________________________________________
dense (Dense)                (None, 10)                6410
=================================================================
Total params: 1,381,180
Trainable params: 1,378,664
Non-trainable params: 2,516
_________________________________________________________________
Train on 50000 samples, validate on 10000 samples
Epoch 1/25
50000/50000 [==============================] - 65s 1ms/sample - loss: 1.3901 - acc: 0.5062 - val_loss: 1.2472 - val_acc: 0.5844
Epoch 2/25
50000/50000 [==============================] - 60s 1ms/sample - loss: 0.8892 - acc: 0.6850 - val_loss: 1.1548 - val_acc: 0.6043
Epoch 3/25
50000/50000 [==============================] - 58s 1ms/sample - loss: 0.7163 - acc: 0.7501 - val_loss: 0.7644 - val_acc: 0.7397
Epoch 4/25
50000/50000 [==============================] - 60s 1ms/sample - loss: 0.5862 - acc: 0.7953 - val_loss: 0.7281 - val_acc: 0.7497
Epoch 5/25
50000/50000 [==============================] - 58s 1ms/sample - loss: 0.4689 - acc: 0.8385 - val_loss: 0.6285 - val_acc: 0.7845
Epoch 6/25
50000/50000 [==============================] - 60s 1ms/sample - loss: 0.3610 - acc: 0.8739 - val_loss: 0.6346 - val_acc: 0.7972
Epoch 7/25
50000/50000 [==============================] - 57s 1ms/sample - loss: 0.2597 - acc: 0.9098 - val_loss: 0.6999 - val_acc: 0.7813
Epoch 8/25
50000/50000 [==============================] - 57s 1ms/sample - loss: 0.1846 - acc: 0.9364 - val_loss: 0.7594 - val_acc: 0.7948
Epoch 9/25
50000/50000 [==============================] - 58s 1ms/sample - loss: 0.1354 - acc: 0.9529 - val_loss: 0.7938 - val_acc: 0.7912
Epoch 10/25
50000/50000 [==============================] - 59s 1ms/sample - loss: 0.1102 - acc: 0.9615 - val_loss: 0.8753 - val_acc: 0.7911
Epoch 11/25
50000/50000 [==============================] - 57s 1ms/sample - loss: 0.0888 - acc: 0.9678 - val_loss: 0.8236 - val_acc: 0.8080
Epoch 12/25
50000/50000 [==============================] - 58s 1ms/sample - loss: 0.0781 - acc: 0.9719 - val_loss: 0.9165 - val_acc: 0.8022
Epoch 13/25
50000/50000 [==============================] - 58s 1ms/sample - loss: 0.0755 - acc: 0.9737 - val_loss: 0.9990 - val_acc: 0.7870
Epoch 14/25
50000/50000 [==============================] - 57s 1ms/sample - loss: 0.0698 - acc: 0.9753 - val_loss: 0.9725 - val_acc: 0.7974
Epoch 15/25
50000/50000 [==============================] - 60s 1ms/sample - loss: 0.0582 - acc: 0.9797 - val_loss: 0.9911 - val_acc: 0.8022
Epoch 16/25
50000/50000 [==============================] - 59s 1ms/sample - loss: 0.0623 - acc: 0.9777 - val_loss: 0.9775 - val_acc: 0.8022
Epoch 17/25
50000/50000 [==============================] - 59s 1ms/sample - loss: 0.0518 - acc: 0.9816 - val_loss: 1.0650 - val_acc: 0.7886
Epoch 18/25
50000/50000 [==============================] - 60s 1ms/sample - loss: 0.0489 - acc: 0.9829 - val_loss: 1.0873 - val_acc: 0.7943
Epoch 19/25
50000/50000 [==============================] - 59s 1ms/sample - loss: 0.0485 - acc: 0.9833 - val_loss: 0.9752 - val_acc: 0.8086
Epoch 20/25
50000/50000 [==============================] - 60s 1ms/sample - loss: 0.0448 - acc: 0.9846 - val_loss: 1.0749 - val_acc: 0.8067
Epoch 21/25
50000/50000 [==============================] - 60s 1ms/sample - loss: 0.0415 - acc: 0.9854 - val_loss: 1.0268 - val_acc: 0.8094
Epoch 22/25
50000/50000 [==============================] - 59s 1ms/sample - loss: 0.0422 - acc: 0.9849 - val_loss: 1.0714 - val_acc: 0.8046
Epoch 23/25
50000/50000 [==============================] - 61s 1ms/sample - loss: 0.0380 - acc: 0.9869 - val_loss: 1.0735 - val_acc: 0.8005
Epoch 24/25
50000/50000 [==============================] - 59s 1ms/sample - loss: 0.0350 - acc: 0.9877 - val_loss: 1.0389 - val_acc: 0.8195
Epoch 25/25
50000/50000 [==============================] - 59s 1ms/sample - loss: 0.0360 - acc: 0.9876 - val_loss: 1.1168 - val_acc: 0.8076
10000/10000 [==============================] - 3s 337us/sample - loss: 1.1168 - acc: 0.8076