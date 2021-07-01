from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.preprocessing import LabelBinarizer
import pickle
import cv2
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.70)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tensorboard = TensorBoard(log_dir = '{}'.format('check'))
img_size = 224

train_data = np.load('Dataset_v1.npy', allow_pickle=True)

x = np.array([i[0] for i in train_data]).reshape(-1,img_size,img_size,3)
y = [i[1] for i in train_data]

print(y[:10])

lb = LabelBinarizer()
labels = lb.fit_transform(y)
data = np.array(x)
labels = np.array(y)
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)

trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
valAug = ImageDataGenerator()
mean = np.array([123.68, 116.779, 103.939,92.743], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

baseModel = ResNet50(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / 50)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])
#model.summary()
#h = model.fit_generator(trainAug.flow(trainX, trainY, batch_size=32),steps_per_epoch=len(trainX) // 32,validation_data=valAug.flow(testX, testY),validation_steps=len(testX) // 32,epochs=1)
#model.save('crime_model1.model')
model.fit(trainX, trainY, batch_size=64, epochs=5 ,validation_data = (testX,testY),callbacks = [tensorboard])
model.save('dummy_check.model')
#model.save_weights("model_weights.h5")
