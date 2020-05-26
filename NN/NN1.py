#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.config
for device in tf.config.list_physical_devices('GPU'):
    config.experimental.set_visible_devices(device, 'GPU')
    config.experimental.set_memory_growth(device, True)  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.compat.v1.Session()

#set_session(sess)  # set this TensorFlow session as the default session for Keras

# Load data
data_file = "../Delphes_to_NN/prod/merged.h5"
df = pd.read_hdf(data_file)

# define target and input variables
target = "Higgs_Mass_gen"
inputs = list(df.keys())
inputs.remove(target)

inputs = [i for i in inputs if not "_gen" == i[-4:]]
inputs = [i for i in inputs if not "METcov" in i]
inputs = [i for i in inputs if not "DM" == i[:2]]
inputs = [i for i in inputs if not "channel" == i[:7]]

# Split index ranges into training and testing parts with shuffle
train_size = .7
test_size = max([0, 1 - train_size])
df_train, df_test = train_test_split(
    df,
    test_size = test_size,
    train_size = train_size,
    random_state = 1,
    shuffle = True)

df_x_train = df_train.drop(columns=[k for k in df_train.keys() if not k in inputs])
df_y_train = df_train[target]
df_x_test = df_test.drop(columns=[k for k in df_test.keys() if not k in inputs])
df_y_test = df_test[target]

print('Size of training set: ', len(df_x_train))
print('Size of test set: ', len(df_x_test))

# Create model
NN_model = Sequential()
NN_model.add(Dense(10, activation="linear", input_shape=(len(df_x_train.keys()),)))
NN_model.add(Dense(10, activation="linear"))
NN_model.add(Dense(1))
print(NN_model.summary())
NN_model.compile(loss='mean_squared_error',
           optimizer=Adam(),
           metrics=[metrics.mae])

# Train model
epochs = 500
batch_size = 128
print('Epochs: ', epochs)
print('Batch size: ', batch_size)

keras_callbacks = [EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=0)]

history = NN_model.fit(df_x_train, df_y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       shuffle=True,
                       verbose=0, # Change it to 2, if wished to observe execution
                       validation_data=(df_x_test, df_y_test),
                       #callbacks=keras_callbacks,
)

# Evaluate and report performance of the trained model
train_score = NN_model.evaluate(df_x_train, df_y_train, verbose=0)
valid_score = NN_model.evaluate(df_x_test, df_y_test, verbose=0)

print('Train MAE: ', round(train_score[1], 4), ', Train Loss: ', round(train_score[0], 4))
print('Val MAE: ', round(valid_score[1], 4), ', Val Loss: ', round(valid_score[0], 4))

def plot_hist(h, xsize=6, ysize=10):
    # Prepare plotting
    fig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = [xsize, ysize]
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)
    
    # summarize history for MAE
    plt.subplot(211)
    plt.plot(h['mean_absolute_error'])
    plt.plot(h['val_mean_absolute_error'])
    plt.title('Training vs Validation MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # summarize history for loss
    plt.subplot(212)
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot it all in IPython (non-interactive)
    plt.draw()
    plt.show()
    
    return

plot_hist(history.history, xsize=8, ysize=12)
