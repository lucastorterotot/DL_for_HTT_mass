#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

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

import matplotlib.pyplot as plt

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

# Normalize entries
# for i in inputs:
#     print(i, np.mean(df[i]), np.std(df[i]))
#     df[i] = (df[i] - np.mean(df[i]))/np.std(df[i])
#     print(i, np.mean(df[i]), np.std(df[i]))

# Split index ranges into training and testing parts with shuffle
train_size = .7
valid_size = .2
test_size = 100 / len(df[target])

test_size = max([test_size, 1 - train_size - valid_size])

def train_valid_test_split(df, train_part=.6, valid_part=.2, test_part=.2, seed=None):
    np.random.seed(seed)
    total_size = train_part + valid_part + test_part
    train_percent = train_part / total_size
    valid_percent = valid_part / total_size
    test_percent = test_part / total_size
    perm = np.random.permutation(df.index)
    m = len(df)
    train_end = int(train_percent * m)
    valid_end = int(valid_percent * m) + train_end
    train = perm[:train_end]
    valid = perm[train_end:valid_end]
    test = perm[valid_end:]
    return train, valid, test

np_train, np_valid, np_test = train_valid_test_split(
    df,
    train_part = train_size,
    valid_part = valid_size,
    test_part = test_size,
    seed = 1)

df_x_train = df.loc[np_train, :].drop(columns=[k for k in df.keys() if not k in inputs])
df_y_train = df.loc[np_train, [target]]
df_x_valid = df.loc[np_valid, :].drop(columns=[k for k in df.keys() if not k in inputs])
df_y_valid = df.loc[np_valid, [target]]
df_x_test  = df.loc[np_test, :].drop(columns=[k for k in df.keys() if not k in inputs])
df_y_test  = df.loc[np_test, [target]]

print('Size of training set: ', len(df_x_train))
print('Size of valid set: ', len(df_x_valid))
print('Size of test set:     ', len(df_x_test))

arr_x_train = np.c_[df_x_train]
arr_y_train = np.r_[df_y_train[target]]
arr_x_valid = np.c_[df_x_valid]
arr_y_valid = np.r_[df_y_valid[target]]
arr_x_test  = np.c_[df_x_test]
arr_y_test  = np.r_[df_y_test[target]]

# Create model
NN_model = Sequential()
NN_model.add(Dense(100, activation="linear", input_shape=(len(df_x_train.keys()),)))
NN_model.add(Dense(200, activation="linear"))
NN_model.add(Dense(100, activation="linear"))
NN_model.add(Dense(50, activation="linear"))
NN_model.add(Dense(1, activation="linear"))
print(NN_model.summary())
NN_model.compile(loss='mean_squared_error',
           optimizer=Adam(),
           metrics=[metrics.mae])

# Train model
epochs = 50
batch_size = 128
print('Epochs: ', epochs)
print('Batch size: ', batch_size)

keras_callbacks = [EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=0)]

history = NN_model.fit(arr_x_train, arr_y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       shuffle=True,
                       verbose=2, # Change it to 2, if wished to observe execution
                       validation_data=(arr_x_valid, arr_y_valid),
                       callbacks=keras_callbacks,
)

# Evaluate and report performance of the trained model
train_score = NN_model.evaluate(arr_x_train, arr_y_train, verbose=0)
valid_score = NN_model.evaluate(arr_x_valid, arr_y_valid, verbose=0)

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

#plot_hist(history.history, xsize=8, ysize=12)

predictions, answers = NN_model.predict(arr_x_test), arr_y_test
plt.scatter(answers, predictions, color="C0")
plt.plot(answers, answers, color="C3")
plt.xlabel("Generated Higgs Mass (GeV)")
plt.y_label("Predicted Higgs Mass (GeV)")
plt.show()
