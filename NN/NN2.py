#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras import layers, optimizers, regularizers
from keras.layers import Flatten , Activation
from keras.layers import Dense
from keras.utils import multi_gpu_model

import keras.backend.tensorflow_backend as tfback

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).
    # Source of this function: https://github.com/keras-team/keras/issues/13684
    # Returns
    A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.set_visible_devices(gpus[0], 'GPU')

tfback._get_available_gpus = _get_available_gpus
print(_get_available_gpus())

import matplotlib.pyplot as plt

# Load data
data_file = "../FastSim_NanoAOD_to_NN/nevents_1000/Htt_merged_NanoAODSIM.h5"
df = pd.read_hdf(data_file)

# make transverse masses
for ana in ["reco", "gen"]:
    df["mT1_{ana}".format(ana=ana)] = (2*df["tau1_pt_{ana}".format(ana=ana)]*df["MET_pt_{ana}".format(ana=ana)]*(1-np.cos(df["tau1_phi_{ana}".format(ana=ana)]-df["MET_phi_{ana}".format(ana=ana)])))**.5
    df["mT2_{ana}".format(ana=ana)] = (2*df["tau2_pt_{ana}".format(ana=ana)]*df["MET_pt_{ana}".format(ana=ana)]*(1-np.cos(df["tau2_phi_{ana}".format(ana=ana)]-df["MET_phi_{ana}".format(ana=ana)])))**.5
    df["mTtt_{ana}".format(ana=ana)] = (2*df["tau1_pt_{ana}".format(ana=ana)]*df["tau2_pt_{ana}".format(ana=ana)]*(1-np.cos(df["tau1_phi_{ana}".format(ana=ana)]-df["tau2_phi_{ana}".format(ana=ana)])))**.5
    df["mTtot_{ana}".format(ana=ana)] = (df["mT1_{ana}".format(ana=ana)]**2+df["mTtt_{ana}".format(ana=ana)]**2)**.5

# select only good points
min_mass = 100
max_mass = 500
df = df.loc[(df['Higgs_mass_gen'] >= min_mass) & (df['Higgs_mass_gen'] <= max_mass)]

# only with 2 jets
df = df.loc[(df['jet2_pt_reco'] > 0)]

# define target and input variables
target = "Higgs_mass_gen"
inputs = list(df.keys())
inputs.remove(target)

inputs = [i for i in inputs if not "_gen" == i[-4:]]
inputs = [i for i in inputs if not "MET_cov" in i]
inputs = [i for i in inputs if not "DM" == i[:2]]
inputs = [i for i in inputs if not "channel" == i[:7]]
inputs = [i for i in inputs if not "_pdgId_" in i]
inputs = [i for i in inputs if not "_mass_reco" in i]
inputs = [i for i in inputs if not "MET_significance_reco" == i]
inputs = [i for i in inputs if not "_btagDeepB_reco" in i]
inputs = [i for i in inputs if not "charge_" in i]

# look for variables distributions
df.hist(figsize = (24,20), bins = 500, log=True)
plt.plot()
plt.savefig("variables.png")
C_mat = df.corr()
fig = plt.figure(figsize = (15,15))
mask = np.zeros_like(C_mat)
mask[np.triu_indices_from(mask)] = True
import seaborn as sb
sb.heatmap(C_mat, vmax = 1, square = True, center=0, cmap='coolwarm', mask=mask)
fig.savefig("correlations.png")
plt.clf()

# Normalize inputs
def norm_factor(input_var):
    if "eta" in input_var:
        return 1/5
    if "phi" in input_var:
        return 1/3.15
    if "MET_cov" in input_var:
        return 10**(-3)
    if "MET_signif" in input_var:
        return 10**(-3)
    if "pt_reco" in input_var:
        return 10**(-3)
    if "mT" in input_var:
        return 10**(-3)
    return 1

for i in inputs:
    print(i)
    df[i] *= norm_factor(i)

# look for variables distributions
df_inputs = df.drop(columns=[k for k in df.keys() if not k in inputs])
df_inputs.hist(figsize = (24,20), bins = 500, log=True)
plt.plot()
plt.savefig("variables_inputs_after_norm.png")
C_mat = df_inputs.corr()
fig = plt.figure(figsize = (15,15))
mask = np.zeros_like(C_mat)
mask[np.triu_indices_from(mask)] = True
sb.heatmap(C_mat, vmax = 1, square = True, center=0, cmap='coolwarm', mask=mask)
fig.savefig("correlations_inputs_after_norm.png")
plt.clf()

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

arr_x_train = np.r_[df_x_train]
arr_y_train = np.r_[df_y_train[target]]
arr_x_valid = np.r_[df_x_valid]
arr_y_valid = np.r_[df_y_valid[target]]
arr_x_test  = np.r_[df_x_test]
arr_y_test  = np.r_[df_y_test[target]]

# min_in = 0
# max_in = 0
# for k in inputs:
#     min_in = min([min_in, df_x_train[k].min()])
#     max_in = max([max_in, df_x_train[k].max()])

# print("Training inputs ranges from {} to {}".format(min_in, max_in))

# Create model
NN_model = Sequential()
NN_model.add(Dense(1, activation="linear", input_shape=(len(df_x_train.keys()),)))
NN_model.add(Dense(1000, activation="relu"))
NN_model.add(Dense(1000, activation="relu"))
NN_model.add(Dense(1000, activation="relu"))
NN_model.add(Dense(1000, activation="relu"))
NN_model.add(Dense(1000, activation="relu"))
NN_model.add(Dense(1, activation="linear"))
print(NN_model.summary())
NN_model.compile(loss='mean_squared_error',
           optimizer=optimizers.Adam(),
           metrics=[keras.metrics.mae])

# Train model
epochs = 500
batch_size = 128
print('Epochs: ', epochs)
print('Batch size: ', batch_size)

keras_callbacks = [keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=50, verbose=0)]

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
    ax=plt.subplot(211)
    ax.set_yscale('log')
    plt.plot(h['mean_absolute_error'])
    plt.plot(h['val_mean_absolute_error'])
    plt.title('Training vs Validation MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.draw()
    
    # summarize history for loss
    ax=plt.subplot(212)
    ax.set_yscale('log')
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot it all in IPython (non-interactive)
    # plt.draw()
    # plt.show()
    
    fig.savefig("history.png")

plot_hist(history.history, xsize=8, ysize=12)

# Plot predicted vs answer on a test sample
plt.clf()
plt.rcParams["figure.figsize"] = [16, 10]
fig, ax = plt.subplots()
predictions, answers = NN_model.predict(arr_x_test), arr_y_test
# Calculate the point density
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
data , x_e, y_e = np.histogram2d( answers, predictions[:,0], bins = [30,30], density = True )
z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([answers, predictions[:,0]]).T , method = "splinef2d", bounds_error = False)
z[np.where(np.isnan(z))] = 0.0
# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = answers[idx], predictions[idx,0], z[idx]
ax.scatter(x,y, c=z, edgecolor='', label="Test")
ax.plot(answers, answers, color="C3")
plt.xlabel("Generated Higgs Mass (GeV)")
plt.ylabel("Predicted Higgs Mass (GeV)")

# linear regression on trained output
xerr_for_reg = 1
yerr_for_reg = 1
# linear function to adjust
def f(x,p):
    a,b = p
    return a*x+b

# its derivative
def Dx_f(x,p):
    a,b = p
    return a

# difference to data
def residual(p, y, x):
    return (y-f(x,p))/np.sqrt(yerr_for_reg**2 + (Dx_f(x,p)*xerr_for_reg)**2)

# initial estimation
# usually OK but sometimes one need to give a different
# starting point to make it converge
p0 = np.array([0,0])
# minimizing algorithm
import scipy.optimize as spo
x, y = answers, np.r_[predictions][:,0]
try:
    result = spo.leastsq(residual, p0, args=(y, x), full_output=True)
except:
    import pdb; pdb.set_trace()
# optimized parameters a and b
popt = result[0];
# variance-covariance matrix
pcov = result[1];
# uncetainties on parameters (1 sigma)
#uopt = np.sqrt(np.abs(np.diagonal(pcov)))
x_aj = np.linspace(min(x),max(x),100)
y_aj = popt[0]*np.linspace(min(x),max(x),100)+popt[1]

ax.plot(x_aj, y_aj, color="C4")
y_info = 0.95
x_info = 0.025
multialignment='left'
horizontalalignment='left'
verticalalignment='top'
ax.text(x_info, y_info,
        '\n'.join([
            '$f(x) = ax+b$',
            '$a = {{ {0:.2e} }}$'.format(popt[0]),
            '$b = {{ {0:.2e} }}$'.format(popt[1])
        ]),
        transform = ax.transAxes, multialignment=multialignment, verticalalignment=verticalalignment, horizontalalignment=horizontalalignment)

#plt.show()
fig.savefig("predicted_vs_answers.png")

