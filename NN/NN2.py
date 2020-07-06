#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from optparse import OptionParser
usage = "usage: %prog [options]"
parser = OptionParser(usage=usage)
parser.add_option("-o", "--output", dest = "output",
                  default = "NN")
parser.add_option("-L", "--Nlayers", dest = "Nlayers",
                  default = 5)
parser.add_option("-N", "--Nneurons", dest = "Nneurons",
                  default = 1000)
parser.add_option("-E", "--Nevents", dest = "Nevents",
                  default = 20000)
parser.add_option("-g", "--gpu", dest = "gpu",
                  default = 0)

(options,args) = parser.parse_args()

options.Nlayers = int(options.Nlayers)
options.Nneurons = int(options.Nneurons)
options.gpu = int(options.gpu)

output = "_".join([options.output, str(options.Nlayers), "layers", str(options.Nneurons), "neurons"])

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
tf.config.experimental.set_memory_growth(gpus[options.gpu], True)
tf.config.set_visible_devices(gpus[options.gpu], 'GPU')

tfback._get_available_gpus = _get_available_gpus
print(_get_available_gpus())

import matplotlib.pyplot as plt

# Load data
data_file = "../FastSim_NanoAOD_to_NN/nevents_{}/Htt_merged_NanoAODSIM.h5".format(options.Nevents)
df = pd.read_hdf(data_file)

# make transverse masses
for ana in ["reco", "gen"]:
    df["mT1_{ana}".format(ana=ana)] = (2*df["tau1_pt_{ana}".format(ana=ana)]*df["MET_pt_{ana}".format(ana=ana)]*(1-np.cos(df["tau1_phi_{ana}".format(ana=ana)]-df["MET_phi_{ana}".format(ana=ana)])))**.5
    df["mT2_{ana}".format(ana=ana)] = (2*df["tau2_pt_{ana}".format(ana=ana)]*df["MET_pt_{ana}".format(ana=ana)]*(1-np.cos(df["tau2_phi_{ana}".format(ana=ana)]-df["MET_phi_{ana}".format(ana=ana)])))**.5
    df["mTtt_{ana}".format(ana=ana)] = (2*df["tau1_pt_{ana}".format(ana=ana)]*df["tau2_pt_{ana}".format(ana=ana)]*(1-np.cos(df["tau1_phi_{ana}".format(ana=ana)]-df["tau2_phi_{ana}".format(ana=ana)])))**.5
    df["mTtot_{ana}".format(ana=ana)] = (df["mT1_{ana}".format(ana=ana)]**2+df["mT2_{ana}".format(ana=ana)]**2+df["mTtt_{ana}".format(ana=ana)]**2)**.5

# select only good points
min_mass = 100
max_mass = 500

# define target and input variables
target = "Higgs_mass_gen"
inputs = list(df.keys())
inputs.remove(target)

inputs = [i for i in inputs if not "_gen" == i[-4:]]
inputs = [i for i in inputs if not "file" == i]
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

def train_valid_test_split(df, train_size=.6, valid_size=.2, test_size=.2, seed=None):
    np.random.seed(seed)
    total_size = train_size + valid_size + test_size
    train_percent = train_size / total_size
    valid_percent = valid_size / total_size
    test_percent = test_size / total_size
    perm = np.random.permutation(df.index)
    m = len(df)
    train_end = int(train_percent * m)
    valid_end = int(valid_percent * m) + train_end
    train = perm[:train_end]
    valid = perm[train_end:valid_end]
    test = perm[valid_end:]
    return train, valid, test

def norm_factor(input_var):
    if "MET_cov" in input_var:
        return 10**(-3)
    if "MET_signif" in input_var:
        return 10**(-3)
    if "pt_reco" in input_var:
        return 10**(-3)
    if "mT" in input_var:
        return 10**(-3)
    return 1

def plot_hist(h, NNname, xsize=6, ysize=10):
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
    
    fig.savefig("history_{}.png".format(NNname))

# Split index ranges into training and testing parts with shuffle
train_size = .7
valid_size = .2
test_size = .1
test_size = max([test_size, 1 - train_size - valid_size])

np_train, np_valid, np_test = train_valid_test_split(
    df,
    train_size = train_size,
    valid_size = valid_size,
    test_size = test_size,
    seed = 2020)

df["is_train"] = np.zeros(len(df[target]))
df["is_valid"] = np.zeros(len(df[target]))
df["is_test"] = np.zeros(len(df[target]))
df.loc[np_train, ["is_train"]] = 1
df.loc[np_valid, ["is_valid"]] = 1
df.loc[np_test, ["is_test"]] = 1


def NN_make_train_predict(df, inputs, channel = "inclusive", Njets = 2, Nlayers = options.Nlayers, Nneurons = options.Nneurons):

    NNname = "_".join([channel, str(Njets), "jets", str(Nlayers), "layers", str(Nneurons), "neurons"])

    print(NNname)

    if Njets == 0:
        inputs = [i for i in inputs if not 'jet' in i]
    elif Njets == 1:
        inputs = [i for i in inputs if not 'jet2' in i]

    for i in inputs:
        print(i)
        df[i] *= norm_factor(i)

    df_select = df

    df_select = df_select.loc[(df_select['Higgs_mass_gen'] >= min_mass) & (df_select['Higgs_mass_gen'] <= max_mass)]

    if channel != "inclusive":
        df_select = df_select.loc[(df_select['channel_reco'] == channel)]

    if Njets == 0:
        df_select = df_select.loc[df_select['jet1_pt_reco'] == 0]
    elif Njets == 1:
        df_select = df_select.loc[((df_select['jet2_pt_reco'] == 0) & (df_select['jet1_pt_reco'] > 0))]

    df_x_train = df_select.loc[(df_select['is_train'] == 1)].drop(columns=[k for k in df_select.keys() if not k in inputs])
    df_y_train = df_select.loc[(df_select['is_train'] == 1), [target]]
    df_x_valid = df_select.loc[(df_select['is_valid'] == 1), :].drop(columns=[k for k in df_select.keys() if not k in inputs])
    df_y_valid = df_select.loc[(df_select['is_valid'] == 1), [target]]

    print('Size of training set: ', len(df_x_train))
    print('Size of valid set: ', len(df_x_valid))

    if len(df_x_train) == 0 or len(df_x_valid) == 0:
        print("Empty set, aborting...")
        return None, False

    arr_x_train = np.r_[df_x_train]
    arr_y_train = np.r_[df_y_train[target]]
    arr_x_valid = np.r_[df_x_valid]
    arr_y_valid = np.r_[df_y_valid[target]]

    # Create model
    NN_model = Sequential()
    from tensorflow.keras.constraints import max_norm
    NN_model.add(Dense(Nneurons, activation="relu", input_shape=(len(df_x_train.keys()),), kernel_constraint=max_norm(2.)))
    for k in range(Nlayers):
        NN_model.add(Dense(Nneurons, activation="relu", kernel_constraint=max_norm(2.)))
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

    keras_callbacks = [keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=10, verbose=0)]

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


    plot_hist(history.history, NNname, xsize=8, ysize=12)

    # Plot predicted vs answer on a test sample
    plt.clf()
    plt.rcParams["figure.figsize"] = [16, 10]
    fig, ax = plt.subplots()
    predictions, answers = NN_model.predict(arr_x_valid), arr_y_valid
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
    fig.savefig("predicted_vs_answers_{}.png".format(NNname))


    # Plot NN output / mH and mTtot / mH histograms as function of mH
    plt.clf()
    plt.rcParams["figure.figsize"] = [10, 10]
    fig, ax = plt.subplots()
    plt.xlabel("Discriminator / Generated mass")

    NN_output_on_mH = predictions[:,0]/answers
    mTtot_on_mH = np.array(df_x_valid["mTtot_reco"] / norm_factor("mTtot_reco"))/answers
    
    h_NN = ax.hist(NN_output_on_mH, bins=200, range = [0,2], label = 'Deep NN output', alpha=0.5, color = 'C0')
    h_mTtot = ax.hist(mTtot_on_mH, bins=200, range = [0,2], label = 'Classic mTtot', alpha=0.5, color = 'C1')
    plt.legend()

    # Gaussian fits
    # def gaus(x,a,x0,sigma):
    #     return a*np.exp(-(x-x0)**2/(2*sigma**2))

    # from scipy.optimize import curve_fit
    # x, y = (h_NN[1][1:]+h_NN[1][:-1])/2,h_NN[0]
    # popt,pcov = curve_fit(gaus, x, y, p0=[1,1,1])
    # plt.plot(x,gaus(x,*popt), color = 'C0')
    # y_info = 0.5
    # x_info = .75
    # multialignment='left'
    # horizontalalignment='left'
    # verticalalignment='top'
    # ax.text(x_info, y_info,
    #         '\n'.join([
    #             "Deep NN",
    #             'Mean $ = {}$'.format(np.round(popt[1], 3)),
    #             '$\\sigma = {}$'.format(np.round(abs(popt[2]), 3))
    #         ]),
    #         transform = ax.transAxes, multialignment=multialignment, verticalalignment=verticalalignment, horizontalalignment=horizontalalignment)
    
    # x, y = (h_mTtot[1][1:]+h_mTtot[1][:-1])/2,h_mTtot[0]
    # popt,pcov = curve_fit(gaus, x, y, p0=[1,1,1])
    # plt.plot(x,gaus(x,*popt), color = 'C1')
    # y_info = 0.9
    # x_info = 0.025
    # multialignment='left'
    # horizontalalignment='left'
    # verticalalignment='top'
    # ax.text(x_info, y_info,
    #         '\n'.join([
    #             "Classic mTtot",
    #             'Mean $ = {}$'.format(np.round(popt[1], 3)),
    #             '$\\sigma = {}$'.format(np.round(abs(popt[2]), 3))
    #         ]),
    #         transform = ax.transAxes, multialignment=multialignment, verticalalignment=verticalalignment, horizontalalignment=horizontalalignment)
    
    plt.xlim(0,2)
    
    fig.savefig("NN_vs_mTtot_histos_{}.png".format(NNname))

    df["{}_output".format(NNname)] = NN_model.predict(df.drop(columns=[k for k in df_select.keys() if not k in inputs]))

    for i in inputs:
        df[i] *= 1./norm_factor(i)

    return df, True


channels = ["inclusive", "tt", "mt", "et", "mm", "em", "ee"]
Njets_list = [0, 1, 2]

for channel in channels:
    for Njets in Njets_list:
        df_out, valid = NN_make_train_predict(df, inputs, channel = channel, Njets = Njets,
                                              Nlayers = options.Nlayers, Nneurons = options.Nneurons)
        if valid:
            df = df_out

df.to_csv("{}.csv".format(output))
