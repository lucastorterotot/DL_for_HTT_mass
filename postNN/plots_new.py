#!/usr/bin/env python3
# -*- coding: utf-8 -*-

mH_min = .090
mH_max = .8

from optparse import OptionParser
usage = "usage: %prog [options] <NN JSON file> <NN input file>"
parser = OptionParser(usage=usage)
# parser.add_option("-s", "--small", dest = "small_test",
#                   default = False, action = 'store_true')

(options,args) = parser.parse_args()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import model_from_json

# load json and create model
input_json = args[0]
NN_weights_path_and_file = input_json.split('/')
NN_weights_path_and_file[-1] = "NN_weights_{}".format(NN_weights_path_and_file[-1].replace('.json', '.h5'))
NN_weights_file = "/".join(NN_weights_path_and_file)

json_file = open(input_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(NN_weights_file)
print("Loaded model from disk:")
print("\t{}".format(input_json))

# Get infos on the trained NN
infos = NN_weights_path_and_file[-1]
infos = infos.replace('.h5', '')
infos = infos.replace('NN_weights_', '')

is_bottleneck = ("_bottleneck" == infos[-11:])

if is_bottleneck:
    infos = infos.replace('_bottleneck', '')

Nneurons = infos.split("_")[-2]
Nlayers = infos.split("_")[-4]
channel = infos.split("_")[-5]

print(
    "{} channel, {} hidden layers of {} neurons with{} bottleneck".format(
        channel,
        Nlayers,
        Nneurons,
        "" if is_bottleneck else "out",
    )
)

# Load NN input file
df = pd.read_hdf(args[1])

# evaluate loaded model on test data
inputs = [
    "tau1_pt_reco",
    "tau1_eta_reco",
    "tau1_phi_reco",
    "tau2_pt_reco",
    "tau2_eta_reco",
    "tau2_phi_reco",
    "jet1_pt_reco",
    "jet1_eta_reco",
    "jet1_phi_reco",
    "jet2_pt_reco",
    "jet2_eta_reco",
    "jet2_phi_reco",
    # "recoil_pt_reco",
    # "recoil_eta_reco",
    # "recoil_phi_reco",
    "MET_pt_reco",
    "MET_phi_reco",
    # "MET_covXX_reco",
    # "MET_covXY_reco",
    # "MET_covYY_reco",
    # "MET_significance_reco",
    "mT1_reco",
    "mT2_reco",
    "mTtt_reco",
    "mTtot_reco",
    ]

predictions = loaded_model.predict(df[inputs])

if True:

    df = df.loc[(df["Higgs_mass_gen"] <= 130) & (df["Higgs_mass_gen"] >= 110)]
    df = df.loc[(df["is_valid"] == 1)]
    # Plot predicted vs answer on a test sample
    plt.clf()
    plt.rcParams["figure.figsize"] = [16, 10]
    fig, ax = plt.subplots()
    predictions, answers = loaded_model.predict(df[inputs]), df["Higgs_mass_gen"]
    # Calculate the point density
    from matplotlib.colors import Normalize
    from scipy.interpolate import interpn
    # data , x_e, y_e = np.histogram2d( answers, predictions[:,0], bins = [50,50], density = True )
    # z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([answers, predictions[:,0]]).T , method = "splinef2d", bounds_error = False)
    # z[np.where(np.isnan(z))] = 0.0
    # # Sort the points by density, so that the densest points are plotted last
    # idx = z.argsort()
    # x, y, z = answers[idx], predictions[idx,0], z[idx]
    # ax.scatter(x,y, c=z, edgecolor='', label="Test")
    ax.scatter(answers, predictions)
    ax.plot(answers, answers, color="C3")
    plt.xlabel("Generated Higgs Mass (TeV)")
    plt.ylabel("Predicted Higgs Mass (TeV)")
    
    # # linear regression on trained output
    # xerr_for_reg = 1
    # yerr_for_reg = 1
    # # linear function to adjust
    # def f(x,p):
    #     a,b = p
    #     return a*x+b
    
    # # its derivative
    # def Dx_f(x,p):
    #     a,b = p
    #     return a
    
    # # difference to data
    # def residual(p, y, x):
    #     return (y-f(x,p))/np.sqrt(yerr_for_reg**2 + (Dx_f(x,p)*xerr_for_reg)**2)
    
    # # initial estimation
    # # usually OK but sometimes one need to give a different
    # # starting point to make it converge
    # p0 = np.array([0,0])
    # # minimizing algorithm
    # import scipy.optimize as spo
    # x, y = answers, np.r_[predictions][:,0]
    # try:
    #     result = spo.leastsq(residual, p0, args=(y, x), full_output=True)
    #     # optimized parameters a and b
    #     popt = result[0];
    #     # variance-covariance matrix
    #     pcov = result[1];
    #     # uncetainties on parameters (1 sigma)
    #     #uopt = np.sqrt(np.abs(np.diagonal(pcov)))
    #     x_aj = np.linspace(min(x),max(x),100)
    #     y_aj = popt[0]*np.linspace(min(x),max(x),100)+popt[1]
        
    #     ax.plot(x_aj, y_aj, color="C4")
    #     y_info = 0.95
    #     x_info = 0.025
    #     multialignment='left'
    #     horizontalalignment='left'
    #     verticalalignment='top'
    #     ax.text(x_info, y_info,
    #             '\n'.join([
    #                 '$f(x) = ax+b$',
    #                 '$a = {{ {0:.2e} }}$'.format(popt[0]),
    #                 '$b = {{ {0:.2e} }}$'.format(popt[1])
    #             ]),
    #             transform = ax.transAxes, multialignment=multialignment, verticalalignment=verticalalignment, horizontalalignment=horizontalalignment)
    # except:
    #     #import pdb; pdb.set_trace()
    #     print("No extrapolation possible")
        
    #plt.show()
    fig.savefig("predicted_vs_answers2.png")
    plt.close('all') 

# import pdb; pdb.set_trace()
# macros.NN_responses(df, channel, Nneurons, Nlayers, bottleneck, mH_min, mH_max)            
