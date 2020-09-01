#!/usr/bin/env python3
# -*- coding: utf-8 -*-

mH_min = .090
mH_max = .8

from optparse import OptionParser
usage = "usage: %prog [options]"
parser = OptionParser(usage=usage)
parser.add_option("-s", "--small", dest = "small_test",
                  default = False, action = 'store_true')

(options,args) = parser.parse_args()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# NN structures
Nlayers_list = [k for k in range(2,11)]+[15]
Nneurons_list = [500, 1000, 1500, 2000]
bottleneck_list = ["", "_bottleneck"]

if options.small_test:
    Nlayers_list = Nlayers_list[:2]
    Nneurons_list = Nneurons_list[:2]
    bottleneck_list = bottleneck_list[:2]

# Load data
data_dir = "/data2/ltorterotot/ML/NN/latest_Delphes/"#latest"
file_basename = "PROD_{}_layers_{}_neurons{}.h5"

df = None
for Nlayers in Nlayers_list:
    for Nneurons in Nneurons_list:
        for bottleneck in bottleneck_list:
            if True:
                print("\t{} {} {}".format(Nlayers, Nneurons, bottleneck))
                _df = pd.read_hdf(
                    "/".join([
                        data_dir,
                        file_basename.format(str(Nlayers), str(Nneurons), bottleneck)
                    ])
                )
                if df is None:
                    df = _df
                else:
                    for k in _df:
                        if k not in df:
                            df[k] = _df[k]

df = df.loc[(df["Higgs_Mass_gen"] >= mH_min) & (df["Higgs_Mass_gen"] <= mH_max)]
df = df.loc[(df["is_valid"] == 1)]

# Get available channels and create the combined NN output
channels = list(set(df.channel_reco)) + ["lt", "ll"]

# Get missing NN structures due to duplications with bottleneck
# if "_bottleneck" in bottleneck_list:
#     for Nlayers in [2,3]:
#         for Nneurons in [2000]:
#             for channel in channels + ["inclusive"]:
#                 key = "{}_{}_layers_{}_neurons{}_output".format(channel, str(Nlayers), str(Nneurons), "_bottleneck")
#                 if key not in df.keys() and Nlayers in Nlayers_list:
#                     df[key] = df[key.replace("2000_neurons", "1000_neurons")]
            
                            
# Create the combined NN outputs
for Nlayers in Nlayers_list:
    for Nneurons in Nneurons_list:
        for bottleneck in bottleneck_list:
            df["{}_{}_layers_{}_neurons{}_output".format("combined6", str(Nlayers), str(Nneurons), bottleneck)] = df["{}_{}_layers_{}_neurons{}_output".format("inclusive", str(Nlayers), str(Nneurons), bottleneck)]
            df["{}_{}_layers_{}_neurons{}_output".format("combined3", str(Nlayers), str(Nneurons), bottleneck)] = df["{}_{}_layers_{}_neurons{}_output".format("inclusive", str(Nlayers), str(Nneurons), bottleneck)]
            for channel in channels:
                df.loc[(df.channel_reco == channel), "{}_{}_layers_{}_neurons{}_output".format("combined6", str(Nlayers), str(Nneurons), bottleneck)] = df.loc[(df.channel_reco == channel)]["{}_{}_layers_{}_neurons{}_output".format(channel, str(Nlayers), str(Nneurons), bottleneck)]
                if channel in ["tt"]:
                    df.loc[(df.channel_reco == channel), "{}_{}_layers_{}_neurons{}_output".format("combined3", str(Nlayers), str(Nneurons), bottleneck)] = df.loc[(df.channel_reco == channel)]["{}_{}_layers_{}_neurons{}_output".format(channel, str(Nlayers), str(Nneurons), bottleneck)]
                elif channel in ["mt", "et"]:
                    df.loc[(df.channel_reco == channel), "{}_{}_layers_{}_neurons{}_output".format("combined3", str(Nlayers), str(Nneurons), bottleneck)] = df.loc[(df.channel_reco == channel)]["{}_{}_layers_{}_neurons{}_output".format("lt", str(Nlayers), str(Nneurons), bottleneck)]
                elif channel in ["mm", "em", "ee"]:
                    df.loc[(df.channel_reco == channel), "{}_{}_layers_{}_neurons{}_output".format("combined3", str(Nlayers), str(Nneurons), bottleneck)] = df.loc[(df.channel_reco == channel)]["{}_{}_layers_{}_neurons{}_output".format("ll", str(Nlayers), str(Nneurons), bottleneck)]

channels.sort(reverse=True)
channels = ["inclusive", "combined6", "combined3"] + channels

if options.small_test:
    channels = ["inclusive", "combined3", "tt"]

import DL_for_HTT.post_training.utils as utils
import DL_for_HTT.post_training.macros as macros

for channel in channels:
    macros.mean_sigma_mae(df, channel, Nneurons_list, Nlayers_list, bottleneck_list, mH_min, mH_max)
    for bottleneck in bottleneck_list:
        for Nlayers in Nlayers_list:
            for Nneurons in Nneurons_list:
                macros.NN_responses(df, channel, Nneurons, Nlayers, bottleneck, mH_min, mH_max)
                macros.plot_pred_vs_ans(df, channel, Nneurons, Nlayers, bottleneck, mH_min, mH_max)
            
