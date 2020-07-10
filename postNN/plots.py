#!/usr/bin/env python3
# -*- coding: utf-8 -*-

mH_min = .1
mH_max = .5

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
Nlayers_list = [2, 3, 4, 5, 10, 15]
Nneurons_list = [1000, 2000]

if options.small_test:
    Nlayers_list = Nlayers_list[:2]
    Nneurons_list = Nneurons_list[:2]

# Load data
data_dir = "/data2/ltorterotot/ML/NN/TeV_outputs"
file_basename = "PROD_{}_layers_{}_neurons.h5"

df = None
for Nlayers in Nlayers_list:
    for Nneurons in Nneurons_list:
        _df = pd.read_hdf(
            "/".join([
                data_dir,
                file_basename.format(str(Nlayers), str(Nneurons))
            ])
        )
        if df is None:
            df = _df
        else:
            for k in _df:
                if k not in df:
                    df[k] = _df[k]

df = df.loc[(df["Higgs_mass_gen"] >= mH_min) & (df["Higgs_mass_gen"] <= mH_max)]
df = df.loc[(df["is_valid"] == 1)]

# Get available channels and create the combined NN output
channels = list(set(df.channel_reco))

for Nlayers in Nlayers_list:
    for Nneurons in Nneurons_list:
        df["{}_{}_layers_{}_neurons_output".format("combined", str(Nlayers), str(Nneurons))] = df["{}_{}_layers_{}_neurons_output".format("inclusive", str(Nlayers), str(Nneurons))]
        for channel in channels:
            df.loc[(df.channel_reco == channel), "{}_{}_layers_{}_neurons_output".format("combined", str(Nlayers), str(Nneurons))] = df.loc[(df.channel_reco == channel)]["{}_{}_layers_{}_neurons_output".format(channel, str(Nlayers), str(Nneurons))]

channels.sort(reverse=True)
channels = ["inclusive", "combined"] + channels

if options.small_test:
    channels = channels[:3]

import postNN.utils as utils
import postNN.macros as macros

for channel in channels:
    macros.mean_sigma_mae(df, channel, Nneurons_list, Nlayers_list, mH_min, mH_max)
    for Nlayers in Nlayers_list:
        for Nneurons in Nneurons_list:
            macros.NN_responses(df, channel, Nneurons, Nlayers, mH_min, mH_max)
