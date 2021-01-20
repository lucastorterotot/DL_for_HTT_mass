#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import DL_for_HTT.common.NN_settings as NN_default_settings

from optparse import OptionParser
usage = "usage: %prog [options] <NN JSON file> <NN input file>"
parser = OptionParser(usage=usage)
parser.add_option("-m", "--minmass", dest = "min_mass",
                  default = NN_default_settings.min_mass)
parser.add_option("-M", "--maxmass", dest = "max_mass",
                  default = NN_default_settings.max_mass)

(options,args) = parser.parse_args()

options.min_mass = float(options.min_mass)
options.max_mass = float(options.max_mass)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Name = args[0].split('/')[-1].replace('.json', '')
if  Name[:3] == 'XGB':
    model_type = 'XGBoost'
    import xgboost as xgb
    loaded_model = xgb.Booster()
    loaded_model.load_model(args[0])

    # XGB-XGB-inclusive-max_depth-20-eta-0.1-num_round-20.json
    # Get infos on the trained XGB
    infos = Name
    infos = infos.replace('.json', '')

    num_round = infos.split("-")[-1]
    eta = infos.split("-")[-3]
    max_depth = infos.split("-")[-5]
    channel = infos.split("-")[-7]

    print("Properties:")

    print(
        "\t{} channel, eta={}, max_depth={}, num_round={}".format(
            channel,
            eta,
            max_depth,
            num_round,
        )
    )
else:
    model_type = 'DNN'
    from keras.models import model_from_json

    # load json and create model
    input_json = args[0]
    NN_weights_path_and_file = input_json.split('/')
    NN_weights_path_and_file[-1] = "NN_weights-{}".format(NN_weights_path_and_file[-1].replace('.json', '.h5'))
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
    infos = infos.replace('NN_weights-', '')
    
    is_bottleneck = ("-bottleneck" == infos[-11:])

    bottleneck = ""
    if is_bottleneck:
        infos = infos.replace('-bottleneck', '')
        bottleneck = "-bottleneck"

    Nneurons = infos.split("-")[-2]
    Nlayers = infos.split("-")[-4]
    channel = infos.split("-")[-5]

    w_init_mode = infos.split("-")[-6]
    optimizer = infos.split("-")[-7]
    loss = infos.split("-")[-8]

    print("Properties:")

    print(
        "\t{} channel, {} hidden layers of {} neurons with{} bottleneck".format(
            channel,
            Nlayers,
            Nneurons,
            "" if is_bottleneck else "out",
        )
    )
    print(
        "\ttrained with {} optimizer, w_init {} and {} loss.".format(
            optimizer,
            w_init_mode,
            loss,
        )
    )

# Load NN input file
df = pd.read_hdf(args[1])
df = df.loc[df['is_test']==1]
# evaluate loaded model on test data
inputs = NN_default_settings.inputs
target = NN_default_settings.target
Name+="test"
if model_type == 'XGBoost':
    df["predictions"] = loaded_model.predict(xgb.DMatrix(data = np.r_[df[inputs]], feature_names=inputs))
else:
    df["predictions"] = loaded_model.predict(df[inputs])

import DL_for_HTT.post_training.macros as macros
macros.get_distributions(df, channel, Name)
macros.plot_pred_vs_ans(df, channel, Name, options.min_mass, options.max_mass)
macros.NN_responses(df, channel, Name, options.min_mass, options.max_mass)
if model_type == 'XGBoost':
    macros.feature_importance(loaded_model, inputs, Name)
