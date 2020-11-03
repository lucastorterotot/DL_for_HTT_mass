import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import DL_for_HTT.common.NN_settings as NN_default_settings

def load_model_from_json(input_json):
    model_name = input_json.split('/')[-1].replace('.json', '')
    if  model_name[:3] == 'XGB':
        model_type = model_name.split("-")[0]
        import xgboost as xgb
        loaded_model = xgb.XGBRegressor()
        loaded_model.load_model(input_json)
        
        # ... -inclusive-max_depth-5-eta-0.1-n_estimators-500-es-5-gamma-0-min_child_weight-1-loss-rmse
        # Get infos on the trained XGB
        infos = model_name
        infos = infos.replace('.json', '')

        loss = infos.split("-")[-1]
        min_child_weight = infos.split("-")[-3]
        gamma = infos.split("-")[-5]
        early_stopping_rounds = infos.split("-")[-7]
        n_estimators = infos.split("-")[-9]
        eta = infos.split("-")[-11]
        max_depth = infos.split("-")[-13]
        channel = infos.split("-")[-15]
        
        print("Properties:")
        
        print(
            "\t{} channel, eta = {}, max_depth = {}, n_estimators = {}, early_stopping_rounds = {},".format(
                channel,
                eta,
                max_depth,
                n_estimators,
                early_stopping_rounds,
            )
        )
        print(
            "\t loss = {}, gamma = {}, min_child_weight = {}".format(
                loss,
                gamma,
                min_child_weight,
            )
        )
    else:
        model_type = 'DNN'
        from keras.models import model_from_json
        
        # load json and create model
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
    return loaded_model, model_type, model_name

def load_h5_file_and_predict(input_h5, loaded_model, model_type, inputs = NN_default_settings.inputs, target = NN_default_settings.target):
    df = pd.read_hdf(input_h5)
    if model_type == 'XGBoost':
        from xgboost import DMatrix
        df["predictions"] = loaded_model.predict(DMatrix(data = np.r_[df[inputs]], feature_names=inputs))
    else:
        df["predictions"] = loaded_model.predict(df[inputs])
    return df

from scipy.optimize import curve_fit

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def make_gaussian_fit(ax_hist):
    x, y = (ax_hist[1][1:]+ax_hist[1][:-1])/2, ax_hist[0]
    popt,pcov = curve_fit(gaus, x, y, p0=[1,1,1])
    return x, popt
