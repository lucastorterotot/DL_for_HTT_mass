import postNN.utils as utils

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = [7, 7]

def NN_responses(df, channel, Nneurons, Nlayers, mH_min, mH_max):

    if channel in ["inclusive", "combined"]:
        df1 = df
    else:
        df1 = df.loc[(df["channel_reco"] == channel)]
        
    means = []
    sigmas = []
    xpos = []
    xerr = []
    
    mHcuts = [.200, .350]
    mHranges = [[mH_min, mHcuts[0]]]
    for mHcut in mHcuts[1:]:
        mHranges.append([mHranges[-1][1], mHcut])
    mHranges.append([mHranges[-1][1], mH_max])
    for mHrange in mHranges:
        xpos.append((mHrange[1]+mHrange[0])/2)
        xerr.append((mHrange[1]-mHrange[0])/2)
                
        df2 = df1.loc[(df1["Higgs_mass_gen"] >= mHrange[0]) & (df1["Higgs_mass_gen"] <= mHrange[1])]
        
        predictions = np.r_[df2["{}_{}_layers_{}_neurons_output".format(channel, str(Nlayers), str(Nneurons))]]
        mHs = np.r_[df2["Higgs_mass_gen"]]
        values = predictions/mHs
        
        fig, ax = plt.subplots()
        fig.suptitle("{} NN response in range {} to {} TeV for {} layers of {} neurons".format(channel, mHrange[0], mHrange[1], str(Nlayers), str(Nneurons)))
        plt.xlabel("Discriminator / Generated mass")
        plt.ylabel("Probability")
        
        hist = ax.hist(values, bins=300, range = [0,3], label = 'Deep NN output', alpha=0.5, color = 'C0')
        x, popt = utils.make_gaussian_fit(hist)
        
        means.append(popt[1])
        sigmas.append(popt[2])
        
        plt.plot(x,utils.gaus(x,*popt), color = 'C0')
        y_info = 0.75
        x_info = 0.65
        multialignment='left'
        horizontalalignment='left'
        verticalalignment='top'
        ax.text(x_info, y_info,
                '\n'.join([
                    "Deep NN",
                    'Mean $ = {}$'.format(np.round(popt[1], 3)),
                    '$\\sigma = {}$'.format(np.round(abs(popt[2]), 3))
                ]),
                transform = ax.transAxes, multialignment=multialignment, verticalalignment=verticalalignment, horizontalalignment=horizontalalignment)
        
        plt.xlim(0,3)
        
        fig.savefig("NN_response_{}".format("_".join([channel, str(Nlayers), "layers", str(Nneurons), "neurons", str(mHrange[0]), "to", str(mHrange[1]), "TeV"])).replace(".", "_")+".png")
        
        plt.close('all')
        
    fig, ax = plt.subplots()
    fig.suptitle("{} NN response for {} layers of {} neurons".format(channel, str(Nlayers), str(Nneurons)))
    plt.xlabel("Generated mass (TeV)")
    plt.ylabel("NN output / Generated mass")
    
    plt.plot([mH_min, mH_max], [1,1], color='C3')
    
    ax.errorbar(
        xpos, means, xerr = xerr, yerr = sigmas,
        marker='+', markersize=4, linewidth=.4, elinewidth=1,
        fmt=' ', capsize = 3, capthick = .4)
    
    plt.ylim(0.7,1.7)
    plt.xlim(mH_min, mH_max)
    
    fig.savefig("NN_response_{}.png".format("_".join([channel, str(Nlayers), "layers", str(Nneurons), "neurons"])))
    plt.close('all')

def mean_sigma_mae(df, channel, Nneurons_list, Nlayers_list, mH_min, mH_max):
    for Nneurons in Nneurons_list:
        mean_sigma_mae_fct_Nlayers(df, channel, Nneurons, Nlayers_list, mH_min, mH_max)
    for Nlayers in Nlayers_list:
        mean_sigma_mae_fct_Nneurons(df, channel, Nneurons_list, Nlayers, mH_min, mH_max)

def mean_sigma_mae_fct_Nlayers(df, channel, Nneurons, Nlayers_list, mH_min, mH_max):
    mean_sigma_mae_fct(df, channel, Nlayers_list, mH_min, mH_max, fixed = "{} neurons per layer".format(str(Nneurons)), at = Nneurons, type = "n")

def mean_sigma_mae_fct_Nneurons(df, channel, Nneurons_list, Nlayers, mH_min, mH_max):
    mean_sigma_mae_fct(df, channel, Nneurons_list, mH_min, mH_max, fixed = "{} hidden layers".format(str(Nlayers)), at = Nlayers, type = "l")

def mean_sigma_mae_fct(df, channel, list, mH_min, mH_max, fixed = "?", at = 0, type = "?"):
                       
    if channel in ["inclusive", "combined"]:
        df1 = df
    else:
        df1 = df.loc[(df["channel_reco"] == channel)]
        
    means = []
    sigmas = []
    maes = []
    xpos = []

    for val in list:
        if type == "n":
            var = "{}_{}_layers_{}_neurons_output".format(channel, str(val), str(at))
        elif type == "l":
            var = "{}_{}_layers_{}_neurons_output".format(channel, str(at), str(val))
        predictions = np.r_[df1[var]]
        mHs = np.r_[df1["Higgs_mass_gen"]]
        values = predictions/mHs
        
        fig, ax = plt.subplots()

        hist = ax.hist(values, bins=200, range = [0,2], label = 'Deep NN output', alpha=0.5, color = 'C0')
        x, popt = utils.make_gaussian_fit(hist)
        
        means.append(popt[1])
        sigmas.append(popt[2])
        maes.append(abs(predictions - mHs).sum()/len(predictions))
        xpos.append(val)

    if type == "n":
        uxerr = .5
    elif type == "l":
        uxerr = 100
    xerr = [uxerr for x in xpos]

    fig, ax = plt.subplots()
    fig.suptitle("{} performances with {}".format(channel, fixed))
    if type == "n":
        plt.xlabel("Number of hidden layers")
    elif type == "l":
        plt.xlabel("Number of neurons per hidden layer")
    
    
    ax.errorbar(
        xpos, means, xerr = xerr, yerr = sigmas,
        marker='+', markersize=4, linewidth=.4, elinewidth=1,
        fmt=' ', capsize = 6, capthick = 1,
        color = "C0", label = "$\\sigma$")
    ax.errorbar(
        xpos, means, xerr = xerr, yerr = maes,
        marker='+', markersize=4, linewidth=.4, elinewidth=1,
        fmt=' ', capsize = 6, capthick = 1,
        color = "C3", label = "MAE")
    ax.legend(loc='upper right')

    xmin = int(xpos[0]-xerr[0])
    xmax = int(xpos[-1]+xerr[-1])+1
    
    plt.plot([xmin, xmax], [1,1], color='C3')

    plt.ylim(0.75,1.25)
    plt.xlim(xmin, xmax)
    
    if type == "n":
        fig.savefig("NN_mean_{}_at_fixed_{}_Nneurons.png".format(channel, str(at)))
    elif type == "l":
        fig.savefig("NN_mean_{}_at_fixed_{}_Nlayers.png".format(channel, str(at)))    
    plt.close('all')
