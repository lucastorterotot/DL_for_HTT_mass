import DL_for_HTT.post_training.utils as utils

from DL_for_HTT.common.NN_settings import target

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = [7, 7]

def filter_channel(df, channel = None):
    df1 = df
    if channel in set(df['channel_reco']):
        df1 = df.loc[(df['channel_reco'] == channel)]
    elif channel == "lt":
        df1 = df.loc[(df['channel_reco'] == "mt") | (df['channel_reco'] == "et")]
    elif channel == "ll":
        df1 = df.loc[(df['channel_reco'] == "mm") | (df['channel_reco'] == "em") | (df['channel_reco'] == "ee")]
    return df1    

def NN_responses(df, channel, Nneurons, Nlayers, bottleneck, mH_min, mH_max):
    df1 = filter_channel(df, channel)
        
    medians_NN = []
    CL68s_NN_up = []
    CL68s_NN_do = []
    CL95s_NN_up = []
    CL95s_NN_do = []
    medians_mTtot = []
    CL68s_mTtot_up = []
    CL68s_mTtot_do = []
    CL95s_mTtot_up = []
    CL95s_mTtot_do = []
    xpos = []
    xerr = []
    
    mHcuts = np.arange(mH_min, mH_max, 10e-3) # [.200, .350]
    mHranges = [[mH_min, mHcuts[0]]]
    for mHcut in mHcuts[1:]:
        mHranges.append([mHranges[-1][1], mHcut])
    mHranges.append([mHranges[-1][1], mH_max])
    for mHrange in mHranges:
        mHrange[0] = np.round(mHrange[0],3)
        mHrange[1] = np.round(mHrange[1],3)
        
        df2 = df1.loc[(df1[target] >= mHrange[0]) & (df1[target] <= mHrange[1])]
        
        predictions = np.r_[df2["{}_{}_layers_{}_neurons{}_output".format(channel, str(Nlayers), str(Nneurons), bottleneck)]]
        if len(predictions) == 0:
            continue

        xpos.append((mHrange[1]+mHrange[0])/2)
        xerr.append((mHrange[1]-mHrange[0])/2)

        mTtots = np.r_[df2["mTtot_reco"]]
        mHs = np.r_[df2[target]]
        values_NN = predictions/mHs
        values_mTtot = mTtots/mHs
        
        values_NN = [v for v in values_NN]
        values_mTtot = [v for v in values_mTtot]
        values_NN.sort()
        values_mTtot.sort()

        try:
            medians_NN.append(values_NN[int(len(values_NN)/2)])
            medians_mTtot.append(values_mTtot[int(len(values_mTtot)/2)])
        except:
            import pdb; pdb.set_trace()

        above_NN = [v for v in values_NN if v >= medians_NN[-1]]
        below_NN = [v for v in values_NN if v <= medians_NN[-1]]
        above_mTtot = [v for v in values_mTtot if v >= medians_mTtot[-1]]
        below_mTtot = [v for v in values_mTtot if v <= medians_mTtot[-1]]

        above_NN.sort()
        below_NN.sort(reverse = True)
        above_mTtot.sort()
        below_mTtot.sort(reverse = True)

        CL68s_NN_up.append(above_NN[int(0.68 * len(above_NN))])
        CL68s_NN_do.append(below_NN[int(0.68 * len(below_NN))])
        CL95s_NN_up.append(above_NN[int(0.95 * len(above_NN))])
        CL95s_NN_do.append(below_NN[int(0.95 * len(below_NN))])
        CL68s_mTtot_up.append(above_mTtot[int(0.68 * len(above_mTtot))])
        CL68s_mTtot_do.append(below_mTtot[int(0.68 * len(below_mTtot))])
        CL95s_mTtot_up.append(above_mTtot[int(0.95 * len(above_mTtot))])
        CL95s_mTtot_do.append(below_mTtot[int(0.95 * len(below_mTtot))])
        
    fig, ax = plt.subplots()
    fig.suptitle("{} channel, {} layers of {} neurons{}".format(channel, str(Nlayers), str(Nneurons), " with bottleneck" if bottleneck != "" else ""))
    plt.xlabel("Generated mass (GeV)")
    plt.ylabel("Discriminator / Generated mass")
    
    ax.fill_between(
        xpos, CL95s_NN_do, CL68s_NN_do,
        color = "yellow", alpha = .5
    )
    ax.fill_between(
        xpos, CL68s_NN_up, CL95s_NN_up,
        color = "yellow", alpha = .5
    )
    ax.fill_between(
        xpos, CL68s_NN_do, CL68s_NN_up,
        color = "green", alpha = .5
    )
    ax.errorbar(
        xpos, medians_NN, xerr = xerr, #yerr = sigmas,
        marker='.', markersize=4, linewidth=0, elinewidth=1,
        fmt=' ', capsize = 3, capthick = 0, color = "black", label = "DNN",
    )
    # ax.errorbar(
    #     xpos, medians_NN, xerr = xerr, #yerr = sigmas,
    #     marker='+', markersize=4, linewidth=.4, elinewidth=1,
    #     fmt=' ', capsize = 3, capthick = .4, color = "black", #label = "DNN",
    # )

    ax.plot(
        xpos, CL95s_mTtot_do,
        color = "C7", #alpha = .5,
        dashes = [1,1],
    )
    ax.plot(
        xpos, CL95s_mTtot_up,
        color = "C7", #alpha = .5,
        dashes = [1,1],
    )
    ax.plot(
        xpos, CL68s_mTtot_do,
        color = "C7", #alpha = .5,
        dashes = [2,2],
    )
    ax.plot(
        xpos, CL68s_mTtot_up,
        color = "C7", #alpha = .5,
        dashes = [2,2],
    )
    ax.plot(
        xpos, medians_mTtot,
        color = "C7", #alpha = .5,
        #dashes = [2,1],
        label = "mTtot",
    )
    
    plt.plot([mH_min, mH_max], [1,1], color='C3')    

    plt.ylim(0,3)
    plt.xlim(mH_min, mH_max)

    ax.legend(loc='upper right')
    
    fig.savefig("NN_response_{}{}.png".format("_".join([channel, str(Nlayers), "layers", str(Nneurons), "neurons"]), bottleneck))
    plt.close('all')

def mean_sigma_mae(df, channel, Nneurons_list, Nlayers_list, bottleneck_list, mH_min, mH_max):
    for bottleneck in bottleneck_list:
        for Nneurons in Nneurons_list:
            mean_sigma_mae_fct_Nlayers(df, channel, Nneurons, Nlayers_list, bottleneck, mH_min, mH_max)
        for Nlayers in Nlayers_list:
            mean_sigma_mae_fct_Nneurons(df, channel, Nneurons_list, Nlayers, bottleneck, mH_min, mH_max)

def mean_sigma_mae_fct_Nlayers(df, channel, Nneurons, Nlayers_list, bottleneck, mH_min, mH_max):
    mean_sigma_mae_fct(df, channel, Nlayers_list, bottleneck, mH_min, mH_max, fixed = "{} neurons per layer".format(str(Nneurons)), at = Nneurons, type = "n")

def mean_sigma_mae_fct_Nneurons(df, channel, Nneurons_list, Nlayers, bottleneck, mH_min, mH_max):
    mean_sigma_mae_fct(df, channel, Nneurons_list, bottleneck, mH_min, mH_max, fixed = "{} hidden layers".format(str(Nlayers)), at = Nlayers, type = "l")

def mean_sigma_mae_fct(df, channel, list, bottleneck, mH_min, mH_max, fixed = "?", at = 0, type = "?"):
    df1 = filter_channel(df, channel)
        
    means = []
    sigmas = []
    maes = []
    xpos = []

    for val in list:

        if bottleneck == "_bottleneck":
            if (type == "l" and val == 2000 and at == 2) or (type == "n" and val == 2 and at == 2000):
                continue
            
        if type == "n":
            var = "{}_{}_layers_{}_neurons{}_output".format(channel, str(val), str(at), bottleneck)
        elif type == "l":
            var = "{}_{}_layers_{}_neurons{}_output".format(channel, str(at), str(val), bottleneck)
            
        predictions = np.r_[df1[var]]
        mHs = np.r_[df1[target]]
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
    fig.suptitle("{} performances with {}{}".format(channel, fixed, " and bottleneck" if bottleneck != "" else ""))
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
        fig.savefig("NN_mean_{}_at_fixed_{}_Nneurons{}.png".format(channel, str(at), bottleneck))
    elif type == "l":
        fig.savefig("NN_mean_{}_at_fixed_{}_Nlayers{}.png".format(channel, str(at), bottleneck))    
    plt.close('all')

def plot_pred_vs_ans(predictions, answers, channel, Nneurons, Nlayers, bottleneck, mH_min, mH_max):

    # Plot predicted vs answer on a test sample
    plt.clf()
    fig, ax = plt.subplots()

    import seaborn as sns
    sns.kdeplot(answers, predictions, cmap="viridis", n_levels=30, shade=True, bw=.15)

    ax.plot(answers, answers, color="C3")
    plt.xlabel("Generated Higgs Mass (GeV)")
    plt.ylabel("Predicted Higgs Mass (GeV)")
    
    #plt.show()
    plt.xlim(mH_min, mH_max)
    plt.ylim(mH_min, mH_max)

    fig.savefig(
        "predicted_vs_answers_{}_{}_layers_{}_neurons{}.png".format(
            channel, Nlayers, Nneurons, bottleneck
        )
    )
