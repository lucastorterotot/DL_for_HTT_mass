import postNN.utils as utils

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
        
        predictions = np.r_[df2["{}_{}_layers_{}_neurons{}_output".format(channel, str(Nlayers), str(Nneurons), bottleneck)]]
        mHs = np.r_[df2["Higgs_mass_gen"]]
        values = predictions/mHs
        
        fig, ax = plt.subplots()
        fig.suptitle("{} NN response in range {} to {} TeV\n for {} layers of {} neurons{}".format(channel, mHrange[0], mHrange[1], str(Nlayers), str(Nneurons), " with bottleneck" if bottleneck != "" else ""))
        plt.xlabel("Discriminator / Generated mass")
        plt.ylabel("Number of events")
        
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
        
        fig.savefig("NN_response_{}{}".format("_".join([channel, str(Nlayers), "layers", str(Nneurons), "neurons", str(mHrange[0]), "to", str(mHrange[1]), "TeV"]), bottleneck).replace(".", "_")+".png")
        
        plt.close('all')
        
    fig, ax = plt.subplots()
    fig.suptitle("{} NN response for {} layers of {} neurons{}".format(channel, str(Nlayers), str(Nneurons), " with bottleneck" if bottleneck != "" else ""))
    plt.xlabel("Generated mass (TeV)")
    plt.ylabel("NN output / Generated mass")
    
    plt.plot([mH_min, mH_max], [1,1], color='C3')
    
    ax.errorbar(
        xpos, means, xerr = xerr, yerr = sigmas,
        marker='+', markersize=4, linewidth=.4, elinewidth=1,
        fmt=' ', capsize = 3, capthick = .4)
    
    plt.ylim(0.7,1.7)
    plt.xlim(mH_min, mH_max)
    
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

def plot_pred_vs_ans(df, channel, Nneurons, Nlayers, bottleneck, mH_min, mH_max):
    _df = filter_channel(df, channel)
    # Plot predicted vs answer on a test sample
    plt.clf()
    fig, ax = plt.subplots()

    predictions, answers = np.r_[_df["{}_{}_layers_{}_neurons{}_output".format(channel, str(Nlayers), str(Nneurons), bottleneck)]], np.r_[_df["Higgs_mass_gen"]]

    # Calculate the point density
    from matplotlib.colors import Normalize
    from scipy.interpolate import interpn
    data , x_e, y_e = np.histogram2d( answers, predictions, bins = [50,50], density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([answers, predictions]).T , method = "splinef2d", bounds_error = False)
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = answers[idx], predictions[idx], z[idx]
    ax.scatter(x,y, c=z, edgecolor='', label="Test")
    ax.plot(answers, answers, color="C3")
    plt.xlabel("Generated Higgs Mass (TeV)")
    plt.ylabel("Predicted Higgs Mass (TeV)")
    
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
    x, y = answers, np.r_[predictions]
    try:
        result = spo.leastsq(residual, p0, args=(y, x), full_output=True)
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
    except:
        #import pdb; pdb.set_trace()
        print("No extrapolation possible")
    
    #plt.show()
    plt.xlim(mH_min, mH_max)
    plt.ylim(-.1, 1)

    fig.savefig(
        "predicted_vs_answers_{}_{}_layers_{}_neurons{}.png".format(
            channel, Nlayers, Nneurons, bottleneck
        )
    )
