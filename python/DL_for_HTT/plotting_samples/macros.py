import DL_for_HTT.post_training.utils as utils

import locale
import matplotlib.pyplot as plt
plt.rcdefaults()

import numpy as np

plt.rcParams["figure.figsize"] = [7, 7]
plt.rcParams['axes.formatter.use_locale'] = True

labels = {
    'GenHiggsMassGeV' : {
        'fr' : "Masse générée du Higgs (GeV)",
        'en' : "Generated Higgs Mass (GeV)",
    },
    'ModelPredGeV' : {
        'fr' : "Prédiction du modèle (GeV)",
        'en' : "Predicted Mass (GeV)",
    },
    'median' : {
        'fr' : "Médiane",
        'en' : "Median",
    },
    'average' : {
        'fr' : "Moyenne",
        'en' : "Average",
    },
    'Pred_on_HiggsGenMass' : {
        'fr' : "Prédiction du modèle / Masse générée du Higgs",
        'en' : "Predicted Mass / Generated Mass",
    },
    'Pred_minus_HiggsGenMassGeV' : {
        'fr' : "Prédiction du modèle - Masse générée du Higgs (GeV)",
        'en' : "Predicted Mass - Generated Mass (GeV)",
    },
    'Calibr_response' : {
        'fr' : "Réponse calibrée du modèle",
        'en' : "Calibrated Model Response",
    },
    'Nevents' : {
        'fr' : "Nombre d'événements",
        'en' : 'N events',
    },
    "Probability" : {
        'fr' : 'Probabilité',
        'en' : "Probability",
    },
    'Score' : {
        'fr' : 'Score',
        'en' : 'Score',
    },
    'Variable' : {
        'fr' : 'Variable',
        'en' : 'Variable',
    },
    'CutEfficiency' : {
        'fr' : 'Efficacité de sélection',
        'en' : 'Cut efficiency',
    },
}

channel_str = {
    "tt" : r"$\tau_{\rm{h}}\tau_{\rm{h}}$",
    "mt" : r"$\mu\tau_{\rm{h}}$",
    "et" : r"$e\tau_{\rm{h}}$",
    "mm" : r"$\mu\mu$",
    "em" : r"$e\mu$",
    "ee" : r"$ee$",
    "lt" : r"$\ell\tau_{\rm{h}}$",
    "ll" : r"$\ell\ell$",
    "any": "tous",
}

labels["Higgs_mass_gen"] = labels["GenHiggsMassGeV"]

vars_with_y_log_scale = [
    'tau1_pt_reco',
    'tau2_pt_reco',
]

def filter_channel(df, channel = None):
    df1 = df
    if channel in set(df['channel_reco']):
        df1 = df.loc[(df['channel_reco'] == channel)]
    elif channel == "lt":
        df1 = df.loc[(df['channel_reco'] == "mt") | (df['channel_reco'] == "et")]
    elif channel == "ll":
        df1 = df.loc[(df['channel_reco'] == "mm") | (df['channel_reco'] == "em") | (df['channel_reco'] == "ee")]
    return df1    

def analysis_cuts_efficiency(df, min_mass, max_mass, language, **kwargs):
    files = set(df.file.array)

    N_raw = {}
    files_for_masses = {}
    for f in files:
        mass = int(f.split("_")[1])
        if mass == 750: # bugged root file
            continue
        if not mass in N_raw.keys():
            N_raw[mass] = 0
            files_for_masses[mass] = []
        N_raw[mass] += int(f.split("_")[5])
        files_for_masses[mass].append(f)

    masses = [k for k in N_raw.keys()]
    masses.sort()

    N_raw = [N_raw[mass] for mass in masses]

    N_channels = {}
    for channel in ["tt", "mt", "et", "mm", "em", "ee"]:
        df1 = df.loc[(df['channel_reco'] == channel)]
        N_channels[channel] = []
        for mass in masses:
            N_passing = 0
            for f in files_for_masses[mass]:
                N_passing += len(df1.loc[(df1['file'] == f)])
            N_channels[channel].append(N_passing)

    masses = np.array(masses)
    N_raw = np.array(N_raw)
    for channel in N_channels:
        N_channels[channel] = np.array(N_channels[channel])

    N_channels["lt"] = N_channels["mt"] + N_channels["et"]
    N_channels["ll"] = N_channels["mm"] + N_channels["em"] + N_channels["ee"]
    N_channels["any"] = N_channels["tt"] + N_channels["lt"] + N_channels["ll"]

    fig, ax = plt.subplots()
    plt.xlabel(labels["GenHiggsMassGeV"][language])
    plt.ylabel(labels["CutEfficiency"][language])

    for channel in ["any", "tt", "mt", "et", "mm", "em", "ee"]:
        if language == 'en':
            channel_str['any'] = 'any'
        _x = masses
        _y = N_channels[channel]/N_raw
        Delta_mass = 10
        x = np.arange(_x[0], _x[-1]+2*Delta_mass, Delta_mass, dtype='float')
        y = np.zeros(len(x))
        for k in range(len(y)-1):
            y[k] = _y[(_x >= x[k]) & (_x < x[k+1])].mean()
        x += Delta_mass/2
        ax.plot(
            x, y,
            label = channel_str[channel],
        )

    plt.ylim(1e-4,1)
    plt.xlim(50, 800)
    ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.legend(loc='lower right')

    fig.savefig("analysis_cuts_efficiency{}.png".format("-en" if language=='en' else ""))
    
available_plots = {
    "analysis_cuts_efficiency" : analysis_cuts_efficiency,
}
