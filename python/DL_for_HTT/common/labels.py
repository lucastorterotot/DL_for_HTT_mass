labels = {
    'selected' : {
        'fr' : "Sélectionné",
        'en' : "Selected",
    },
    'DNN' : {
        'fr' : "Réseau de neurones",
        'en' : "Neural Network",
    },
    'XGB' : {
        'fr' : "XGBoost",
        'en' : "XGBoost",
    },
    'full' : {
        'fr' : "toute masse",
        'en' : "any mass",
    },
    'low' : {
        'fr' : "basse masse",
        'en' : "low mass",
    },
    'medium' : {
        'fr' : "moyenne masse",
        'en' : "medium mass",
    },
    'high' : {
        'fr' : "haute masse",
        'en' : "high mass",
    },
    'mse' : {
        'fr' : r"MSE (GeV$^2$)",
        'en' : r"MSE (GeV$^2$)",
    },
    'mae' : {
        'fr' : "MAE (GeV)",
        'en' : "MAE (GeV)",
    },
    'mape' : {
        'fr' : "MAPE",
        'en' : "MAPE",
    },
    'NPUshift' : {
        'fr' : r"Variation de $N_{{\rm PU}}$",
        'en' : r"$N_{{\rm PU}}$ shift",
    },
    'All inputs' : {
        'fr' : "Toutes",
        'en' : "All",
    },
    'Other inputs sets' : {
        'fr' : "Sous-ensembles",
        'en' : "Subsets",
    },
    'No Npu' : {
        'fr' : r"Sans $N_{{\rm PU}}$",
        'en' : r"No $N_{{\rm PU}}$",
    },
    'No Nnu' : {
        'fr' : r"Sans $N_{{\nu}}^{{\rm reco}}$",
        'en' : r"No $N_{{\nu}}^{{\rm reco}}$",
    },
    'No jr' : {
        'fr' : "Sans AHA",
        'en' : "No RHA",
    },
    'No jets' : {
        'fr' : "Sans jets",
        'en' : "No jets",
    },
    'No mT' : {
        'fr' : r"Sans $m_{{\rm T}}$",
        'en' : r"No $m_{{\rm T}}$",
    },
    'No METcov' : {
        'fr' : "Sans METcov",
        'en' : "No METcov",
    },
    'GenHiggsMassGeV' : {
        'fr' : "Masse générée du Higgs (GeV)",
        'en' : "Generated Higgs Mass (GeV)",
    },
    'ModelPredGeV' : {
        'fr' : "Prédiction du modèle (GeV)",
        'en' : "Predicted Mass (GeV)",
    },
    'median_diff' : {
        'fr' : "Écart sur la médiane",
        'en' : "Median difference",
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
    'relative_resolution' : {
        'fr' : "Résolution relative (%)",
        'en' : "Relative resolution (%)",
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
    'Density' : {
        'fr' : "Densité",
        'en' : "Density",
    },
    'eta' : {
        'fr' : r"Taux d'apprentissage ($\eta$)",
        'en' : r"Learning Rate ($\eta$)",
    },
    'max_depth' : {
        'fr' : "Profondeur maximale",
        'en' : "Maximum Depth",
    },
    '1sig_width' : {
        'fr' : r"Largeur à $1\sigma$",
        'en' : r"$1\sigma$ width",
    },
    '1sig_calibr_width' : {
        'fr' : r"Largeur à $1\sigma$ calibrée",
        'en' : r"$1\sigma$ calibrated width",
    },
    '2sig_width' : {
        'fr' : r"Largeur à $2\sigma$",
        'en' : r"$2\sigma$ width",
    },
    '2sig_calibr_width' : {
        'fr' : r"Largeur à $2\sigma$ calibrée",
        'en' : r"$2\sigma$ calibrated width",
    },
    'gu' : {
        'fr' : "Glorot uniforme",
        'en' : "Glorot uniform",
    },
    'gn' : {
        'fr' : "Glorot normale",
        'en' : "Glorot normal",
    },
    'u' : {
        'fr' : "Uniforme",
        'en' : "Uniform",
    },
    'n' : {
        'fr' : "Normale",
        'en' : "Normal",
    },
    'Nlayers' : {
        'fr' : "couches cachées",
        'en' : "Hidden Layers",
    },
    'Nneurons' : {
        'fr' : "neurones",
        'en' : "Neurons per layer",
    },
    'relu' : {
        'fr' : "ReLU",
        'en' : "ReLU",
    },
    'elu' : {
        'fr' : "ELU",
        'en' : "ELU",
    },
    'selu' : {
        'fr' : "SELU",
        'en' : "SELU",
    },
    'softplus' : {
        'fr' : "Softplus",
        'en' : "Softplus",
    },
}

for key in ["Adam", "Adadelta", "SGD"]:
    labels[key] = {
        'fr' : key,
        'en' : key,
    }

labels["Higgs_mass_gen"] = labels["GenHiggsMassGeV"]
labels["predictions"] = labels["ModelPredGeV"]

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
