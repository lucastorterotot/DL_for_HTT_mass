# NN settings

# Higgs mass range in GeV
min_mass = 90
max_mass = 800

# channels to process
channels = "inclusive"

# NN structure
Nlayers = 3
Nneurons = 1000

# NN training
loss = "cosine_similarity"
optimizer = "Adadelta"
w_init_mode = "uniform"

# Dataset splitting
train_frac = 0.7
valid_frac = 0.2
random_seed = 2020

# Target and inputs
target = "Higgs_mass_gen"
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
    "MET_covXX_reco",
    "MET_covXY_reco",
    "MET_covYY_reco",
    # "MET_significance_reco",
    "mT1_reco",
    "mT2_reco",
    "mTtt_reco",
    "mTtot_reco",
    ]
