# NN settings

# Higgs mass range in GeV
min_mass = 50
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
    # "jet1_pt_reco",
    # "jet1_eta_reco",
    # "jet1_phi_reco",
    # "jet2_pt_reco",
    # "jet2_eta_reco",
    # "jet2_phi_reco",
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

inputs_from_Heppy = {
    "tau1_pt_reco" : "l1_pt",
    "tau1_eta_reco" : "l1_eta",
    "tau1_phi_reco" : "l1_phi",
    "tau2_pt_reco" : "l2_pt",
    "tau2_eta_reco" : "l2_eta",
    "tau2_phi_reco" : "l2_phi",
    "jet1_pt_reco" : "j1_pt",
    "jet1_eta_reco" : "j1_eta",
    "jet1_phi_reco" : "j1_phi",
    "jet2_pt_reco" : "j2_pt",
    "jet2_eta_reco" : "j2_eta",
    "jet2_phi_reco" : "j2_phi",
    "MET_pt_reco" : "met",
    "MET_phi_reco" : "metphi",
    "mT1_reco" : "l1_mt",
    "mT2_reco" : "l2_mt",
    "mTtt_reco" : "mt_tt",
    "mTtot_reco" : "mt_tot",
}
