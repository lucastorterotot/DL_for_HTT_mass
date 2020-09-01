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
