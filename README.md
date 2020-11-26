# Machine Learning for Higgs bosons mass reconstruction in di-tau events

[![Delphes](https://img.shields.io/badge/Delphes-3.4.2-red.svg)](https://cp3.irmp.ucl.ac.be/projects/delphes)
[![Pythia](https://img.shields.io/badge/Pythia-8.235-blue.svg)](http://home.thep.lu.se/Pythia/)

## Installation

### This repository
Fork the repository and clone it on your machine
```
mkdir -p <YOUR_DIRECTORY_NAME>
git clone <YOUR_FORKED_REPOSITORY> ./<YOUR_DIRECTORY_NAME>
cd <YOUR_DIRECTORY_NAME>
git remote add lucas git@github.com:lucastorterotot/DL_for_HTT_mass.git
```
Run the provided installation script to ensure setting variables properly:
```
./install
```

### Delphes 3.4.2
```
mkdir -p $DL_for_HTT/Delphes && cd $DL_for_HTT/Delphes
wget http://cp3.irmp.ucl.ac.be/downloads/Delphes-3.4.2.tar.gz && tar -zxf Delphes-3.4.2.tar.gz
cd Delphes-3.4.2
make
```

### Pythia 8.235
```
mkdir -p $DL_for_HTT/Pythia8 && cd $DL_for_HTT/Pythia8
wget http://home.thep.lu.se/~torbjorn/pythia8/pythia8235.tgz && tar xzvf pythia8235.tgz
cd pythia8235 && ./configure --prefix=$(pwd)
make install
export PYTHIA8=$(pwd)
cd $DL_for_HTT/Delphes/Delphes-3.4.2/ && make HAS_PYTHIA8=true
```

### Run a test
```
cd $DL_for_HTT/Delphes/Delphes-3.4.2
./DelphesPythia8 cards/delphes_card_CMS.tcl examples/Pythia8/configNoLHE.cmnd delphes_nolhe.root
```

## HTT events generation

### Using Delphes
Generate HTT events. It takes roughly 1 hour for 100000 events, try with 1000:
```
cd $DL_for_HTT/Event_generation_with_Delphes
gen_HTT_at_mass -m <mh in GeV> -N <number of events to generate>
```

### Using FastSim
Instead of Delphes, one can use NanoAOD from CMS FastSim. See [this repository](https://github.com/lucastorterotot/cmssw/tree/HTT_generator).

## Analyze the samples and get a `.txt` output table
To run the `root` to `txt` analysis on a file, do
```
HTT_Delphes_tree_analysis <FILE> <OUTPUT_NAME>
```
or, if FastSim was used,
```
HTT_FastSim_NanoAOD_tree_analysis <FILE> <OUTPUT_NAME>
```
Then you have a table in `OUTPUT_NAME.txt` that you can import in a python script using `numpy`, `pandas`, etc.

Now you would need the `tf` environment from `conda` on `lyovis10`:
```
conda activate tf
```

To merge the different `.txt` outputs (in principle one by root file, corresponding to one mass point and one amount of events) one can use the `txt_merger` script:
```
txt_merger -o <OUTPUT_NAME> <LIST_OF_INPUTS>
```
Make sure the columns are the same! To avoid typing all the files names, if for example you want to merge all `Htt_XXX_NanoAODSIM.txt` files where `XXX` is the Higgs mass into `Htt_merged_NanoAODSIM.txt` then you can do
```
txt_merger -o Htt_merged_NanoAODSIM.txt $(ls | grep Htt_.*_NanoAODSIM.txt | grep -ve Htt_merged_)
```

Once this is done, this txt file can be converted to `hdf5` format as it uses less disk space:
```
txt_to_hdf5 Htt_merged_NanoAODSIM.txt Htt_merged_NanoAODSIM
```
Then you may delete `root` and `txt` files
```
find . -type f -iname Htt_\*_NanoAODSIM\*.{root,txt} -delete
```

## Prepare data for the NN
One has to define which data the NN will be trained on, which will be kept for testing, and so on. To do so, a dedicated script is provided:
```
analyzed_events_to_NN <input h5 file from previous step>
```
It will update the previous file with new information (train, valid or test event).

Options are available for this script:

- `-m` (`min_mass`), minimum mass point to consider;
- `-M` (`max_mass`), maximum mass point to consider;
- `-t` (`train_frac`), training fraction in the dataset;
- `-v` (`valid_frac`), validation fraction in the dataset, testing will be the remaining;
- `-r` (`random_seed`), random seed to use for splitting the dataset into train, valid and test parts;
- `-F` (`Flat`), wether to make a flat target distribution or not.

## Train ML models
### Deep Neural Networks
You can run as a test
```
NN_trainer -L 1 -N 1 -o TEST -i no_METcov <input h5 file from previous step>
```
This will run a training on the events stored in the input h5 file from previous step with 1 (`-L` or `--Nlayers`) hidden layer containing 1 (`-N` or `--Nneurons`) neuron, so this would be quite quick (and not really a good model).

The used list of the model inputs will be the `no_METcov` list (`-i` or `--model_inputs`). Available inputs lists are stored in `$DL_for_HTT/python/DL_for_HTT/common/model_inputs/`.

In the working directory, a file named `inputs_for_models_in_this_dir.py` will make it possible to restore the list of inputs. You may create one directory for each group of models using a different list of inputs.

Output `.json` and `.h5` files containing the NN structure will have a name containing `TEST` (`-o` or `--output`), the channel the NN has been trained on, the number of hidden layers and the base number of neurons per layer.

Others options are:

- `-g` or `--gpu`: the GPU unit to use. If several are available this makes it possible to give one GPU for two parallelized processes;
- `-l` or `--loss`: the loss function to use for training;
- `-O` or `--optimizer`: the optimizer to use for training, from `tensorflow.keras.optimizers`;
- `-w` or `--w_init_mode`: the weight initialisation mode;
- `-m` or `--minmass`: the minimum Higgs mass to consider for training;
- `-M` or `--maxmass`: the maximum Higgs mass to consider for training;
- `-a` or `--last_activation`: the activation function to use for the output layer;
- `-c` or `--channels`: the channels to train the constructed model on (one model obtained for each channel). It can be tt, mt, et, mm, em, ee, lt, ll, inclusive.

### XGBoost regressor
You can run as a test
```
xgboost_trainer -d 1 -n 1 -o TEST -i no_METcov <input h5 file from previous step>
```
This will run a training on the events stored in the input h5 file from previous step with maximum 1 (`-n` or `--n_estimators`) of maximum 1 (`-d` or `--max_depth`) depth, so this would be quite quick (and not really a good model).

The used list of the model inputs will be the `no_METcov` list (`-i` or `--model_inputs`). Available inputs lists are stored in `$DL_for_HTT/python/DL_for_HTT/common/model_inputs/`.

In the working directory, a file named `inputs_for_models_in_this_dir.py` will make it possible to restore the list of inputs. You may create one directory for each group of models using a different list of inputs.

Output `.json` file containing the model structure will have a name containing `TEST` (`-o` or `--output`) and other information on the model.

Others options are:

- `-e` or `--eta`: the learning rate;
- `-s` or `--early_stopping_rounds`: the number of rounds to wait before stopping training once evaluation reaches a plateau;
- `-E` or `--eval`: the evaluation metric for the model (used for the early stopping);
- `-g` or `--gamma`: minimum loss reduction required to make a further partition on a leaf node of the tree;
- `-w` or `--min_child_weight`: minimum sum of instance weight (hessian) needed in a child;
- `-j` or `--n_jobs`: the number of parallel threads used to run xgboost;
- `-O` or `--objective`: the loss function to be minimized;
- `-m` or `--minmass`: the minimum Higgs mass to consider for training;
- `-M` or `--maxmass`: the maximum Higgs mass to consider for training;
- `-c` or `--channels`: the channels to train the constructed model on (one model obtained for each channel). It can be tt, mt, et, mm, em, ee, lt, ll, inclusive.

## Test the models
A dedicated script, `ml_plotter`, makes it easy to get plots using the trained models:
```
ml_plotter --model <JSON FILE FOR MODEL> --events <H5 FILE CONTAINING EVENTS TO USE>
```
This will proceed all the possible plots for the provided model on the `test` subsample from the `h5` file, on the inclusive channel (i.e. all at once).
The plots to proceed can be reduced with a comma separated list given to the `--plots` option. The subsample to use (`train`, `valid`, `test`, `all`, `any`) can also be set with `--subsample`.