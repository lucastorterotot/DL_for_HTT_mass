# DL for HTT mass reconstruction

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

## HTT events generation with Delphes
Generate HTT events. It takes roughly 1 hour for 100000 events, try with 1000:
```
cd $DL_for_HTT/Event_generation_with_Delphes
gen_HTT_at_mass -m <mh in GeV> -N <number of events to generate>
```

## HTT events from FastSim NanoAOD analysis
Instead of Delphes, one can use NanoAOD from CMS FastSim.

Get the root NanoAOD input files from  FastSim and go in the directory in which they are stored. A dedicated script has been designed:
```
cd $DL_for_HTT/HTT_analysis_FastSim_NanoAOD
./get_all_NanoAODSIM_to_data2 -u <YOUR_LYOSERV_USERNAME>
```
On `lyovis10`, it creates a directory `/data2/${USER}/ML/HTT_analysis_FastSim_NanoAOD` and run `rsync` on `/gridgroup/cms/htt/shared_files/Data/NanoAODSIM/nevents_*` from `lyoserv`.

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

Now you need the `tf` environment from `conda`:
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

A dedicated script to run all the commands described in this section is provided:
```
cd $DL_for_HTT/HTT_analysis_FastSim_NanoAOD
./convert_NanoAODSIM_in_data2
```
This script will ensure to have a good naming scheme so that any event can be found back in the original root file.

## Prepare data for the NN
One has to define which data the NN will be trained on, which will be kept for testing, and so on. To do so, a dedicated script is provided:
```
analyzed_events_to_NN <input h5 file from previous step> <output_name>
```
Options are available for this script:

- `-m` (`min_mass`), minimum mass point to consider;
- `-M` (`max_mass`), maximum mass point to consider;
- `-T` (`TeV_switch`), converts GeV to TeV (make sure the previous options are then given in the correct unit);
- `-t` (`train_frac`), training fraction in the dataset;
- `-v` (`valid_frac`), validation fraction in the dataset, testing will be the remaining;
- `-r` (`random_seed`), random seed to use for splitting the dataset into train, valid and test parts;
- `-F` (`Flat`), wether to make a flat target distribution or not.

## Train the deep NN
Go in the NN directory and activate the conda environment if not already done.

You can run as a test
```
NN_trainer -L 1 -N 1 -o TEST <input h5 file from previous step>
```
This will run a training on the events stored in the input h5 file from previous step with 1 (`-L`) hidden layer containing 1 (`-N`) neuron, so this would be quite quick.
Output json and h5 files containing the NNs structure will have a name containing `TEST` (`-o`), the channel the NN has been trained on, the number of hidden layers and the base number of neurons per layer.
