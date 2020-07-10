# DL for HTT mass reconstruction

[![Delphes](https://img.shields.io/badge/Delphes-3.4.2-red.svg)](https://cp3.irmp.ucl.ac.be/projects/delphes)
[![Pythia](https://img.shields.io/badge/Pythia-8.235-blue.svg)](http://home.thep.lu.se/Pythia/)

## Installation

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

## HTT events from FastSim NanoAOD analysis
Get the root NanoAOD input files from  FastSim and go in the directory in which they are stored.
To run on the files, if named `Htt_${mass}_NanoAODSIM.root`, do
```
for f in $(ls | grep Htt_.*_NanoAODSIM.root) ; do HTT_FastSim_NanoAOD_tree_analysis $f ${f%.*} ; done
```
Then you have a table in `Htt_${mass}_NanoAODSIM.txt` that you can import in a python script using `numpy`, `pandas`, etc.

You can merge the different mass points outputs by using
```
conda activate tf
txt_merger -o Htt_merged_NanoAODSIM.txt $(ls | grep Htt_.*_NanoAODSIM.txt | grep -ve Htt_merged_)
```
and convert it to hdf5 format using less disk space with
```
txt_to_hdf5 Htt_merged_NanoAODSIM.txt Htt_merged_NanoAODSIM
```
Then you may delete `root` and `txt` files
```
find . -type f -iname Htt_\*_NanoAODSIM.{root,txt} -delete
```

## Train NN
Go in the NN directory and activate the conda environment if not already done
```
cd $DL_for_HTT/NN
conda activate tf
```
If the hdf5 output files from the previous step are stored in `$DL_for_HTT/FastSim_NanoAOD_to_NN/nevents_${Nevt}/Htt_merged_NanoAODSIM.h5` you can run as a test
```
./NN2.py -E 10 -L 1 -N 1 -o TEST
```
This will run a training on the 10 (`-E`) events per mass point samples with 1 (`-L`) hidden layer containing 1 (`-N`) neuron. Output hdf5 files containing the NNs outputs will be named starting with `TEST` (`-o`).

If this runs properly, the full production cna be done in parallel on the two GPUs by using `NN_prods.sh`:
```
NN_prods.sh 0 & NN_prods.sh 1 & wait
```
This takes some time (~ 1.5 days at this point).
Outputs will be named `PROD_X_layers_Y_neurons.h5` and may take ~500 Mo each.
`X` will be in `[2, 3, 4, 5, 10, 15]` and `Y` in `[1000, 2000]`, so that's a total of 12 files i.e. 6 Go in total.

_NB_ For training, all GeV inputs are converted to TeV so that the NNs handle values mostly in `[0,1]`.

## Get plots about the NNs performances
Go in the post NN directory and activate the conda environment if not already done
```
cd $DL_for_HTT/postNN
conda activate tf
```
One script runs on provided outputs from previous step. To be kind with the disk space, these outputs are stored in `/data2`:
```
mkdir -p /data2/ltorterotot/ML/NN/TeV_outputs/
mv $DL_for_HTT/NN/*.h5 /data2/ltorterotot/ML/NN/TeV_outputs/
```
Then the plotting script can be tested with its small option enabled:
```
./plots.py -s
```
If everything works fine you may get *lots* of png outputs, eventually some `tex` files containing tables ready to be used in a document (to be implemented if wanted).
To get the full analysis of the NNs outputs, do not use the small option:
```
./plots.py
```