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
cd $DL_for_HTT/Delphes
./DelphesPythia8 cards/delphes_card_CMS.tcl examples/Pythia8/configNoLHE.cmnd delphes_nolhe.root
```

## HTT events generation and analysis
Generate HTT events. It takes roughly 1 hour for 100000 events, try with 1000:
```
cd $DL_for_HTT/Event_generation_with_Delphes
DelphesPythia8 delphes_card_CMS.tcl event_gen_cfgs/Higgs_to_tau_tau.cmnd SM_HTT.root
```
Analyse them to get a NN-friendly input. It takes roughly 1 hour for 100000 generated events. Selections leave around 10 percent of them in the final output.
```
cd $DL_for_HTT/Delphes_to_NN/
HTT_Delphes_tree_analysis ../Event_generation_with_Delphes/SM_HTT.root SM_HTT
```
Then you have a table in `$DL_for_HTT/Delphes_to_NN/SM_HTT.txt` that you can import in a python script using `numpy`, `pandas`, etc.
