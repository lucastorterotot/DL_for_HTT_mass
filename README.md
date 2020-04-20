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