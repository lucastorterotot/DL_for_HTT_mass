#!/bin/bash

gpu=$1

events=20000

for Nlayers in {2..5}
do
    if [[ $gpu == 0 ]]
    then
        Nneurons=1000
    else
        Nneurons=2000
    fi
    cd $DL_for_HTT/NN/
    ./NN2.py -E $events -L $Nlayers -N $Nneurons -o PROD -g $gpu
    if [[ $Nlayers != 2 ]] || [[ $Nneurons != 2000 ]]
    then
        ./NN2.py -E $events -L $Nlayers -N $Nneurons -o PROD -g $gpu -b
    fi
done

for Nlayers in 10 15
do
    if [[ $gpu == 0 ]]
    then
        Nneurons=2000
    else
        Nneurons=1000
    fi
    cd $DL_for_HTT/NN/
    ./NN2.py -E $events -L $Nlayers -N $Nneurons -o PROD -g $gpu
    ./NN2.py -E $events -L $Nlayers -N $Nneurons -o PROD -g $gpu -b
done
