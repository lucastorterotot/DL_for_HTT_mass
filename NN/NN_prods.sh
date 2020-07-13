#!/bin/bash

mkdir -p /data2/ltorterotot/ML/NN/latest
cd /data2/ltorterotot/ML/NN/latest

gpu=$1

input=ALL

for Nlayers in {2..5}
do
    if [[ $gpu == 0 ]]
    then
        Nneurons=1000
    else
        Nneurons=2000
    fi

    $DL_for_HTT/NN/NN2.py -i $input -L $Nlayers -N $Nneurons -o PROD -g $gpu
    if [[ $Nlayers != 2 ]] || [[ $Nneurons != 2000 ]]
    then
        $DL_for_HTT/NN/NN2.py -i $input -L $Nlayers -N $Nneurons -o PROD -g $gpu -b
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

    $DL_for_HTT/NN/NN2.py -i $input -L $Nlayers -N $Nneurons -o PROD -g $gpu
    $DL_for_HTT/NN/NN2.py -i $input -L $Nlayers -N $Nneurons -o PROD -g $gpu -b
done
