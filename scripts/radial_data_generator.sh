#!/bin/bash

export MYPYTHON=/afs/cern.ch/work/c/camontan/public/miniconda3

source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev4cuda9/latest/x86_64-centos7-gcc62-opt/setup.sh
unset PYTHONHOME
unset PYTHONPATH
source $MYPYTHON/bin/activate
export PATH=$MYPYTHON/bin:$PATH

which python

python3 radial_data_generator.py $1 $2 $3 $4 $5 $6
