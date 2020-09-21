# Big data generator for Hénon fitting procedures

# Base libraries
import numpy as np
import sys

# Personal library
import henon_map as hm

# Set savepath (ON AFS FOR HTCONDOR!!!)
savepath = "/afs/cern.ch/work/c/camontan/public/loss_studies/data/"
tempdir = "/afs/cern.ch/work/c/camontan/public/loss_studies/tmp/"

# Load parameters
epsilon         = float(sys.argv[1])
top             = float(sys.argv[2])
steps           = int(sys.argv[3])
starting_radius = float(sys.argv[4])
max_turns       = int(sys.argv[5])

# Building the data
engine = hm.uniform_scan.generate_instance(
    epsilon,
    top,
    steps,
    starting_radius,
    tempdir=tempdir
)

engine.scan(max_turns)

# Saving
engine.save_values(savepath + "unif_henon_eps_{}.hdf5".format(int(epsilon)))
