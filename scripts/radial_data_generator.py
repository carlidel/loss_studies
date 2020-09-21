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
angular_samples = int(sys.argv[1])
radial_samples  = int(sys.argv[2])
max_turns       = int(sys.argv[3])
epsilon         = float(sys.argv[4])
max_radius      = float(sys.argv[5])
starting_radius = float(sys.argv[6])

# Building the data
alpha = np.linspace(0, np.pi/2, angular_samples)
theta1 = np.linspace(0, np.pi * 2, angular_samples, endpoint=False)
theta2 = np.linspace(0, np.pi * 2, angular_samples, endpoint=False)

engine = hm.radial_block.generate_instance(
    radial_samples,
    alpha,
    theta1,
    theta2,
    epsilon,
    max_radius,
    starting_radius,
    tempdir=tempdir
)

engine.scan(max_turns)

# Saving
engine.save_values(savepath + "henon_eps_{}.hdf5".format(int(epsilon)))
