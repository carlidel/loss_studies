# Big data generator for Hénon fitting procedures

# Base libraries
import numpy as np

# Personal library
import henon_map as hm

# For timing
import time

# Set savepath (ON AFS FOR HTCONDOR!!!)
savepath = "/afs/cern.ch/work/c/camontan/public/loss_studies/data"

# Set sampling
samples = 25
dr = 0.0025

# Set times
max_turns = 1000000
min_turns = 100

# Set parameters
starting_epsilon = 1.0
epsilon_step = 1.0

# 23h in seconds
max_execution_time = 23 * 60 * 60

# Building the data
alpha = np.linspace(0, np.pi/2, samples)
theta1 = np.linspace(0, np.pi * 2, samples, endpoint=False)
theta2 = np.linspace(0, np.pi * 2, samples, endpoint=False)

A, TH1, TH2 = np.meshgrid(alpha, theta1, theta2, indexing='ij')

A = A.flatten()
TH1 = TH1.flatten()
TH2 = TH2.flatten()

# Start loops
time_start = time.time()
time_mid = time.time()

epsilon = starting_epsilon

while time.time() - time_start < max_execution_time:
    engine = hm.radial_scan.generate_instance(
        dr,
        A, TH1, TH2,
        epsilon,
        0.1
    )
    engine.block_compute(max_turns, min_turns)
    engine.save_values(savepath + "henon_eps_{}.pkl".format(int(epsilon)))
    
    print("Done Epsilon {}, seconds passed: {}".format(
        epsilon, time.time() - time_start))
    print("Iteration took {} seconds".format(time.time() - time_mid))
    time_mid = time.time()
    
    epsilon += epsilon_step