# Base libraries
import math
import numpy as np
import scipy.integrate as integrate
from tqdm import tqdm
import pickle
import itertools

# Personal libraries
import sixtrackwrap as sx

import time

savepath = "/afs/cern.ch/work/c/camontan/public/loss_studies/data/"

min_turns = 100
max_turns = 100000
n_turn_samples = 500

turn_sampling = np.linspace(
    min_turns, max_turns, n_turn_samples, dtype=np.int_)[::-1]

d_r = 1.0
starting_step = 10  # USE IT CAREFULLY AS IT REQUIRES PRIOR KNOWLEDGE ON DA

batch_size = 1000000

# BASELINE COMPUTING
baseline_samples = 33
baseline_total_samples = baseline_samples ** 3

# Building vectors
alpha_preliminary_values = np.linspace(-1.0, 1.0, baseline_samples)
alpha_values = np.arccos(alpha_preliminary_values) / 2
theta1_values = np.linspace(0.0, np.pi * 2.0, baseline_samples, endpoint=False)
theta2_values = np.linspace(0.0, np.pi * 2.0, baseline_samples, endpoint=False)

d_preliminar_alpha = alpha_preliminary_values[1] - alpha_preliminary_values[0]
d_theta1 = theta1_values[1] - theta1_values[0]
d_theta2 = theta2_values[1] - theta2_values[0]

alpha_mesh, theta1_mesh, theta2_mesh = np.meshgrid(
    alpha_values, theta1_values, theta2_values, indexing='ij')

alpha_flat = alpha_mesh.flatten()
theta1_flat = theta1_mesh.flatten()
theta2_flat = theta2_mesh.flatten()

# Data generation
engine = sx.radial_scanner(alpha_flat, theta1_flat,
                           theta2_flat, d_r, starting_step=starting_step)

engine.scan(max_turns, min_turns, batch_size=batch_size)

engine.save_values(savepath + "big_scan.pkl")
