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

max_turns = 100000

top = 35.0
samples = 35
starting_radius = 10.0  # USE IT CAREFULLY AS IT REQUIRES PRIOR KNOWLEDGE ON DA

engine = sx.uniform_scanner(
    top,
    samples,
    starting_radius
)
engine.scan(max_turns=max_turns)
engine.save_values(savepath + "big_uniform_scan.pkl")
