# Big data generator for Hénon fitting procedures

# Personal library
import henon_map as hm

# For timing
import time

# Set savepath (ON AFS FOR HTCONDOR!!!)
savepath = "/afs/cern.ch/work/c/camontan/public/loss_studies/data/"

# Set sampling
samples = 30

# Set times
max_turns = 1000000
min_turns = 100

# Set parameters
starting_epsilon = 1.0
epsilon_step = 1.0

# 23h in seconds
max_execution_time = 23 * 60 * 60

# Start loops
time_start = time.time()
time_mid = time.time()

epsilon = starting_epsilon

while time.time() - time_start < max_execution_time:
    engine = hm.uniform_scan.generate_instance(
        epsilon,
        1.0,
        samples,
        starting_radius=0.1
    )
    engine.scan(max_turns)
    engine.save_values(savepath + "unif_henon_eps_{}.pkl".format(int(epsilon)))
    
    print("Done Epsilon {}, seconds passed: {}".format(
        epsilon, time.time() - time_start))
    print("Iteration took {} seconds".format(time.time() - time_mid))
    time_mid = time.time()
    
    epsilon += epsilon_step