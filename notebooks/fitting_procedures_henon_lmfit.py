#!/usr/bin/env python
# coding: utf-8

# # General DA fittings and Fokker-Planck comparisons with lmfit(Hénon map)

# ## Imports

# In[2]:


# In[3]:


# In[1]:


# Base libraries
import math
import numpy as np
import scipy.integrate as integrate
from scipy.special import erf
import pickle
import itertools
from scipy.optimize import curve_fit

from numba import njit, prange

from tqdm.notebook import tqdm
import time
import matplotlib.pyplot as plt
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from math import gcd

import pandas as pd

from scipy.special import lambertw
from scipy.interpolate import interp1d

import os

# Personal libraries
import sixtrackwrap as sx
import crank_nicolson_numba.nekhoroshev as nk
import henon_map as hm

# Personal modules
import fit_utils as fit

# Let's get freaky
import multiprocessing


# *if we want quiet operations... even though it's not a good practice at all...*

# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


PROCESSES = 24


# In[4]:


def wrap_fit(f, turn, da, eps):
    r1, r2 = f(turn, da)
    return r1, r2, eps

def wrap_fp_fit(f, turn, loss, I0, I_max, k, eps):
    r1, r2 = f(turn, loss, 1.0, I0, I_max, 1, k, 0.26, "lstsqr")
    return r1, r2, eps


# ## Load data and setup original DA

# In[5]:


savepath = "../data/"

sigma = 0.25
turn_samples = 500
min_turns = 1000
max_turns = 5000000
turn_sampling = np.linspace(min_turns, max_turns, turn_samples, dtype=np.int)[::-1]
# Do we want to load ALL the Hénon data files or should we skip some?
skipper = 1

cut_point = 0.9

files = list(sorted(list(filter(lambda f: "henon_eps_" in f and "unif" not in f and "hdf5" in f, os.listdir(savepath))), key=lambda f: float(f[10: -5])))[::skipper]


# In[6]:


epsilons = []

real_DAs = []
real_DAs_err = []
gaussian_DAs = []

real_losses = []
gaussian_losses = []


# In[7]:


def callback(results):
    epsilons.append(results[0])
    real_DAs.append(results[1])
    real_DAs_err.append(results[2])
    d_r = results[3]
    gaussian_losses.append(results[4])
    
def preprocess(f):
    epsilon = float(f[10: -4])
    epsilons.append(epsilon)
    engine = hm.uniform_radial_scanner(savepath + f)
    #print(epsilon)
    d_r = engine.dr
    DA, DA_err = engine.compute_DA_standard(turn_sampling)
    print(epsilon, "Done DA")
    engine.assign_weights_from_file(savepath + "weights.hdf5")
    gaussian_losses.append(engine.compute_loss(turn_sampling, cut_point, True))
    print(epsilon, "Done Loss")
    return(epsilon, DA, DA_err, d_r, gaussian_losses)


# In[ ]:


with multiprocessing.Pool(PROCESSES) as pool:
    results = [pool.apply_async(preprocess, args=(f,), callback=callback) for f in files[:]]
    pool.close()
    pool.join()
    
TMP = sorted(zip(epsilons, real_DAs, real_DAs_err, gaussian_losses), key=lambda x: x[0])

epsilons = [T[0] for T in TMP]
real_DAs = [T[1] for T in TMP]
real_DAs_err = [T[2] for T in TMP]
gaussian_losses = [T[3] for T in TMP]


# ### Save data (since it takes a long time to generate)

# In[ ]:


with open("../data/henon_loss_data.pkl", "wb") as f:
    pickle.dump(
        {
            "sigma": sigma,
            "turn_samples": turn_samples,
            "min_turns": min_turns,
            "max_turns": max_turns,
            "epsilons": epsilons,
            "real_losses": real_losses,
            "gaussian_losses": gaussian_losses,
            "real_DAs": real_DAs,
            "real_DAs_err": real_DAs_err,
            "gaussian_DAs": gaussian_DAs
        },
        f
    )


# ### Load data (if it was already computed before)

# In[ ]:


with open("../data/henon_loss_data.pkl", "rb") as f:
    dictionary = pickle.load(f)
    sigma = dictionary["sigma"]
    turn_samples = dictionary["turn_samples"]
    min_turns = dictionary["min_turns"]
    max_turns = dictionary["max_turns"]
    epsilons = dictionary["epsilons"]
    real_losses = dictionary["real_losses"]
    gaussian_losses = dictionary["gaussian_losses"]
    real_DAs = dictionary["real_DAs"]
    real_DAs_err = dictionary["real_DAs_err"]
    gaussian_DAs = dictionary["gaussian_DAs"]


# ## Visualize the various DA and Loss plots

# In[ ]:


plt.figure()

for i in range(len(gaussian_losses)):
    plt.plot(turn_sampling, gaussian_losses[i], c="C1")


# In[ ]:


plt.figure()

for i in range(len(real_DAs)):
    plt.plot(turn_sampling, real_DAs[i], c="C0")


# ## General fitting of all the Hénon maps across all models

# In[ ]:


real_results_model_2 = []
real_finals_model_2 = []


# In[ ]:


temp_result = []
temp_final = []
temp_eps = []

def log_result(result):
    temp_result.append(result[0])
    temp_final.append(result[1])
    temp_eps.append(result[2])

with multiprocessing.Pool(PROCESSES) as pool:
    results = [pool.apply_async(wrap_fit, args=(fit.fit_model_2, turn_sampling[::-1], real_DAs[i][::-1], epsilons[i]), callback=log_result) for i in range(len(real_DAs))]
    pool.close()
    pool.join()
    
TMP = sorted(zip(temp_eps, temp_result, temp_eps), key=lambda x: x[0])

real_results_model_2 = [T[1] for T in TMP]
real_finals_model_2 = [T[2] for T in TMP]


# In[ ]:


k1 = []
k1_err = []
n0_1 = []
n0_1_err = [] 
rho1 = []
rho1_err = []
for i, eps in enumerate(epsilons):
    k1.append(real_results_model_2[i].params.get("k").value)
    k1_err.append(real_results_model_2[i].params.get("k").stderr)
    n0_1.append(real_results_model_2[i].params.get("n0").value)
    n0_1_err.append(real_results_model_2[i].params.get("n0").stderr)
    rho1.append(real_results_model_2[i].params.get("rho").value)
    rho1_err.append(real_results_model_2[i].params.get("rho").stderr)


# In[ ]:


with open("../data/henon_fit_2.pkl", 'wb') as f:
    pickle.dump((
        real_results_model_2,
        real_finals_model_2,
    ), f)


# In[ ]:


with open("../data/henon_fit_2.pkl", 'rb') as f:
    real_results_model_2, real_finals_model_2 = pickle.load(f)


# In[ ]:


# Lmfit
from lmfit import Minimizer, Parameters, report_fit

def fit_model_2_fixed_n0(turns, DA):
    params = Parameters()
    params.add("rho", value=1, min=0, vary=True)
    params.add("n0", value=1, vary=False)
    params.add("k", value=1, min=0, vary=True)
    minner = Minimizer(fit.model_2_lmfit, params, fcn_args=(turns, DA))
    result = minner.minimize(method="basinhopping")
    final = DA + result.residual
    return result, final


# In[ ]:


real_results_model_2_fixed_n0 = []
real_finals_model_2_fixed_n0 = []


# In[ ]:


temp_result = []
temp_final = []
temp_eps = []

def log_result(result):
    temp_result.append(result[0])
    temp_final.append(result[1])
    temp_eps.append(result[2])

with multiprocessing.Pool(PROCESSES) as pool:
    results = [pool.apply_async(wrap_fit, args=(fit_model_2_fixed_n0, turn_sampling[::-1], real_DAs[i][::-1], epsilons[i]), callback=log_result) for i in range(len(real_DAs))]
    pool.close()
    pool.join()
    
TMP = sorted(zip(temp_eps, temp_result, temp_eps), key=lambda x: x[0])

real_results_model_2_fixed_n0 = [T[1] for T in TMP]
real_finals_model_2_fixed_n0 = [T[2] for T in TMP]


# In[ ]:


k1_fix = []
k1_err_fix = []
rho1_fix = []
rho1_err_fix = []
for i, eps in enumerate(epsilons):
    k1_fix.append(real_results_model_2_fixed_n0[i].params.get("k").value)
    k1_err_fix.append(real_results_model_2_fixed_n0[i].params.get("k").stderr)
    rho1_fix.append(real_results_model_2_fixed_n0[i].params.get("rho").value)
    rho1_err_fix.append(real_results_model_2_fixed_n0[i].params.get("rho").stderr)


# In[ ]:


with open("../data/henon_fit_2_fix.pkl", 'wb') as f:
    pickle.dump((
        real_results_model_2_fixed_n0,
        real_finals_model_2_fixed_n0,
    ), f)


# In[ ]:


with open("../data/henon_fit_2_fix.pkl", 'rb') as f:
    real_results_model_2_fixed_n0, real_finals_model_2_fixed_n0 = pickle.load(f)


# In[ ]:


real_results_model_4 = []
real_finals_model_4 = []


# In[ ]:


temp_result = []
temp_final = []
temp_eps = []

def log_result(result):
    temp_result.append(result[0])
    temp_final.append(result[1])
    temp_eps.append(result[2])

with multiprocessing.Pool(PROCESSES) as pool:
    results = [pool.apply_async(wrap_fit, args=(fit.fit_model_4, turn_sampling[::-1], real_DAs[i][::-1], epsilons[i]), callback=log_result) for i in range(len(real_DAs))]
    pool.close()
    pool.join()
    
TMP = sorted(zip(temp_eps, temp_result, temp_eps), key=lambda x: x[0])

real_results_model_4 = [T[1] for T in TMP]
real_finals_model_4 = [T[2] for T in TMP]


# In[ ]:


k2 = []
k2_err = []
rho2 = []
rho2_err = []

for i, eps in enumerate(epsilons):
    k2.append(real_results_model_4[i].params.get("k").value)
    k2_err.append(real_results_model_4[i].params.get("k").stderr)
    rho2.append(real_results_model_4[i].params.get("rho").value)
    rho2_err.append(real_results_model_4[i].params.get("rho").stderr)


# In[ ]:


with open("../data/henon_fit_4.pkl", 'wb') as f:
    pickle.dump((
        real_results_model_4,
        real_finals_model_4,
    ), f)


# In[ ]:


with open("../data/henon_fit_4.pkl", 'rb') as f:
    real_results_model_4, real_finals_model_4 = pickle.load(f)


# In[ ]:


real_results_model_4_free = []
real_finals_model_4_free = []


# In[ ]:


temp_result = []
temp_final = []
temp_eps = []

def log_result(result):
    temp_result.append(result[0])
    temp_final.append(result[1])
    temp_eps.append(result[2])

with multiprocessing.Pool(PROCESSES) as pool:
    results = [pool.apply_async(wrap_fit, args=(fit.fit_model_4_free, turn_sampling[::-1], real_DAs[i][::-1], epsilons[i]), callback=log_result) for i in range(len(real_DAs))]
    pool.close()
    pool.join()
    
TMP = sorted(zip(temp_eps, temp_result, temp_eps), key=lambda x: x[0])

real_results_model_4_free = [T[1] for T in TMP]
real_finals_model_4_free = [T[2] for T in TMP]


# In[ ]:


k3 = []
k3_err = []
rho3 = []
rho3_err = []
n0_3 = []
n0_3_err = []
for i, eps in enumerate(epsilons):
    k3.append(real_results_model_4_free[i].params.get("k").value)
    k3_err.append(real_results_model_4_free[i].params.get("k").stderr)
    rho3.append(real_results_model_4_free[i].params.get("rho").value)
    rho3_err.append(real_results_model_4_free[i].params.get("rho").stderr)
    n0_3.append(real_results_model_4_free[i].params.get("n0").value)
    n0_3_err.append(real_results_model_4_free[i].params.get("n0").stderr)


# In[ ]:


with open("../data/henon_fit_4_free.pkl", 'wb') as f:
    pickle.dump((
        real_results_model_4_free,
        real_finals_model_4_free,
    ), f)


# In[ ]:


with open("../data/henon_fit_4_free.pkl", 'rb') as f:
    real_results_model_4, real_finals_model_4 = pickle.load(f)


# In[ ]:


plt.figure()
plt.errorbar(epsilons, rho1_fix, yerr=rho1_err_fix, linewidth=0, elinewidth=2, marker="x", label="Model 2, two parameters")
plt.errorbar(epsilons, rho1, yerr=rho1_err, linewidth=0, elinewidth=2, marker="x", label="Model 2, three parameters")
plt.errorbar(epsilons, rho2, yerr=rho2_err, linewidth=0, elinewidth=2, marker="x", label="Model 4, two parameters")
plt.errorbar(epsilons, rho3, yerr=rho3_err, linewidth=0, elinewidth=2, marker="x", label="Model 4, three parameters")

plt.yscale("log")

plt.legend()
plt.xlabel("$\\varepsilon$")
plt.ylabel("$\\rho$")
plt.title("Hénon map fitting results for different $\\epsilon$ values")
plt.tight_layout()
plt.savefig("../img/rho_values_henon_err.png", dpi=600)


# In[ ]:


plt.figure()
plt.errorbar(epsilons, k1_fix, yerr=k1_err_fix, linewidth=0, elinewidth=2, marker="x", label="Model 2, three parameters")
plt.errorbar(epsilons, k1, yerr=k1_err, linewidth=0, elinewidth=2, marker="x", label="Model 2, three parameters")
plt.errorbar(epsilons, k2, yerr=k2_err, linewidth=0, elinewidth=2, marker="x", label="Model 4, two parameters")
plt.errorbar(epsilons, k3, yerr=k3_err, linewidth=0, elinewidth=2, marker="x", label="Model 4, three parameters")

plt.legend()
plt.xlabel("$\\varepsilon$")
plt.ylabel("$\\kappa$")
plt.title("Hénon map fitting results for different $\\epsilon$ values")
plt.tight_layout()
plt.savefig("../img/k_values_henon_err.png", dpi=600)


# In[ ]:


plt.figure()
plt.errorbar(epsilons, n0_1, yerr=n0_1_err, linewidth=0, elinewidth=2, marker="x", label="Model 2, three parameters")
plt.errorbar(epsilons, n0_3, yerr=n0_3_err, linewidth=0, elinewidth=2, marker="x", label="Model 4, three parameters")

plt.yscale("log")

plt.legend()
plt.xlabel("$\\varepsilon$")
plt.ylabel("$N_0$")
plt.title("Hénon map fitting results for different $\\epsilon$ values")
plt.tight_layout()
plt.savefig("../img/n0_values_henon_err.png", dpi=600)


# ### New fitting models

# In[ ]:


def model_2_bis(params, x, data):
    b_tilde = params["b_tilde"]
    B = params["B"]
    N_0 = params["N_0"]
    k = params["k"]
    temp = np.power(B * np.log(x / N_0), k)
    temp[np.isnan(temp)] = 0.0
    model = b_tilde / temp
    if np.any(np.isnan(model)):
        print(b_tilde, B, N_0, k)
    return model - data

def model_4_bis(params, x, data):
    b_tilde = params["b_tilde"]
    B = params["B"]
    N_0 = params["N_0"]
    k = params["k"]
    lambert = lambertw(
        -(2 / (k * B)) * np.power(x / N_0, - 2 / k), -1
    )
    lambert[np.isnan(lambert)] = -np.inf
    model = b_tilde / np.power(- (k * B / 2) * np.real(lambert), k)
    
    if np.any(np.isnan(model)):
        print(b_tilde, B, N_0, k)
        try:
            print(params["delta"])
        except:
            pass
    #print(model)
    return model - data


# In[ ]:


def fit_model_2_bis(turns, DA):
    params = Parameters()
    params.add("b_tilde", value=1, min=0, vary=True)
    params.add("B", value=1, min=0, vary=True)
    params.add("N_0", expr="7 * sqrt(6) / 48 * (b_tilde ** (1/2))")
    params.add("k", value=1, min=0, vary=True)
    minner = Minimizer(model_2_bis, params, fcn_args=(turns, DA))
    result = minner.minimize(method="basinhopping")
    final = DA + result.residual
    return result, final

def fit_model_2_bis_free(turns, DA):
    params = Parameters()
    params.add("b_tilde", value=1, min=0, vary=True)
    params.add("B", value=1, min=0, vary=True)
    params.add("N_0", value=1, min=0, max=turns.min() * 1.0, vary=True)
    params.add("k", value=1, min=0, vary=True)
    minner = Minimizer(model_2_bis, params, fcn_args=(turns, DA))
    result = minner.minimize(method="basinhopping")
    final = DA + result.residual
    return result, final

def fit_model_4_bis(turns, DA):
    params = Parameters()
    params.add("b_tilde", value=1, min=0, vary=True)
    params.add("delta", value=0.2, min=0.0, max=1/np.e, vary=True)
    params.add("N_0", expr="7 * sqrt(6) / 48 * (b_tilde ** (1/2))")
    params.add("k", value=0.5, min=0.1, max=3, vary=True)
    params.add("B", expr="(2/(k*delta)) * (1000 / N_0) ** (-2/k)")
    minner = Minimizer(model_4_bis, params, fcn_args=(turns, DA))
    result = minner.minimize(method="basinhopping")
    final = DA + result.residual
    return result, final

def fit_model_4_bis_free(turns, DA):
    params = Parameters()
    params.add("b_tilde", value=1, min=0, vary=True)
    params.add("delta", value=0.2, min=0.0, max=1/np.e, vary=True)
    params.add("N_0", value=1, min=0, max=turns.min() * 1.0, vary=True)
    params.add("k", value=0.5, min=0.1, max=3, vary=True)
    params.add("B", expr="(2/(k*delta)) * (1000 / N_0) ** (-2/k)")
    minner = Minimizer(model_4_bis, params, fcn_args=(turns, DA))
    result = minner.minimize(method="basinhopping")
    final = DA + result.residual
    return result, final


# In[ ]:


real_results_model_2_bis = []
real_finals_model_2_bis = []


# In[ ]:


temp_result = []
temp_final = []
temp_eps = []

def log_result(result):
    temp_result.append(result[0])
    temp_final.append(result[1])
    temp_eps.append(result[2])

with multiprocessing.Pool(PROCESSES) as pool:
    results = [pool.apply_async(wrap_fit, args=(fit_model_2_bis, turn_sampling[::-1], real_DAs[i][::-1], epsilons[i]), callback=log_result) for i in range(len(real_DAs))]
    pool.close()
    pool.join()
    
TMP = sorted(zip(temp_eps, temp_result, temp_eps), key=lambda x: x[0])

real_results_model_2_bis = [T[1] for T in TMP]
real_finals_model_2_bis = [T[2] for T in TMP]


# In[ ]:


for i, eps in tqdm(list(enumerate(epsilons))):
    turns = turn_sampling[::-1]
    real_DA = real_DAs[i][::-1]
    
    result, final = fit_model_2_bis(turns, real_DA)
    real_results_model_2_bis.append(result)
    real_finals_model_2_bis.append(final)


# In[ ]:


k4 = []
k4_err = []
b_tilde4 = []
b_tilde4_err = []
B4 = []
B4_err = []
N_0_4 = []
N_0_4_err = []
for i, eps in enumerate(epsilons):
    k4.append(real_results_model_2_bis[i].params.get("k").value)
    k4_err.append(real_results_model_2_bis[i].params.get("k").stderr)
    b_tilde4.append(real_results_model_2_bis[i].params.get("b_tilde").value)
    b_tilde4_err.append(real_results_model_2_bis[i].params.get("b_tilde").stderr)
    B4.append(real_results_model_2_bis[i].params.get("B").value)
    B4_err.append(real_results_model_2_bis[i].params.get("B").stderr)
    N_0_4.append(real_results_model_2_bis[i].params.get("N_0").value)
    N_0_4_err.append(real_results_model_2_bis[i].params.get("N_0").stderr)


# In[ ]:


with open("../data/henon_fit_2_new.pkl", 'wb') as f:
    pickle.dump((
        real_results_model_2_bis,
        real_finals_model_2_bis
    ), f)


# In[ ]:


with open("../data/henon_fit_2_new.pkl", 'rb') as f:
    real_results_model_2_bis, real_finals_model_2_bis = pickle.load(f)


# In[ ]:


real_results_model_4_bis = []
real_finals_model_4_bis = []


# In[ ]:


temp_result = []
temp_final = []
temp_eps = []

def log_result(result):
    temp_result.append(result[0])
    temp_final.append(result[1])
    temp_eps.append(result[2])

with multiprocessing.Pool(PROCESSES) as pool:
    results = [pool.apply_async(wrap_fit, args=(fit_model_4_bis, turn_sampling[::-1], real_DAs[i][::-1], epsilons[i]), callback=log_result) for i in range(len(real_DAs))]
    pool.close()
    pool.join()
    
TMP = sorted(zip(temp_eps, temp_result, temp_eps), key=lambda x: x[0])

real_results_model_4_bis = [T[1] for T in TMP]
real_finals_model_4_bis = [T[2] for T in TMP]


# In[ ]:


k5 = []
k5_err = []
b_tilde5 = []
b_tilde5_err = []
B5 = []
B5_err = []
N_0_5 = []
N_0_5_err = []
for i, eps in enumerate(epsilons):
    k5.append(real_results_model_4_bis[i].params.get("k").value)
    k5_err.append(real_results_model_4_bis[i].params.get("k").stderr)
    b_tilde5.append(real_results_model_4_bis[i].params.get("b_tilde").value)
    b_tilde5_err.append(real_results_model_4_bis[i].params.get("b_tilde").stderr)
    B5.append(real_results_model_4_bis[i].params.get("B").value)
    B5_err.append(real_results_model_4_bis[i].params.get("B").stderr)
    N_0_5.append(real_results_model_4_bis[i].params.get("N_0").value)
    N_0_5_err.append(real_results_model_4_bis[i].params.get("N_0").stderr)


# In[ ]:


with open("../data/henon_fit_4_new.pkl", 'wb') as f:
    pickle.dump((
        real_results_model_4_bis,
        real_finals_model_4_bis
    ), f)


# In[ ]:


with open("../data/henon_fit_4_new.pkl", 'rb') as f:
    real_results_model_4_bis, real_finals_model_4_bis = pickle.load(f)


# In[ ]:


real_results_model_2_bis_free = []
real_finals_model_2_bis_free = []


# In[ ]:


temp_result = []
temp_final = []
temp_eps = []

def log_result(result):
    temp_result.append(result[0])
    temp_final.append(result[1])
    temp_eps.append(result[2])

with multiprocessing.Pool(PROCESSES) as pool:
    results = [pool.apply_async(wrap_fit, args=(fit_model_2_bis_free, turn_sampling[::-1], real_DAs[i][::-1], epsilons[i]), callback=log_result) for i in range(len(real_DAs))]
    pool.close()
    pool.join()
    
TMP = sorted(zip(temp_eps, temp_result, temp_eps), key=lambda x: x[0])

real_results_model_2_bis_free = [T[1] for T in TMP]
real_finals_model_2_bis_free = [T[2] for T in TMP]


# In[ ]:


k6 = []
k6_err = []
b_tilde6 = []
b_tilde6_err = []
B6 = []
B6_err = []
N_0_6 = []
N_0_6_err = []
for i, eps in enumerate(epsilons):
    k6.append(real_results_model_2_bis_free[i].params.get("k").value)
    k6_err.append(real_results_model_2_bis_free[i].params.get("k").stderr)
    b_tilde6.append(real_results_model_2_bis_free[i].params.get("b_tilde").value)
    b_tilde6_err.append(real_results_model_2_bis_free[i].params.get("b_tilde").stderr)
    B6.append(real_results_model_2_bis_free[i].params.get("B").value)
    B6_err.append(real_results_model_2_bis_free[i].params.get("B").stderr)
    N_0_6.append(real_results_model_2_bis_free[i].params.get("N_0").value)
    N_0_6_err.append(real_results_model_2_bis_free[i].params.get("N_0").stderr)


# In[ ]:


with open("../data/henon_fit_2_new_free.pkl", 'wb') as f:
    pickle.dump((
        real_results_model_2_bis_free,
        real_finals_model_2_bis_free
    ), f)


# In[ ]:


with open("../data/henon_fit_2_new_free.pkl", 'rb') as f:
    real_results_model_2_bis_free, real_finals_model_2_bis_free = pickle.load(f)


# In[ ]:


real_results_model_4_bis_free = []
real_finals_model_4_bis_free = []


# In[ ]:


temp_result = []
temp_final = []
temp_eps = []

def log_result(result):
    temp_result.append(result[0])
    temp_final.append(result[1])
    temp_eps.append(result[2])

with multiprocessing.Pool(PROCESSES) as pool:
    results = [pool.apply_async(wrap_fit, args=(fit_model_4_bis_free, turn_sampling[::-1], real_DAs[i][::-1], epsilons[i]), callback=log_result) for i in range(len(real_DAs))]
    pool.close()
    pool.join()
    
TMP = sorted(zip(temp_eps, temp_result, temp_eps), key=lambda x: x[0])

real_results_model_4_bis_free = [T[1] for T in TMP]
real_finals_model_4_bis_free = [T[2] for T in TMP]


# In[ ]:


k7 = []
k7_err = []
b_tilde7 = []
b_tilde7_err = []
B7 = []
B7_err = []
N_0_7 = []
N_0_7_err = []
for i, eps in enumerate(epsilons):
    k7.append(real_results_model_4_bis_free[i].params.get("k").value)
    k7_err.append(real_results_model_4_bis_free[i].params.get("k").stderr)
    b_tilde7.append(real_results_model_4_bis_free[i].params.get("b_tilde").value)
    b_tilde7_err.append(real_results_model_4_bis_free[i].params.get("b_tilde").stderr)
    B7.append(real_results_model_4_bis_free[i].params.get("B").value)
    B7_err.append(real_results_model_4_bis_free[i].params.get("B").stderr)
    N_0_7.append(real_results_model_4_bis_free[i].params.get("N_0").value)
    N_0_7_err.append(real_results_model_4_bis_free[i].params.get("N_0").stderr)


# In[ ]:


with open("../data/henon_fit_4_new_free.pkl", 'wb') as f:
    pickle.dump((
        real_results_model_4_bis_free,
        real_finals_model_4_bis_free
    ), f)


# In[ ]:


with open("../data/henon_fit_4_new_free.pkl", 'rb') as f:
    real_results_model_4_bis_free, real_finals_model_4_bis_free = pickle.load(f)


# ## Fokker-Planck stuff

# In[ ]:


# Lmfit
from lmfit import Minimizer, Parameters, report_fit
from pynverse import inversefunc
# Fokker-Plank

def fp_lmfit(params, x, data, dt, I0, I_max, iter_step):
    # Gathering parameters
    k = params["k"]
    I_star = params["I_star"]

    #print("k={}, I_star={}".format(k, I_star))

    # Declaring the engine
    engine_nk = nk.cn_nekhoroshev(I_max, 1.0, I_star, 1 / (k * 2), 0, I0, dt)
    multiplier = 0.01 / integrate.simps(engine_nk.diffusion, np.linspace(0, I_max, len(I0)))

    while True:
        engine_nk = nk.cn_nekhoroshev(I_max, multiplier, I_star, 1 / (k * 2), 0, I0, dt)
        # Allocating lists and counter
        t = []
        survival = []
        # Starting while loop for fitting procedure
        step = 0
        reached = True
        while(1.0 - engine_nk.get_particle_loss() >= data[-1]):
            # Append the data
            t.append(step * iter_step * dt * multiplier)
            survival.append(1.0 - engine_nk.get_particle_loss())
            # Iterate
            engine_nk.iterate(iter_step)
            # Evolve counter
            step += 1
            if step == 10000:
                #print("End not reached!")
                reached = False
                break
        # Append one last time
        t.append(step * iter_step * dt * multiplier)
        survival.append(1.0 - engine_nk.get_particle_loss())
        if len(t) > 10:
            break
        else:
            #print("decrease multiplier")
            multiplier /= 10
            
    # Post processing
    if reached:
        f = interp1d(t, survival, kind="cubic")
        inv_f = inversefunc(
            f,
            domain=(t[0], t[-1]),
            image=(survival[-1], survival[0])
        )
        point_t = inv_f(data[-1])
        point_s = inv_f(data[0])
        points_t = np.linspace(point_s, point_t, len(x))
        values_f = f(points_t)
    else:
        f = interp1d(t, survival, kind="cubic")
        points_t = np.linspace(0, t[-1], len(x))
        values_f = f(points_t) * 10
    return values_f - data


def autofit_fp(turns, losses, dt, I0, I_max, iter_step, k_0, I_star_0, method):
    params = Parameters()
    params.add("k", value=k_0, min=0, vary=True)
    params.add("I_star", value=I_star_0, min=0, vary=True)

    minner = Minimizer(fp_lmfit, params, fcn_args=(
        turns, losses, dt, I0, I_max, iter_step))
    result = minner.minimize(method=method)
    final = losses + result.residual
    return result, final


# In[ ]:


I_max = cut_point**2 / 2
I = np.linspace(0, I_max, 500)
I0 = I * np.exp(-(I/sigma**2))
I0 /= integrate.trapz(I0, I)


# ### All

# In[ ]:


temp_result = []
temp_final = []
temp_eps = []

def log_result(result):
    temp_result.append(result[0])
    temp_final.append(result[1])
    temp_eps.append(result[2])
    
with multiprocessing.Pool(PROCESSES) as pool:
    results = [pool.apply_async(
        wrap_fp_fit,
        args=(
            autofit_fp,
            turn_sampling[::-1],
            gaussian_losses[i][::-1],
            I0,
            I_max,
            k3[i],
            epsilons[i]),
        callback=log_result) for i in range(len(gaussian_losses))]
    pool.close()
    pool.join()
    
TMP = sorted(zip(temp_eps, temp_result, temp_eps), key=lambda x: x[0])

results = [T[1] for T in TMP]
finals = [T[2] for T in TMP]


# In[ ]:


with open("../data/fp_henon_fit.pkl", 'wb') as f:
    pickle.dump((results, finals), f)


# In[ ]:


with open("../data/fp_henon_fit.pkl", 'rb') as f:
    results, finals = pickle.load(f)


# ### No begin

# In[ ]:


temp_result = []
temp_final = []
temp_eps = []

def log_result(result):
    temp_result.append(result[0])
    temp_final.append(result[1])
    temp_eps.append(result[2])
    
with multiprocessing.Pool(PROCESSES) as pool:
    results = [pool.apply_async(
        wrap_fp_fit,
        args=(
            autofit_fp,
            (turn_sampling[turn_sampling > 10000])[::-1],
            (gaussian_losses[i][turn_sampling > 10000])[::-1],
            I0,
            I_max,
            k3[i],
            epsilons[i]),
        callback=log_result) for i in range(len(gaussian_losses))]
    pool.close()
    pool.join()
    
TMP = sorted(zip(temp_eps, temp_result, temp_eps), key=lambda x: x[0])

results_no_beg = [T[1] for T in TMP]
finals_no_beg = [T[2] for T in TMP]


# In[ ]:


with open("../data/fp_henon_fit_no_beg.pkl", 'wb') as f:
    pickle.dump((results_no_beg, finals_no_beg), f)


# In[ ]:


with open("../data/fp_henon_fit_no_beg.pkl", 'rb') as f:
    results_no_beg, finals_no_beg = pickle.load(f)


# ### No end

# In[ ]:


temp_result = []
temp_final = []
temp_eps = []

def log_result(result):
    temp_result.append(result[0])
    temp_final.append(result[1])
    temp_eps.append(result[2])
    
with multiprocessing.Pool(PROCESSES) as pool:
    results = [pool.apply_async(
        wrap_fp_fit,
        args=(
            autofit_fp,
            (turn_sampling[len(turn_sampling)//2:])[::-1],
            (gaussian_losses[i][len(turn_sampling)//2:])[::-1],
            I0,
            I_max,
            k3[i],
            epsilons[i]),
        callback=log_result) for i in range(len(gaussian_losses))]
    pool.close()
    pool.join()
    
TMP = sorted(zip(temp_eps, temp_result, temp_eps), key=lambda x: x[0])

results_no_end = [T[1] for T in TMP]
finals_no_end = [T[2] for T in TMP]


# In[ ]:


with open("../data/fp_henon_fit_no_end.pkl", 'wb') as f:
    pickle.dump((results_no_end, finals_no_end), f)


# In[ ]:


with open("../data/fp_henon_fit_no_end.pkl", 'rb') as f:
    results_no_end, finals_no_end = pickle.load(f)


# ### No beg no end

# In[ ]:


temp_result = []
temp_final = []
temp_eps = []

def log_result(result):
    temp_result.append(result[0])
    temp_final.append(result[1])
    temp_eps.append(result[2])
    
with multiprocessing.Pool(PROCESSES) as pool:
    results = [pool.apply_async(
        wrap_fp_fit,
        args=(
            autofit_fp,
            ((turn_sampling[turn_sampling > 10000])[len(turn_sampling)//2:])[::-1],
            ((gaussian_losses[i][turn_sampling > 10000])[len(turn_sampling)//2:])[::-1],
            I0,
            I_max,
            k3[i],
            epsilons[i]),
        callback=log_result) for i in range(len(gaussian_losses))]
    pool.close()
    pool.join()
    
TMP = sorted(zip(temp_eps, temp_result, temp_eps), key=lambda x: x[0])

results_all_mods = [T[1] for T in TMP]
finals_all_mods = [T[2] for T in TMP]


# In[ ]:


with open("../data/fp_henon_fit_all_mods.pkl", 'wb') as f:
    pickle.dump((results_all_mods, finals_all_mods), f)


# In[ ]:


with open("../data/fp_henon_fit_all_mods.pkl", 'rb') as f:
    results_all_mods, finals_all_mods = pickle.load(f)


# ### Plots FP

# In[ ]:


plt.figure()
plt.plot(turns, gaussian_losses[0][::-1], label="loss data")
plt.plot(turns, finals[0], label="FP fitting")

plt.legend()
plt.xlabel("$N$ turns")
plt.ylabel("Losses")
plt.title("Hénon map with gaussian weights $(\\sigma={:.2}, \\varepsilon={:.4})$".format(sigma, epsilons[0]))

plt.tight_layout()

plt.savefig("../img/fp_henon_fitting_1.png", dpi=600)


# In[ ]:


plt.figure()
plt.plot(turns, gaussian_losses[5][::-1], label="loss data")
plt.plot(turns, finals[5], label="FP fitting")

plt.legend()
plt.xlabel("$N$ turns")
plt.ylabel("Losses")
plt.title("Hénon map with gaussian weights $(\\sigma={:.2}, \\varepsilon={:.4})$".format(sigma, epsilons[5]))

plt.tight_layout()

plt.savefig("../img/fp_henon_fitting_2.png", dpi=600)


# In[ ]:


plt.figure()
plt.plot(turns, gaussian_losses[10][::-1], label="loss data")
plt.plot(turns, finals[10], label="FP fitting")

plt.legend()
plt.xlabel("$N$ turns")
plt.ylabel("Losses")
plt.title("Hénon map with gaussian weights $(\\sigma={:.2}, \\varepsilon={:.4})$".format(sigma, epsilons[10]))

plt.tight_layout()

plt.savefig("../img/fp_henon_fitting_3.png", dpi=600)


# In[ ]:


plt.figure()
plt.plot(turns, gaussian_losses[-2][::-1], label="loss data")
plt.plot(turns, finals[-2], label="FP fitting")

plt.legend()
plt.xlabel("$N$ turns")
plt.ylabel("Losses")
plt.title("Hénon map with gaussian weights $(\\sigma={:.2}, \\varepsilon={:.4})$".format(sigma, epsilons[-2]))

plt.tight_layout()

plt.savefig("../img/fp_henon_fitting_4.png", dpi=600)


# In[ ]:


plt.figure()
plt.plot(turn_sampling[::-1], gaussian_losses[-1][::-1], label="loss data")
plt.plot(turn_sampling[::-1], finals[-1], label="FP fitting")

plt.legend()
plt.xlabel("$N$ turns")
plt.ylabel("Losses")
plt.title("Hénon map with gaussian weights $(\\sigma={:.2}, \\varepsilon={:.4})$".format(sigma, epsilons[-1]))

plt.tight_layout()

plt.savefig("../img/fp_henon_fitting_5.png", dpi=600)


# In[ ]:


eps_fp = []
k_fp = []
k_err_fp = []
I_star_fp = []
I_star_err_fp = []

for i, result in enumerate(results):
    k_fp.append(result.params.get("k").value)
    k_err_fp.append(result.params.get("k").stderr)
    I_star_fp.append(result.params.get("I_star").value)
    I_star_err_fp.append(result.params.get("I_star").stderr)
    eps_fp.append(epsilons[i])


# In[ ]:


eps_fp_no_beg = []
k_fp_no_beg = []
k_err_fp_no_beg = []
I_star_fp_no_beg = []
I_star_err_fp_no_beg = []

for i, result in enumerate(results_no_beg):
    k_fp_no_beg.append(result.params.get("k").value)
    k_err_fp_no_beg.append(result.params.get("k").stderr)
    I_star_fp_no_beg.append(result.params.get("I_star").value)
    I_star_err_fp_no_beg.append(result.params.get("I_star").stderr)
    eps_fp_no_beg.append(epsilons[i])


# In[ ]:


eps_fp_no_end = []
k_fp_no_end = []
k_err_fp_no_end = []
I_star_fp_no_end = []
I_star_err_fp_no_end = []

for i, result in enumerate(results_no_end):
    k_fp_no_end.append(result.params.get("k").value)
    k_err_fp_no_end.append(result.params.get("k").stderr)
    I_star_fp_no_end.append(result.params.get("I_star").value)
    I_star_err_fp_no_end.append(result.params.get("I_star").stderr)
    eps_fp_no_end.append(epsilons[i])


# In[ ]:


eps_fp_all_mods = []
k_fp_all_mods = []
k_err_fp_all_mods = []
I_star_fp_all_mods = []
I_star_err_fp_all_mods = []

for i, result in enumerate(results_all_mods):
    k_fp_all_mods.append(result.params.get("k").value)
    k_err_fp_all_mods.append(result.params.get("k").stderr)
    I_star_fp_all_mods.append(result.params.get("I_star").value)
    I_star_err_fp_all_mods.append(result.params.get("I_star").stderr)
    eps_fp_all_mods.append(epsilons[i])


# In[ ]:


plt.figure()

plt.errorbar(eps_fp[:-1], k_fp[:-1], yerr=k_err_fp[:-1], linewidth=0, elinewidth=2, marker="x", label="FP all data")
plt.errorbar(eps_fp[:-1], k_fp_no_beg[:-1], yerr=k_err_fp_no_beg[:-1], linewidth=0, elinewidth=2, marker="x", label="FP no begin")
plt.errorbar(eps_fp[:-1], k_fp_no_end[:-1], yerr=k_err_fp_no_end[:-1], linewidth=0, elinewidth=2, marker="x", label="FP half data missing")
plt.errorbar(eps_fp[:-1], k_fp_all_mods[:-1], yerr=k_err_fp_all_mods[:-1], linewidth=0, elinewidth=2, marker="x", label="FP all modifications")

plt.legend()
plt.xlabel("$\\varepsilon$")
plt.ylabel("$\\kappa$")
plt.title("Comparison between $\\kappa$ values from different fitting methods")
plt.tight_layout()

plt.savefig("../img/fp_variants_henon_k.png", dpi=600)


# In[ ]:


plt.figure()

plt.errorbar(eps_fp[:-1], I_star_fp[:-1], yerr=I_star_err_fp[:-1], linewidth=0, elinewidth=2, marker="x", label="FP all data")
plt.errorbar(eps_fp[:-1], I_star_fp_no_beg[:-1], yerr=I_star_err_fp_no_beg[:-1], linewidth=0, elinewidth=2, marker="x", label="FP no begin")
plt.errorbar(eps_fp[:-1], I_star_fp_no_end[:-1], yerr=I_star_err_fp_no_end[:-1], linewidth=0, elinewidth=2, marker="x", label="FP half data missing")
plt.errorbar(eps_fp[:-1], I_star_fp_all_mods[:-1], yerr=I_star_err_fp_all_mods[:-1], linewidth=0, elinewidth=2, marker="x", label="FP all modifications")

plt.legend()
plt.xlabel("$\\varepsilon$")
plt.ylabel("$I_\\ast$")
plt.title("Comparison between $I_\\ast$ values from different fitting methods")
plt.tight_layout()

plt.savefig("../img/fp_variants_henon_Istar.png", dpi=600)


# In[ ]:


plt.figure()

plt.scatter(epsilons, k2, label="Model 4, two parameters")
plt.scatter(epsilons, k3, label="Model 4, three parameters")
plt.scatter(eps_fp[:-1], k_fp[:-1], label="Fokker-Planck fitting procedure")

plt.legend()
plt.xlabel("$\\varepsilon$")
plt.ylabel("$\\kappa$")
plt.title("Comparison between $\\kappa$ values from different fitting methods")
plt.tight_layout()

plt.savefig("../img/general_henon_clean.png", dpi=600)


# In[ ]:


plt.figure()

plt.errorbar(epsilons, k2, yerr=k2_err, linewidth=0, elinewidth=2, marker="x", label="Model 4, two parameters")
plt.errorbar(epsilons, k3, yerr=k3_err, linewidth=0, elinewidth=2, marker="x", label="Model 4, three parameters")
plt.errorbar(eps_fp[:-1], k_fp[:-1], yerr=k_err_fp[:-1], linewidth=0, elinewidth=2, marker="x", label="Fokker-Planck fitting procedure")

plt.legend()
plt.xlabel("$\\varepsilon$")
plt.ylabel("$\\kappa$")
plt.title("Comparison between $\\kappa$ values from different fitting methods")
plt.tight_layout()

plt.savefig("../img/general_henon_clean_err.png", dpi=600)


# In[ ]:


plt.figure()

plt.errorbar(epsilons, rho2, yerr=rho2_err, linewidth=0, elinewidth=2, marker="x", label="Model 4, two parameters $(\\rho)$")
#plt.errorbar(epsilons, rho3, yerr=rho3_err, linewidth=0, elinewidth=2, marker="x", label="Model 4, three parameters")
plt.errorbar(eps_fp[:-1], np.asarray(I_star_fp[:-1]), yerr=I_star_err_fp[:-1], linewidth=0, elinewidth=2, marker="x", label="Fokker-Planck fitting procedure")

#plt.yscale("log")

plt.legend()
plt.xlabel("$\\varepsilon$")
plt.ylabel("$I_\\ast$")
plt.title("Comparison between $I_\\ast$ values from different fitting methods")
plt.tight_layout()

plt.savefig("../img/general_henon_clean_I_star_err.png", dpi=600)


# In[ ]:


plt.figure()

plt.scatter(epsilons, k2, label="Model 4, two parameters")
plt.scatter(epsilons, k3, label="Model 4, three parameters")
plt.scatter(eps_fp[:], k_fp[:], label="Fokker-Planck fitting procedure")

plt.legend()
plt.xlabel("$\\varepsilon$")
plt.ylabel("$\\kappa$")
plt.title("Comparison between $\\kappa$ values from different fitting methods")
plt.tight_layout()

plt.savefig("../img/general_henon.png", dpi=600)


# ## REDO Fokker-Planck but starting from another k

# ### All

# In[ ]:


bis_results = []
bis_finals = []

for i, g_loss in tqdm(enumerate(gaussian_losses), total=len(gaussian_losses)):
    turns = turn_sampling[::-1]
    loss = g_loss[::-1]
    result, final = autofit_fp(turns, loss, 1.0, I0, I_max, 1, k2[i], 0.26, "lstsqr")
    bis_results.append(result)
    bis_finals.append(final)


# In[ ]:


with open("../data/bis_fp_henon_fit.pkl", 'wb') as f:
    pickle.dump((bis_results, bis_finals), f)


# In[ ]:


with open("../data/bis_fp_henon_fit.pkl", 'rb') as f:
    bis_results, bis_finals = pickle.load(f)


# ### No begin

# In[ ]:


bis_results_no_beg = []
bis_finals_no_beg = []

for i, g_loss in tqdm(enumerate(gaussian_losses), total=len(gaussian_losses)):
    turns = turn_sampling[::-1]
    loss = g_loss[::-1]
    # mods
    loss = loss[turns>5000]
    turns = turns[turns>5000]
    result, final = autofit_fp(turns, loss, 1.0, I0, I_max, 1, k2[i], 0.26, "lstsqr")
    bis_results_no_beg.append(result)
    bis_finals_no_beg.append(final)


# In[ ]:


with open("../data/bis_fp_henon_fit_no_beg.pkl", 'wb') as f:
    pickle.dump((bis_results_no_beg, bis_finals_no_beg), f)


# In[ ]:


with open("../data/bis_fp_henon_fit_no_beg.pkl", 'rb') as f:
    bis_results_no_beg, bis_finals_no_beg = pickle.load(f)


# ### No end

# In[ ]:


bis_results_no_end = []
bis_finals_no_end = []

for i, g_loss in tqdm(enumerate(gaussian_losses), total=len(gaussian_losses)):
    turns = turn_sampling[::-1]
    loss = g_loss[::-1]
    # mods
    loss = loss[:len(turns)//2]
    turns = turns[:len(turns)//2]
    result, final = autofit_fp(turns, loss, 1.0, I0, I_max, 1, k2[i], 0.26, "lstsqr")
    bis_results_no_end.append(result)
    bis_finals_no_end.append(final)


# In[ ]:


with open("../data/bis_fp_henon_fit_no_end.pkl", 'wb') as f:
    pickle.dump((bis_results_no_end, bis_finals_no_end), f)


# In[82]:


with open("../data/bis_fp_henon_fit_no_end.pkl", 'rb') as f:
    bis_results_no_end, bis_finals_no_end = pickle.load(f)


# ### No beg no end

# In[ ]:


bis_results_all_mods = []
bis_finals_all_mods = []

for i, g_loss in tqdm(enumerate(gaussian_losses), total=len(gaussian_losses)):
    turns = turn_sampling[::-1]
    loss = g_loss[::-1]
    # mods
    loss = loss[:len(turns)//2]
    turns = turns[:len(turns)//2]
    loss = loss[turns>5000]
    turns = turns[turns>5000]
    result, final = autofit_fp(turns, loss, 1.0, I0, I_max, 1, k2[i], 0.26, "lstsqr")
    bis_results_all_mods.append(result)
    bis_finals_all_mods.append(final)


# In[ ]:


with open("../data/bis_fp_henon_fit_all_mods.pkl", 'wb') as f:
    pickle.dump((bis_results_all_mods, bis_finals_all_mods), f)


# In[99]:


with open("../data/fp_henon_fit_all_mods.pkl", 'rb') as f:
    results_all_mods, finals_all_mods = pickle.load(f)


# In[111]:


bis_eps_fp = []
bis_k_fp = []
bis_k_err_fp = []
bis_I_star_fp = []
bis_I_star_err_fp = []

for i, result in enumerate(bis_results):
    bis_k_fp.append(result.params.get("k").value)
    bis_k_err_fp.append(result.params.get("k").stderr)
    bis_I_star_fp.append(result.params.get("I_star").value)
    bis_I_star_err_fp.append(result.params.get("I_star").stderr)
    bis_eps_fp.append(epsilons[i])


# In[101]:


bis_eps_fp_no_beg = []
bis_k_fp_no_beg = []
bis_k_err_fp_no_beg = []
bis_I_star_fp_no_beg = []
bis_I_star_err_fp_no_beg = []

for i, result in enumerate(bis_results_no_beg):
    bis_k_fp_no_beg.append(result.params.get("k").value)
    bis_k_err_fp_no_beg.append(result.params.get("k").stderr)
    bis_I_star_fp_no_beg.append(result.params.get("I_star").value)
    bis_I_star_err_fp_no_beg.append(result.params.get("I_star").stderr)
    bis_eps_fp_no_beg.append(epsilons[i])


# In[102]:


eps_fp_no_end = []
k_fp_no_end = []
k_err_fp_no_end = []
I_star_fp_no_end = []
I_star_err_fp_no_end = []

for i, result in enumerate(bis_results_no_end):
    k_fp_no_end.append(result.params.get("k").value)
    k_err_fp_no_end.append(result.params.get("k").stderr)
    I_star_fp_no_end.append(result.params.get("I_star").value)
    I_star_err_fp_no_end.append(result.params.get("I_star").stderr)
    eps_fp_no_end.append(epsilons[i])


# In[103]:


bis_eps_fp_all_mods = []
bis_k_fp_all_mods = []
bis_k_err_fp_all_mods = []
bis_I_star_fp_all_mods = []
bis_I_star_err_fp_all_mods = []

for i, result in enumerate(bis_results_all_mods):
    bis_k_fp_all_mods.append(result.params.get("k").value)
    bis_k_err_fp_all_mods.append(result.params.get("k").stderr)
    bis_I_star_fp_all_mods.append(result.params.get("I_star").value)
    bis_I_star_err_fp_all_mods.append(result.params.get("I_star").stderr)
    bis_eps_fp_all_mods.append(epsilons[i])


# In[119]:


plt.figure()

plt.errorbar(eps_fp[:-1], k_fp[:-1], yerr=k_err_fp[:-1], linewidth=0, elinewidth=2, marker="x", label="FP all data (from below)")
plt.errorbar(eps_fp[:-1], bis_k_fp[:-1], yerr=bis_k_err_fp[:-1], linewidth=0, elinewidth=2, marker="x", label="FP all data (from above)")

plt.legend()
plt.xlabel("$\\varepsilon$")
plt.ylabel("$\\kappa$")
plt.title("Comparison between $\\kappa$ values from different fitting methods")
plt.tight_layout()

plt.savefig("../img/fp_variants_henon_above_below.png", dpi=600)


# ## Other Plots...

# In[243]:


plt.figure()
plt.scatter(epsilons, k1, label="Model 2, three parameters")
plt.scatter(epsilons, k2, label="Model 4, two parameters")
plt.scatter(epsilons, k3, label="Model 4, three parameters")
plt.scatter(epsilons, k4, label="Model 2 new, n0 fixed")
plt.scatter(epsilons, k5, label="Model 4 new, n0 fixed")
plt.scatter(epsilons, k6, label="Model 2 new, n0 free")
plt.scatter(epsilons, k7, label="Model 4 new, n0 free")

plt.legend(fontsize="x-small")
plt.xlabel("$\\varepsilon$")
plt.ylabel("$\\kappa$")
plt.title("Hénon map fitting results for different $\\epsilon$ values")
plt.tight_layout()

plt.savefig("../img/all_fits_henon.png", dpi=600)


# In[264]:


plt.figure()
plt.errorbar(epsilons, k1, yerr=k1_err, linewidth=0, elinewidth=2, marker="x", label="Model 2, three parameters")
plt.errorbar(epsilons, k2, yerr=k2_err, linewidth=0, elinewidth=2, marker="x", label="Model 4, two parameters")
plt.errorbar(epsilons, k3, yerr=k3_err, linewidth=0, elinewidth=2, marker="x", label="Model 4, three parameters")
plt.errorbar(epsilons, k4, yerr=k4_err, linewidth=0, elinewidth=2, marker="x", label="Model 2 new, n0 fixed")
plt.errorbar(epsilons, k5, yerr=k5_err, linewidth=0, elinewidth=2, marker="x", label="Model 4 new, n0 fixed")
plt.errorbar(epsilons, k6, yerr=k6_err, linewidth=0, elinewidth=2, marker="x", label="Model 2 new, n0 free")
plt.errorbar(epsilons, k7, yerr=k7_err, linewidth=0, elinewidth=2, marker="x", label="Model 4 new, n0 free")

plt.legend(fontsize="x-small")
plt.xlabel("$\\varepsilon$")
plt.ylabel("$\\kappa$")
plt.title("Hénon map fitting results for different $\\epsilon$ values")
plt.tight_layout()

plt.savefig("../img/all_fits_henon_err.png", dpi=600)


# In[244]:


with open("../data/all_henon_fit.pkl", 'wb') as f:
    pickle.dump((
        real_results_model_2,
        real_finals_model_2,
        real_results_model_4,
        real_finals_model_4,
        real_results_model_4_free,
        real_finals_model_4_free,
        real_results_model_2_bis,
        real_finals_model_2_bis,
        real_results_model_4_bis,
        real_finals_model_4_bis,
        real_results_model_2_bis_free,
        real_finals_model_2_bis_free,
        real_results_model_4_bis_free,
        real_finals_model_4_bis_free
    ), f)


# In[9]:


with open("../data/all_henon_fit.pkl", 'rb') as f:
    real_results_model_2, real_finals_model_2, real_results_model_4, real_finals_model_4, real_results_model_4_free, real_finals_model_4_free, real_results_model_2_bis, real_finals_model_2_bis, real_results_model_4_bis, real_finals_model_4_bis, real_results_model_2_bis_free, real_finals_model_2_bis_free, real_results_model_4_bis_free, real_finals_model_4_bis_free = pickle.load(f)


# In[ ]:




