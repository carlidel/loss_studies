# Base libraries
import math
import numpy as np
import scipy.integrate as integrate
import pickle
import itertools
from scipy.optimize import curve_fit
from math import gcd
import pandas as pd
from scipy.special import lambertw
from scipy.interpolate import interp1d
from pynverse import inversefunc
from numba import njit, prange
from tqdm.autonotebook import tqdm

# Personal libraries
import sixtrackwrap as sx
import crank_nicolson_numba.nekhoroshev as nk
import crank_nicolson_numba.polynomial as pl

# Lmfit
from lmfit import Minimizer, Parameters, report_fit

## Prepare inverse functions for obtaining DA from loss values

# Uniform 4D distribution

@njit
def uniform_loss(DA, DA0):
    """Return the survived beam estimation for an uniform beam

    Parameters
    ----------
    DA : ndarray
        values of DA to evaluate
    DA0 : float
        baseline value of DA

    Returns
    -------
    ndarray
        survived beam values
    """    
    return (DA ** 4 / DA0 ** 4)

@njit
def DA_from_unifom_loss(loss, DA0):
    """Given the amount of survived beam, returns the corresponding DA for a uniform beam

    Parameters
    ----------
    loss : ndarray
        survived beam values
    DA0 : ndarray
        baseline value of DA

    Returns
    -------
    ndarray
        corresponding values of DA
    """    
    return np.power((loss * DA0 ** 4), 1/4)

# Symmetric 4D gaussian

@njit
def symmetric_gaussian_loss(DA, sigma, DA0):
    """Return the survived beam estimation for a symmetric gaussian beam, given the DA.

    Parameters
    ----------
    DA : ndarray
        DAs to consider
    sigma : float
        sigma of the gaussian distribution
    DA0 : float
        DA baseline

    Returns
    -------
    ndarray
        list of survived values
    """    
    baseline = - np.exp(- 0.5 * (DA0 / sigma) ** 2) * (DA0 ** 2 + 2 * sigma ** 2) + 2 * sigma ** 2
    return (- np.exp(- 0.5 * (DA / sigma) ** 2) * (DA ** 2 + 2 * sigma ** 2) + 2 * sigma ** 2) / baseline


def DA_from_symmetric_gaussian_loss(loss, sigma, DA0):
    """Given the amount of survived beam, returns the corresponding DA of a given gaussian beam

    Parameters
    ----------
    loss : ndarray
        survival values to consider
    sigma : float
        sigma of the distribution
    DA0 : float
        baseline DA

    Returns
    -------
    ndarray
        DA values
    """    
    func = inversefunc(
        lambda x: symmetric_gaussian_loss(x, sigma, DA0),
        domain=[0.0, DA0]
    )
    return func(loss)


### $\chi^2$ function
def chi_2(original, estimate):
    """Chi squared function

    Parameters
    ----------
    original : ndarray
        ideal values
    estimate : ndarray
        estimated values

    Returns
    -------
    float
        chi squared value
    """    
    return np.sum(np.power(estimate - original, 2) / original)


### Model 2

def model_2(x, rho, n0, k):
    temp = (np.power(np.log(x / n0), k))
    temp[np.isnan(temp)] = 0.0
    return rho * np.power(k / (2 * np.exp(1)), k) / temp


def explore_k_model_2(turns, da, k_min, k_max, n_samples):
    ks = np.linspace(k_min, k_max, n_samples)
    pars = []
    errs = []
    co_pars = []
    for k in tqdm(ks):
        par, co_par = curve_fit(
            lambda x, a, b : model_2(x, a, b, k),
            turns,
            da,
            bounds=([0, 0.00001],[np.inf, turns[-1]-0.0001])
        )
        pars.append(par)
        co_pars.append(co_par)
        errs.append(chi_2(da, model_2(turns, par[0], par[1], k)))
    return np.asarray(pars), np.asarray(errs), np.asarray(co_pars)


def model_2_lmfit(params, x, data, err=None):
    rho = params["rho"]
    n0 = params["n0"]
    k = params["k"]
    model = rho * np.power(k / (2 * np.exp(1)), k) / (np.power(np.log(x / n0), k))
    if err is None:
        res = model - data
    else:
        res = (model - data) / err
    res[np.isnan(res)] = 2.0
    res[np.isinf(res)] = 100.0
    return res


def fit_model_2(turns, DA, err=None):
    params = Parameters()
    params.add("rho", value=1, min=0, vary=True)
    params.add("n0", value=1, min=0, max=turns.min() * 0.5, vary=True)
    params.add("k", value=1, min=0, vary=True)
    if err is None:
        minner = Minimizer(model_2_lmfit, params, fcn_args=(turns, DA))
    else:
        minner = Minimizer(model_2_lmfit, params, fcn_args=(turns, DA, err))
    result = minner.minimize(method="basinhopping")
    final = DA + result.residual
    return result, final


def fit_model_2_fixed_n0(turns, DA, err=None):
    params = Parameters()
    params.add("rho", value=1, min=0, vary=True)
    params.add("n0", value=1, vary=False)
    params.add("k", value=1, min=0, vary=True)
    if err is None:
        minner = Minimizer(model_2_lmfit, params, fcn_args=(turns, DA))
    else:
        minner = Minimizer(model_2_lmfit, params, fcn_args=(turns, DA, err))
    result = minner.minimize(method="basinhopping")
    final = DA + result.residual
    return result, final


### Model 4

def model_4(x, rho, k):
    lamb = 0.5
    lambert = lambertw(- (1 / (2 * np.exp(1) * lamb)) * np.power(rho / 6, 1 / k) * np.power((8/7) * x, -1 / (lamb * k)), -1)
    lambert[np.isnan(lambert)] = -np.inf
    lambert = np.real(lambert)
    return (rho / np.power(-2 * np.exp(1) * lamb * lambert, k))

def explore_k_model_4(turns, da, k_min, k_max, n_samples):
    ks = np.linspace(k_min, k_max, n_samples)
    pars = []
    errs = []
    co_pars = []
    for k in tqdm(ks):
        par, co_par = curve_fit(
            lambda x, a : model_4(x, a, k),
            turns,
            da,
            bounds=([0.1],[np.inf])
        )
        pars.append(par)
        co_pars.append(co_par)
        errs.append(chi_2(da, model_4(turns, par[0], k)))
    return np.asarray(pars), np.asarray(errs), np.asarray(co_pars)


def model_4_lmfit(params, x, data, err=None):
    rho = params["rho"]
    k = params["k"]
    model = model_4(x, rho, k)
    if err is None:
        res = model - data
    else:
        res = (model - data) / err
    res[np.isnan(res)] = 2.0
    res[np.isinf(res)] = 100.0
    return res


def fit_model_4(turns, DA, err=None):
    params = Parameters()
    params.add("rho", value=1, min=0, vary=True)
    params.add("k", value=1, min=0, vary=True)
    if err is None:
        minner = Minimizer(model_4_lmfit, params, fcn_args=(turns, DA))
    else:
        minner = Minimizer(model_4_lmfit, params, fcn_args=(turns, DA, err))
    result = minner.minimize(method="basinhopping")
    final = DA + result.residual
    return result, final


### A more free Model 4

def model_4_free(x, rho, n0, k):
    #print(rho, n0, k)
    lamb = 0.5
    lambert = lambertw(- (1 / (lamb * k)) *
                       np.power(x / n0, - 1 / (lamb * k)), -1)
    lambert[np.isnan(lambert)] = -np.inf
    lambert = np.real(lambert)
    return (rho / (np.power(- 2 * lamb * np.exp(1) * lambert, k)))


def explore_model_4_free(turns, da, k_min, k_max, k_samples):
    ks = np.linspace(k_min, k_max, k_samples)
    pars = []
    errs = []
    co_pars = []
    for k in tqdm(ks):
        par, co_par = curve_fit(
            lambda x, a, b : model_4_free(x, a, b, k),
            turns,
            da,
            bounds=([0.01, 0.01], [np.inf, turns[-1] - 0.1])
        )
        pars.append(par)
        co_pars.append(co_par)
        errs.append(chi_2(da, model_4_free(turns, par[0], par[1], k)))
    return np.asarray(pars), np.asarray(errs), np.asarray(co_pars)


def model_4_free_lmfit(params, x, data, err=None):
    rho = params["rho"]
    k = params["k"]
    n0 = params["n0"]
    model = model_4_free(x, rho, n0, k)
    if err is None:
        res = model - data
    else:
        res = (model - data) / err
    res[np.isnan(res)] = 2.0
    res[np.isinf(res)] = 100.0
    return res


def fit_model_4_free(turns, DA, err=None):
    params = Parameters()
    params.add("rho", value=1, min=0, vary=True)
    params.add("n0", value=1, min=0, max=turns.min()*0.5, vary=True)
    params.add("k", value=1, min=0, vary=True)
    if err is None:
        minner = Minimizer(model_4_free_lmfit, params, fcn_args=(turns, DA))
    else:
        minner = Minimizer(model_4_free_lmfit, params, fcn_args=(turns, DA, err))
    result = minner.minimize(method="basinhopping")
    final = DA + result.residual
    return result, final

# Fokker-Plank

def fp(params, dt, I0, I_max, iter_step, multiplier, iters, iter_mult, actual_iters, losses, t_start, t_end):
    # Gathering parameters
    k = params["k"]
    I_star = params["I_star"]
    
    engine_nk = nk.cn_nekhoroshev(I_max, multiplier, I_star, 1 / (k * 2), 0, I0, dt)
    survival = []
    for i in range(len(iters) * iter_mult):
        survival.append(1.0 - engine_nk.get_particle_loss())
        engine_nk.iterate(iter_step)
    f = interp1d(range(len(survival)), survival, kind="cubic")
    
    t = np.linspace(t_start, t_start + (t_end - t_start) * iter_mult, actual_iters)
    return f(t)
        
def new_fp(engine_nk, iters, iter_mult, t_start, t_end, iter_step, actual_iters):
    engine_nk.reset()
    survival = []
    for i in range(len(iters) * iter_mult):
        survival.append(1.0 - engine_nk.get_particle_loss())
        engine_nk.iterate(iter_step)
    f = interp1d(range(len(survival)), survival, kind="cubic")
    
    t = np.linspace(t_start, t_start + (t_end - t_start) * iter_mult, actual_iters)
    return f(t)
    
def fp_lmfit(params, x, data, dt, I0, I_max, iter_step, hold_tight=False, more_data=False):
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
            t.append(step)
            survival.append(1.0 - engine_nk.get_particle_loss())
            # Iterate
            engine_nk.iterate(iter_step)
            # Evolve counter
            step += 1
            if not hold_tight and step == 10000:
                #print("End not reached!")
                reached = False
                break
        # Append one last time
        t.append(step)
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
        point_s = 0
        point_t = t[-1]
        values_f = f(points_t) * 10
    if more_data:
        return engine_nk, t, point_s, point_t, iter_step
    else:
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

# Fokker-Plank (POLYNOMIAL)

def fp_poly(params, dt, I0, I_max, iter_step, multiplier, iters, iter_mult, actual_iters, losses, t_start, t_end):
    # Gathering parameters
    exponent = params["exponent"]
    
    engine_nk = pl.cn_polynomial(I_max, multiplier, exponent, I0, dt)
    survival = []
    for i in range(len(iters) * iter_mult):
        survival.append(1.0 - engine_nk.get_particle_loss())
        engine_nk.iterate(iter_step)
    f = interp1d(range(len(survival)), survival, kind="cubic")
    
    t = np.linspace(t_start, t_start + (t_end - t_start) * iter_mult, actual_iters)
    return f(t)

def fp_lmfit_poly(params, x, data, dt, I0, I_max, iter_step, hold_tight=False, more_data=False):
    # Gathering parameters
    exponent = params["exponent"]
    
    # Declaring the engine
    engine_nk = pl.cn_polynomial(I_max, 1.0, exponent, I0, dt)
    
    multiplier = 0.01 / integrate.simps(engine_nk.diffusion, np.linspace(0, I_max, len(I0)))

    while True:
        engine_nk = pl.cn_polynomial(I_max, multiplier, exponent, I0, dt)
        
        # Allocating lists and counter
        t = []
        survival = []
        # Starting while loop for fitting procedure
        step = 0
        reached = True
        while(1.0 - engine_nk.get_particle_loss() >= data[-1]):
            # Append the data
            t.append(step)
            survival.append(1.0 - engine_nk.get_particle_loss())
            # Iterate
            engine_nk.iterate(iter_step)
            # Evolve counter
            step += 1
            if not hold_tight and step == 10000:
                #print("End not reached!")
                reached = False
                break
        # Append one last time
        t.append(step)
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
    if more_data:
        return engine_nk, t, point_s, point_t, iter_step
    else:
        return values_f - data


def autofit_fp_poly(turns, losses, dt, I0, I_max, iter_step, exp_0, method):
    params = Parameters()
    params.add("exponent", value=exp_0, min=0, vary=True)

    minner = Minimizer(fp_lmfit_poly, params, fcn_args=(
        turns, losses, dt, I0, I_max, iter_step))
    result = minner.minimize(method=method)
    final = losses + result.residual
    return result, final

# Fokker-Plank (POLYNOMIAL WITH I_STAR)

def fp_poly_istar(params, dt, I0, I_max, iter_step, multiplier, iters, iter_mult, actual_iters, losses, t_start, t_end):
    # Gathering parameters
    exponent = params["exponent"]
    istar = params["istar"]
    
    engine_nk = pl.cn_polynomial(I_max, multiplier, exponent, I0, dt, I_star=istar)
    survival = []
    for i in range(len(iters) * iter_mult):
        survival.append(1.0 - engine_nk.get_particle_loss())
        engine_nk.iterate(iter_step)
    f = interp1d(range(len(survival)), survival, kind="cubic")
    
    t = np.linspace(t_start, t_start + (t_end - t_start) * iter_mult, actual_iters)
    return f(t)

def fp_lmfit_poly_istar(params, x, data, dt, I0, I_max, iter_step, hold_tight=False, more_data=False):
    # Gathering parameters
    exponent = params["exponent"]
    istar = params["istar"]
    
    # Declaring the engine
    engine_nk = pl.cn_polynomial(I_max, 1.0, exponent, I0, dt, I_star=istar)
    
    multiplier = 0.01 / integrate.simps(engine_nk.diffusion, np.linspace(0, I_max, len(I0)))

    while True:
        engine_nk = pl.cn_polynomial(I_max, multiplier, exponent, I0, dt, I_star=istar)
        
        # Allocating lists and counter
        t = []
        survival = []
        # Starting while loop for fitting procedure
        step = 0
        reached = True
        while(1.0 - engine_nk.get_particle_loss() >= data[-1]):
            # Append the data
            t.append(step)
            survival.append(1.0 - engine_nk.get_particle_loss())
            # Iterate
            engine_nk.iterate(iter_step)
            # Evolve counter
            step += 1
            if not hold_tight and step == 10000:
                #print("End not reached!")
                reached = False
                break
        # Append one last time
        t.append(step)
        survival.append(1.0 - engine_nk.get_particle_loss())
        if len(t) > 10:
            break
        else:
            #print("decrease multiplier")
            multiplier /= 10
            # If everything is just... shitty
            if multiplier < 1e-12:
                if more_data:
                    return engine_nk, t, 0, t[-1], iter_step
                else:
                    return data
            
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
    if more_data:
        return engine_nk, t, point_s, point_t, iter_step
    else:
        return values_f - data


def autofit_fp_poly_istar(turns, losses, dt, I0, I_max, iter_step, exp_0, method):
    params = Parameters()
    params.add("exponent", value=exp_0, min=0, vary=True)
    params.add("istar", value=1.0, min=0, vary=True)

    minner = Minimizer(fp_lmfit_poly_istar, params, fcn_args=(
        turns, losses, dt, I0, I_max, iter_step))
    result = minner.minimize(method=method)
    final = losses + result.residual
    return result, final