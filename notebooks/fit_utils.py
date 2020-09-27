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


def model_2_lmfit(params, x, data):
    rho = params["rho"]
    n0 = params["n0"]
    k = params["k"]
    model = model_2(x, rho, n0, k)
    return model - data


def fit_model_2(turns, DA):
    params = Parameters()
    params.add("rho", value=1, min=0, vary=True)
    params.add("n0", value=1, min=0, max=turns.min() * 0.5, vary=True)
    params.add("k", value=1, min=0, vary=True)
    minner = Minimizer(model_2_lmfit, params, fcn_args=(turns, DA))
    result = minner.minimize(method="basinhopping")
    final = DA + result.residual
    return result, final


def fit_model_2_fixed_n0(turns, DA):
    params = Parameters()
    params.add("rho", value=1, min=0, vary=True)
    params.add("n0", value=1, vary=False)
    params.add("k", value=1, min=0, vary=True)
    minner = Minimizer(model_2_lmfit, params, fcn_args=(turns, DA))
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


def model_4_lmfit(params, x, data):
    rho = params["rho"]
    k = params["k"]
    model = model_4(x, rho, k)
    return model - data


def fit_model_4(turns, DA):
    params = Parameters()
    params.add("rho", value=1, min=0, vary=True)
    params.add("k", value=1, min=0, vary=True)
    minner = Minimizer(model_4_lmfit, params, fcn_args=(turns, DA))
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


def model_4_free_lmfit(params, x, data):
    rho = params["rho"]
    k = params["k"]
    n0 = params["n0"]
    model = model_4_free(x, rho, n0, k)
    return model - data


def fit_model_4_free(turns, DA):
    params = Parameters()
    params.add("rho", value=1, min=0, vary=True)
    params.add("n0", value=1, min=0, max=turns.min()*0.5, vary=True)
    params.add("k", value=1, min=0, vary=True)
    minner = Minimizer(model_4_free_lmfit, params, fcn_args=(turns, DA))
    result = minner.minimize(method="basinhopping")
    final = DA + result.residual
    return result, final

# Fokker-Plank

def fp_lmfit(params, x, data, dt, I0, I_max, iter_step):
    # Gathering parameters
    k = params["k"]
    I_star = params["I_star"]

    print("k={}, I_star={}".format(k, I_star))

    # Declaring the engine
    engine_nk = nk.cn_nekhoroshev(I_max, 1.0, I_star, 1 / (k * 2), 0, I0, dt)
    multiplier = 0.1 / integrate.simps(engine_nk.diffusion, np.linspace(0, I_max, len(I0)))
    print(multiplier)
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
        print(t[-1])
        survival.append(1.0 - engine_nk.get_particle_loss())
        # Iterate
        engine_nk.iterate(iter_step)
        # Evolve counter
        step += 1
        if step == 50000:
            print("End not reached!")
            reached = False
            break
    # Append one last time
    t.append(step * iter_step * dt * multiplier)
    survival.append(1.0 - engine_nk.get_particle_loss())

    # Post processing
    if reached:
        f = interp1d(t, survival, kind="cubic")
        inv_f = inversefunc(
            f,
            domain=(t[0], t[-1]),
            image=(survival[-1], survival[0])
        )
        point_t = inv_f(data[-1])
        points_t = np.linspace(0, point_t, len(x))
        values_f = f(points_t)
    else:
        f = interp1d(t, survival, kind="cubic")
        points_t = np.linspace(0, t[-1], len(x))
        values_f = f(points_t)
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


def fp_fitting(lost_table, I_max, I0, k, I_star, dt, iter_step, turn_sampling):
    """Function for Fokker-Planck fitting given the fixed parameters

    Parameters
    ----------
    lost_table : ndarray
        lost data (without the 0.0 point!)
    I_max : float
        absorbing barrier location
    I0 : ndarray
        initial distribution in action
    k : float
        nek parameter
    I_star : float
        nek parameter
    dt : float
        dt in crank-nicolson
    iter_step : int
        number of iterations per step to do in crank-nicolson
    turn_sampling : ndarray
        turns at which the sampling was done (without the t_max point!)

    Returns
    -------
    tuple of data
        values_f, np.asarray(t)[::-1], error, step
    """    
    # Let's print!
    print("k={}, I_star={}, dt={}".format(k, I_star, dt))
    
    # fixing the data
    lost_table = np.concatenate((lost_table, [1.0]))
    turn_sampling = np.concatenate((turn_sampling, [0.0]))
    
    # Declaring the engine
    engine_nk = nk.cn_nekhoroshev(I_max, 1.0, I_star, 1 / (k * 2), 0, I0, dt)
    
    # Allocating lists and counter
    t = []
    survival = []
    
    # Starting while loop for fitting procedure
    step = 0
    while(1.0 - engine_nk.get_particle_loss() >= lost_table[0]):
        # Append the data
        t.append(step * iter_step * dt)
        survival.append(1.0 - engine_nk.get_particle_loss())
        # Iterate
        engine_nk.iterate(iter_step)
        # Evolve counter
        step += 1
        if step == 10000:
            print("This thing is going for a LONG ride!")
    # Append one last time
    t.append(step * iter_step * dt)
    survival.append(1.0 - engine_nk.get_particle_loss())
    # Post processing
    f = interp1d(t, survival, kind="cubic")
    inv_f = inversefunc(
        f,
        domain=(t[0], t[-1]),
        image=(survival[-1], survival[0])
    )
    point_t = inv_f(lost_table[0])
    points_t = np.linspace(0, point_t, len(turn_sampling))
    values_f = f(points_t)
    values_f = values_f[::-1]
    error = chi_2(lost_table, values_f)
    # Returning the thing
    return values_f, np.asarray(t)[::-1], error, step


def scan_I_star(lost_table, turn_sampling, I_max, I0, k, I_star_start, dt, iter_step, step_proportion=0.05):
    """Automated I_star fitting procedure for FP process.

    Parameters
    ----------
    lost_table : ndarray
        lost data
    turn_sampling : ndarray
        turn_sampling for lost_table
    I_max : float
        absorbing barrier location
    I0 : ndarray
        initial distribution
    k : flaot
        nek parameter
    I_star_start : float
        starting point for the search
    dt : float
        starting time scale for cn
    iter_step : int
        number of cn iterations per step
    step_proportion : float, optional
        multiplier factor for the research procedure, by default 0.05

    Returns
    -------
    tuple
        values, I_star, t, error
    """    
    # Setup lists
    I_star_sampled = []
    errors = []
    # Setup first extra 2 I_star values
    I_star_up = I_star_start * (1 + step_proportion)
    I_star_down = I_star_start * (1 - step_proportion)

    # first 3 analysis
    values_down, t_down, error_down, step_down = fp_fitting(
        lost_table, I_max, I0, k, I_star_down, dt, iter_step, turn_sampling)
    if step_down > 1000:
        print("down increase!")
        dt *= 10
    values_start, t_start, error_start, step_start = fp_fitting(
        lost_table, I_max, I0, k, I_star_start, dt, iter_step, turn_sampling)
    if step_start > 1000:
        print("start increase!")
        dt *= 10
    values_up, t_up, error_up, step_up = fp_fitting(
        lost_table, I_max, I0, k, I_star_up, dt, iter_step, turn_sampling)
    if step_up > 1000:
        print("up increase!")
        dt *= 10

    I_star_sampled.append(I_star_start)
    errors.append(error_start)

    if error_up < error_start and error_up < error_down:
        print("Going UP!")
        I_star_sampled.append(I_star_up)
        errors.append(error_up)
        I_star_now = I_star_up
        values_now = values_up
        t_now = t_up
        while errors[-1] < errors[-2]:
            I_star_new = I_star_now * (1 + step_proportion)
            values, t, error, step = fp_fitting(
                lost_table, I_max, I0, k, I_star_new, dt, iter_step, turn_sampling)
            if error > errors[-1]:
                I_star_sampled.append(I_star_new)
                errors.append(error)
                break
            values_now = values
            t_now = t
            I_star_now = I_star_new
            I_star_sampled.append(I_star_now)
            errors.append(error)
            if step > 1000:
                print("increase!")
                dt *= 10
        return values_now, I_star_now, t_now, np.min(errors)

    elif error_down < error_start and error_down < error_up:
        print("Going DOWN!")
        I_star_sampled.append(I_star_down)
        errors.append(error_down)
        I_star_now = I_star_down
        values_now = values_down
        t_now = t_down
        while errors[-1] < errors[-2]:
            I_star_new = I_star_now * (1 - step_proportion)
            values, t, error, step = fp_fitting(
                lost_table, I_max, I0, k, I_star_new, dt, iter_step, turn_sampling)
            if error > errors[-1]:
                I_star_sampled.append(I_star_new)
                errors.append(error)
                break
            values_now = values
            t_now = t
            I_star_now = I_star_new
            I_star_sampled.append(I_star_now)
            errors.append(error)
            if step > 1000:
                print("increase!")
                dt *= 2
        return values_now, I_star_now, t_now, np.min(errors)

    else:
        print("STAYNG HERE!")
        return values_start, I_star_start, t_start, np.min(errors)


def scan_k(lost_table, turn_sampling, I_max, I0, k_start, I_star_start, dt, iter_step, step_proportion=0.05, k_proportion_step=0.05):
    """Automated k fitting procedure for FP process (with nested I_star fitting)

    Parameters
    ----------
    lost_table : ndarray
        lost data
    turn_sampling : ndarray
        turn sampling for lost_table
    I_max : float
        absorbing barrier location
    I0 : ndarray
        initial distribution
    k_start : float
        starting point for k
    I_star_start : float
        general starting point for I_star
    dt : float
        starting time scale for cn
    iter_step : int
        number of cn iterations per step
    step_proportion : float, optional
        multplier factor for research procedure in I_star, by default 0.05
    k_proportion_step : float, optional
        multiplier factor for research procedure in k, by default 0.05

    Returns
    -------
    tuple
        values, k, I_star, t, error
    """    
    k_up = k_start * (1 + k_proportion_step)
    k_down = k_start * (1 - k_proportion_step)

    I_star_default = 0.2

    k_sampled = []
    I_star_selected = []
    errors = []

    dt = 0.001

    #first 3 steps
    values_down, I_star_down, t_down, error_down = scan_I_star(
        lost_table, turn_sampling, I_max, I0, k_down, I_star_default, dt, iter_step)
    values_start, I_star_start, t_start, error_start = scan_I_star(
        lost_table, turn_sampling, I_max, I0, k_start, I_star_default, dt, iter_step)
    values_up, I_star_up, t_up, error_up = scan_I_star(
        lost_table, turn_sampling, I_max, I0, k_up, I_star_default, dt, iter_step)

    k_sampled.append(k_start)
    I_star_selected.append(I_star_start)
    errors.append(error_start)

    if error_up < error_start and error_up < error_down:
        print("Going UP with k!!!")
        k_sampled.append(k_up)
        I_star_selected.append(I_star_up)
        errors.append(error_up)
        k_now = k_up
        I_star_now = I_star_up
        values_now = values_up
        t_now = t_up
        while errors[-1] < errors[-2]:
            k_new = k_now * (1 + k_proportion_step)
            values, I_star, t, error = scan_I_star(
                lost_table, turn_sampling, I_max, I0, k_new, I_star_default, dt, iter_step)
            if error > errors[-1]:
                k_sampled.append(k_new)
                I_star_selected.append(I_star)
                errors.append(error)
                break
            values_now = values
            t_now = t
            k_now = k_new
            I_star_now = I_star
            k_sampled.append(k_now)
            I_star_selected.append(I_star_now)
            errors.append(error)
        return values_now, k_now, I_star_now, t_now, np.min(errors)

    elif error_down < error_start and error_down < error_up:
        print("Going DOWN with k!!!")
        k_sampled.append(k_down)
        I_star_selected.append(I_star_down)
        errors.append(error_down)
        k_now = k_down
        I_star_now = I_star_down
        values_now = values_down
        t_now = t_down
        while errors[-1] < errors[-2]:
            k_new = k_now * (1 - k_proportion_step)
            values, I_star, t, error = scan_I_star(
                lost_table, turn_sampling, I_max, I0, k_new, I_star_default, dt, iter_step)
            if error > errors[-1]:
                k_sampled.append(k_new)
                I_star_selected.append(I_star)
                errors.append(error)
                break
            values_now = values
            t_now = t
            k_now = k_new
            I_star_now = I_star
            k_sampled.append(k_now)
            I_star_selected.append(I_star_now)
            errors.append(error)
        return values_now, k_now, I_star_now, t_now, np.min(errors)
    else:
        print("STAYNG HERE!")
        return values_start, k_start, I_star_start, t_start, error_start
