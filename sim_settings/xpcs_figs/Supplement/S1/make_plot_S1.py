import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
import tifffile

from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy import optimize as opt

import emcee
import corner

# Configure plot settings
from matplotlib import rcParams
import os
#rcParams.update({'figure.autolayout': True})
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


def go1():
    offset = 0
    step = 0.22
    cmap = cm.viridis_r
    colors_for_qs = [cmap(offset),cmap(step+offset),cmap(2*step+offset),cmap(3*step+offset),cmap(4*step+offset)]
    markers_for_qs = ['s','o','^', 'D', 'P']
    
    # ~ data = np.loadtxt('data_beta.csv', delimiter = ',', skiprows = 1)
    data = np.loadtxt('data_tau_beta.csv', delimiter = ',', skiprows = 2)
    temps = data[:, 0]
    indices = np.argsort(temps)
    temps = temps[indices]
    data = data[indices, :]
    
    print(np.mean(data[:, 4:7]))
    
    tau_avg = np.mean(data[:, 1:4], axis = 1)
    beta_avg = np.mean(data[:, 4:7], axis = 1)
    print('data.shape', data.shape)
    print('total average beta ', np.mean(beta_avg))
    print('total average beta ', np.std(beta_avg))
    
    # Create a figure and specify the gridspec
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])
    
    # First column, first figure (top)
    ax1 = plt.subplot(gs[0, 0])
    
    # First column, second figure (bottom)
    ax2 = plt.subplot(gs[1, 0])
    
    # Second column, spanning both rows (single large figure)
    ax3 = plt.subplot(gs[:, 1])
    
    for i in range(3):
        ax2.scatter(temps, data[:, i+4], fc = colors_for_qs[i], ec = 'black', s = 25, alpha = 0.7)
    
    ax2.set_ylabel(r'$\beta$')
    ax2.plot(temps, beta_avg, color = 'black')
    ax2.set_xlabel(r'Temperature (K)')
    
    for i in range(3):
        ax1.scatter(temps, data[:, i+1]/60, fc = colors_for_qs[i], ec = 'black', s = 25, alpha = 0.7, label = f'{round(10.4*(i+1), 2)}' + r' $\mu$m$^{-1}$')
    
    ax1.set_ylabel(r'$\tau$ (Minutes)')
    ax1.plot(temps, tau_avg/60, color = 'black')
    ax1.legend()
    
    
    flat_beta = np.ravel(data[:, 4:7])
    beta_flat_1 = data[:,4]
    beta_flat_2 = data[:,5]
    beta_flat_3 = data[:,6]
    
    # ~ ax3.hist(beta_flat_1, bins=5, histtype='step', linewidth=2, facecolor='none', 
             # ~ edgecolor=colors_for_qs[0],fill=True, label = f'mean = {round(np.mean(flat_beta), 2)}\nstdev = {round(np.std(flat_beta), 2)}')
             
    # ~ ax3.hist(beta_flat_2, bins=5, histtype='step', linewidth=2, facecolor='none', 
             # ~ edgecolor=colors_for_qs[1],fill=True, label = f'mean = {round(np.mean(flat_beta), 2)}\nstdev = {round(np.std(flat_beta), 2)}')
             
    # ~ ax3.hist(beta_flat_3, bins=5, histtype='step', linewidth=2, facecolor='none', 
             # ~ edgecolor=colors_for_qs[2],fill=True, label = f'mean = {round(np.mean(flat_beta), 2)}\nstdev = {round(np.std(flat_beta), 2)}')
    
    ax3.hist(flat_beta, bins=10, histtype='step', linewidth=2, facecolor='c', 
             hatch='/', edgecolor='k',fill=True, label = f'mean = {round(np.mean(flat_beta), 2)}\nstdev = {round(np.std(flat_beta), 2)}')
             
    # ~ ax3.hist(beta_avg, bins=10, histtype='step', linewidth=2, facecolor='k', 
             # ~ hatch='/', edgecolor='k',fill=True, label = f'mean = {round(np.mean(beta_avg), 2)}\nstdev = {round(np.std(beta_avg), 2)}')
    
    ax3.set_xlabel(r'$\beta$')
    ax3.set_ylabel('Counts')
    
    ax3.legend()
    plt.show()
    
def fit_func(t, tau, beta):
    return np.exp(-1*(t/tau)**beta)

def log_likelihood(theta, x, y, yerr):
    tau, beta = theta
    model = fit_func(x, tau, beta)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2)
    
def log_prior(theta):
    tau, beta = theta
    if tau < 0 or beta < 0:
        return -np.inf
    return 0
    
def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

def go2():
    g2 = np.loadtxt('95K_g2.txt', delimiter = ',')
    F2 = (g2-1)/np.mean(g2[:, :10, None] - 1, axis = 1)
    lagsteps = np.loadtxt('lag_steps.txt')[1:]
    
    p0 = [1000, 1]
    popt, pcov = curve_fit(fit_func, lagsteps, F2[1, :], p0 = p0)
    
    
    pos = popt + 1e-4 * np.random.randn(32, 2)
    nwalkers, ndim = pos.shape
    
    yerr = 2*0.025*np.ones(F2[1, :].shape)
    sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args = (lagsteps, F2[1, :], yerr))
    sampler.run_mcmc(pos, 5000, progress=True)
    
    fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = [r"$\tau$ (min)", r"$\beta$"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("step number");
    
    plt.show()
    
    flat_samples = sampler.get_chain(discard=100, thin=1, flat=True)
    print(flat_samples.shape)

    fig = corner.corner(
        flat_samples, labels=labels, quantiles = [0.5], smooth = True, plot_datapoints = False, levels = [.393, 0.864])
        
    plt.show()
    
    tt = np.linspace(min(lagsteps), max(lagsteps), 1000)
    yy = fit_func(tt, *popt)
    
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        plt.plot(tt, fit_func(tt, *sample), "-k", alpha=0.1)
    
    print(lagsteps.shape)
    print(F2.shape)
    plt.scatter(lagsteps, F2[1, :], facecolor = 'none', edgecolor = 'green', zorder = 10)
    plt.plot(tt, yy, '-b')
    plt.xscale('log')
    plt.ylabel(r'$F^2$')
    plt.xlabel(r'$\Delta t$ (s)')
    plt.show()
    
    
go2()
