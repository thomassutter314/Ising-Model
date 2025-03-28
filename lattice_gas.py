"""
Description:
    Code simulates a lattice gas, this is used to create the initial random Tc and random field disorder distributions for the Ising model
"""

# Path for imageJ: C:\Users\thoma\Documents\GitHub\Ising-Model\results\default\time_sequence

import numpy as np
from numpy.random import rand
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import tifffile
from scipy.ndimage import gaussian_filter
from scipy import optimize as opt
import scipy as sci
import os
import time
from numba import jit

# Configure plot settings
from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=12)

from scikit_beam_master import skbeam
from scikit_beam_master.skbeam.core import correlation as corr

import utils
import plotting

@jit(nopython = True)
def compute_config_energy(pars, L, alpha = 100):
    U = 0
    for id1 in range(pars.shape[0]):
        for i in range(pars.shape[0] - 1 - id1):
            id2 = (id1 + 1 + i)%pars.shape[0] # index of the 2nd particle
            Dsqr = 0
            for j in range(3):
                D_temp = pars[id1, j] - pars[id2, j]
                if L > 2*abs(D_temp):
                    Dsqr += D_temp**2
                else:
                    Dsqr += (L - abs(D_temp))**2
            
            if Dsqr > 0:
                U += -alpha/np.sqrt(Dsqr)
            else:
                U += np.inf
        
        # ~ Dsqr = 0
        # ~ s_vec = [30, 25, 5]
        # ~ for j in range(3):
            # ~ D_temp = pars[id1, j] - s_vec[j]
            # ~ if L > 2*abs(D_temp):
                # ~ Dsqr += D_temp**2
            # ~ else:
                # ~ Dsqr += (L - abs(D_temp))**2
                
        # ~ U += 0.2*Dsqr
                
    return U

@jit(nopython = True)
def evolve(pars, L, steps = 500, beta = 1):
    for n in range(pars.shape[0]):
        U0 = compute_config_energy(pars, L)
        
        pars_probe = np.copy(pars)
        
        # Choose a random face of the cube to be the probe direction
        dir_probe = np.random.randint(0,6)
        i = dir_probe//2
        j = dir_probe%2
        
        pars_probe[n, i] = (pars[n, i] + (2*j - 1))%L # move the particle along the probe direction in the probe configuration
        
        Udiff = compute_config_energy(pars_probe, L) - U0 # compute the energy difference between probe config and current config
        
        # ~ tprob = 1/(1 + np.exp(beta*Udiff)) # compute transition probability
        tprob = np.exp(-beta*Udiff)
        
        if tprob >= 1 or np.random.random() < tprob:
            pars[n, i] = (pars[n, i] + (2*j - 1))%L # with probability tprob, execute motion
        
def simulate_lattice_gas(L = 100, N = 1000, steps = 300):
    # Generate a collection of N particles in 3D
    pars = np.random.randint(0, L, size = [N, 3])
    
    for i in range(steps):
        if i%100 == 0:
            print(i)
        evolve(pars, L)
    
    np.save('pars.npy',pars)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(pars[:, 0], pars[:, 1], pars[:, 2])
    
    ax.set_xlim3d(0, L)
    ax.set_ylim3d(0, L) 
    ax.set_zlim3d(0, L) 
    plt.show()

def load_pars_and_combine(pars_locs, L = 30):
    pars = np.load(pars_locs[0])
    for i in range(len(pars_locs) - 1):
        pars_new = np.load(pars_locs[i + 1])
        pars = np.append(pars, pars_new, axis = 0)
    
    np.save('pars_combine.npy', pars)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(pars[:, 0], pars[:, 1], pars[:, 2])
    
    ax.set_xlim3d(0, L)
    ax.set_ylim3d(0, L) 
    ax.set_zlim3d(0, L) 
    plt.show()
    
# Loads in the pars and makes random mass voxel images
def load_and_construct_mass(pars_loc = 'pars.npy', L = 100, A = 0.25, sigma = 4, power = 4):
    pars = np.load(pars_loc)
    print(f'pars shape: {pars.shape}')
    
    mass = np.zeros([L, L, L])
    
    Z, Y, X = np.indices(mass.shape)
    for n in range(pars.shape[0]):
        x0, y0, z0 = pars[n]
        
        mass += A*np.exp(-0.5*((X - x0)**power + (Y - y0)**power + (Z - z0)**power)/sigma**power)
    
    print(f'max mass = {np.max(mass)}')
    tifffile.imwrite('sim_settings//mass.tiff', np.array(mass, dtype = np.float32))
    
    fig = plt.figure(figsize=(15, 3))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plotting.box_plot_3d(255*mass, ax)
    plt.show()

# Loads in the pars and makes random field voxel images
def load_and_construct_field(pars_loc = 'pars.npy', L = 100, A = 1, sigma = 1, power = 2):
    pars = np.load(pars_loc)
    signs = 2*np.random.randint(0, 2, len(pars)) - 1
    print(f'pars shape: {pars.shape}')
    
    field = np.zeros([L, L, L])
    
    Z, Y, X = np.indices(field.shape)
    for n in range(pars.shape[0]):
        x0, y0, z0 = pars[n]
        
        field += signs[n]*A*np.exp(-0.5*((X - x0)**power + (Y - y0)**power + (Z - z0)**power)/sigma**power)
    
    print(f'max field = {np.max(field)}, min field = {np.min(field)}')
    tifffile.imwrite('sim_settings//field.tiff', np.array(field, dtype = np.float32))
    
    fig = plt.figure(figsize=(15, 3))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plotting.box_plot_3d(255*field, ax)
    plt.show()

# ~ simulate_lattice_gas(steps = 1)

# ~ load_pars_and_combine(['pars_1.npy', 'pars_2.npy', 'pars_3.npy', 'pars_4.npy', 'pars_5.npy', 'pars_6.npy', 'pars_7.npy', 'pars_8.npy', 'pars_9.npy', 'pars_10.npy'])
# ~ load_pars_and_combine(['pars_1.npy','pars_2.npy', 'pars_4.npy', 'pars_5.npy', 'pars_6.npy', 'pars_7.npy', 'pars_11.npy', 'pars_12.npy', 'pars_13.npy', 'pars_14.npy', 'pars_15.npy', 'pars_16.npy'])

# ~ load_and_construct_field(pars_loc = 'results//pars_combine.npy')
# ~ load_and_construct_mass(pars_loc = 'pars.npy')

load_and_construct_field(pars_loc = 'pars.npy')

