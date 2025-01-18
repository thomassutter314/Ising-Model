"""
Author: Thomas Sutter
Description:
    Code here creates the class "Model" that allows one 
    to initialize an Ising model and execute time evolution upon it.
    This script contains most of the heavier computational functions of this project
    
    This mini version focuses on having a region surrounded by an infinite temperature boundary condition
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

class Model_2d():
    # temperature: dimensionless temp of Ising model, Tc of the prisinte model is around 2
    # mass: 32 bit float square image that contains the map of the local "mass" (stricly positive). This is related to the local Tc by Tc ~ 1/mass
    # field: 32 bit float square image that contains the map of the local "field" (either sign). This linearly couples to the order parameter.
    def __init__(self, temperature, mass, field,
                 step_number = 5000, initial_step_number = 10000, exposure_time = 10,
                 save_loc = 'results//default', real_space_images_to_save = 100):
        
        # Make sure step_number is a multiple of the exposure time
        if step_number%exposure_time != 0:
            self.step_number = int(exposure_time*(step_number//exposure_time))
            print(f'Step_number must be a multiple of the exposure_time \nchanging step_number from {step_number} to {self.step_number}')
        else:
            self.step_number = step_number
        
        self.initial_step_number = initial_step_number
        
        self.temp = temperature
        print(f'Temperature = {self.temp}')
        
        self.exposure_time = exposure_time
        print(f'Exposure Time = {self.exposure_time}')
        
        # Where to save the data from the run
        self.save_loc = save_loc
        # check if the directories actually exists
        if not os.path.isdir(self.save_loc):
            os.makedirs(self.save_loc)
            os.makedirs(f'{self.save_loc}//time_sequence')
        if not os.path.isdir(f'{self.save_loc}//time_sequence'):
            os.makedirs(f'{self.save_loc}//time_sequence')
        print(f'Save Loc = {self.save_loc}')
       
        self.save_distance = self.step_number//real_space_images_to_save
        if self.save_distance == 0:
            self.save_distance = 1
        
        self.update_fraction = 1
        
        if mass.shape != field.shape:
            print('ERROR: mass and field must be the same shape')
            return
            
        if mass.shape[0] != mass.shape[1]:
            print('ERROR: mass and field must have square shape')
            return
        
        self.L = mass.shape[0]
        self.config = 2*np.random.randint(2, size=(self.L,self.L)) - 1 # initial state of the simulation is random
        self.config_save = np.zeros([real_space_images_to_save + 2, self.config.shape[0], self.config.shape[1]])
        self.field = field
        self.mass = mass
        
        self.m = np.empty(self.step_number) # place to store the pixel sum inside an roi of the C_size radius
        self.m_mask = np.ones(self.config.shape)
        
        # Run the simulation
        self.simulate()
        
        # Save an average of all the diffraction images
        tifffile.imwrite(f'{self.save_loc}//avg_diffraction.tiff', np.array(np.average(self.diffraction, axis = 0), dtype = np.float32))
        
        # Save diffraction intensity vs. step number
        np.savetxt(f'{self.save_loc}//diffraction_intensity.csv', np.mean(self.diffraction, axis = (1,2)))
        
        # Save the magnetization vs. step data
        np.savetxt(f'{self.save_loc}//magnetization.csv', self.m)
        
        # Computes the g2 values
        self.measure_g2()
  
    def simulate(self):
        print('Starting Initial Evolution')
        t0 = time.perf_counter()
        
        # Perform some initial steps to generate a good initial state for the real simulation
        for i in range(self.initial_step_number):
            update_2d(self.config, self.field, self.mass, self.temp, int(self.update_fraction*self.L**2))
            
        # Create an array to store the diffraction images:
        self.diffraction = np.zeros([self.step_number//self.exposure_time, self.L, self.L])
        print('self.diffraction',self.diffraction.shape)
        image_index = -1
        save_index = -1
        
        # Perform the actual simulation steps
        for i in range(self.step_number):
            # Move to the next diffraction frame when this is triggered
            if i%self.exposure_time == 0:
                image_index += 1
            
            # Save a certain number of real space images
            if i%self.save_distance == 0:
                if save_index >= 0:
                    print('save_index', save_index)
                    tifffile.imwrite(f'{self.save_loc}\\time_sequence\\{i//self.save_distance - 1}.tiff', np.array(255*self.config_save[save_index], dtype = 'uint8'))
                save_index += 1
                
            update_2d(self.config, self.field, self.mass, self.temp, int(self.update_fraction*self.L**2))
            self.m[i] = np.sum(self.config*self.m_mask)
            
            self.diffraction[image_index] += (np.abs(np.fft.fftshift(np.fft.fft2((self.config[:self.L,:self.L]))))**2 - self.diffraction[image_index])/(i%self.exposure_time + 1)
            self.config_save[save_index] += ((self.config+1)//2 - self.config_save[save_index])/(i%self.save_distance + 1)
            
            if i%(int(0.01*self.step_number)) == 0:
                print(f'{int(100*i/self.step_number)} %, working on image index: {image_index}')
        
        tf = time.perf_counter()
        
        print(f'Simulation Run Time = {round(tf-t0,2)} s')
        print(f'Time per micro-step: {1e9*(tf-t0)/((self.step_number + self.initial_step_number)*int(self.update_fraction*self.L**2))} ns')
        print(f'Micro-steps per macro-step: {int(self.update_fraction*self.L**2)}')
        
    def measure_g2(self, flatten = False, normalize = False):
        # Get the times of the images from the file names
        times = np.linspace(0,self.diffraction.shape[0],self.diffraction.shape[0])
        Iframes = np.copy(self.diffraction)
        
        if flatten:
            # Gaussian filter the images
            Iframes_avg_gf = gaussian_filter(np.mean(Iframes, axis = 0), 1)
            Iframes = Iframes/Iframes_avg_gf
            
        if normalize:
            Iframes = Iframes/(np.sum(Iframes, axis = (1,2))[:,None,None])
    
        # Now compute the g2
        N = 3
        
        xcentroid = self.diffraction[0].shape[1]//2
        ycentroid = self.diffraction[0].shape[0]//2

        g2_all = []
        colormap = np.zeros(Iframes.shape[1:3], dtype = np.int64)
        
        for i in range(N):
            labeled_roi_array = np.zeros(Iframes.shape[1:], dtype = int)
            I = i + 1
            labeled_roi_array[ycentroid, xcentroid + I] = 1
            labeled_roi_array[ycentroid, xcentroid - I] = 1
            labeled_roi_array[ycentroid + I, xcentroid] = 1
            labeled_roi_array[ycentroid - I, xcentroid] = 1
    
            colormap = colormap + labeled_roi_array*(i+1)
            
            #XPCS parameters and run calculation
            num_levels = 10
            num_bufs = 12
            g2n, lag_steps = skbeam.core.correlation.multi_tau_auto_corr(num_levels, num_bufs, labeled_roi_array, Iframes)
            
            g2_all.append(g2n)
            
        # Convert g2_all to a sensible shape and structure
        g2_all = np.array(g2_all)[:,:,0]
        
        data_to_save = np.empty([g2_all.shape[0] + 1, g2_all.shape[1]])
        data_to_save[0,:] = lag_steps
        data_to_save[1:,:] = g2_all
        
        # Save the g2 values from the run
        np.savetxt(f'{self.save_loc}\\g2.csv', data_to_save, delimiter = ',', header = f'lag_steps, g2_vals')

class Model_3d():
    # temperature: dimensionless temp of Ising model
    # mass: 32 bit float square image that contains the map of the local "mass" (stricly positive). This is related to the local Tc by Tc ~ 1/mass
    # field: 32 bit float square image that contains the map of the local "field" (either sign). This linearly couples to the order parameter.
    def __init__(self, temperature, mass, field,
                 step_number = 5000, initial_step_number = 10000, exposure_time = 10,
                 save_loc = 'results//default', real_space_images_to_save = 100):
        
        # Make sure step_number is a multiple of the exposure time
        if step_number%exposure_time != 0:
            self.step_number = int(exposure_time*(step_number//exposure_time))
            print(f'Step_number must be a multiple of the exposure_time \nchanging step_number from {step_number} to {self.step_number}')
        else:
            self.step_number = step_number
        
        self.initial_step_number = initial_step_number
        
        self.temp = temperature
        print(f'Temperature = {self.temp}')
        
        self.exposure_time = exposure_time
        print(f'Exposure Time = {self.exposure_time}')
        
        # Where to save the data from the run
        self.save_loc = save_loc
        # check if the directories actually exists
        if not os.path.isdir(self.save_loc):
            os.makedirs(self.save_loc)
            os.makedirs(f'{self.save_loc}//time_sequence')
        if not os.path.isdir(f'{self.save_loc}//time_sequence'):
            os.makedirs(f'{self.save_loc}//time_sequence')
        print(f'Save Loc = {self.save_loc}')
       
        self.save_distance = self.step_number//real_space_images_to_save
        if self.save_distance == 0:
            self.save_distance = 1
        
        self.update_fraction = 1
        
        if mass.shape != field.shape:
            print('ERROR: mass and field must be the same shape')
            return
            
        if mass.shape[0] != mass.shape[1]:
            print('ERROR: mass and field must have square shape')
            return
        
        self.L = mass.shape[0]
        self.config = 2*np.random.randint(2, size=(self.L, self.L, self.L)) - 1 # initial state of the simulation is random
        self.config_save = np.zeros([real_space_images_to_save + 2, self.config.shape[0], self.config.shape[1], self.config.shape[2]])
        self.field = field
        self.mass = mass
        
        self.m = np.empty(self.step_number) # place to store the pixel sum inside an roi of the C_size radius
        self.m_mask = tifffile.imread('sim_settings//mask.tiff')
        # ~ self.m_mask = self.mass
        
        # Run the simulation
        self.simulate()
        
        # Save an average of all the diffraction images
        tifffile.imwrite(f'{self.save_loc}//avg_diffraction.tiff', np.array(np.average(self.diffraction, axis = 0), dtype = np.float32))
        
        # Save diffraction intensity vs. step number
        np.savetxt(f'{self.save_loc}//diffraction_intensity.csv', np.mean(self.diffraction, axis = (1,2)))
        
        # Save the magnetization vs. step data
        np.savetxt(f'{self.save_loc}//magnetization.csv', self.m)
        
        # Computes the g2 values
        self.measure_g2()
  
    def simulate(self):
        print('Starting Initial Evolution')
        t0 = time.perf_counter()
        
        # Perform some initial steps to generate a good initial state for the real simulation
        for i in range(self.initial_step_number):
            update_3d(self.config, self.field, self.mass, self.temp, int(self.update_fraction*self.L**3))
            
        # Create an array to store the diffraction images:
        self.diffraction = np.zeros([self.step_number//self.exposure_time, self.L, self.L, self.L])
        print('self.diffraction',self.diffraction.shape)
        image_index = -1
        save_index = -1
        
        # Perform the actual simulation steps
        for i in range(self.step_number):
            # Move to the next diffraction frame when this is triggered
            if i%self.exposure_time == 0:
                image_index += 1
            
            # Save a certain number of real space images
            if i%self.save_distance == 0:
                if save_index >= 0:
                    print('save_index', save_index)
                    tifffile.imwrite(f'{self.save_loc}\\time_sequence\\{i//self.save_distance - 1}.tiff', np.array(255*self.config_save[save_index], dtype = 'uint8'))
                save_index += 1
                
            update_3d(self.config, self.field, self.mass, self.temp, int(self.update_fraction*self.L**3))
            self.m[i] = np.sum(self.config*self.m_mask)
            
            # ~ self.diffraction[image_index] += (np.abs(np.fft.fftshift(np.fft.fft2((self.config[:self.L,:self.L,:self.L]))))**2 - self.diffraction[image_index])/(i%self.exposure_time + 1)
            self.diffraction[image_index] += (np.abs(sci.fft.fftn(self.config))**2 - self.diffraction[image_index])/(i%self.exposure_time + 1)
            
            self.config_save[save_index] += ((self.config+1)//2 - self.config_save[save_index])/(i%self.save_distance + 1)
            
            if i%(int(0.01*self.step_number)) == 0:
                print(f'{int(100*i/self.step_number)} %, working on image index: {image_index}')
        
        tf = time.perf_counter()
        
        print(f'Simulation Run Time = {round(tf-t0,2)} s')
        print(f'Time per micro-step: {1e9*(tf-t0)/((self.step_number + self.initial_step_number)*int(self.update_fraction*self.L**2))} ns')
        print(f'Micro-steps per macro-step: {int(self.update_fraction*self.L**2)}')
        
    def measure_g2(self, flatten = False, normalize = False):
        print(self.diffraction.shape)
        # ~ times = np.linspace(0, self.diffraction.shape[0], self.diffraction.shape[0])
        xcentroid = 0
        ycentroid = 0
        zcentroid = 0
        
        Iframes = np.copy(self.diffraction)[:, zcentroid, :, :] # Take and x-y cut
        print(f'Iframes shape: {Iframes.shape}')
        
        if flatten:
            # Gaussian filter the images
            Iframes_avg_gf = gaussian_filter(np.mean(Iframes, axis = 0), 1)
            Iframes = Iframes/Iframes_avg_gf
            
        if normalize:
            Iframes = Iframes/(np.sum(Iframes, axis = (1,2))[:,None,None])
    
        # Now compute the g2
        N = 3
        
        g2_all = []
        colormap = np.zeros(Iframes.shape[1:3], dtype = np.int64)
        
        for i in range(N):
            labeled_roi_array = np.zeros(Iframes.shape[1:], dtype = int)
            I = i
            labeled_roi_array[ycentroid, xcentroid + I] = 1
            # ~ labeled_roi_array[ycentroid, xcentroid - I] = 1
            # ~ labeled_roi_array[ycentroid + I, xcentroid] = 1
            # ~ labeled_roi_array[ycentroid - I, xcentroid] = 1
    
            colormap = colormap + labeled_roi_array*(i+1)
            
            #XPCS parameters and run calculation
            num_levels = 10
            num_bufs = 12
            g2n, lag_steps = skbeam.core.correlation.multi_tau_auto_corr(num_levels, num_bufs, labeled_roi_array, Iframes)
            
            g2_all.append(g2n[1:])
            
        # Convert g2_all to a sensible shape and structure
        g2_all = np.array(g2_all)[:,:,0]
        
        data_to_save = np.empty([g2_all.shape[0] + 1, g2_all.shape[1]])
        data_to_save[0,:] = lag_steps[1:]
        data_to_save[1:,:] = g2_all
        
        # Save the g2 values from the run
        np.savetxt(f'{self.save_loc}\\g2.csv', data_to_save, delimiter = ',', header = f'lag_steps, g2_vals')


@jit(nopython = True)
def update_2d(config, field, mass, temp, N):
    # Updates the input configuration by N many steps on random indices
    I = np.random.randint(0, config.shape[0], N)
    J = np.random.randint(0, config.shape[1], N)
    
    for n in range(N):
        nn = config[(I[n] + 1)%config.shape[0], J[n]] + config[(I[n] - 1)%config.shape[0], J[n]] + config[I[n], (J[n] + 1)%config.shape[1]] + config[I[n], (J[n] - 1)%config.shape[1]]
        rbf = np.exp(-2*config[I[n], J[n]]*(nn + field[I[n], J[n]])/(temp/(1+mass[I[n], J[n]]))) # Reciprocal of Boltzmann factor
        
        # Glauber dynamics
        if np.random.random() > 1/(1 + rbf):
            config[I[n], J[n]] *= -1

    return None # The config object is updated inside the function, no return

@jit(nopython = True)
def update_3d(config, field, mass, temp, N):
    # Updates the input configuration by N many steps on random indices
    I = np.random.randint(0, config.shape[0], N)
    J = np.random.randint(0, config.shape[1], N)
    K = np.random.randint(0, config.shape[2], N)
    
    for n in range(N):
        # Compute the sum over the 6 nearest neighbors
        nn = config[(I[n] + 1)%config.shape[0], J[n], K[n]] + config[(I[n] - 1)%config.shape[0], J[n], K[n]] + \
             config[I[n], (J[n] + 1)%config.shape[1], K[n]] + config[I[n], (J[n] - 1)%config.shape[1], K[n]] + \
             config[I[n], J[n], (K[n] + 1)%config.shape[2]] + config[I[n], J[n], (K[n] - 1)%config.shape[2]]
        
        rbf = np.exp(-2*config[I[n], J[n], K[n]]*(nn + field[I[n], J[n], K[n]])/(temp/(1+mass[I[n], J[n], K[n]]))) # Reciprocal of Boltzmann factor
        
        # Glauber dynamics
        if np.random.random() > 1/(1 + rbf):
            config[I[n], J[n], K[n]] *= -1

    return None # The config object is updated inside the function, no return

# Generate the quenched random field and random mass that are used in the simulation
def generate_field_2d(L, impurities, impurity_size, impurity_field_strength, impurity_mass_strhgength, impurity_type):
    x, y = np.arange(0, L), np.arange(0, L)
    Y, X = np.meshgrid(x, y)
    field = np.zeros(X.shape)
    mass = np.zeros(X.shape)
    print(impurity_size)
    sigma_dist = np.random.normal(loc = impurity_size, scale = 0.5, size = len(impurities[0, :]))
    for n in range(len(impurities[0, :])):
        R = np.sqrt((X - impurities[0, n])**2 + (Y - impurities[1, n])**2)
        field += impurity_type[n]*impurity_field_strength*np.exp(-0.5*(R/sigma_dist[n])**2)
        mass += impurity_mass_strength*np.exp(-0.5*(R/sigma_dist[n])**20)
        
    # ~ mass[X > 0.7*mass.shape[0]] = (impurity_mass_strength - temp)
    # ~ mass[X < 0.1*mass.shape[0]] = (impurity_mass_strength - temp)
    # ~ mass[0,:] = -0.5
    # ~ mass[:,0] = -0.5
    
    return field, mass 

# Generate the quenched random field and random mass that are used in the simulation
def generate_field_3d(L, impurities, impurity_size, impurity_field_strength, impurity_mass_strhgength, impurity_type):
    x, y, z = np.arange(0, L), np.arange(0, L), np.arange(0, L)
    Z, Y, X = np.meshgrid(x, y, z)
    field = np.zeros(X.shape)
    mass = np.zeros(X.shape)
    # ~ print(impurity_size)
    # ~ sigma_dist = np.random.normal(loc = impurity_size, scale = 0.5, size = len(impurities[0, :]))
    # ~ for n in range(len(impurities[0, :])):
        # ~ R = np.sqrt((X - impurities[0, n])**2 + (Y - impurities[1, n])**2)
        # ~ field += impurity_type[n]*impurity_field_strength*np.exp(-0.5*(R/sigma_dist[n])**2)
        # ~ mass += impurity_mass_strength*np.exp(-0.5*(R/sigma_dist[n])**20)
        
    # ~ mass[X > 0.7*mass.shape[0]] = (impurity_mass_strength - temp)
    # ~ mass[X < 0.1*mass.shape[0]] = (impurity_mass_strength - temp)
    # ~ mass[0,:] = -0.5
    # ~ mass[:,0] = -0.5
    
    return field, mass 

def analyze_g2_curves(fname = r"results//default//g2.csv", cmap = cm.plasma):
    data = np.loadtxt(fname, delimiter = ',')
    lag_steps, g2 = data[0, :], data[1:, :]
    
    # compute the square of the intermediate scattering function from the data
    F2 = ((g2-1)/(g2[:, 0, None] - 1))
    print(f'F2.shape = {F2.shape}')
    
    fig, axs = plt.subplots(2)
    
    p0 = [10, 1]
    for i in range(F2.shape[0]):
        try:
            popt, pcov = opt.curve_fit(utils.stretched_exp, lag_steps, F2[i], p0 = p0)
        except:
            popt = p0
        tt = np.linspace(min(lag_steps), max(lag_steps), 1000)
        yy = utils.stretched_exp(tt, *popt)
        
        color = cmap(i/(F2.shape[0] - 1))
        axs[0].scatter(lag_steps, F2[i], color = color)
        axs[0].plot(tt, yy, c = color, label = f'tau = {round(popt[0], 3)}\nbeta = {round(popt[1], 3)}')
        
        axs[1].scatter(lag_steps, g2[i], color = color)
        
    axs[0].legend()
    plt.show()
        
def analyze_magnetization(fname = r"results//default//magnetization.csv"):
    data = np.loadtxt(fname)
    print('<m^2>', np.mean(data**2))
    plt.plot(data)
    plt.show()

def go(temperatures):
    field = tifffile.imread(f'sim_settings//field.tiff')
    mass = tifffile.imread(f'sim_settings//mass.tiff')
    
    for i in range(len(temperatures)):
        temperature = temperatures[i]
        
        # ~ model = Model_3d(temperature = temperature,
                  # ~ step_number = 100000, initial_step_number = 2500, exposure_time = 10,
                  # ~ real_space_images_to_save = 100,
                  # ~ mass = mass, field = field)
                  
        model = Model_3d(temperature = temperature,
            step_number = 2000000, initial_step_number = 2500, exposure_time = 100,
            real_space_images_to_save = 1000,
            mass = mass, field = field)
                  
        plotting.set_box_plots_3d(r'results\default\time_sequence',  diagF = False)
    
        # ~ analyze_g2_curves(r'results\default\g2.csv')
        
        analyze_magnetization(r'results//default//magnetization.csv')
                  
        string = "{:.2f}".format(temperature)
        os.rename('results//default', f'results//T{string}_1')


    
# ~ go(temperatures = [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0])
# ~ go(temperatures = [4.90, 4.95, 5.00, 5.05])
go(temperatures = [5.0])


# ~ analyze_magnetization(r'C:\Users\kogar\OneDrive\Documents\GitHub\Ising-Model\results\random_mass\T5.0_1\magnetization.csv')


