"""
Author: Thomas Sutter
Description:
    Code here creates the class "Model" that allows one 
    to initialize an Ising model and execute time evolution upon it.
    This script contains most of the heavier computational functions of this project
"""

# Path for imageJ: C:\Users\thoma\Documents\GitHub\Ising-Model\results\default\time_sequence

import numpy as np
from numpy.random import rand
from numba import jit
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import tifffile
from scipy.ndimage import gaussian_filter
from scipy import optimize as opt
import os

# Configure plot settings
from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

# Package for computing g2, I'm a dummy so I literally just copied the package files in the directory here because I was having trouble installing it with pip....
import skbeam
import skbeam.core.correlation as corr

class Model_1():
    def __init__(self, L, step_number = 5000, initial_step_number = 10000, save_distance = 10,\
                 temperature = 2, C_concentration = 0, C_size = 1, \
                 config = None, save_loc = 'results//default', save_images = False):
        
        self.save_images = save_images             
        
        self.skip_fraction = 0.8 # % of sites skipped per update to remove pathological time crystal states
        
        self.C_motion_energy = 15
        # ~ self.C_motion_energy = 140
        
        # Where to save the data from the run
        self.save_loc = save_loc             
        
        self.L = L # Size of the lattice
        self.beta = 1/temperature
        
        # If no initial configuration is supplied, then one is randomly generated (sample of the infinite temperature model)
        # In the event that an initial configuration is supplied, then self.L is overridden by the shape of the supplied configuration (only square configurations should be supplied)
        if type(config) == type(None):
            self.config = 2*np.random.randint(2, size=(self.L,self.L))-1
        else:
            self.L = len(self.config) # Model length overridden to supplied configuration
            
        if self.config.shape[0] != self.config.shape[1]:
            print('ERROR: Initial configuration shape is not square')
            exit
        
        # For a given impurity concentration, convert to the total number of impurities on the lattice
        self.C_concentration = C_concentration
        self.C_count = int(self.L**2*C_concentration)
        
        # Make impurities, C gives their locations and C_type gives whether they are +1 or -1 impurities
        self.C = np.random.randint(self.L, size = (2,self.C_count))
        self.C_size_sqr = C_size**2
        self.C_type = (2*np.random.randint(2, size = (self.C_count)) - 1)
        
        
        self.config, self.affected_indices, self.impurity_to_site_list = set_impurities(self.config, self.C, self.C_size_sqr, self.C_type)
        
        self.average_impurity_site_number = 0
        for i in range(len(self.impurity_to_site_list)):
            self.average_impurity_site_number += len(self.impurity_to_site_list[i])
            
        self.average_impurity_site_number = self.average_impurity_site_number/len(self.impurity_to_site_list)
            
        self.settings_dict = {'lattice_length':self.L, 'temperature':1/self.beta,
                    'impurity_concentration': self.C_concentration, 'impurity_count': self.C_count, \
                    'impurity_size': C_size}
                    
        for keys in self.settings_dict:
            print(f'{keys}:{self.settings_dict[keys]}')
            
        # Save the metadata
        with open(f'{self.save_loc}//metadata.txt','w') as ssf:
            # Write the input settings
            ssf.write('Input Settings: \n \n')
            for key, value in self.settings_dict.items():
                ssf.write('%s:%s\n' % (key, value))
            
        # Make a mask for displaying where the impurities are located in the imshow plots
        self.C_mask = np.zeros(self.config.shape)
        self.C_mask[self.C[0], self.C[1]] = self.C_type
        
        # Save the impuriy locations
        tifffile.imwrite(f'{self.save_loc}\\impurities.tiff', np.array(self.C_mask, dtype = np.float32))
        
        plt.figure()
        plt.imshow(self.config, cmap = 'grey', vmin = -1, vmax = 1)
        plt.imshow(self.C_mask, cmap = 'cool', alpha = np.abs(self.C_mask))
        plt.title('Initial Configuration')
        plt.savefig(f'{self.save_loc}\\initial_config.tiff')
        plt.close()
        
        self.simulate(step_number, initial_step_number, save_distance)
        
        # Computes the g2 values
        self.measure_g2()
    
    def evolve(self):
        # Executes a single Monte Carlo step in which 1 minus the skip fraction percent of sites are updated
        
        # Compute map of coercion from all the nearest neighbors
        # ~ nn = (np.roll(self.config, 1, axis = 0) + np.roll(self.config, -1, axis = 0) + np.roll(self.config, 1, axis = 1) + np.roll(self.config, -1, axis = 1))
        
        nn = 1*(np.roll(self.config, 1, axis = 0) + np.roll(self.config, -1, axis = 0) + np.roll(self.config, 1, axis = 1) + np.roll(self.config, -1, axis = 1))
             # ~ (0.5)**(3)*(np.roll(self.config, (1,1), axis = (0,1)) + np.roll(self.config, (-1,1), axis = (0,1)) + np.roll(self.config, (1,-1), axis = (0,1)) + np.roll(self.config, (-1,-1), axis = (0,1))) + \
             # ~ (0.2)**(3)*(np.roll(self.config, (2,1), axis = (0,1)) + np.roll(self.config, (2,-1), axis = (0,1)) + np.roll(self.config, (-2,1), axis = (0,1)) + np.roll(self.config, (-2,-1), axis = (0,1))) + \
             # ~ (0.2)**(3)*(np.roll(self.config, (1,2), axis = (0,1)) + np.roll(self.config, (-1,2), axis = (0,1)) + np.roll(self.config, (1,-2), axis = (0,1)) + np.roll(self.config, (-1,-2), axis = (0,1))) + \
             # ~ (0.5)**(6)*(np.roll(self.config, 2, axis = 0) + np.roll(self.config, -2, axis = 0) + np.roll(self.config, 2, axis = 1) + np.roll(self.config, -2, axis = 1))
        
        rbf = np.exp(2*self.config*nn*self.beta) # Reciprocal of Boltzmann factor
        
        # Make a transfer map according to a sigmoid function
        s = 2*np.array(np.random.random(np.shape(self.config)) > 1/(1 + rbf), dtype = int) - 1
        # ~ s = 2*np.array(np.random.random(np.shape(self.config)) > 1/rbf, dtype = int) - 1
        
        # We need to skip some fraction of sites in the update to remove pathological "time crystal" solutions
        skip_indices = np.where(np.random.random(s.shape) < self.skip_fraction)
        s[skip_indices[0], skip_indices[1]] = 1
        
        # ~ s[self.C[0,:],self.C[1,:]] = 1
                
        for i in range(len(self.impurity_to_site_list)):
            impurity_price = np.sum(s[self.affected_indices[self.impurity_to_site_list[i], 0], self.affected_indices[self.impurity_to_site_list[i], 1]])/self.average_impurity_site_number
            rbf = np.exp(-1*(self.C_motion_energy)*self.beta)
            if np.random.random() > 1/(1 + rbf):
                self.C_type[i] *= -1
        
        # Execute the operation of the transfer map
        self.config *= s
        
        self.config[self.affected_indices[:,0], self.affected_indices[:,1]] = self.C_type[self.affected_indices[:,2]]
    
    def simulate(self, step_number, initial_step_number = 5000, save_distance = 100):
        print('Starting Initial Evolution')
        for i in range(initial_step_number):
            self.evolve()
        print('Initial Evolution Finished')
        
        self.diffraction = np.zeros([step_number//save_distance + 1, self.config.shape[0], self.config.shape[1]])
        image_index = 0
        for i in range(step_number):
            self.evolve()
            self.diffraction[image_index] += np.abs(np.fft.fftshift(np.fft.fft2(self.config)))**2
            if i%save_distance == 0:
                if i%(int(0.01*step_number)) == 0:
                    print(f'{int(100*i/step_number)} %, finished image index: {image_index}')
                if self.save_images:
                    # ~ tifffile.imwrite(f'{self.save_loc}\\diffraction\\{i}.tiff', np.array(np.fft.ifft2(np.sqrt(self.diffraction[image_index])), dtype = np.float32))
                    tifffile.imwrite(f'{self.save_loc}\\diffraction\\{i}.tiff', np.array(self.diffraction[image_index], dtype = np.float32))
                    tifffile.imwrite(f'{self.save_loc}\\time_sequence\\{i}.tiff', np.array((self.config+1)//2, dtype = bool))
                
                # ~ self.diffraction[image_index] += np.random.normal(loc = 0, scale = 1000, size = self.diffraction[image_index].shape)
                image_index += 1
    
    def measure_g2(self):
        # Get the times of the images from the file names
        times = np.linspace(0,self.diffraction.shape[0],self.diffraction.shape[0])
        Iframes = np.copy(self.diffraction)

        
        # ~ magnetization = np.average(realFrames, axis = (1,2))
        # ~ fig, ax = plt.subplots()
        # ~ ax.plot(magnetization)
        # ~ ax.set_ylabel('Magnetization')
        # ~ ax.set_xlabel('Time')
        
        # Gaussian filter the images
        Iframes_avg_gf = gaussian_filter(np.mean(Iframes, axis = 0), 1)
        # ~ Iframes_flat = Iframes/Iframes_avg_gf
        Iframes_flat = Iframes
        
        # Now compute the g2
        q_radius = 1
        # ~ q_radius = 1
        N = 4
        xcentroid = self.diffraction[0].shape[1]//2
        ycentroid = self.diffraction[0].shape[0]//2

        g2_all = []
        colormap = np.zeros(Iframes.shape[1:3], dtype = np.int64)
        
        for i in range(N):
            #data = Icrop.copy()/(np.sum(Icrop, axis = (1,2))[:,None,None]) # Normalized images
            # ~ data = Iframes_flat.copy()/(np.sum(Iframes_flat, axis = (1,2))[:,None,None]) # Normalized images
            data = Iframes_flat.copy()
            
            labeled_roi_array = getAnnulusMask(data[0],xcentroid,ycentroid,q_radius,i,i+1)
    
            colormap = colormap + labeled_roi_array*(i+1)
            
            #XPCS parameters and run calculation
            num_levels = 10
            num_bufs = 12
            g2n, lag_steps = corr.multi_tau_auto_corr(num_levels, num_bufs, labeled_roi_array, data[1:])
            
            g2_all.append(g2n[1:])
    
        # Convert g2_all to a sensible shape and structure
        g2_all = np.array(g2_all)[:,:,0]
        
        data_to_save = np.empty([g2_all.shape[0] + 1, g2_all.shape[1]])
        data_to_save[0,:] = lag_steps[1:]
        data_to_save[1:,:] = g2_all
        # Save the g2 values from the run
        np.savetxt(f'{self.save_loc}\\g2.csv', data_to_save, delimiter = ',', header = f'q_radius = {q_radius}')
            
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(np.log(np.mean(Iframes, axis = 0) + 1))
        axs[0].set_title('Peak, Log Scale')
        axs[1].imshow(colormap, cmap = cm.jet)
        axs[1].set_title('ROI Regions')
        #plt.imshow(labeled_roi_array,alpha=0.2,cmap=cm.gray)
    
    
        #################################
        fig, ax = plt.subplots(2)
        print(np.shape(g2_all))
        #F = np.sqrt((g2_all-1)/(g2_all[:,:2].mean(axis = 1)[:, None] - 1))
        F = np.sqrt((g2_all-1)/(g2_all[:,0][:, None] - 1))
    
        for i in range(F.shape[0]):
            ax[0].scatter(lag_steps[1:], F[i], label = f'q_radius = {i*q_radius}')
            ax[1].scatter(lag_steps[1:], g2_all[i], label = f'q_radius = {i*q_radius}')
            #plt.semilogx(lag_steps[1:], g2_all[i], label = f'q_radius = {i*q_radius}')
        
            # ~ popt, pcov = opt.curve_fit(sc_exponential, lag_steps[1:], F[i])
            # ~ tt = np.linspace(min(lag_steps[1:]), max(lag_steps[1:],1000))
            # ~ yy = sc_exponential(tt, *popt)
            # ~ ax.plot(tt, yy)
        
        # ~ ax[0].set_xscale('log')
        # ~ ax[1].set_xscale('log')
        # ~ ax.set_xlim([0,100])
        ax[0].set_xlabel(r'$\tau$ (s)')
        ax[0].set_ylabel(r'$F$')
        ax[1].set_ylabel(r'$g_2$')
        plt.legend()
        plt.show()
        
class Model_2():
    def __init__(self, L, step_number = 5000, initial_step_number = 10000, save_distance = 10,\
                 temperature = 2, C_concentration = 0, C_size = 1, \
                 config = None, save_loc = 'results//default', save_images = False):
        
        self.save_images = save_images             
        
        self.skip_fraction = 0.5 # % of sites skipped per update to remove pathological time crystal states
        
        self.C_motion_energy = 20
        # ~ self.C_motion_energy = 140
        
        # Where to save the data from the run
        self.save_loc = save_loc             
        
        self.L = L # Size of the lattice
        self.beta = 1/temperature
        
        # If no initial configuration is supplied, then one is randomly generated (sample of the infinite temperature model)
        # In the event that an initial configuration is supplied, then self.L is overridden by the shape of the supplied configuration (only square configurations should be supplied)
        if type(config) == type(None):
            self.config = 2*np.random.randint(2, size=(self.L,self.L))-1
        else:
            self.L = len(self.config) # Model length overridden to supplied configuration
            
        if self.config.shape[0] != self.config.shape[1]:
            print('ERROR: Initial configuration shape is not square')
            exit
        
        # For a given impurity concentration, convert to the total number of impurities on the lattice
        self.C_concentration = C_concentration
        self.C_count = int(self.L**2*C_concentration)
        
        # Make impurities, C gives their locations and C_type gives whether they are +1 or -1 impurities
        self.C = np.random.randint(self.L, size = (2,self.C_count))
        self.C_size_sqr = C_size**2
        self.C_type = (2*np.random.randint(2, size = (self.C_count)) - 1)
        
        
        self.config, self.affected_indices, self.impurity_to_site_list = set_impurities(self.config, self.C, self.C_size_sqr, self.C_type)
        
        self.average_impurity_site_number = 0
        for i in range(len(self.impurity_to_site_list)):
            self.average_impurity_site_number += len(self.impurity_to_site_list[i])
            
        self.average_impurity_site_number = self.average_impurity_site_number/len(self.impurity_to_site_list)
            
        self.settings_dict = {'lattice_length':self.L, 'temperature':1/self.beta,
                    'impurity_concentration': self.C_concentration, 'impurity_count': self.C_count, \
                    'impurity_size': C_size}
                    
        for keys in self.settings_dict:
            print(f'{keys}:{self.settings_dict[keys]}')
            
        # Save the metadata
        with open(f'{self.save_loc}//metadata.txt','w') as ssf:
            # Write the input settings
            ssf.write('Input Settings: \n \n')
            for key, value in self.settings_dict.items():
                ssf.write('%s:%s\n' % (key, value))
            
        # Make a mask for displaying where the impurities are located in the imshow plots
        self.C_mask = np.zeros(self.config.shape)
        self.C_mask[self.C[0], self.C[1]] = self.C_type
        
        # Save the impuriy locations
        tifffile.imwrite(f'{self.save_loc}\\impurities.tiff', np.array(self.C_mask, dtype = np.float32))
        
        plt.figure()
        plt.imshow(self.config, cmap = 'grey', vmin = -1, vmax = 1)
        plt.imshow(self.C_mask, cmap = 'cool', alpha = np.abs(self.C_mask))
        plt.title('Initial Configuration')
        plt.savefig(f'{self.save_loc}\\initial_config.tiff')
        plt.close()
        
        self.simulate(step_number, initial_step_number, save_distance)
        
        # Computes the g2 values
        self.measure_g2()
    
    def evolve(self):
        # Executes a single Monte Carlo step in which 1 minus the skip fraction percent of sites are updated
        
        # Compute map of coercion from all the nearest neighbors
        # ~ nn = (np.roll(self.config, 1, axis = 0) + np.roll(self.config, -1, axis = 0) + np.roll(self.config, 1, axis = 1) + np.roll(self.config, -1, axis = 1))
        
        nn = 1*(np.roll(self.config, 1, axis = 0) + np.roll(self.config, -1, axis = 0) + np.roll(self.config, 1, axis = 1) + np.roll(self.config, -1, axis = 1))
             # ~ (0.5)**(3)*(np.roll(self.config, (1,1), axis = (0,1)) + np.roll(self.config, (-1,1), axis = (0,1)) + np.roll(self.config, (1,-1), axis = (0,1)) + np.roll(self.config, (-1,-1), axis = (0,1))) + \
             # ~ (0.2)**(3)*(np.roll(self.config, (2,1), axis = (0,1)) + np.roll(self.config, (2,-1), axis = (0,1)) + np.roll(self.config, (-2,1), axis = (0,1)) + np.roll(self.config, (-2,-1), axis = (0,1))) + \
             # ~ (0.2)**(3)*(np.roll(self.config, (1,2), axis = (0,1)) + np.roll(self.config, (-1,2), axis = (0,1)) + np.roll(self.config, (1,-2), axis = (0,1)) + np.roll(self.config, (-1,-2), axis = (0,1))) + \
             # ~ (0.5)**(6)*(np.roll(self.config, 2, axis = 0) + np.roll(self.config, -2, axis = 0) + np.roll(self.config, 2, axis = 1) + np.roll(self.config, -2, axis = 1))
        
        rbf = np.exp(2*self.config*nn*self.beta) # Reciprocal of Boltzmann factor
        
        # Make a transfer map according to a sigmoid function
        s = 2*np.array(np.random.random(np.shape(self.config)) > 1/(1 + rbf), dtype = int) - 1
        
        # We need to skip some fraction of sites in the update to remove pathological "time crystal" solutions
        skip_indices = np.where(np.random.random(s.shape) < self.skip_fraction)
        s[skip_indices[0], skip_indices[1]] = 1
        
        # ~ s[self.C[0,:],self.C[1,:]] = 1
        
        # Execute the operation of the transfer map
        self.config *= s
        
        # ~ self.config[self.affected_indices[:,0], self.affected_indices[:,1]] = self.C_type[self.affected_indices[:,2]]
    
    def simulate(self, step_number, initial_step_number = 5000, save_distance = 100):
        print('Starting Initial Evolution')
        for i in range(initial_step_number):
            self.evolve()
        print('Initial Evolution Finished')
        
        self.diffraction = np.zeros([step_number//save_distance + 1, self.config.shape[0], self.config.shape[1]])
        image_index = 0
        for i in range(step_number):
            self.evolve()
            self.diffraction[image_index] += np.abs(np.fft.fftshift(np.fft.fft2(self.config)))**2
            if i%save_distance == 0:
                if i%(int(0.01*step_number)) == 0:
                    print(f'{int(100*i/step_number)} %, finished image index: {image_index}')
                if self.save_images:
                    # ~ tifffile.imwrite(f'{self.save_loc}\\diffraction\\{i}.tiff', np.array(np.fft.ifft2(np.sqrt(self.diffraction[image_index])), dtype = np.float32))
                    tifffile.imwrite(f'{self.save_loc}\\diffraction\\{i}.tiff', np.array(self.diffraction[image_index], dtype = np.float32))
                    tifffile.imwrite(f'{self.save_loc}\\time_sequence\\{i}.tiff', np.array((self.config+1)//2, dtype = bool))
                
                # ~ self.diffraction[image_index] += np.random.normal(loc = 0, scale = 1000, size = self.diffraction[image_index].shape)
                image_index += 1
    
    def measure_g2(self):
        # Get the times of the images from the file names
        times = np.linspace(0,self.diffraction.shape[0],self.diffraction.shape[0])
        Iframes = np.copy(self.diffraction)

        
        # ~ magnetization = np.average(realFrames, axis = (1,2))
        # ~ fig, ax = plt.subplots()
        # ~ ax.plot(magnetization)
        # ~ ax.set_ylabel('Magnetization')
        # ~ ax.set_xlabel('Time')
        
        # Gaussian filter the images
        Iframes_avg_gf = gaussian_filter(np.mean(Iframes, axis = 0), 1)
        # ~ Iframes_flat = Iframes/Iframes_avg_gf
        Iframes_flat = Iframes
        
        # Now compute the g2
        q_radius = 3
        # ~ q_radius = 1
        N = 4
        xcentroid = self.diffraction[0].shape[1]//2
        ycentroid = self.diffraction[0].shape[0]//2

        g2_all = []
        colormap = np.zeros(Iframes.shape[1:3], dtype = np.int64)
        
        for i in range(N):
            #data = Icrop.copy()/(np.sum(Icrop, axis = (1,2))[:,None,None]) # Normalized images
            # ~ data = Iframes_flat.copy()/(np.sum(Iframes_flat, axis = (1,2))[:,None,None]) # Normalized images
            data = Iframes_flat.copy()
            
            labeled_roi_array = getAnnulusMask(data[0],xcentroid,ycentroid,q_radius,i,i+1)
    
            colormap = colormap + labeled_roi_array*(i+1)
            
            #XPCS parameters and run calculation
            num_levels = 10
            num_bufs = 12
            g2n, lag_steps = corr.multi_tau_auto_corr(num_levels, num_bufs, labeled_roi_array, data[1:])
            
            g2_all.append(g2n[1:])
    
        # Convert g2_all to a sensible shape and structure
        g2_all = np.array(g2_all)[:,:,0]
        
        data_to_save = np.empty([g2_all.shape[0] + 1, g2_all.shape[1]])
        data_to_save[0,:] = lag_steps[1:]
        data_to_save[1:,:] = g2_all
        # Save the g2 values from the run
        np.savetxt(f'{self.save_loc}\\g2.csv', data_to_save, delimiter = ',', header = f'q_radius = {q_radius}')
            
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(np.log(np.mean(Iframes, axis = 0) + 1))
        axs[0].set_title('Peak, Log Scale')
        axs[1].imshow(colormap, cmap = cm.jet)
        axs[1].set_title('ROI Regions')
        #plt.imshow(labeled_roi_array,alpha=0.2,cmap=cm.gray)
    
    
        #################################
        fig, ax = plt.subplots(2)
        print(np.shape(g2_all))
        #F = np.sqrt((g2_all-1)/(g2_all[:,:2].mean(axis = 1)[:, None] - 1))
        F = np.sqrt((g2_all-1)/(g2_all[:,0][:, None] - 1))
    
        for i in range(F.shape[0]):
            ax[0].scatter(lag_steps[1:], F[i], label = f'q_radius = {i*q_radius}')
            ax[1].scatter(lag_steps[1:], g2_all[i], label = f'q_radius = {i*q_radius}')
            #plt.semilogx(lag_steps[1:], g2_all[i], label = f'q_radius = {i*q_radius}')
        
            # ~ popt, pcov = opt.curve_fit(sc_exponential, lag_steps[1:], F[i])
            # ~ tt = np.linspace(min(lag_steps[1:]), max(lag_steps[1:],1000))
            # ~ yy = sc_exponential(tt, *popt)
            # ~ ax.plot(tt, yy)
        
        # ~ ax[0].set_xscale('log')
        # ~ ax[1].set_xscale('log')
        # ~ ax.set_xlim([0,100])
        ax[0].set_xlabel(r'$\tau$ (s)')
        ax[0].set_ylabel(r'$F$')
        ax[1].set_ylabel(r'$g_2$')
        plt.legend()
        plt.show()
        
    
    
def getAnnulusMask(image,xc,yc,R,n,m):
    #set labeled_roi_arry to 1 in ROI
    print(np.shape(image))
    ylim, xlim = np.shape(image)
    x = np.arange(0,xlim,1)
    y = np.arange(0,ylim,1)
    X, Y = np.meshgrid(x, y)
    # 1's for greater than n and less than m

    mask2 = np.array(((X-xc))**2 + ((Y-yc))**2 >= (R*n)**2, dtype=int)
    mask1 = np.array(((X-xc))**2 + ((Y-yc))**2 <= (R*m)**2, dtype=int)
    
    mask = mask1*mask2
    #plt.imshow(labeled_roi_array)
    return mask

def sc_exponential(x, tau, beta):
    return np.exp(-1*((x-1)/tau)**beta)

def make_plots(fname = r"C:\Users\thoma\Documents\GitHub\Ising-Model\results\default\g2.csv"):
    data = np.loadtxt(fname, delimiter = ',')
    
    # ~ data =  np.loadtxt(r"C:\Users\thoma\OneDrive - UCLA IT Services\Desktop\OneDrive - UCLA IT Services\Research\TiSe2_XPCS\exhibits\A\95K_F.txt", delimiter = ',')
    
    print(np.shape(data))
    data = data[:,:-20]
    # ~ lag_steps = np.loadtxt(r"C:\Users\thoma\OneDrive - UCLA IT Services\Desktop\OneDrive - UCLA IT Services\Research\TiSe2_XPCS\exhibits\A\lag_steps.txt")
    # ~ print(np.shape(lag_steps))
    # ~ data[0,:] = lag_steps[1:]
    
    
    for i in range(data[1:].shape[0] - 1):
        plt.scatter(data[0], data[i+1])
    plt.show()
    
    fig, ax = plt.subplots()
    
    # ~ print(data)
    F = np.sqrt((data[1:]-1)/(data[1:,0][:, None] - 1))
    # ~ print(F)
    # ~ F = data[1:]
    

    
    # ~ print(F)
    
    # ~ print(lag_steps)
    
    cmap = cm.plasma
    tau_from_slope_beta1_assumed = []
    taus = []
    for i in range(F.shape[0]):
        tau_from_slope_beta1_assumed.append(-round(5/(F[i,5] - F[i,0]),1))
        xvals = np.log(data[0])[1:len(data[0])//2]
        yvals = np.log(-1*np.log(F[i]))[1:len(data[0])//2]
        
        # ~ F[F>1] = 0.99
        # ~ xvals = np.log(lag_steps[1:])[1:len(lag_steps)]
        # ~ yvals = np.log(-1*np.log(F[i]))[1:len(lag_steps)]
        
        # ~ print(yvals)
        # ~ ax.scatter(xvals, yvals, label = f'q_radius = {i*q_radius}')
        
        pol = np.polyfit(xvals, yvals, deg = 1)
        # ~ print(pol)
        tau, beta = np.exp(-1*pol[1]/pol[0]), pol[0]
        
        ax.scatter(data[0], F[i], color = cmap(i/F.shape[0]))
        
        try:
            popt, pcov = opt.curve_fit(sc_exponential, data[0], F[i], p0 = [tau, beta])
            taus.append(popt[0])
            
            tt = np.linspace(min(data[0]), max(data[0]), 10000)
            # ~ tt = np.linspace(0,1000,1000)
            # ~ yy = sc_exponential(tt, *[tau, beta])
            
            # ~ print(yy)
            yy = sc_exponential(tt, *popt)
            
            # ~ print(yy)
            # ~ ax.plot(tt, yy)
            plt.plot(tt, yy, label = f'tau = {round(popt[0])}, beta = {round(popt[1],2)}', color = cmap(i/F.shape[0]))
            # ~ plt.plot(tt, yy, label = f'tau = {round(popt[0])}', color = cmap(i/F.shape[0]))
            taus
        except:
            print('Error in fit')
        
    # ~ print(tau_from_slope_beta1_assumed)
    print(taus)
    
    
    ax.set_xlabel('time (steps)')
    ax.set_ylabel('F')
    ax.legend()
    # ~ ax.set_xscale('log')
    plt.show()

def set_impurities(config, impurities, impurity_size_sqr, impurity_type):
    affected_indices = []
    impurity_to_site_list = [] # 1st list index is for the impurity, 2nd list gives indices of the associated affected sites
    for i in range(config.shape[0]):
        for j in range(config.shape[1]):
            xsqr = np.min([(impurities[0] - i)**2, (config.shape[0] - abs(impurities[0] - i))**2], axis = 0)
            ysqr = np.min([(impurities[1] - j)**2, (config.shape[1] - abs(impurities[1] - j))**2], axis = 0)
            if impurity_size_sqr > np.min(xsqr + ysqr): # Checks if distance to closest impurity is within impurity size
                index = np.argmin(xsqr + ysqr) # Find index of the closest impurity
                config[i, j] = impurity_type[index]
                affected_indices.append([i, j, index])
    
    affected_indices = np.array(affected_indices)
    
    for i in range(len(impurity_type)):
        index_group = np.where(affected_indices[:,2] == i)[0]
        impurity_to_site_list.append(index_group)
    
    return config, affected_indices, impurity_to_site_list

def metametaplot():
    q = np.arange(0,10,1)
    # ~ tau_60 = np.array([23, 14, 15, 9, 10, 8, 9, 8, 8, 10])*60
    # ~ tau_30 = np.array([51, 34, 25, 22, 18, 14, 13, 14, 14, 15])*30
    # ~ tau_15 = np.array([])*15
    # ~ tau_1 = [1500, 960, 710, 508, 358, 310, 252, 173, 133, 95]
    
    tau_60 = np.array([596.5, 109.3, 25.4, 14.0, 11.4, 10.5, 10.9, 13.0, 16.1, 20.2])
    tau_60 *= tau_60
    
    tau_30 = np.array([45.92140082281508, 53.422192661234746, 38.019570366531674, 25.654510529247954, 21.211909898540263, 21.901743041123186, 22.741739215449922, 22.875607326640534, 23.714494438883637, 25.60509902893903])
    tau_30 *= 30
    
    tau_15 = np.array([14746.4, 219.9, 65.1, 30.3, 17.9, 12.9, 11.0, 10.6, 10.0, 10.1])
    tau_15 *= 15

    tau_1 = np.array([5298, 5298, 576, 143, 41, 16, 8, 5, 4, 3])*1
    
    plt.plot(q, tau_60, label = 'N = 60')
    plt.plot(q, tau_30, label = 'N = 30')
    plt.plot(q, tau_15, label = 'N = 15')
    plt.plot(q, tau_1, label = 'N = 1')
    plt.xlabel('q')
    plt.ylabel('tau')
    plt.legend()
    plt.show()

def metametametaplot():
    Ecu = [0, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 25, 30]
    beta = [0.58, 0.63, 0.62, 0.69, 0.83, 1.02, 1.17, 1.18, 1.16, 1.2, 1.11, 1.03, 0.71, 0.56, 0.56]
    
    plt.scatter(Ecu, beta, marker = 's', ec = 'black', fc = 'green', s = 35)
    plt.xlabel(r'$E_{\text{Cu}}$')
    plt.ylabel(r'$\beta$')
    plt.show()



def metaplot():
    
    # impurity strength = 1
    # ~ temps = [1, 1.5, 2, 2.5, 3, 3.5]
    # ~ tau_1 = [494303, 122360, 58917 , 1280, 40, 15]
    # ~ tau_2 = [147244, 94608, 12528 , 621, 31, 14]
    # ~ tau_3 = [47846, 14113, 5076, 286, 31, 14]
    
    # impurity strength = 5
    # ~ temps = [1, 1.5, 2, 2.5, 3, 3.5]
    # ~ tau_1 = [166954, 70199, 24933, 1900, 121, 33]
    # ~ tau_2 = [46140, 25493, 3372, 622, 69, 21]
    # ~ tau_3 = [12534, 10142, 1561, 263, 67, 28]
    
    temps = [1.5, 1.8, 2, 2.5]
    tau_1 = [951, 131, 35, 9]
    tau_2 = [648, 97, 42, 7]
    tau_3 = [482, 71, 26, 6]
    
    fig, ax = plt.subplots()
    
    ax.plot(temps, tau_1, color = 'blue', linestyle = '-', label = 'q = 5')
    ax.plot(temps, tau_2, color = 'orange', linestyle = '-', label = 'q = 10')
    ax.plot(temps, tau_3, color = 'green', linestyle = '-', label = 'q = 15')
    
    ax.set_xlabel('Temp')
    ax.set_ylabel('Tau')
    ax.legend()
    # ~ ax.set_yscale('log')
    plt.show()
    


model = Model_1(L = 180, temperature = 1.8, C_concentration = 0.02, C_size = 2, 
              step_number = 500*30, initial_step_number = 1000, save_distance = 30,
              save_images = True) # save distance of 30

make_plots()





