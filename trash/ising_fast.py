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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import tifffile
from scipy.ndimage import gaussian_filter
from scipy import optimize as opt
import os
import time
from numba import jit

# Configure plot settings
from matplotlib import rcParams
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

# Package for computing g2, I'm a dummy so I literally just copied the package files in the directory here because I was having trouble installing it with pip....
import skbeam
import skbeam.core.correlation as corr

class Model():
    def __init__(self, L = 150, temperature = 2,\
                 gen_new_fields = False, C_concentration = 0.01, C_size = 4, C_field_strength = 0.03, C_mass_strength = 0., \
                 step_number = 5000, initial_step_number = 10000, exposure_time = 10,
                 save_loc = 'results//default', real_space_images_to_save = 100):
                     
        # Where to save the data from the run
        self.save_loc = save_loc
        self.save_distance = step_number//real_space_images_to_save
        if self.save_distance == 0:
            self.save_distance = 1
        
        self.L = L # Size of the lattice
        self.beta = 1/temperature
        self.step_number = step_number
        self.initial_step_number = initial_step_number
        self.exposure_time = exposure_time
        
        self.update_fraction = 1
        
        self.config = 2*np.random.randint(2, size=(self.L,self.L)) - 1 # initial state of the simulation
        
        # For a given impurity concentration, convert to the total number of impurities on the lattice
        C_count = int(self.L**2*C_concentration)
        if C_count%2 != 0:
            C_count += 1
        # Create impurity configuration
        self.C = np.random.randint(self.L, size = (2,C_count))
        self.C_type = np.ones(C_count)
        self.C_type[::2] = -1
        
        if gen_new_fields:
            self.field, self.mass = generate_field(self.L, self.C, C_size, C_field_strength, C_mass_strength, self.C_type)
            tifffile.imwrite(f'{save_loc}//field.tiff', np.array(self.field, dtype = np.float32))
            tifffile.imwrite(f'{save_loc}//mass.tiff', np.array(self.mass, dtype = np.float32))
        else:
            self.field = tifffile.imread(f'{save_loc}//field.tiff')
            self.mass = tifffile.imread(f'{save_loc}//mass.tiff')
        

        # Run the simulation
        self.simulate()
        
        # Computes the g2 values
        self.measure_g2()
  
    def simulate(self):
        print('Starting Initial Evolution')
        t0 = time.perf_counter()
        
        # Perform some initial steps to generate a good initial state for the real simulation
        for i in range(self.initial_step_number):
            update(self.config, self.field, self.mass, self.beta, int(self.update_fraction*self.L**2))
            
        # Create an array to store the diffraction images:
        self.diffraction = np.zeros([self.step_number//self.exposure_time + 1, self.L, self.L])
        image_index = 0
        
        # Perform the actual simulation steps
        for i in range(self.step_number):
            # Move to the next diffraction frame when this is triggered
            if i%self.exposure_time == 0:
                image_index += 1
                
            update(self.config, self.field, self.mass, self.beta, int(self.update_fraction*self.L**2))
            # ~ self.diffraction[image_index] += (np.abs(np.fft.fftshift(np.fft.fft2(self.config[:self.L,:self.L])))**2 - self.diffraction[image_index])/(i%self.exposure_time + 1)
            self.diffraction[image_index] += (np.abs(np.fft.fftshift(np.fft.fft2(0.5*(1 + self.config[:self.L,:self.L]))))**2 - self.diffraction[image_index])/(i%self.exposure_time + 1)
            # ~ self.diffraction[image_index] += np.sum(self.config[:self.L,:self.L])**2
            
            # ~ self.diffraction[image_index] += np.random.normal(loc = 0,
                                                # ~ scale = 1e3*np.sqrt(self.diffraction[image_index]),
                                                # ~ size = self.config.shape)
            
            # Save a certain number of real space images
            if i%self.save_distance == 0:
                tifffile.imwrite(f'{self.save_loc}\\time_sequence\\{i//self.save_distance}.tiff', np.array((self.config+1)//2, dtype = bool))
                
                
            if i%(int(0.01*self.step_number)) == 0:
                print(f'{int(100*i/self.step_number)} %, working on image index: {image_index}')

        tf = time.perf_counter()
        
        print(f'{100*sum(np.array(self.config[self.C[0,:],self.C[1,:]] == self.C_type, dtype = int))/self.C.shape[1]}')
        print(f'Simulation Run Time = {round(tf-t0,2)} s')
        
        print('Making Omega-Q map')
        self.make_omega_q_map()
       
    def measure_g2(self):
        # Get the times of the images from the file names
        times = np.linspace(0,self.diffraction.shape[0],self.diffraction.shape[0])
        Iframes = np.copy(self.diffraction)

        
        # ~ magnetization = np.average(realFrames, axis = (1,2))
        # ~ fig, ax = plt.subplots()
        # ~ ax.plot(magnetization)
        # ~ ax.set_ylabel('Magnetization')
        # ~ ax.set_xlabel('Time')
        
        # ~ Iframes[:,self.diffraction[0].shape[0]//2, self.diffraction[0].shape[1]//2] = 0
        # Gaussian filter the images
        Iframes_avg_gf = gaussian_filter(np.mean(Iframes, axis = 0), 1)
        Iframes_flat = Iframes/Iframes_avg_gf
        # ~ Iframes_flat = Iframes
        
        # Now compute the g2
        q_radius = 1
        # ~ q_radius = 1
        N = 3
        xcentroid = self.diffraction[0].shape[1]//2
        ycentroid = self.diffraction[0].shape[0]//2

        g2_all = []
        colormap = np.zeros(Iframes.shape[1:3], dtype = np.int64)
        
        for i in range(N):
            #data = Icrop.copy()/(np.sum(Icrop, axis = (1,2))[:,None,None]) # Normalized images
            # ~ data = Iframes_flat.copy()/(np.sum(Iframes_flat, axis = (1,2))[:,None,None]) # Normalized images
            data = Iframes_flat.copy()
            
            # ~ labeled_roi_array = getAnnulusMask(data[0],xcentroid,ycentroid,q_radius,i,i+1)
            labeled_roi_array = np.zeros(np.shape(data[0]), dtype = int)
            labeled_roi_array[ycentroid, xcentroid + i] = 1
            labeled_roi_array[ycentroid, xcentroid - i] = 1
            labeled_roi_array[ycentroid + i, xcentroid] = 1
            labeled_roi_array[ycentroid - i, xcentroid] = 1
    
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
    
    def make_omega_q_map(self):
        Sms = np.abs(np.fft.fft(self.diffraction, axis = 0))**2
        Sms_radsym = np.empty([Sms.shape[0]-1, 10])
        delta_r = 1.5
        for ri in range(Sms_radsym.shape[1]):
            mask = getAnnulusMask(Sms[0], Sms.shape[2]//2, Sms.shape[1]//2, delta_r, ri, ri + 1)
            # ~ Sms_radsym[:, ri] = np.sum(Sms * mask[None, :, :], axis = (1,2))/(2*np.pi*(ri+0.5))*1/(np.sum(Sms * mask[None, :, :]))
            Sms_radsym[:, ri] = np.sum(Sms[1:] * mask[None, :, :], axis = (1,2))/(np.sum(Sms[1:] * mask[None, :, :]))
            
        tifffile.imwrite(f'{self.save_loc}\\S_q_omega.tiff', np.array(Sms_radsym, dtype = np.float32))
        plt.imshow(Sms_radsym, aspect = 'auto')
        plt.show()
        
        
def generate_field(L, impurities, impurity_size, impurity_field_strength, impurity_mass_strength, impurity_type):
    x, y = np.arange(0, L), np.arange(0, L)
    Y, X = np.meshgrid(x, y)
    field = np.zeros(X.shape)
    mass = np.zeros(X.shape)
    print(impurity_size)
    for n in range(len(impurities[0, :])):
        R = np.sqrt((X - impurities[0, n])**2 + (Y - impurities[1, n])**2)
        field += impurity_type[n]*impurity_field_strength*np.exp(-0.5*(R/impurity_size)**2)
        mass += impurity_mass_strength*np.exp(-0.5*(R/impurity_size)**20)
    
    return field, mass

@jit(nopython = True)
def update(config, field, mass, beta, N):
    # Updates the input configuration by N many steps on random indices
    I = np.random.randint(0, config.shape[0], N)
    J = np.random.randint(0, config.shape[1], N)
    
    for n in range(N):
        # ~ if impurity_size_sqr > np.min((impurities[0] - I[n])**2 + (impurities[1] - J[n])**2): # Checks if distance to closest impurity is within impurity size
            # ~ continue
        nn = config[(I[n] + 1)%config.shape[0], J[n]] + config[(I[n] - 1)%config.shape[0], J[n]] + config[I[n], (J[n] + 1)%config.shape[1]] + config[I[n], (J[n] - 1)%config.shape[1]]
        # ~ rbf = np.exp(-2*config[I[n], J[n]]*(nn + field[I[n], J[n]])*beta/(1 + mass[I[n], J[n]])) # Reciprocal of Boltzmann factor
        rbf = np.exp(-2*config[I[n], J[n]]*(nn + field[I[n], J[n]])*(beta + mass[I[n], J[n]])) # Reciprocal of Boltzmann factor
        
        
        # ~ if rbf >= 1:
            # ~ config[I[n], J[n]] *= -1
        # ~ elif np.random.random() < rbf:
            # ~ config[I[n], J[n]] *= -1
        
        if np.random.random() > 1/(1 + rbf):
            config[I[n], J[n]] *= -1

    return None # The config object is updated inside the function, no return

def getAnnulusMask(image,xc,yc,R,n,m):
    #set labeled_roi_arry to 1 in ROI
    print(np.shape(image))
    ylim, xlim = np.shape(image)
    x = np.arange(0,xlim,1)
    y = np.arange(0,ylim,1)
    X, Y = np.meshgrid(x, y)
    # 1's for greater than n and less than m

    mask2 = np.array(((X-xc))**2 + ((Y-yc))**2 > (R*n)**2, dtype=int)
    mask1 = np.array(((X-xc))**2 + ((Y-yc))**2 <= (R*m)**2, dtype=int)
    
    mask = mask1*mask2
    #plt.imshow(labeled_roi_array)
    return np.array(mask, dtype = int)

def sc_exponential(x, tau, beta):
    return np.exp(-1*((x-1)/tau)**beta)

def make_plots(fname = r"C:\Users\thoma\Documents\GitHub\Ising-Model\results\default\g2.csv"):
    data = np.loadtxt(fname, delimiter = ',')
    
    # ~ data =  np.loadtxt(r"C:\Users\thoma\OneDrive - UCLA IT Services\Desktop\OneDrive - UCLA IT Services\Research\TiSe2_XPCS\exhibits\A\95K_F.txt", delimiter = ',')
    
    print(np.shape(data))
    # ~ data = data[:]
    
    
    
    # ~ lag_steps = np.loadtxt(r"C:\Users\thoma\OneDrive - UCLA IT Services\Desktop\OneDrive - UCLA IT Services\Research\TiSe2_XPCS\exhibits\A\lag_steps.txt")
    # ~ print(np.shape(lag_steps))
    # ~ data[0,:] = lag_steps[1:]
    
    
    for i in range(data[1:].shape[0] - 1):
        plt.scatter(data[0], data[i+1])
    plt.show()
    

    fig, ax = plt.subplots()
    g2 = data[1:]
    g2[g2 < 1] = 1
    F = np.sqrt((g2-1)/(g2[:, 0, None] - 1))
    # ~ F = np.sqrt((data[1:]-1)/(data[1:,0][:, None] - 1))

    cmap = cm.plasma
    tau_from_slope_beta1_assumed = []
    taus = []
    
    step_in = 5
    for i in range(F.shape[0]):
        F_new = F[i, step_in:]/F[i, step_in]
        lag_steps_new = data[0,step_in:]
        F_new[F_new >= 1] = 0.99999
        
        tau_from_slope_beta1_assumed.append(-round(5/(F_new[5] - F_new[0]),1))
        xvals = np.log(lag_steps_new)[1:len(lag_steps_new)//2]
        yvals = np.log(-1*np.log(F_new))[1:len(lag_steps_new)//2]
        
        
        # ~ xvals = np.log(lag_steps[1:])[1:len(lag_steps)]
        # ~ yvals = np.log(-1*np.log(F[i]))[1:len(lag_steps)]
        
        # ~ print(yvals)
        # ~ ax.scatter(xvals, yvals, label = f'q_radius = {i*q_radius}')
        
        pol = np.polyfit(xvals, yvals, deg = 1)
        # ~ print(pol)
        tau, beta = np.exp(-1*pol[1]/pol[0]), pol[0]
        
        ax.scatter(data[0,step_in:], F_new, color = cmap(i/F.shape[0]))
        
        try:
            popt, pcov = opt.curve_fit(sc_exponential, lag_steps_new, F_new, p0 = [tau, beta])
            taus.append(popt[0])
            
            tt = np.linspace(min(lag_steps_new), max(lag_steps_new), 10000)
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

    
# 2.26918531421
model = Model(L = 100, temperature = 2.1, 
              step_number = 100000, initial_step_number = 5000, exposure_time = 20,
              C_concentration = 0.02, C_size = 2, C_field_strength = 0.000, C_mass_strength = 0.4,
              real_space_images_to_save = 1000, gen_new_fields = True)
              
# ~ model = Model(L = 100, temperature = 2.2, 
              # ~ step_number = 100000, initial_step_number = 5000, exposure_time = 50,
              # ~ C_concentration = 0.02, C_size = 2, C_field_strength = 0.1500, C_mass_strength = 0,
              # ~ real_space_images_to_save = 1000, gen_new_fields = True)
              
# ~ model = Model(L = 50, temperature = 3, 
              # ~ step_number = 50000, initial_step_number = 5000, exposure_time = 20,
              # ~ C_concentration = 0.0015, C_size = 7, C_field_strength = 0, C_mass_strength = 0.29,
              # ~ real_space_images_to_save = 500, gen_new_fields = True)

make_plots()
