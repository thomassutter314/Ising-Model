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

import skbeam
import skbeam.core.correlation

class Model():
    def __init__(self, temperature = 2,\
                 gen_new_fields = False, C_size = 4, C_field_strength = 0.03, C_mass_strength = 0., \
                 step_number = 5000, initial_step_number = 10000, exposure_time = 10,
                 save_loc = 'results//default', real_space_images_to_save = 100):
                     
        print(f'Temperature = {temperature}')
        print(f'Save Loc = {save_loc}')
        # Where to save the data from the run
        self.save_loc = save_loc
        self.save_distance = step_number//real_space_images_to_save
        if self.save_distance == 0:
            self.save_distance = 1
        
        self.temp = temperature
        self.step_number = step_number
        self.initial_step_number = initial_step_number
        self.exposure_time = exposure_time
        
        self.update_fraction = 1
        
        self.L = 200
        C_count = 3
        self.config = 2*np.random.randint(2, size=(self.L,self.L)) - 1 # initial state of the simulation
        self.config_save = np.zeros([real_space_images_to_save + 2, self.config.shape[0], self.config.shape[1]])
        # ~ self.C = np.array([[self.L//2, self.L//2],[self.L//8, self.L//8],[7*self.L//8, 7*self.L//8],[1*self.L//8, 7*self.L//8],[7*self.L//8, 1*self.L//8]]).transpose()
        # ~ self.C = np.array([[self.L//6, self.L//6],[self.L//6, self.L//2],[self.L//6, 5*self.L//6],[self.L//2, self.L//6],[5*self.L//6, self.L//5],[5*self.L//6, self.L//2],[self.L//2, 5*self.L//6],[5*self.L//6, 5*self.L//6]]).transpose()
        self.C = []
        for i in range(C_count):
            for j in range(C_count):
                randomness = 4
                self.C.append([int((i+0.5)*self.L/C_count + np.random.normal(loc = 0, scale = randomness)), int((j+0.5)*self.L/C_count + np.random.normal(loc = 0, scale = randomness))])
                
        
        self.C = np.array(self.C).transpose()
        print(self.C)
        print(self.C.shape)
                
        self.C_type = np.array([1,-1,-1,1,1,-1,1,-1,1])
        
        if gen_new_fields:
            self.field, self.mass = generate_field(self.L, self.C, C_size, C_field_strength, C_mass_strength, self.C_type)
            tifffile.imwrite(f'{save_loc}//field.tiff', np.array(self.field, dtype = np.float32))
            tifffile.imwrite(f'{save_loc}//mass.tiff', np.array(self.mass, dtype = np.float32))
        else:
            self.field = tifffile.imread(f'{save_loc}//field.tiff')
            self.mass = tifffile.imread(f'{save_loc}//mass.tiff')
        
        self.m = np.empty(self.step_number) # place to store the pixel sum inside an roi of the C_size radius
        # ~ self.m_mask = getAnnulusMask(self.config, self.config.shape[1]//6, self.config.shape[0]//6, C_size, 0, 1)
        self.m_mask = np.ones(self.config.shape)
        
        # ~ x, y = np.arange(0, self.L), np.arange(0, self.L)
        # ~ Y, X = np.meshgrid(x, y)
        # ~ self.m_mask = np.array(X > 0.7*self.config.shape[0], dtype = float)
        
        # Run the simulation
        self.simulate()
        
        tifffile.imwrite(r'C:\Users\thoma\Documents\GitHub\Ising-Model\results\default\7_4_2024_data' + f'//{temperature}.tiff', np.array(np.average(self.diffraction, axis = 0), dtype = np.float32))
        
        # ~ analyze_telegraph(self.m)
        # Computes the g2 values
        self.measure_g2()
  
    def simulate(self):
        print('Starting Initial Evolution')
        t0 = time.perf_counter()
        
        # Perform some initial steps to generate a good initial state for the real simulation
        for i in range(self.initial_step_number):
            update(self.config, self.field, self.mass, self.temp, int(self.update_fraction*self.L**2))
            
        # Create an array to store the diffraction images:
        self.diffraction = np.zeros([self.step_number//self.exposure_time + 2, self.L, self.L])
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
                
                
            update(self.config, self.field, self.mass, self.temp, int(self.update_fraction*self.L**2))
            self.m[i] = np.sum(self.config*self.m_mask)
            # ~ self.m[i] = np.sum(self.config)
            
            # ~ print(image_index)
            self.diffraction[image_index] += (np.abs(np.fft.fftshift(np.fft.fft2((self.config[:self.L,:self.L]))))**2 - self.diffraction[image_index])/(i%self.exposure_time + 1)
            self.config_save[save_index] += ((self.config+1)//2 - self.config_save[save_index])/(i%self.save_distance + 1)
            
            if i%(int(0.01*self.step_number)) == 0:
                print(f'{int(100*i/self.step_number)} %, working on image index: {image_index}')
        
        tf = time.perf_counter()
        
        # ~ self.diffraction = np.array(self.diffraction, dtype = int)
        
        # ~ print(f'{100*sum(np.array(self.config[self.C[0,:],self.C[1,:]] == self.C_type, dtype = int))/self.C.shape[1]}')
        print(f'Simulation Run Time = {round(tf-t0,2)} s')
           
    def measure_g2(self):
        # Get the times of the images from the file names
        times = np.linspace(0,self.diffraction.shape[0],self.diffraction.shape[0])
        Iframes = np.copy(self.diffraction)
        
        
        tifffile.imwrite(r'C:\Users\thoma\Documents\GitHub\Ising-Model\results\default\diffraction\avg_diffraction_image.tiff', np.mean(Iframes, axis = 0))
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
        q_radius = 3
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
            # ~ labeled_roi_array[ycentroid, xcentroid - i] = 1
            # ~ labeled_roi_array[ycentroid + i, xcentroid] = 1
            # ~ labeled_roi_array[ycentroid - i, xcentroid] = 1
    
            colormap = colormap + labeled_roi_array*(i+1)
            
            #XPCS parameters and run calculation
            num_levels = 10
            num_bufs = 12
            # ~ num_levels, num_bufs = 1, len(data[1:])
            # ~ if num_bufs%2 != 0:
                # ~ num_bufs += -1
            g2n, lag_steps = skbeam.core.correlation.multi_tau_auto_corr(num_levels, num_bufs, labeled_roi_array, data[1:])
            
            g2_all.append(g2n[1:])
    
        # Convert g2_all to a sensible shape and structure
        g2_all = np.array(g2_all)[:,:,0]
        
        data_to_save = np.empty([g2_all.shape[0] + 1, g2_all.shape[1]])
        data_to_save[0,:] = lag_steps[1:]
        data_to_save[1:,:] = g2_all
        # Save the g2 values from the run
        np.savetxt(f'{self.save_loc}\\g2.csv', data_to_save, delimiter = ',', header = f'q_radius = {q_radius}')

def make_plots(save_name = '0', fname = r"C:\Users\thoma\Documents\GitHub\Ising-Model\results\default\g2.csv", 
                fit_in = 20, fit_out = 10):
    save_loc = r'C:\Users\thoma\Documents\GitHub\Ising-Model\results\default\7_4_2024_data'
    
    data = np.loadtxt(fname, delimiter = ',')
    np.savetxt(f'{save_loc}\\g2_{save_name}_{int(time.time())}.csv', data, delimiter = ',')
    print('data.shape',data.shape)


    for i in range(data[1:].shape[0]):
        plt.scatter(data[0], data[i+1])
    plt.show()
    

    fig, ax = plt.subplots()
    g2 = data[1:]
    # ~ g2[g2 < 1] = 1.000000000001
    F2 = ((g2-1)/(g2[:, 0, None] - 1))
    print(F2.shape)
    # ~ F[F == np.sqrt(-1)] = 0
    
    np.savetxt(f'{save_loc}\\F2_{save_name}.csv', F2, delimiter = ',')
    # ~ F = np.sqrt((data[1:]-1)/(data[1:,0][:, None] - 1))

    cmap = cm.plasma
    tau_from_slope_beta1_assumed = []
    taus = []
    
    step_in = 2
    for i in range(F2.shape[0]):
        F2_new = F2[i, step_in:]/F2[i, step_in]
        lag_steps_new = data[0,step_in:]
        F2_new[F2_new >= 1] = 0.99999
        
        tau_from_slope_beta1_assumed.append(-round(5/(F2_new[5] - F2_new[0]),1))
        xvals = np.log(lag_steps_new)[1:len(lag_steps_new)//2]
        yvals = np.log(-1*np.log(F2_new))[1:len(lag_steps_new)//2]
        
        
        # ~ xvals = np.log(lag_steps[1:])[1:len(lag_steps)]
        # ~ yvals = np.log(-1*np.log(F[i]))[1:len(lag_steps)]
        
        # ~ print(yvals)
        # ~ ax.scatter(xvals, yvals, label = f'q_radius = {i*q_radius}')
        
        pol = np.polyfit(xvals[fit_out:fit_in], yvals[fit_out:fit_in], deg = 1)
        # ~ print(pol)
        tau, beta = np.exp(-1*pol[1]/pol[0]), pol[0]
        
        ax.scatter(data[0,step_in:], F2_new, color = cmap(i/F2.shape[0]), s = 5)
        p0 = [tau, 1, 1]
        try:
            # ~ popt, pcov = opt.curve_fit(sc_exponential, lag_steps_new[:fit_in], F_new[:fit_in], p0 = [tau, beta])
            popt, pcov = opt.curve_fit(exponential_decay, lag_steps_new[fit_out:fit_in], F2_new[fit_out:fit_in], p0 = p0)
            taus.append(popt[0])
            
            tt = np.linspace(min(lag_steps_new[fit_out:fit_in]), max(lag_steps_new[fit_out:fit_in]), 10000)
            # ~ tt = np.linspace(0,1000,1000)
            # ~ yy = sc_exponential(tt, *[tau, beta])
            
            # ~ print(yy)
            yy = exponential_decay(tt, *popt)
            print('0.5 crossing',tt[np.argmin((yy-0.5)**2)])
            
            # ~ print(yy)
            # ~ ax.plot(tt, yy)
            plt.plot(tt, yy, label = f'tau = {round(popt[0],2)}' + r'$\pm$' + f'{round(np.sqrt(pcov[0,0]),2)}'  + \
                                     f', beta = {round(popt[1],2)}' + r'$\pm$' + f'{round(np.sqrt(pcov[1,1]),2)}', color = cmap(i/F2.shape[0]))
            # ~ plt.plot(tt, yy, label = f'tau = {round(popt[0])}', color = cmap(i/F.shape[0]))
            taus
        except:
            print('Error in fit')
            tt = np.linspace(min(lag_steps_new[fit_out:fit_in]), max(lag_steps_new[fit_out:fit_in]), 10000)
            yy = exponential_decay(tt, tau, 1, 1)
            plt.plot(tt, yy, label = f'tau = {round(p0[0],2)}', color = cmap(i/F2.shape[0]))
        
    # ~ print(tau_from_slope_beta1_assumed)
    print(taus)
    
    
    ax.set_xlabel('time (steps)')
    ax.set_ylabel('F')
    ax.legend()
    # ~ ax.set_xscale('log')
    plt.show()

@jit(nopython = True)
def update(config, field, mass, temp, N):
    # Updates the input configuration by N many steps on random indices
    I = np.random.randint(0, config.shape[0], N)
    J = np.random.randint(0, config.shape[1], N)
    
    for n in range(N):
        # ~ if impurity_size_sqr > np.min((impurities[0] - I[n])**2 + (impurities[1] - J[n])**2): # Checks if distance to closest impurity is within impurity size
            # ~ continue
        nn = config[(I[n] + 1)%config.shape[0], J[n]] + config[(I[n] - 1)%config.shape[0], J[n]] + config[I[n], (J[n] + 1)%config.shape[1]] + config[I[n], (J[n] - 1)%config.shape[1]]
        # ~ rbf = np.exp(-2*config[I[n], J[n]]*(nn + field[I[n], J[n]])*beta/(1 + mass[I[n], J[n]])) # Reciprocal of Boltzmann factor
        rbf = np.exp(-2*config[I[n], J[n]]*(nn + field[I[n], J[n]])/(temp/(1+mass[I[n], J[n]]))) # Reciprocal of Boltzmann factor
        
        
        # ~ if rbf >= 1:
            # ~ config[I[n], J[n]] *= -1
        # ~ elif np.random.random() < rbf:
            # ~ config[I[n], J[n]] *= -1
        
        if np.random.random() > 1/(1 + rbf):
            config[I[n], J[n]] *= -1

    return None # The config object is updated inside the function, no return

# Generate the quenched random field that is used in the simulation
def generate_field(L, impurities, impurity_size, impurity_field_strength, impurity_mass_strength, impurity_type):
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


def getAnnulusMask(image,xc,yc,R,n,m):
    #set labeled_roi_arry to 1 in ROI
    print(np.shape(image))
    ylim, xlim = np.shape(image)
    x = np.arange(0,xlim,1)
    y = np.arange(0,ylim,1)
    X, Y = np.meshgrid(x, y)
    # 1's for greater than n and less than m

    mask1 = np.array(((X-xc))**2 + ((Y-yc))**2 <= (R*m)**2, dtype=int)
    mask2 = np.array(((X-xc))**2 + ((Y-yc))**2 > (R*n)**2, dtype=int)
    
    mask = mask1*mask2
    
    #plt.imshow(labeled_roi_array)
    return np.array(mask, dtype = int)

def sc_exponential(x, tau, beta):
    return np.exp(-1*((x-1)/tau)**beta)
    
def exponential_decay(x, tau, beta, I0):
    return I0*np.exp(-((x-1)/tau)**beta)

def fit_func(t, tau, beta):
    return np.exp(-(np.abs(t)/np.abs(tau))**np.abs(beta))

def analyze_telegraph(signal):
    corr = sci.signal.correlate(signal, signal, mode = 'same')
    corr = corr/np.max(corr)
    # ~ corr[corr < 0] = 0
    t_corr = np.linspace(-len(corr)/2,len(corr)/2,len(corr))
    
    fit_interval = 56005
    tt = np.linspace(-fit_interval,fit_interval,50000)
    p0 = [500, 1]
    yy_guess = fit_func(tt, *p0)
    popt, pcov = sci.optimize.curve_fit(fit_func, t_corr[len(corr)//2:len(corr)//2+fit_interval], corr[len(corr)//2:len(corr)//2+fit_interval], p0 = p0)
    print(popt)
    yy = fit_func(tt, *popt)
    
    fig, axs = plt.subplots(2)
    axs[0].plot(signal, 'k-')
    axs[0].set_xlabel('Time (steps)')
    axs[0].set_ylabel('Magnetization in ROI')
    
    axs[1].plot(t_corr, corr, 'k-')
    axs[1].plot(tt, yy_guess, 'b--')
    axs[1].plot(tt, yy, 'g-', linewidth = 3, alpha = 0.6)
    axs[1].set_xlabel(r'$\Delta t$ (steps)')
    axs[1].set_ylabel('Correlation')
    
    
 

# ~ temp = 2.8
# ~ model = Model(temperature = temp,
              # ~ step_number = 750001, initial_step_number = 10000, exposure_time = 2000,
              # ~ C_size = 5.5, C_field_strength = 0.0, C_mass_strength = 0.4,
              # ~ real_space_images_to_save = 375, gen_new_fields = False)



