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



def measure_g2(Iframes):
    # Gaussian filter the images
    Iframes_avg_gf = gaussian_filter(np.mean(Iframes, axis = 0), 1)
    Iframes_flat = Iframes/Iframes_avg_gf
    # ~ Iframes_flat = Iframes
    
    im_0 = Iframes[0]
    
    # Now compute the g2
    q_radius = 5
    N = 19
    # ~ N = 15
    
    xcentroid = im_0.shape[1]//2
    ycentroid = im_0.shape[0]//2

    g2_all = []
    colormap = np.zeros(Iframes.shape[1:3], dtype = np.int64)
    
    for i in range(N):
        
        #data = Icrop.copy()/(np.sum(Icrop, axis = (1,2))[:,None,None]) # Normalized images
        data = Iframes_flat.copy()/(np.sum(Iframes_flat, axis = (1,2))[:,None,None]) # Normalized images
        #data = Icrop_flat.copy()
        
        labeled_roi_array = getAnnulusMask(data[0],xcentroid,ycentroid,q_radius,i+1,i+2)

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
    # ~ np.savetxt(f'{self.save_loc}\\g2.csv', data_to_save, delimiter = ',', header = f'q_radius = {q_radius}')
        
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
    epsilon = 1e-3
    
    F = np.sqrt(abs((g2_all-1)/(g2_all[:,0][:, None] - 1)))

    for i in range(F.shape[0]):
        ax[0].scatter(lag_steps[1:], F[i], label = f'q_radius = {i*q_radius}')
        ax[1].scatter(lag_steps[1:], g2_all[i], label = f'q_radius = {i*q_radius}')
        #plt.semilogx(lag_steps[1:], g2_all[i], label = f'q_radius = {i*q_radius}')
    
        # ~ popt, pcov = opt.curve_fit(sc_exponential, lag_steps[1:], F[i])
        # ~ tt = np.linspace(min(lag_steps[1:]), max(lag_steps[1:],1000))
        # ~ yy = sc_exponential(tt, *popt)
        # ~ ax.plot(tt, yy)
    
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    # ~ ax.set_xlim([0,100])
    ax[0].set_xlabel(r'$\tau$ (s)')
    ax[0].set_ylabel(r'$F$')
    plt.legend()
    plt.show()
    
    fig, ax = plt.subplots()
    
    step_in = int(0.75*len(lag_steps))
    lag_steps = lag_steps[:-step_in ]
    F = F[:,:-step_in]
    
    popt_list = np.empty([F.shape[0], 2])
    for i in range(F.shape[0]):
        xvals = np.log(lag_steps[1:] + epsilon)[1:len(lag_steps[1:])//2]
        yvals = np.log(np.abs(-1*np.log(F[i]+epsilon))-epsilon)[1:len(lag_steps[1:])//2]
        
        xvals = xvals[3:]
        yvals = yvals[3:]
        
        print('xvals', xvals)
        print('yvals', yvals)
        # ~ ax.scatter(xvals, yvals, label = f'q_radius = {i*q_radius}')
        
        pol = np.polyfit(xvals, yvals, deg = 1)
        # ~ print(pol)
        tau, beta = np.exp(-1*pol[1]/pol[0]), pol[0]
        
        print('tau_preliminary', tau)
        print('beta_preliminary', beta)
        
        ax.scatter(lag_steps[1:], F[i])
        
        print(F[i])
        
        popt, pcov = opt.curve_fit(sc_exponential, lag_steps[1:], F[i], p0 = [tau, beta])
        popt_list[i] = popt
        print('popt',popt)
        
        tt = np.linspace(min(lag_steps[1:]), max(lag_steps[1:]), 10000)
        # ~ tt = np.linspace(0,1000,1000)
        # ~ yy = sc_exponential(tt, *[tau, beta])
        
        # ~ print(yy)
        yy = sc_exponential(tt, *popt)
        
        # ~ print(yy)
        # ~ ax.plot(tt, yy)
        plt.plot(tt, yy, label = f'tau = {round(popt[0])}, beta = {round(popt[1],2)}')
    
    np.savetxt('popt.csv',popt_list, delimiter=',')
    ax.set_xlabel('time (steps)')
    ax.set_ylabel('F')
    ax.legend()
    # ~ ax.set_xscale('log')
    plt.show()

def sc_exponential(x, tau, beta):
    return np.exp(-1*((x-1)/tau)**beta)


class Simulation():
    def __init__(self):
        # Number of particles
        # ~ self.N = 5
        self.N = 50
        self.L = 2500
        
        # Positions of the particles
        self.positions = np.random.normal(loc=0,scale=5,size=[self.N,2])
        self.positions_0 = np.copy(self.positions)
        self.velocity = np.random.normal(loc=0,scale=1,size=[self.N,2])
        
        k_max = 35
        self.k_length = 200
        kx, ky = np.linspace(-k_max,k_max,self.k_length), np.linspace(-k_max,k_max,self.k_length)
        self.k_vecs = np.array(np.meshgrid(kx, ky))

        self.lspread = 0.02
        # ~ self.lspread = 0.1
        # ~ self.lspread = 0.005
        
        self.simulate()
        
    def evolve(self):
        # ~ self.positions += np.random.normal(loc=0,scale=self.lspread,size=[self.N,2])
        # ~ self.positions += 0.1*(self.positions_0 - self.positions)
        self.positions += 0.01*self.velocity
        
        # ~ self.positions += np.random.normal(loc=0,scale=self.lspread,size=[self.N,2])*(np.exp(-10*np.sqrt(np.sum((self.positions_0 - self.positions)**2,axis=1)))[:,None])
        
    def simulate(self):
        ft_images = np.empty([self.L, self.k_length, self.k_length], dtype = np.float32)
        for i in range(self.L):
            if i%50 == 0:
                print(f'{i}/{self.L}')
            ft_images[i] = self.get_ft()
            
            # ~ tifffile.imwrite(r'C:\Users\thoma\Documents\GitHub\Ising-Model\results\beta_example' + f'//{i}.tiff', ft_images[i])
            
            fig, axs = plt.subplots()
            axs.scatter(self.positions[:,0], self.positions[:,1])
            
            axs.set_xlim([-10,10])
            axs.set_ylim([-10,10])
            axs.set_aspect('equal')
            fig.savefig(r'C:\Users\thoma\Documents\GitHub\Ising-Model\results\beta_example' + f'//{i}.tiff')
            plt.close()
            
            self.evolve()
            
        measure_g2(ft_images)
            
    def get_ft(self):        
        ft_amp = np.sum(np.exp(-1.j*np.sum(self.positions[:,:,None,None] * self.k_vecs[None,:,:], axis = 1)),axis = 0)
        return np.abs(ft_amp)**2
        

# ~ sim = Simulation()
    
    
def meta_plot():
    # ~ tau = np.array([149, 28144, 3119, 1410, 732, 395, 233, 152, 104, 77, 52, 45, 38, 27, 25])
    # ~ beta = np.array([0.22, 0.33, 0.63, 0.78, 0.9, 0.99, 1.05, 1.06, 1.1, 1.08, 1.13, 1.07, 0.99, 1.14, 1.03])
    
    data = np.loadtxt(r"C:\Users\thoma\Documents\GitHub\Ising-Model\popt.csv", delimiter = ',')
    print(np.shape(data))
    tau = data[:,0]
    beta = data[:,1]

    
    Gamma = (1/tau)**beta
    Q = np.linspace(0,len(data),len(data))
    
    skip = 10
    popt = np.polyfit(np.log(Q[skip:]), np.log(Gamma[skip:]), deg = 1)
    print(popt)
    
    
    fig, axs = plt.subplots(2)
    
    axs[0].scatter(np.log(Q), np.log(Gamma))
    axs[0].scatter(np.log(Q[skip:]), np.log(Gamma[skip:]))
    
    tt = np.linspace(min(np.log(Q[skip:])),max(np.log(Q[skip:])),2)
    yy = popt[0]*tt + popt[1]
    axs[0].plot(tt,yy,'g-', label = f'slope = {round(popt[0],2)}')
    yy = 2*tt + popt[1]
    axs[0].plot(tt,yy,'g--')
    axs[0].set_ylabel(r'Log($\Gamma$)')
    axs[0].legend()
    
    axs[1].scatter(np.log(Q), beta)
    axs[1].set_ylabel(r'$\beta$')
    axs[1].set_xlabel('Log(Q)')
    axs[1].axhline(y = 1, c = 'black')
    
    plt.show()
    
meta_plot()
