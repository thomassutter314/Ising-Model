import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
import tifffile

from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy import optimize as opt

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

# Style Choices
linewidth = 5
linealpha = 0.5
scatter_size = 30
scatter_alpha = 1
span_alpha = 0.7
offset = 0
step = 0.22
cmap = cm.viridis_r
colors_for_qs = [cmap(offset),cmap(step+offset),cmap(2*step+offset),cmap(3*step+offset),cmap(4*step+offset)]

# Style Choices
linewidth = 5
linealpha = 0.5
scatter_size = 30
scatter_alpha = 1
span_alpha = 0.7
offset = 0
step = 0.22
cmap = cm.viridis_r
colors_for_qs = [cmap(offset),cmap(step+offset),cmap(2*step+offset),cmap(3*step+offset),cmap(4*step+offset)]



def exponential_decay_double(x, tau1, beta1, tau2, beta2, I):
    beta1 = 1
    # ~ beta2 = 1.3
    return (1-I)*np.exp(-(x/tau1)**beta1) + I*np.exp(-(x/tau2)**beta2)
    
    
def exponential_decay_single(x, tau, beta):
    return np.exp(-(x/tau)**beta)


def go1():
    im = tifffile.imread('conversion_example_exp_2000_5_frame.tif')
    
    fig, axs = plt.subplots(1,5)
    L = im.shape[1]
    print(L)
    for i in range(len(axs)):
        axs[i].imshow(im[i], cmap = 'bwr', vmax = 255, vmin = 0)
        axs[i].set_title(f'Step # = {i*2000*L**2*1e-8} X 10$^8$')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        
        
    plt.show()


def go2():
    files_list = ['exp=2000.0.csv','exp=1000.0.csv','exp=500.0.csv','exp=250.0.csv',]
    exp_times = np.array([2000, 1000, 500, 250])
    exp_times_labels = [r'8$\times 10^7$',r'4$\times 10^7$',r'2$\times 10^7$',r'1 $\times 10^7$']
    L = 200
    
    fit_params = np.array([

    [0, 1.00000000e+00, 30000, 1.4, 1],
    
    [7850, 1.00000000e+00, 41500, 2.0, 0.54],
    
    [6700, 1.00000000e+00, 45000, 2.4, 0.456],
    
    [4000, 1.00000000e+00, 47500, 2.8, 0.416],
    ])
    
    p0 = [1200, 1, 30000, 1.5, 0.5]
    
    fig, axs = plt.subplots(1, 2)
    for fi in range(len(files_list)):
        data = np.loadtxt(files_list[fi], delimiter = ',')
        
        popt, pcov = opt.curve_fit(exponential_decay_double, data[:,0], data[:,1], p0 = p0)
        print('popt_double', popt[1], popt[3])
        tt = np.linspace(min(data[:,0]), max(data[:,0]), 1000)
        yy = exponential_decay_double(tt, *popt)
        
        if exp_times[fi] == 2000:
            popt, pcov = opt.curve_fit(exponential_decay_single, data[:,0], data[:,1], p0 = [3000, 1.5])
            print('popt_single', popt[1])
            yy_single = exponential_decay_single(tt, *popt)
            axs[0].plot(L**2*tt, yy_single, 'k--', zorder = -5, alpha = 1)
            axs[1].plot(L**2*tt, yy_single, 'k--', zorder = -5, alpha = 1)
        
        
        axs[1].plot(L**2*tt, yy, c = colors_for_qs[fi], linewidth = linewidth, zorder = -10, alpha = linealpha)
        axs[1].scatter(L**2*data[:,0], data[:,1], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[fi], marker = 'o')
                
        axs[0].plot(L**2*tt, yy, c = colors_for_qs[fi], linewidth = linewidth, zorder = -10, alpha = linealpha)
        axs[0].scatter(L**2*data[:,0], data[:,1], label = exp_times_labels[fi], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[fi], marker = 'o')
    
    # Create a Rectangle patch
    rect = patches.Rectangle((1e7, 1), 1.5e9, -0.7, linewidth=1, edgecolor='k', facecolor='none')

    # Add the patch to the Axes
    axs[0].add_patch(rect)
    
    axs[1].set_xlim([1e7,  1.5e9])
    axs[1].set_ylim([0.3, 1])
    
    # ~ axs[1].set_xlim([L**2, 150000*L**2])
    axs[0].set_xscale('log')
    axs[0].set_ylabel(r'$F^2$')
    axs[0].set_xlabel(r'$\Delta t$ (steps)')
    axs[1].set_xlabel(r'$\Delta t$ (steps $\times 10^9 $)')
    axs[0].legend()
    plt.show()
    
go2()
go1()
        
