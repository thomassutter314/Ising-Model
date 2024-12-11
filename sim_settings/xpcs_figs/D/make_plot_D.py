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
import scipy as sci

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



def exponential_decay(x, tau, beta):
    beta = 1
    return np.exp(-(x/tau)**beta)
    

def fit_func_1(x, A):
    # ~ x0, gamma =  1.20382101e+02, -7.66184239e-02
    x0, gamma =  50, -0.12
    return A*np.exp(gamma * (x - x0)) + 10
    
def fit_func_2(x, A):
    x0, gamma =  2.74, -22
    return A*np.exp(gamma * (x - x0)) + 0.286
    
    
def go1():
    data = np.loadtxt('data.csv', skiprows = 1, delimiter = ',')
    print(np.shape(data))
    temps = data[:, 0]
    tau_1 = data[:, 1]
    err_1 = data[:, 2]
    tau_2 = data[:, 3]
    err_2 = data[:, 4]
    tau_3 = data[:, 5]
    err_3 = data[:, 6]
    tau_4 = data[:, 7]
    err_4 = data[:, 8]
    tau_5 = data[:, 9]
    err_5 = data[:, 10]
    
    sim = np.loadtxt('sim.csv', skiprows = 1, delimiter = ',')
    sim = sim[10:,:]
    print(np.shape(sim))
    temps_sim = sim[:, 0]
    tau_1_sim = sim[:, 1]
    err_1_sim = sim[:, 2]
    tau_2_sim = sim[:, 3]
    err_2_sim = sim[:, 4]
    tau_3_sim = sim[:, 5]
    err_3_sim = sim[:, 6]
    
    telegraph = np.loadtxt('telegraph.csv', skiprows = 1, delimiter = ',')
    
    
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
    markers_for_qs = ['s','o','^', 'D', 'P']
    markers_for_qs = ['o','o','o', 'D', 'P']
    
    fig, axs = plt.subplots(1,3, figsize=(10, 3))
    
    # Plot the real data
    axs[0].errorbar(temps, tau_1/60, 2*err_1/60, color = 'black', fmt= 'none', capsize = 3, zorder = -1)
    axs[0].scatter(temps, tau_1/60, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[0], marker = markers_for_qs[0], label = r'0.002 Å$^{-1}$')
    axs[0].errorbar(temps, tau_2/60, 2*err_2/60, color = 'black', fmt= 'none', capsize = 3, zorder = -1)
    axs[0].scatter(temps, tau_2/60, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[1], marker = markers_for_qs[1], label = r'0.004 Å$^{-1}$')
    axs[0].errorbar(temps, tau_3/60, 2*err_3/60, color = 'black', fmt= 'none', capsize = 3, zorder = -1)
    axs[0].scatter(temps, tau_3/60, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[2], marker = markers_for_qs[2], label = r'0.006 Å$^{-1}$')
    
    axs[0].legend()
    # ~ axs[0].errorbar(temps, tau_4/60, 2*err_4/60, color = 'black', fmt= 'none', capsize = 3, zorder = -1)
    # ~ axs[0].scatter(temps, tau_4/60, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[3], marker = markers_for_qs[2])
    
    # ~ axs[0].errorbar(temps, tau_5/60, 2*err_5/60, color = 'black', fmt= 'none', capsize = 3, zorder = -1)
    # ~ axs[0].scatter(temps, tau_5/60, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[3], marker = markers_for_qs[2])
    
    
    
    # ~ axs.set_yscale('log')
    p0_1 = [4.22090094e+00]
    p0_2 = [3.54954222e+00]
    p0_3 = [3.21447820e+00]
    popt_1, pcov_1 = curve_fit(fit_func_1, temps, tau_1/60, p0 = p0_1)
    popt_2, pcov_2 = curve_fit(fit_func_1, temps, tau_2/60, p0 = p0_2)
    popt_3, pcov_3 = curve_fit(fit_func_1, temps, tau_3/60, p0 = p0_3)
    tt = np.linspace(min(temps) - 1, max(temps) + 1, 1000)
    yy = fit_func_1(tt, *popt_1)
    axs[0].plot(tt, yy, c = colors_for_qs[0], linewidth = linewidth, zorder = -5, alpha = linealpha)
    yy = fit_func_1(tt, *popt_2)
    axs[0].plot(tt, yy, c = colors_for_qs[1], linewidth = linewidth, zorder = -5, alpha = linealpha)
    yy = fit_func_1(tt, *popt_3)
    axs[0].plot(tt, yy, c = colors_for_qs[2], linewidth = linewidth, zorder = -5, alpha = linealpha)
    
    
    N = 1000
    zorder = -10
    cmap = 'bwr'
    x0 = 85
    vmax = 30
    x = np.linspace(min(temps), max(temps), N) - x0
    x[x < 0] = 0
    # ~ x[x > 87] = 87
    X = np.array([x])
    x0, x1 = axs[0].get_xlim()
    y0, y1 = axs[0].get_ylim()
    axs[0].imshow(X, extent = [x0, x1, y0, y1], aspect = 'auto',
                  alpha = 0.7, zorder = zorder, cmap = cmap,
                  vmin = -vmax, vmax = vmax)
    
    L = 80 # number of sites
    # Plot the simulation result
    axs[1].scatter(temps_sim, 1e-6*L**2*tau_1_sim, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[0], marker = markers_for_qs[0])
    axs[1].scatter(temps_sim, 1e-6*L**2*tau_2_sim, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[1], marker = markers_for_qs[1])
    axs[1].scatter(temps_sim, 1e-6*L**2*tau_3_sim, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[2], marker = markers_for_qs[2])
    
    p0_1 = [0.3]
    p0_2 = [0.2]
    p0_3 = [0.1]
    popt_1, pcov_1 = curve_fit(fit_func_2, temps_sim, 1e-6*L**2*tau_1_sim, p0 = p0_1)
    popt_2, pcov_2 = curve_fit(fit_func_2, temps_sim, 1e-6*L**2*tau_2_sim, p0 = p0_2)
    popt_3, pcov_3 = curve_fit(fit_func_2, temps_sim, 1e-6*L**2*tau_3_sim, p0 = p0_3)
    
    # ~ popt_1, popt_2, popt_3 = p0_1, p0_2, p0_3
    tt = np.linspace(min(temps_sim) - 0.005, max(temps_sim) + 0.005, 1000)
    yy = fit_func_2(tt, *popt_1)
    axs[1].plot(tt, yy, c = colors_for_qs[0], linewidth = linewidth, zorder = -5, alpha = linealpha)
    yy = fit_func_2(tt, *popt_2)
    axs[1].plot(tt, yy, c = colors_for_qs[1], linewidth = linewidth, zorder = -5, alpha = linealpha)
    yy = fit_func_2(tt, *popt_3)
    axs[1].plot(tt, yy, c = colors_for_qs[2], linewidth = linewidth, zorder = -5, alpha = linealpha)
    
    
    N = 1000
    zorder = -10
    cmap = 'bwr'
    x0 = 2.53
    vmax = 0.14
    x = np.linspace(min(temps_sim), max(temps_sim), N) - x0
    x[x < 0] = 0
    X = np.array([x])
    x0, x1 = axs[1].get_xlim()
    y0, y1 = axs[1].get_ylim()
    axs[1].imshow(X, extent = [x0, x1, y0, y1], aspect = 'auto',
                  alpha = 0.7, zorder = zorder, cmap = cmap,
                  vmin = -vmax, vmax = vmax)
    
    axs[0].set_xlabel('Temperature (K)')
    axs[0].set_ylabel(r'$\tau$ (Minutes)')
    axs[0].set_title('Data')
    # ~ axs[0].axvline(x = 73.6)
    
    axs[1].set_xlabel('Temperature (a.u.)')
    axs[1].set_ylabel(r'$\tau$ (Steps X $10^6$)')
    axs[1].set_title('Simulation')
    # ~ axs[1].axvline(x = 2.46)
    
    
    # Create an inset axis for the g2 data
    # ~ inset_ax = inset_axes(axs[1], width="30%", height="30%", loc='upper right', bbox_transform=axs[1].transAxes)
    inset_ax = inset_axes(axs[1], width="100%", height="100%", 
                      bbox_to_anchor=(0.5, 0.55, 0.45, 0.45),  # Adjust these values for padding
                      bbox_transform=axs[1].transAxes)
              
    
    inset_ax.set_xlabel(r'$\Delta t$ (steps)')
    inset_ax.set_ylabel(r'$\vert F \vert^2$', rotation = 0)
    # ~ inset_ax.label_params(axis='y', labelrotation=90)
    
    # ~ sim_data = np.loadtxt('F2_2.43_incorrect_step_conversion.csv').transpose()
    # ~ lag_steps_crop = sim_data[0]*L**2/51.65
    # ~ F2_crop = sim_data[1:, :]
    # ~ save_data = np.array([sim_data[0]*L**2/51.65,  sim_data[1], sim_data[2], sim_data[3]]).transpose()
    # ~ np.savetxt('save_F2_2.43.csv', save_data)
    
    sim_data = np.loadtxt('F2_2.43.csv').transpose()
    print('sim_data.shape',sim_data.shape)
    lag_steps_crop = sim_data[0]
    F2_crop = sim_data[1:, :]
      
    p0 = [10000600, 1]
    popt = p0
    colors = ['blue', 'green', 'red']
    labels = ['q0', 'q1', 'q2']
    for i in range(len(F2_crop)):
        try:
            popt, pcov = opt.curve_fit(exponential_decay, lag_steps_crop, F2_crop[i], p0 = p0)
            # ~ print('F2_crop[i]', F2_crop[i])
        except:
            p0 = popt
            print(f'ERROR IN FIT {i}')
        
        print('popt',popt)
        # ~ print(f'{labels[i]}: \n tau = {popt[0]} pm {np.sqrt(pcov[0,0])} \n beta = {popt[1]} pm {np.sqrt(pcov[1,1])} \n I0 = {popt[2]} pm {np.sqrt(pcov[2,2])}')
        
        # ~ print(i)
        color = colors[i]
        tt = np.linspace(0, max(lag_steps_crop), 2000)
        yy = exponential_decay(tt, *popt)
        # ~ print(yy)
        inset_ax.plot(tt, yy, c = colors_for_qs[i], linewidth = linewidth, zorder = -10, alpha = linealpha)
        inset_ax.scatter(lag_steps_crop[1:], F2_crop[i,1:], ec = 'black', s = 10, alpha = scatter_alpha, fc = colors_for_qs[i], marker = 'o')
      
    inset_ax.set_ylim([0,1.02])
    inset_ax.set_xlim([1e4,2e7])
    inset_ax.set_xscale('log')
    # ~ inset_ax.set_xticks([1e4,1e5,1e6,1e7])
    # ~ inset_ax.axvline(x = lag_steps_crop[fit_out])
    
    psi = 2*(1/255*telegraph[:,1] - 0.5)
    print(np.shape(telegraph))
    axs[2].plot(1e-6*L**2*telegraph[:,0],psi, c = 'black', linewidth = 1)
    # ~ axs[2].yaxis.set_major_formatter(FormatStrFormatter('%g $\pi$'))
    # ~ axs[2].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    axs[2].set_xlim([0,3])
    axs[2].set_xlabel(r'Steps X $10^6$')
    # ~ axs[2].set_ylabel('Average Phase of ROI (rad)')
    axs[2].set_ylabel(r'$\langle \psi \rangle$')
    
    # ~ axs[2].axvline(x = 0)
    # ~ axs[2].axvline(x = 100)
    # ~ axs[2].axvline(x = 200)
    # ~ axs[2].axvline(x = 300)
    
    # ~ axs[2].axvline(x = 300)
    
    plt.subplots_adjust(left=0.085, right=0.965, top=0.88, bottom=0.187, wspace=0.35, hspace=0.2)
    plt.show()
    
    labels = ['small q', 'medium q', 'large q']
    for i in range(len(F2_crop)):
        popt, pcov = opt.curve_fit(exponential_decay, lag_steps_crop, F2_crop[i], p0 = p0)
        color = colors[i]
        tt = np.linspace(min(lag_steps_crop[1:]), max(lag_steps_crop), 2000)
        yy = exponential_decay(tt, *popt)
        plt.plot(tt, yy, c = colors_for_qs[i], linewidth = linewidth, zorder = -10, alpha = linealpha)
        plt.scatter(lag_steps_crop[1:], F2_crop[i,1:], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[i], marker = markers_for_qs[i], label = labels[i])
    plt.xscale('log')
    plt.ylabel(r'$F^2$')
    plt.xlabel(r'$\Delta t$ (steps)')
    plt.legend()
    plt.show()
    
def go_frames():
    fig, axs = plt.subplots(1,5, figsize=(10, 2))
    
    mass = tifffile.imread('mass_schematic.tiff')
    mass = mass[::-1,:]
    
    mass_obj = axs[0].imshow(mass, cmap = 'viridis')
    axs[0].set_title(r'$J_{i,j}$')
    # ~ cbar = fig.colorbar(mass_obj, aspect = 10, shrink = 0.25)
    
    frame_0 = tifffile.imread('step_0x1000.tif')
    frame_0 = frame_0[::-1,:]
    # ~ frame_0 = sci.ndimage.gaussian_filter(frame_0, 1)
    axs[1].imshow(frame_0, cmap = 'bwr')
    axs[1].set_title(r'Step 0')
    
    frame_1 = tifffile.imread('step_100x1000.tif')
    frame_1 = frame_1[::-1,:]
    axs[2].imshow(frame_1, cmap = 'bwr')
    axs[2].set_title(r'Step $0.5 \times 10^6$')
    
    frame_2 = tifffile.imread('step_200x1000.tif')
    frame_2 = frame_2[::-1,:]
    axs[3].imshow(frame_2, cmap = 'bwr')
    axs[3].set_title(r'Step $1.0 \times 10^6$')
    
    frame_3 = tifffile.imread('step_300x1000.tif')
    frame_3 = frame_3[::-1,:]
    axs[4].imshow(frame_3, cmap = 'bwr')
    axs[4].set_title(r'Step $1.5 \times 10^6$')
    

    for ax in axs:
        # Create a Rectangle patch
        rect = patches.Rectangle((56.9, 8.5), 5, 5, linewidth=1, edgecolor='black', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        
    
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.subplots_adjust(left=0.085, right=0.965, top=0.88, bottom=0.187, wspace=0.35, hspace=0.2)
    plt.show()
    

def go2():
    data = np.loadtxt('data.csv', skiprows = 1, delimiter = ',')
    print(np.shape(data))
    temps = data[:, 0]
    tau_1 = data[:, 1]
    err_1 = data[:, 2]
    tau_2 = data[:, 3]
    err_2 = data[:, 4]
    tau_3 = data[:, 5]
    err_3 = data[:, 6]
    tau_4 = data[:, 7]
    err_4 = data[:, 8]
    tau_5 = data[:, 9]
    err_5 = data[:, 10]
    
    sim = np.loadtxt('sim.csv', skiprows = 1, delimiter = ',')
    sim = sim[10:,:]
    print(np.shape(sim))
    temps_sim = sim[:, 0]
    tau_1_sim = sim[:, 1]
    err_1_sim = sim[:, 2]
    tau_2_sim = sim[:, 3]
    err_2_sim = sim[:, 4]
    tau_3_sim = sim[:, 5]
    err_3_sim = sim[:, 6]
    
    # Create a figure
    fig = plt.figure(figsize=(12, 8))
    # Create a GridSpec with 2 rows and 4 columns
    gs = GridSpec(2, 4, figure=fig)
    
    # Create subplots for the first row (3 subplots, each taking up the same amount of space)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    
    # Create subplots for the second row (4 subplots)
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[1, 3])

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    # Style Choices
    linewidth = 5
    linealpha = 0.5
    scatter_size = 30
    scatter_alpha = 0.7
    span_alpha = 0.7
    offset = 0
    step = 0.22
    cmap = cm.viridis_r
    colors_for_qs = [cmap(offset),cmap(step+offset),cmap(2*step+offset),cmap(3*step+offset),cmap(4*step+offset)]
    markers_for_qs = ['s','o','^', 'D', 'P']
    
    # Plot the real data
    axes[0].scatter(temps, tau_1/60, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[0], marker = markers_for_qs[0])
    axes[0].scatter(temps, tau_2/60, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[1], marker = markers_for_qs[1])
    axes[0].scatter(temps, tau_3/60, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[2], marker = markers_for_qs[2])
    # ~ axes[0].scatter(temps, tau_4/60)
    # ~ axes[0].scatter(temps, tau_5/60)
    # ~ axes[0].set_yscale('log')
    
    p0_1 = [4.22090094e+00]
    p0_2 = [3.54954222e+00]
    p0_3 = [3.21447820e+00]
    popt_1, pcov_1 = curve_fit(fit_func_1, temps, tau_1/60, p0 = p0_1)
    popt_2, pcov_2 = curve_fit(fit_func_1, temps, tau_2/60, p0 = p0_2)
    popt_3, pcov_3 = curve_fit(fit_func_1, temps, tau_3/60, p0 = p0_3)
    tt = np.linspace(min(temps) - 1, max(temps) + 1, 1000)
    yy = fit_func_1(tt, *popt_1)
    axes[0].plot(tt, yy, c = colors_for_qs[0], linewidth = linewidth, zorder = -10, alpha = linealpha)
    yy = fit_func_1(tt, *popt_2)
    axes[0].plot(tt, yy, c = colors_for_qs[1], linewidth = linewidth, zorder = -10, alpha = linealpha)
    yy = fit_func_1(tt, *popt_3)
    axes[0].plot(tt, yy, c = colors_for_qs[2], linewidth = linewidth, zorder = -10, alpha = linealpha)
    
    # Plot the simulation result
    axes[1].scatter(temps_sim, 50/1000*tau_1_sim, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[0], marker = markers_for_qs[0])
    axes[1].scatter(temps_sim, 50/1000*tau_2_sim, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[1], marker = markers_for_qs[1])
    axes[1].scatter(temps_sim, 50/1000*tau_3_sim, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[2], marker = markers_for_qs[2])
    
    p0_1 = [3]
    p0_2 = [2]
    p0_3 = [1]
    popt_1, pcov_1 = curve_fit(fit_func_2, temps_sim, 50/1000*tau_1_sim, p0 = p0_1)
    popt_2, pcov_2 = curve_fit(fit_func_2, temps_sim, 50/1000*tau_2_sim, p0 = p0_2)
    popt_3, pcov_3 = curve_fit(fit_func_2, temps_sim, 50/1000*tau_3_sim, p0 = p0_3)
    
    # ~ popt_1, popt_2, popt_3 = p0_1, p0_2, p0_3
    tt = np.linspace(min(temps_sim) - 0.005, max(temps_sim) + 0.005, 1000)
    # ~ yy = fit_func_2(tt, *popt_1)
    s_1 = CubicSpline(temps_sim[::-1], 50/1000*tau_1_sim[::-1])
    yy = s_1(tt)
    axes[1].plot(tt, yy, c = colors_for_qs[0], linewidth = linewidth, zorder = -10, alpha = linealpha)
    # ~ s_2 = CubicSpline(temps_sim[::-1], 50/1000*tau_2_sim[::-1])
    # ~ yy = s_2(tt)
    yy = fit_func_2(tt, *popt_2)
    axes[1].plot(tt, yy, c = colors_for_qs[1], linewidth = linewidth, zorder = -10, alpha = linealpha)
    # ~ s_3 = CubicSpline(temps_sim[::-1], 50/1000*tau_3_sim[::-1])
    # ~ yy = s_3(tt)
    yy = fit_func_2(tt, *popt_3)
    axes[1].plot(tt, yy, c = colors_for_qs[2], linewidth = linewidth, zorder = -10, alpha = linealpha)
    
    axes[0].set_xlabel('Temperature (K)')
    axes[0].set_ylabel(r'$\tau$ (Minutes)')
    axes[0].set_title('Data')
    
    axes[1].set_xlabel('Temperature (a.u.)')
    axes[1].set_ylabel(r'$\tau$ (Steps X 1000)')
    axes[1].set_title('Simulation')
    
    
    # Show the plot
    plt.show()

def go3():
    # Style Choices
    linewidth = 5
    linealpha = 0.5
    scatter_size = 30
    scatter_alpha = 0.7
    span_alpha = 0.7
    offset = 0
    step = 0.22
    cmap = cm.viridis_r
    colors_for_qs = [cmap(offset),cmap(step+offset),cmap(2*step+offset),cmap(3*step+offset),cmap(4*step+offset)]
    markers_for_qs = ['s','o','^', 'D', 'P']
    
    
    fit_in = 6
    fit_out = 35
    temp = '2.43'
    # ~ temp = '1.9'
    fileNames = os.listdir()
    
    data_all = []
    for i in range(len(fileNames)):
        if f'g2_{temp}' in fileNames[i]:
            print(fileNames[i])
            data_all.append(np.loadtxt(fileNames[i], delimiter = ','))
            
    data = np.zeros(data_all[0].shape)
    
    weights = np.ones(len(data_all))
    
    for j in range(len(weights)):
        w = weights[j]
        data += w*data_all[j]/(np.sum(weights))
        
    lag_steps = data[0]
    g2 = data[1:]
    F2 = (g2-1)/(g2[:, 0, None] - 1)
    
    # ~ print('F2.shape',F2.shape)
    # Remove some of the data
    lag_steps_crop = lag_steps[fit_in:] - lag_steps[fit_in]
    lag_steps_crop = 50*lag_steps_crop # 50 is the exposure time
    
    # ~ print('F2[:, fit_in, None]',F2[:, fit_in, None])
    # ~ normalization = np.array([0.81867587,0.6950753,0.54176673])
    F2_crop = F2[:, fit_in : F2.shape[1]]/F2[:, fit_in, None]
    F_crop = np.sqrt(F2_crop * np.sign(F2_crop))
    
    print('F_crop', np.shape(F_crop))
    
    # ~ F2_crop = F2[:, fit_in : F2.shape[1] - fit_out]/normalization[:, None]
    
    # ~ print('F2_crop.shape',F2_crop.shape)
    
    
    # ~ print('F2_crop', F2_crop)
    
    fig, axs = plt.subplots()
    
    p0 = [26*50, 1]
    colors = ['blue', 'green', 'red']
    labels = ['q0', 'q1', 'q2']
    for i in range(len(F2)):
        try:
            popt, pcov = opt.curve_fit(exponential_decay, lag_steps_crop[:fit_out], F_crop[i, :fit_out], p0 = p0)
            # ~ print('F2_crop[i]', F2_crop[i])
        except:
            p0 = popt
            print(f'ERROR IN FIT {i}')
        
        print('popt',popt)
        # ~ print(f'{labels[i]}: \n tau = {popt[0]} pm {np.sqrt(pcov[0,0])} \n beta = {popt[1]} pm {np.sqrt(pcov[1,1])} \n I0 = {popt[2]} pm {np.sqrt(pcov[2,2])}')
        
        # ~ print(i)
        color = colors[i]
        tt = np.linspace(min(lag_steps_crop[1:]), max(lag_steps_crop), 2000)
        yy = exponential_decay(tt, *popt)
        # ~ print(yy)
        axs.plot(tt, yy, c = colors_for_qs[i], linewidth = linewidth, zorder = -10, alpha = linealpha)
        axs.scatter(lag_steps_crop[1:], F_crop[i,1:], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[i], marker = markers_for_qs[i])
      
    # ~ axs.set_ylim([0,1.02])
    # ~ axs.set_xlim([0.9,1000])
    axs.set_xscale('log')
    # ~ inset_ax.axvline(x = lag_steps_crop[fit_out])
    
    plt.show()



go1()
go_frames()


# ~ go3()

