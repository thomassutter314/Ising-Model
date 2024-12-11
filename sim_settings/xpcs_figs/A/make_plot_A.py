import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import cm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
import tifffile

from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

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

# Define the functions for measuring the intensity and beam width
# We also will want a function for diagonalizing a 2D matrix

def gaussian_offset(x, mu, A, sigma, offset):
    return A*np.exp(-0.5*((x-mu)/sigma)**2) + offset

def compute_moments_1d(data, offset = 0, bg_subtract = False):
    if bg_subtract:
        bg_mean = (np.mean(data[:5]) + np.mean(data[-6:]))/2
        print(bg_mean)
        data = data - bg_mean # subtract off the bg value
        
    data = data - offset
    data[data < 0] = 0 # make values strictly positive
    
    # Compute 0th and 1st moments
    I0 = np.sum(data)
    x = np.arange(0,len(data),1)
    I1 = np.array(np.sum(x*data))/I0
    I2 = np.sum((x-I1)**2*data)/I0 

    return I0, I1, I2

def compute_moments_2d(im, bg_subtract = True):
    if bg_subtract:
        bg_mean = (np.mean(im[-1,:]) + np.mean(im[0,:]) + np.mean(im[:,-1]) + np.mean(im[0,:]))/4
        print(bg_mean)
        im = im - bg_mean # subtract off the bg value
        im[im < 0] = 0 # make values strictly positive
    
    # Compute 0th and 1st moments
    I0 = np.sum(im)
    x = range(np.shape(im)[1])
    y = range(np.shape(im)[0])
    X, Y = np.meshgrid(x, y)
    I1 = np.array([np.sum(X*im)/I0,np.sum(Y*im)/I0])

    # return an array with structure [[var(x), covar(x,y)],[covar(y,x), var(y)]]
    var_x = np.sum((X-I1[0])**2*im)/I0 
    var_y = np.sum((Y-I1[1])**2*im)/I0 
    covar_xy = np.sum((X-I1[0])*(Y-I1[1])*im)/I0 
    I2 = np.array([[var_x, covar_xy],[covar_xy, var_y]])

    return I0, I1, I2

def diag_rank2(A):
    a = A[0,0]
    b = A[0,1]
    c = A[1,0]
    d = A[1,1]

    T1 = (a+d)/2
    T2 = np.sqrt(b*c-a*d+(a+d)**2/4)

    return T1 + T2, T1 - T2

def get_sigma_2d(im):
    I0, I1, I2 = compute_moments_2d(im)
    v1, v2 = diag_rank2(I2)
    return (v1*v2)**(1/4)
    
def get_sigma_1d(data,offset=0):
    I0, I1, I2 = compute_moments_1d(data,offset=offset)
    return np.sqrt(I2)
    
def plot_regions(ax, colors_for_qs, delta = 1, alpha = 0.1):
    # ~ ax.axvline(x = 0, color = 'orange', linestyle = '--')
    ax.axvspan(-delta, delta, color = colors_for_qs[0], alpha = alpha)
    for i in range(len(colors_for_qs)-1):
        ax.axvspan((i+1)*delta, (i+2)*delta ,color = colors_for_qs[i+1], alpha = alpha)
        ax.axvspan(-(i+2)*delta, -(i+1)*delta ,color = colors_for_qs[i+1], alpha = alpha)

def bin_data(data, n):
    if len(np.shape(data)) == 1:
        if len(data)%n == 0:
            output = np.zeros(len(data)//n)
        else:
            output = np.zeros(len(data)//n + 1)
            
        for i in range(len(data)):
            output[i//n] = (data[i] + (i%n)*output[i//n])/(i%n + 1)
            
    if len(np.shape(data)) == 2:    

        if data.shape[1]%n == 0:
            output = np.zeros([data.shape[0],data.shape[1]//n])
        else:
            output = np.zeros([data.shape[0],data.shape[1]//n + 1])
        
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                output[i, j//n] = (data[i, j] + (j%n)*output[i, j//n])/(j%n + 1)
            
    return output
   
def fit_func_1(t, tau, beta):
    return 1 + np.exp(-1*(t/tau)**beta)
    
def fit_func_2(t, tau, beta):
    beta = 1.2
    return np.exp(-1*(t/tau)**beta)
   
# Conversion to inverse angstrom
# Conversion to inverse angstrom
L_det = 1.5 # in meter
pix_size = 75e-6 # in meter
λ = 0.96847 # in angstrom
pl = 2.355*2*np.pi/λ*(pix_size/L_det)



print(f'pl = {pl}')

# Load Data
lagsteps = np.loadtxt('lag_steps.txt', delimiter = ',')[1:]

g2_75 = np.loadtxt('75K_g2.txt', delimiter = ',')
waterfall_75 = tifffile.imread('75K_waterfall.tiff')
waterfall_75 = np.array(waterfall_75, dtype = float)  - np.min(waterfall_75)
# ~ waterfall_75[waterfall_75 < 0] = 0

g2_95 = np.loadtxt('95K_g2.txt', delimiter = ',')
waterfall_95 = tifffile.imread('95K_waterfall.tiff')
waterfall_95 = np.array(waterfall_95, dtype = float)  - np.min(waterfall_95)
# ~ waterfall_95[waterfall_95 < 0] = 0

g2_100 = np.loadtxt('100K_g2.txt', delimiter = ',')
waterfall_100 = tifffile.imread('100K_waterfall.tiff')
waterfall_100 = np.array(waterfall_100, dtype = float)  - np.min(waterfall_100)


# Bin the data
bin_size = 2
lagsteps = bin_data(lagsteps, bin_size)
g2_75 = bin_data(g2_75, bin_size)
g2_95 = bin_data(g2_95, bin_size)
g2_100 = bin_data(g2_100, bin_size)

plt.plot(g2_95[1,:])
plt.show()

data_75 = (g2_75-1)/np.mean(g2_75[:, :10, None] - 1, axis = 1)
data_95 = (g2_95-1)/np.mean(g2_95[:, :10, None] - 1, axis = 1)
data_100 = (g2_100-1)/np.mean(g2_100[:, :10, None] - 1, axis = 1)

lagsteps_full = np.copy(lagsteps)
data_100_full = np.copy(data_100)
# remove the last d data point from all scans
d = 2
data_75 = data_75[:, :-d]
data_95 = data_95[:, :-d]
data_100 = data_100[:, :-d]
lagsteps = lagsteps[:-d]
    
    
# Style Choices
scatter_size = 25
linewidth = 5
scatter_alpha = 0.7
span_alpha = 0.5

offset = 0
step = 0.22
cmap = cm.viridis_r
colors_for_qs = [cmap(offset),cmap(step+offset),cmap(2*step+offset),cmap(3*step+offset),cmap(4*step+offset)]
markers_for_qs = ['s','o','^', 'D', 'P']
yaxis_lims = [0,1.1]
axins_ymin, axins_ymax = 9, 200
axins_size = 25

fig, axs = plt.subplots(3,3, sharey = 'row', figsize = (10,7))
plt.subplots_adjust(left = 0.078, bottom = 0.11, right = 0.978, top = 0.94, wspace = 0.076, hspace = 0.2)

# For all plots to have the same visual scale
waterfall_max = 130
waterfall_min = 0

mag_factor_75 = 4.5
bg_75 = 2.9
# 75 K
axs[0,0].set_title('75 K')
# ~ axs[2,0].scatter(lagsteps, data_75[0], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[0], marker = markers_for_qs[0])
# ~ axs[2,0].scatter(lagsteps, data_75[1], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[1], marker = markers_for_qs[1])
# ~ axs[2,0].scatter(lagsteps, data_75[2], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[2], marker = markers_for_qs[2])
# ~ axs[2,0].scatter(lagsteps, data_75[3], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[3], marker = markers_for_qs[3])
# ~ axs[2,0].scatter(lagsteps, data_75[4], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[4], marker = markers_for_qs[4])
axs[2,0].set_xscale('log')
axs[1,0].imshow(mag_factor_75*waterfall_75, aspect = 'auto', extent = [-waterfall_75.shape[1]/2,waterfall_75.shape[1]/2,waterfall_75.shape[0]/60,0], vmax = waterfall_max, vmin = waterfall_min)
Q_75 = pl*(np.arange(0,waterfall_75.shape[1],1) - 38)
axs[0,0].plot(Q_75, mag_factor_75*(np.mean(waterfall_75, axis = 0) - bg_75), 'k-', linewidth = 2)
plot_regions(axs[0,0], colors_for_qs, delta = 2.75*pl, alpha = span_alpha)

p0 = [1000, 1]
tt = np.linspace(min(lagsteps), max(lagsteps), 1000)
taus = np.array([10576.0,5484.69,3837.15,3961.47,4314.42])
for i in range(5):
    # ~ popt, pcov = curve_fit(fit_func_2, lagsteps, data_75[i], p0 = p0)
    # ~ print(popt)
    yy = fit_func_2(tt, taus[i], beta = 1.2)
    # ~ axs[2,1].scatter(lagsteps, data_75[i], ec = 'black', s = 7, alpha = scatter_alpha, fc = colors_for_qs[i], marker = markers_for_qs[i])
    axs[2,0].scatter(lagsteps, data_75[i], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[i])
    axs[2,0].plot(tt, yy, color = colors_for_qs[i], linewidth = linewidth, alpha = 0.75, zorder = -10)



# ~ # Create an inset
# ~ # inset axes....
x_inset = 0.2
y_inset = 0.23
w_inset = 0.3
h_inset = 0.3
axins = axs[2,0].inset_axes([x_inset, y_inset, w_inset, h_inset])
for i in range(5):
    print(f'mean q = {2.75*pl*(i+0.5)}', f'max q = {2.75*pl*(i+1)}')
    axins.scatter([2.75*pl*(i+0.5)], [taus[i]/60], ec = 'black', s = axins_size, alpha = scatter_alpha, fc = colors_for_qs[i])
axins.set_ylabel(r'$\tau$ (min)', fontsize=9)
axins.set_xlabel(r'$q_{\text{avg}}$ (Å$^{-1}$)', fontsize=9, labelpad = -3)
# ~ axins.set_xticks([])
axins.set_xticks([0,0.008])
axins.set_yscale('log')
# ~ axins.yaxis.set_major_formatter(NullFormatter())
# ~ axins.yaxis.set_minor_formatter(NullFormatter())
axins.tick_params(axis='both', which='major', labelsize=9)
axins.set_ylim([axins_ymin,axins_ymax])



# 95 K
mag_factor_95 = mag_factor_75*2
bg_95 = 3.9
axs[0,1].set_title('95 K')

# ~ axs[2,1].scatter(lagsteps, data_95[0], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[0], marker = markers_for_qs[0])
# ~ axs[2,1].scatter(lagsteps, data_95[1], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[1], marker = markers_for_qs[1])
# ~ axs[2,1].scatter(lagsteps, data_95[2], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[2], marker = markers_for_qs[2])
# ~ axs[2,1].scatter(lagsteps, data_95[3], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[3], marker = markers_for_qs[3])
# ~ axs[2,1].scatter(lagsteps, data_95[4], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[4], marker = markers_for_qs[4])

axs[2,1].set_xscale('log')
axs[1,1].imshow(mag_factor_95*waterfall_95, aspect = 'auto', extent = [-waterfall_95.shape[1]/2,waterfall_95.shape[1]/2,waterfall_95.shape[0]/60,0], vmax = waterfall_max, vmin = waterfall_min)
axs[1,1].text(0.75,0.82,s = '2 X', transform = axs[1,1].transAxes, color = 'white', fontsize = 14)
axs[0,1].text(0.75,0.82,s = '2 X', transform = axs[0,1].transAxes, color = 'black', fontsize = 14)
Q_95 = pl*(np.arange(0,waterfall_95.shape[1],1) - 36)
axs[0,1].plot(Q_95, mag_factor_95*(np.mean(waterfall_95, axis = 0) - bg_95), 'k-', linewidth = 2)
plot_regions(axs[0,1], colors_for_qs, delta = 2.75*pl, alpha = span_alpha)

p0 = [1000, 1]
tt = np.linspace(min(lagsteps), max(lagsteps), 1000)
taus = np.array([2182.82,1627.06,725.96,746.03,733.16])
for i in range(5):
    yy = fit_func_2(tt, taus[i], beta = 1.2)
    # ~ axs[2,1].scatter(lagsteps, data_95[i], ec = 'black', s = 7, alpha = scatter_alpha, fc = colors_for_qs[i], marker = markers_for_qs[i])
    axs[2,1].scatter(lagsteps, data_95[i], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[i])
    axs[2,1].plot(tt, yy, color = colors_for_qs[i], linewidth = linewidth, alpha = 0.75, zorder = -10)


# ~ # Create an inset
# ~ # inset axes....
x_inset = 0.2
y_inset = 0.23
w_inset = 0.3
h_inset = 0.3
axins = axs[2,1].inset_axes([x_inset, y_inset, w_inset, h_inset])
for i in range(5):
    axins.scatter([2.75*pl*(i+0.5)], [taus[i]/60], ec = 'black', s = axins_size, alpha = scatter_alpha, fc = colors_for_qs[i])
axins.set_ylabel(r'$\tau$ (min)', fontsize=9)
axins.set_xlabel(r'$q_{\text{avg}}$ (Å$^{-1}$)', fontsize=9, labelpad = -3)
# ~ axins.set_xticks([])
axins.set_xticks([0,0.008])
axins.set_yscale('log')
axins.yaxis.set_major_formatter(NullFormatter())
axins.yaxis.set_minor_formatter(NullFormatter())
axins.tick_params(axis='both', which='major', labelsize=9)
axins.set_ylim([axins_ymin,axins_ymax])


mag_factor_100 = mag_factor_75*4
bg_100 = 3.9
# 100 K
axs[0,2].set_title('100 K')
# ~ axs[2,2].scatter(lagsteps, data_100[0], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[0], marker = markers_for_qs[0], label = f'{int(2.75*3.77)} ' + r'$\mu$m$^{-1}$')
# ~ axs[2,2].scatter(lagsteps, data_100[1], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[1], marker = markers_for_qs[1], label = f'{int(2*2.75*3.77)} ' + r'$\mu$m$^{-1}$')
# ~ axs[2,2].scatter(lagsteps, data_100[2], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[2], marker = markers_for_qs[2], label = f'{int(3*2.75*3.77)} ' + r'$\mu$m$^{-1}$')
# ~ axs[2,2].scatter(lagsteps, data_100[1], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[3], marker = markers_for_qs[3], label = f'{int(4*2.75*3.77)} ' + r'$\mu$m$^{-1}$')
# ~ axs[2,2].scatter(lagsteps, data_100[2], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[4], marker = markers_for_qs[4], label = f'{int(5*2.75*3.77)} ' + r'$\mu$m$^{-1}$')
axs[1,2].imshow(mag_factor_100*waterfall_100, aspect = 'auto', extent = [-waterfall_100.shape[1]/2,waterfall_100.shape[1]/2,waterfall_100.shape[0]/60,0], vmax = waterfall_max, vmin = waterfall_min)
axs[1,2].text(0.75,0.82,s = '4 X', transform = axs[1,2].transAxes, color = 'white', fontsize = 14)
axs[0,2].text(0.75,0.82,s = '4 X', transform = axs[0,2].transAxes, color = 'black', fontsize = 14)
axs[2,2].set_xscale('log')
Q_100 = pl*(np.arange(0,waterfall_100.shape[1],1) - 33)
axs[0,2].plot(Q_100, mag_factor_100*(np.mean(waterfall_100, axis = 0) - bg_100), 'k-', linewidth = 2)
plot_regions(axs[0,2], colors_for_qs, delta = 2.75*pl, alpha = span_alpha)

p0 = [1000, 1]
tt = np.linspace(min(lagsteps), max(lagsteps), 1000)
taus = [697.79,1036.77,852.25,757.83,1034.4]
for i in range(5):
    yy = fit_func_2(tt, taus[i], beta = 1.2)
    # ~ axs[2,1].scatter(lagsteps, data_100[i], ec = 'black', s = 7, alpha = scatter_alpha, fc = colors_for_qs[i], marker = markers_for_qs[i])
    axs[2,2].scatter(lagsteps, data_100[i], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[i])
    axs[2,2].plot(tt, yy, color = colors_for_qs[i], linewidth = linewidth, alpha = 0.75, zorder = -10)

    
# ~ # Create an inset
# ~ # inset axes....
x_inset = 0.2
y_inset = 0.23
w_inset = 0.3
h_inset = 0.3
axins = axs[2,2].inset_axes([x_inset, y_inset, w_inset, h_inset])
for i in range(5):
    axins.scatter([2.75*pl*(i+0.5)], [taus[i]/60], ec = 'black', s = axins_size, alpha = scatter_alpha, fc = colors_for_qs[i])
    
    
axins.set_ylabel(r'$\tau$ (min)', fontsize=9)
axins.set_xlabel(r'$q_{\text{avg}}$ (Å$^{-1}$)', fontsize=9, labelpad = -3)
# ~ axins.set_xticks([])
axins.tick_params(axis='both', which='major', labelsize=9)
axins.set_yscale('log')
axins.set_xticks([0,0.008])
axins.yaxis.set_major_formatter(NullFormatter())
axins.yaxis.set_minor_formatter(NullFormatter())
axins.tick_params(axis='both', which='major', labelsize=9)
axins.set_ylim([axins_ymin,axins_ymax])

# Axis labels
axs[2,2].set_xlabel(r'$\Delta$t (s)')
axs[1,1].set_xlabel(r'$q$ (Å$^{-1}$)')
axs[2,1].set_xlabel(r'$\Delta$t (s)')
axs[1,0].set_xlabel(r'$q$ (Å$^{-1}$)')
axs[2,0].set_xlabel(r'$\Delta$t (s)')
axs[1,2].set_xlabel(r'$q$ (Å$^{-1}$)')
axs[0,0].set_ylabel('Intensity (a.u.)')
axs[1,0].set_ylabel('Time (minute)')
axs[2,0].set_ylabel('$|F|^2$')

# Axis limits
axs[2,1].set_ylim(yaxis_lims)
axs[0,0].set_xlim([min(Q_75),max(Q_75)])
axs[0,1].set_xlim([min(Q_95),max(Q_95)])
axs[0,2].set_xlim([min(Q_100),max(Q_100)])


# Axis ticks marks
# ~ axs[0,2].set_xticks([])
# ~ axs[0,0].set_xticks([])
# ~ axs[0,1].set_xticks([])
    
for a in range(3):
    axs[2,a].axhline(y=1,color='black',linestyle='--', alpha = 0.5)


plt.show()
