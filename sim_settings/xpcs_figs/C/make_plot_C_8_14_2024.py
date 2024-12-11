import numpy as np
import matplotlib.pyplot as plt
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
        
def plot_regions(ax, colors_for_qs, delta = 1*3.77, alpha = 0.5):
    # ~ ax.axvline(x = 0, color = 'orange', linestyle = '--')
    ax.axvspan(-delta,delta, color = colors_for_qs[0], alpha = 0.5)
    
    ax.axvspan(delta, 2*delta ,color = colors_for_qs[1], alpha = 0.5)
    ax.axvspan(-2*delta, -delta ,color = colors_for_qs[1], alpha = 0.5)
    
    ax.axvspan(2*delta, 3*delta ,color = colors_for_qs[2], alpha = 0.5)
    ax.axvspan(-3*delta, -2*delta ,color = colors_for_qs[2], alpha = 0.5)
    
    ax.axvspan(3*delta, 4*delta ,color = colors_for_qs[3], alpha = 0.5)
    ax.axvspan(-4*delta, -3*delta ,color = colors_for_qs[3], alpha = 0.5)
    
    ax.axvspan(4*delta, 5*delta ,color = colors_for_qs[4], alpha = 0.5)
    ax.axvspan(-5*delta, -4*delta ,color = colors_for_qs[4], alpha = 0.5)

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
    
# Conversion to inverse angstrom
pl = 3.77 # pixel lengths to inverse microns

# Load Data
lagsteps = np.loadtxt('lag_steps.txt', delimiter = ',')[1:]

data_202 = np.loadtxt('202K_g2.txt', delimiter = ',')

waterfall_202 = tifffile.imread('202K_waterfall.tiff')
waterfall_202 = waterfall_202[:, 18:-15]
print(waterfall_202.shape)
image_202 = tifffile.imread('202K_image.tiff')

data_206 = np.loadtxt('206K_g2.txt', delimiter = ',')
waterfall_206 = tifffile.imread('206K_waterfall.tiff')
# ~ print(waterfall_206.shape)
image_206 = tifffile.imread('206K_image.tiff')

scale_factor = 5
image_202, image_206 = scale_factor*image_202, scale_factor*image_206


# Bin the data
bin_size = 3
lagsteps = bin_data(lagsteps, bin_size)
data_202 = bin_data(data_202, bin_size)
data_206 = bin_data(data_206, bin_size)

# ~ plt.imshow(image_206)
# ~ plt.axhline(y=70)
# ~ plt.show()


# Style Choices
scatter_size = 20
scatter_alpha = 0.7
span_alpha = 0.7

# ~ offset = 0.2
# ~ step = 0.19
# ~ cmap = cm.winter
offset = 0
step = 0.22
cmap = cm.viridis_r
colors_for_qs = [cmap(offset),cmap(step+offset),cmap(2*step+offset),cmap(3*step+offset),cmap(4*step+offset)]
markers_for_qs = ['s','o','^', 'D', 'P']
yaxis_lims = [0.97,1.07]

fig, axs = plt.subplots(3,2)

# 202 K
axs[0,0].set_title('202 K')
axs[2,0].scatter(lagsteps, data_202[0], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[0], marker = markers_for_qs[0])
axs[2,0].scatter(lagsteps, data_202[1], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[1], marker = markers_for_qs[1])
axs[2,0].scatter(lagsteps, data_202[2], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[2], marker = markers_for_qs[2])
axs[2,0].scatter(lagsteps, data_202[3], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[3], marker = markers_for_qs[3])
axs[2,0].scatter(lagsteps, data_202[4], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[4], marker = markers_for_qs[4])
axs[2,0].set_xscale('log')
axs[2,0].set_xlabel(r'$\Delta$t (s)')
axs[2,0].set_ylim(yaxis_lims)
axs[0,0].set_xlim([-154,126])

waterfall_vmax = 14
waterfall_vmin = 1

axs[1,0].imshow(waterfall_202, aspect = 'auto', extent = [0,waterfall_202.shape[1],waterfall_202.shape[0]/60,0], vmax = waterfall_vmax, vmin = waterfall_vmin, cmap = 'viridis')
# ~ axs[0,0].imshow(image_75)
# ~ Q_202 = pl*(np.arange(0,len(image_202[35,:]),1) - 40)
Q_202 = pl*(np.arange(0,len(waterfall_202[35,:]),1) - 22)

# ~ axs[0,0].plot(Q_202, image_202[35,:], 'k-', linewidth = 2)
axs[0,0].plot(Q_202, 0.75*np.mean(waterfall_202, axis = 0), 'k-', linewidth = 2)

plot_regions(axs[0,0], colors_for_qs, delta = 1*pl)
axs[0,0].set_xlabel(r'q ($\mu$m$^{-1}$)')
axs[1,0].set_xticks([])

# 206 K
axs[0,1].set_title('206 K')
axs[2,1].scatter(lagsteps, data_206[0], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[0], marker = markers_for_qs[0], label = f'{int(1*5*3.77)} ' + r'$\mu$m$^{-1}$')
axs[2,1].scatter(lagsteps, data_206[1], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[1], marker = markers_for_qs[1], label = f'{int(2*5*3.77)} ' + r'$\mu$m$^{-1}$')
axs[2,1].scatter(lagsteps, data_206[2], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[2], marker = markers_for_qs[2], label = f'{int(3*5*3.77)} ' + r'$\mu$m$^{-1}$')
axs[2,1].scatter(lagsteps, data_206[3], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[3], marker = markers_for_qs[3], label = f'{int(4*5*3.77)} ' + r'$\mu$m$^{-1}$')
axs[2,1].scatter(lagsteps, data_206[4], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[4], marker = markers_for_qs[4], label = f'{int(5*5*3.77)} ' + r'$\mu$m$^{-1}$')
axs[2,1].set_xscale('log')
axs[2,1].set_xlabel(r'$\Delta$t (s)')
axs[2,1].set_ylim(yaxis_lims)
# ~ axs[0,1].legend()


axs[1,1].imshow(5*waterfall_206, aspect = 'auto', extent = [0,waterfall_206.shape[1],waterfall_206.shape[0]/60,0], vmax = waterfall_vmax, vmin = waterfall_vmin)
axs[1,1].text(0.6,0.8,s = '5 X', transform = axs[1,1].transAxes, color = 'white', fontsize = 14)
# ~ axs[0,0].imshow(image_75)
Q_206 = pl*(np.arange(0,len(image_206[70,:]),1) - 75)
# ~ axs[0,1].plot(Q_206, image_206[70,:], 'k-', linewidth = 2)
axs[0,1].plot(Q_206, np.mean(6*waterfall_206, axis = 0), 'k-', linewidth = 2)
plot_regions(axs[0,1], colors_for_qs, delta = 5*pl)
axs[0,1].set_xlabel(r'q ($\mu$m$^{-1}$)')
axs[1,1].set_xticks([])

axs[0,0].set_ylabel('Intensity (a.u.)')
axs[1,0].set_ylabel('Time (minute)')
axs[2,0].set_ylabel(r'$g_2$')

axs[0,0].set_yticks([5,10])
axs[0,1].set_yticks([8,10])
axs[1,1].set_yticks([])
axs[2,1].set_yticks([])
axs[0,0].set_xlim([min(Q_202),max(Q_202)])
axs[0,1].set_xlim([min(Q_206), max(Q_206)])
   
axs[0,1].tick_params(axis="y",direction="in", pad=-22)   


axs[0,0].set_yscale('log')
axs[0,1].set_yscale('log')

for a in range(2):
    axs[2,a].axhline(y=1,color='black',linestyle='--')

plt.show()

Q = [0,1,2,3,4]
G = [np.mean(data_202[0]), np.mean(data_202[1]), np.mean(data_202[2]), np.mean(data_202[3]), np.mean(data_202[4])]


plt.scatter(Q,G)
plt.show()
