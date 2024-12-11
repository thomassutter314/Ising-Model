import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from matplotlib.ticker import ScalarFormatter
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

def plot_regions_skip_n(ax, colors_for_qs, delta = 1*3.77, alpha = 0.5, n = 3):
    for i in range(5):
        ax.axvspan((i+n)*delta, (i+n+1)*delta ,color = colors_for_qs[i], alpha = 0.5)
        ax.axvspan(-(i+n+1)*delta, -(i+n)*delta ,color = colors_for_qs[i], alpha = 0.5)
        
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
L_det = 1.5 # in meter
pix_size = 75e-6 # in meter
λ = 0.96847 # in angstrom
pl = 2.355*2*np.pi/λ*(pix_size/L_det)

print(2*pl, 7*pl)

# Load Data
lagsteps = np.loadtxt('lag_steps.txt', delimiter = ',')[1:]
data_200 = np.loadtxt('200K_g2.txt', delimiter = ',')

waterfall_200 = tifffile.imread('200K_waterfall.tiff')
waterfall_200 = waterfall_200[:, 13:-12]
print(waterfall_200.shape)
image_200 = tifffile.imread('200K_image.tiff')

data_204 = np.loadtxt('204K_g2.txt', delimiter = ',')
waterfall_204 = tifffile.imread('204K_waterfall.tiff')
# ~ print(waterfall_206.shape)
image_204 = tifffile.imread('204K_image.tiff')
# For the 204 K data, the bg is important
bg = 0.8
waterfall_204 = np.array(waterfall_204, dtype = float) - bg
waterfall_204_display = np.copy(waterfall_204) # A separate array for the actual waterfall plot because we can't have negative pixels on a log scale plot like this
waterfall_204_display[waterfall_204_display < 0] = 0
image_204 = image_204 - bg


# Bin the data
bin_size = 2
lagsteps = bin_data(lagsteps, bin_size)
data_200 = bin_data(data_200, bin_size)
data_204 = bin_data(data_204, bin_size)


# Style Choices
scatter_size = 25
linewidth = 5
scatter_alpha = 0.7
span_alpha = 0.7


offset = 0
step = 0.22
cmap = cm.viridis_r
colors_for_qs = [cmap(offset),cmap(step+offset),cmap(2*step+offset),cmap(3*step+offset),cmap(4*step+offset)]
markers_for_qs = ['s','o','^', 'D', 'P']


fig, axs = plt.subplots(3,2, sharey = 'row', figsize = (5,7))
plt.subplots_adjust(left = 0.18, bottom = 0.08, right = 0.978, top = 0.963, wspace = 0.076, hspace = 0.2)
# ~ yaxis_lims = [0.9985,1.015]
# ~ axs[2,0].set_ylim(yaxis_lims)
# ~ axs[2,1].set_ylim(yaxis_lims)

# for both plots to have same visual scale
waterfall_vmax = 4.5
waterfall_vmin = 0

# 200 K
axs[0,0].set_title('200 K')
# ~ axs[2,0].scatter(lagsteps, data_200[0], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[0], marker = markers_for_qs[0])
# ~ axs[2,0].scatter(lagsteps, data_200[1], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[1], marker = markers_for_qs[1])
# ~ axs[2,0].scatter(lagsteps, data_200[2], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[2], marker = markers_for_qs[2])
# ~ axs[2,0].scatter(lagsteps, data_200[3], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[3], marker = markers_for_qs[3])
# ~ axs[2,0].scatter(lagsteps, data_200[4], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[4], marker = markers_for_qs[4])
axs[2,0].set_xscale('log')
axs[2,0].set_xlabel(r'$\Delta$t (s)')
axs[1,0].imshow(np.log(waterfall_200+1)/np.log(10), aspect = 'auto', extent = [-waterfall_200.shape[1]/2,waterfall_200.shape[1]/2,waterfall_200.shape[0]/60,0], cmap = 'viridis', vmax = waterfall_vmax, vmin = waterfall_vmin)
Q_200 = pl*(np.arange(0,len(waterfall_200[35,:]),1) - 24)
axs[0,0].plot(Q_200, np.mean(waterfall_200, axis = 0), 'k-', linewidth = 2)
plot_regions(axs[0,0], colors_for_qs, delta = 2*pl)
tt = np.linspace(min(lagsteps), max(lagsteps), 1000)
for i in range(5):
    yy = np.ones(len(tt))*np.mean(data_200[i] - 1)
    # ~ axs[2,1].scatter(lagsteps, data_75[i], ec = 'black', s = 7, alpha = scatter_alpha, fc = colors_for_qs[i], marker = markers_for_qs[i])
    axs[2,0].scatter(lagsteps, data_200[i] - 1, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[i])
    axs[2,0].plot(tt, yy, color = colors_for_qs[i], linewidth = linewidth, alpha = 0.75, zorder = -10)


mag_factor = 1e4 # magnification factor for the intensity of the 204 K data wrt the 200 K data
# 204 K
axs[0,1].set_title('204 K')
# ~ axs[2,1].scatter(lagsteps, data_204[0], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[0], marker = markers_for_qs[0], label = f'{int(1*5*3.77)} ' + r'$\mu$m$^{-1}$')
# ~ axs[2,1].scatter(lagsteps, data_204[1], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[1], marker = markers_for_qs[1], label = f'{int(2*5*3.77)} ' + r'$\mu$m$^{-1}$')
# ~ axs[2,1].scatter(lagsteps, data_204[2], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[2], marker = markers_for_qs[2], label = f'{int(3*5*3.77)} ' + r'$\mu$m$^{-1}$')
# ~ axs[2,1].scatter(lagsteps, data_204[3], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[3], marker = markers_for_qs[3], label = f'{int(4*5*3.77)} ' + r'$\mu$m$^{-1}$')
# ~ axs[2,1].scatter(lagsteps, data_204[4], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[4], marker = markers_for_qs[4], label = f'{int(5*5*3.77)} ' + r'$\mu$m$^{-1}$')
axs[2,1].set_xscale('log')
axs[2,1].set_xlabel(r'$\Delta$t (s)')
# ~ axs[0,1].legend()
axs[1,1].imshow(np.log(mag_factor*waterfall_204_display+1)/np.log(10), aspect = 'auto', extent = [-waterfall_204.shape[1]/2,waterfall_204.shape[1]/2,waterfall_204.shape[0]/60,0], vmax = waterfall_vmax, vmin = waterfall_vmin)
axs[1,1].text(0.75,0.89,s = r'$10^4$ X', transform = axs[1,1].transAxes, color = 'white', fontsize = 12)
axs[0,1].text(0.75,0.89,s = r'$10^4$ X', transform = axs[0,1].transAxes, color = 'black', fontsize = 12)
Q_204 = pl*(np.arange(0,len(waterfall_204[35,:]),1) - 75)
axs[0,1].plot(Q_204, np.mean(mag_factor*waterfall_204, axis = 0), 'k-', linewidth = 2)
plot_regions(axs[0,1], colors_for_qs, delta = 7*pl)

tt = np.linspace(min(lagsteps), max(lagsteps), 1000)
for i in range(5):
    yy = np.ones(len(tt))*np.mean(data_204[i]) - 1
    # ~ axs[2,1].scatter(lagsteps, data_75[i], ec = 'black', s = 7, alpha = scatter_alpha, fc = colors_for_qs[i], marker = markers_for_qs[i])
    axs[2,1].scatter(lagsteps, data_204[i] - 1, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[i])
    axs[2,1].plot(tt, yy, color = colors_for_qs[i], linewidth = linewidth, alpha = 0.75, zorder = -10)


# inset Axes....
x1, x2, y1, y2 = 1, 1280, -.001, .001  # subregion of the original image
axins = axs[2,1].inset_axes(
    [0.3, 0.6, 0.6, 0.3], xticks=[],
    xlim=(x1, x2), ylim=(y1, y2))
    
for i in range(5):
    yy = np.ones(len(tt))*np.mean(data_204[i]) - 1
    axins.scatter(lagsteps, data_204[i] - 1, ec = 'black', s = 0.5*scatter_size, alpha = scatter_alpha, fc = colors_for_qs[i])
    axins.plot(tt, yy, color = colors_for_qs[i], linewidth = 3, alpha = 0.75, zorder = -10)
# ~ axins.set_xscale('log')
axs[2,1].indicate_inset_zoom(axins, edgecolor="black")
axins.set_xscale('log')
axins.tick_params(axis='both', which='major', labelsize=9)




# Axis labels
axs[1,1].set_xlabel(r'$q$ (Å$^{-1}$)')
axs[1,0].set_xlabel(r'$q$ (Å$^{-1}$)')
axs[0,0].set_ylabel('Intensity (a.u.)')
axs[1,0].set_ylabel('Time (minute)')
axs[2,0].set_ylabel(r'$g_2 - 1$')


# Axis scales
axs[0,0].set_yscale('log')
axs[0,1].set_yscale('log')

# Axis ticks
# ~ axs[0,1].set_xticks([])
# ~ axs[0,0].set_xticks([])

# ~ axs[1,1].set_yticks([])
# ~ axs[2,1].set_yticks([])

axs[0,0].set_xlim([min(Q_200),max(Q_200)])
axs[0,1].set_xlim([min(Q_204), max(Q_204)])
# ~ axs[0,1].tick_params(axis="y",direction="in", pad=-22)


for a in range(2):
    axs[2,a].axhline(y=0,color='black',linestyle='--')

# ~ subplots_adjust(left = )
plt.show()

fig, axs = plt.subplots()
Q = np.array([0,1,2,3,4])
G = np.array([np.mean(data_200[0]), np.mean(data_200[1]), np.mean(data_200[2]), np.mean(data_200[3]), np.mean(data_200[4])])
axs.scatter(Q,G)

# Change the y-axis to scientific notation
axs.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
axs.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Optionally, adjust the font size and other properties
axs.yaxis.get_offset_text().set_fontsize(12)

plt.show()
