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
        
def plot_regions(ax, colors_for_qs, delta = 1*3.77):
    # ~ ax.axvline(x = 0, color = 'orange', linestyle = '--')
    ax.axvspan(-delta,delta, color = colors_for_qs[0])
    
    ax.axvspan(delta, 2*delta ,color = colors_for_qs[1])
    ax.axvspan(-2*delta, -delta ,color = colors_for_qs[1])
    
    ax.axvspan(2*delta, 3*delta ,color = colors_for_qs[2])
    ax.axvspan(-3*delta, -2*delta ,color = colors_for_qs[2])
    
    ax.axvspan(3*delta, 4*delta ,color = colors_for_qs[3])
    ax.axvspan(-4*delta, -3*delta ,color = colors_for_qs[3])
    
    ax.axvspan(4*delta, 5*delta ,color = colors_for_qs[4])
    ax.axvspan(-5*delta, -4*delta ,color = colors_for_qs[4])

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
cmap = cm.jet

# Load Data
#lagsteps = np.loadtxt('lag_steps.txt', delimiter = ',')[1:]
#data_202 = np.loadtxt('202K_g2.txt', delimiter = ',')
#waterfall_202 = tifffile.imread('202K_waterfall.tiff')
def go1():
    image_70 = tifffile.imread('data//70K_image.tiff')
    image_75 = tifffile.imread('data//75K_image.tiff')
    image_95 = tifffile.imread('data//95K_image.tiff')
    image_100 = tifffile.imread('data//100K_image.tiff')
    
    # Pure Samples
    image_202 = tifffile.imread('data//pure//202K_image.tiff')
    image_206 = tifffile.imread('data//pure//206K_image.tiff')
    
    fig, axs = plt.subplots(1,2)
    
    fig.suptitle('206 K')
    image = image_206
    
    axs[0].imshow(image)
    xc = 85
    for i in range(5):
        I = 4*(2-i)
        axs[0].axvline(x = xc + I, c = cmap(i/5))
        axs[1].plot(image[:, xc + I], c = cmap(i/5))
    
    
    plt.show()
    
def fit_function(x, tau, beta):
    return np.exp(-1*(x/tau)**beta)
    
def fit_data(x, y):
    xt = x[-8:]
    yt = y[-8:]
    
    coefs = np.polyfit(np.log(xt), np.log(-1*np.log(yt)), deg = 1)
    print(coefs)
    
    xx = np.linspace(min(np.log(xt)), max(np.log(xt)), 100)
    yy = coefs[0]*xx + coefs[1]
    
    plt.plot(xx, yy)
    plt.scatter(np.log(xt), np.log(-1*np.log(yt)))
    plt.show()


def go2():

    lagsteps = np.loadtxt('data//lag_steps.txt', delimiter = ',')[1:]
    data = np.loadtxt('data//100K_F.txt', delimiter = ',')
    
    end_remove = 5
    data = data[:,:-end_remove] # Remove the last "end_remove" points
    lagsteps = lagsteps[:-end_remove]
    
    fig, axs = plt.subplots()
    cmap = cm.jet
    
    
    for i in range(3):
        F = data[i,:]
        # ~ fit_data(lagsteps, F)
        p0 = [10000, 1]
        try:
            popt, pcov = curve_fit(fit_function, lagsteps, F, p0 = [500, 1])
        except:
            popt = p0
            
        tt = np.linspace(min(lagsteps), max(lagsteps), 1000)
        yy_guess = fit_function(tt, *p0)
        yy_fit = fit_function(tt, *popt)
        print(popt)
        
    
        axs.scatter(lagsteps, F, color = cmap(i/3))
        # ~ ax.plot(tt, yy_guess, 'k--', color = cmap(i/3))
        axs.plot(tt, yy_fit, 'g-', label = r'$\tau$ = ' + f'{round(popt[0],1)} s' + '\n' + r'$\beta$ = ' + f'{round(popt[1],3)}', color = cmap(i/3))

        
    

    axs.set_ylabel('F')
    axs.set_xlabel(r'$\Delta t$')
    axs.legend()
    axs.set_xscale('log')
    
    plt.show()




go2()








