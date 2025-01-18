import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
import tifffile

from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import scipy.integrate as integrate
import scipy as sci

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

def gaussian(x, mu, sigma, A):
    return A*np.exp(-0.5*((x-mu)/sigma)**2)
    
def lorentzian(x, mu, FWHM, A):
    return A/(1 + ((x-mu)/(0.5*FWHM))**2)

def f1(T,Tc,A):
    # Fit function for the pristine sample intensity
    two_beta = 0.68 # The two beta value measured here https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.110.196404
    return A*abs(Tc-T)**two_beta*np.heaviside(Tc-T,0.5)
    
def f2(T,nu,A):
    # Fit function for the pristine sample peak width
    Tc = 200.31675431
    return A*abs(T-Tc)**nu*np.heaviside(T-Tc, 0.5)
    
def f4(T,Tc,sigma,A):
    beta = 0.34
    large_number = 150
    y = np.empty(len(T))
    for i in range(len(T)):
        # ~ y[i] = A*abs(Tc-T[i])**two_beta*np.heaviside(Tc-T[i],0.5)
        result = integrate.quad(lambda x: A*abs(x-T[i])**beta*np.exp(-0.5*((x-Tc)/sigma)**2)/(np.sqrt(2*np.pi)*sigma), T[i], large_number)
        y[i] = result[0]
    return y**2
    
def f3(T,Tc,sigma,A):
    two_beta = 0.68
    if two_beta < 0.1:
        two_beta = 0.1
    large_number = 50
    y = np.empty(len(T))
    for i in range(len(T)):
        # ~ y[i] = A*abs(Tc-T[i])**two_beta*np.heaviside(Tc-T[i],0.5)
        result = integrate.quad(lambda x: A*abs(x-T[i])**two_beta*np.exp(-0.5*((x-Tc)/sigma)**2)/(np.sqrt(2*np.pi)*sigma), T[i], large_number)
        y[i] = result[0]
    return y + 0.06

def f5(T,Tc,sigma,A,O):
    nu = 0.63
    y = np.empty(len(T))
    for i in range(len(T)):
        # ~ y[i] = A*abs(Tc-T[i])**two_beta*np.heaviside(Tc-T[i],0.5)
        result = integrate.quad(lambda x: A*abs(T[i]-x)**nu*np.exp(-0.5*((x-Tc)/sigma)**2)/(np.sqrt(2*np.pi)*sigma), 0, T[i])
        y[i] = result[0] + O
    return y

def f6(T,Tc,A,O):
    nu = 0.63
    y = np.empty(len(T))
    for i in range(len(T)):
        # ~ y[i] = A*abs(Tc-T[i])**two_beta*np.heaviside(Tc-T[i],0.5)
        result = integrate.quad(lambda x: A*((T[i]-x)**2 + (O*A)**(2/nu))**(-nu/2)*np.exp(-0.5*((x-Tc)/sigma)**2)/(np.sqrt(2*np.pi)*sigma), 0, T[i])
        y[i] = result[0]
    
    y += O**(-1)*0.5*(1-sci.special.erf((T-Tc)/(np.sqrt(2)*sigma)))
    return 1/y

def f7(T,Tc,A,O):
    large_number = 150    
    nu = 0.63
    sigma = 19.1
    y = np.empty(len(T))
    for i in range(len(T)):
        # ~ y[i] = A*abs(Tc-T[i])**two_beta*np.heaviside(Tc-T[i],0.5)
        result = integrate.quad(lambda x: (A*((T[i]-x)**2 + (O/A)**(2/nu))**(nu/2)*np.heaviside(T[i]-x, 0) + O*np.heaviside(x-T[i], 0))*np.exp(-0.5*((x-Tc)/sigma)**2)/(np.sqrt(2*np.pi)*sigma), 0, large_number)
        y[i] = result[0]
    return y

def f8(T,Tc,sigma,A,O):
    Tc = 7.35785415e+01
    sigma = 1.91317683e+01
    O = 0.0069
    return 0.5*A*(1 + sci.special.erf((T-Tc)/(np.sqrt(2)*sigma))) + O
    

def go2(fit_func = lorentzian, fits = True, verbose = False):
    
    pixel_sums = []
    amplitudes = []
    FWHMs = []
    temps = [4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5]

    for j in range(len(temps)):
        im = tifffile.imread(f'{temps[j]}.tiff')
        for i in range(3):
            im = np.roll(im, im.shape[i]//2, axis = i)
            
        L = 3
        im_2d = np.mean(im[:, :, im.shape[2]//2 - L//2 : im.shape[2]//2 + L//2], axis = 2)
        im_1d = np.mean(im_2d[:, im.shape[1]//2 - L//2 : im.shape[1]//2 + L//2], axis = 1)
        im_1d_x = range(len(im_1d))
        
        if fits:
            # Fit to function
            p0 = [im.shape[0]//2, 1, 2.2e6]
            popt, pcov = curve_fit(fit_func, im_1d_x, im_1d, p0 = p0)
    
            FWHMs.append(popt[1])
            amplitudes.append(popt[2])
            
        pixel_sums.append(np.mean(im[im.shape[0]//2 - L//2 : im.shape[0]//2 + L//2, im.shape[1]//2 - L//2 : im.shape[1]//2 + L//2, im.shape[2]//2 - L//2 : im.shape[2]//2 + L//2]))
        
        
        if verbose and fits:
            # interpolate for plotting fit
            tt = np.linspace(min(im_1d_x), max(im_1d_x), 1000)
            yy = fit_func(tt, *popt)
            
            plt.scatter(im_1d_x, im_1d)
            plt.plot(tt, yy)
            plt.show()
    
    if fits:
        amplitudes = np.array(amplitudes)
        amplitudes /= max(amplitudes)
    
    pixel_sums = np.array(pixel_sums)
    pixel_sums /= max(pixel_sums)
            
            
    Y = amplitudes
    
    plt.scatter(temps, Y)
    # ~ plt.plot(temps, FWHMs)
    
    # Fit the 8% sample intensity vs temp
    p0 = [5.0, 0.2, 2]
    popt, pcov = curve_fit(f3, temps, Y, p0 = p0)
    # ~ popt = p0
    
    print('popt',popt)
    tt = np.linspace(min(temps), max(temps), 1000)
    intensity_fit = f3(tt, *popt)
    tc_dist = np.exp(-0.5*((tt-popt[0])/popt[1])**2)
    
    plt.plot(tt, intensity_fit)
    
    plt.show()
        
    

def go3():
    scatter_marker_size = 50
    scatter_alpha = 1

    data = np.loadtxt('intensities.csv', delimiter = ',', skiprows = 1)
    intensity = data[:, 1]/max(data[:, 1])
    
    temps = [4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6]
    
    # ~ temps = temps[:5]
    # ~ intensity = intensity[:5]
    
    print(len(intensity), len(temps))
  
    # Fit the 8% sample intensity vs temp
    p0 = [5, 0.1, 2]
    popt, pcov = curve_fit(f3, temps, intensity, p0 = p0)
    # ~ popt = p0
    
    print('popt',popt)
    tt = np.linspace(min(temps), max(temps), 1000)
    intensity_fit = f3(tt, *popt)
    tc_dist = np.exp(-0.5*((tt-popt[0])/popt[1])**2)
    
    plt.scatter(temps, intensity)
    plt.plot(tt, intensity_fit)
 
    # ~ fig.subplots_adjust(left = 0.16, bottom = 0.11, right = 0.964, top = 0.938, wspace = 0.355, hspace = 0.2)
    plt.show()
    
    
# ~ go1()
go2()
# ~ go3()
