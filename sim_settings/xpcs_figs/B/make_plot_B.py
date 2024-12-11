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
    large_number = 250
    y = np.empty(len(T))
    for i in range(len(T)):
        # ~ y[i] = A*abs(Tc-T[i])**two_beta*np.heaviside(Tc-T[i],0.5)
        result = integrate.quad(lambda x: A*abs(x-T[i])**two_beta*np.exp(-0.5*((x-Tc)/sigma)**2)/(np.sqrt(2*np.pi)*sigma), T[i], large_number)
        y[i] = result[0]
    return y

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
    
    
def go2():
    pure_data = np.loadtxt('pure_data.csv', delimiter = ',')
    cu8p_data = np.loadtxt('cu8p_data.csv', delimiter = ',')
    
    pure_intensity = pure_data[:,1]/max(pure_data[:,1])
    cu8p_intensity = cu8p_data[:,1]/max(pure_data[:,1])
    
    # ~ pure_lc = 1e3*4*np.pi/(np.sqrt(3)*pure_data[:,2]) # in nm
    # ~ cu8p_lc = 1e3*4*np.pi/(np.sqrt(3)*cu8p_data[:,2])*6 # in nm
    
    pure_lc = pure_data[:,2] # in inverse microns
    cu8p_lc = cu8p_data[:,2] # in inverse microns
    
    
    # ~ pure_lc = pure_data[:,2]
    # ~ cu8p_lc = cu8p_data[:,2]
    
    # plot general settings
    scatter_marker_size = 30
    pure_color = 'green'
    doped_color = 'orange'
    pure_marker = 's'
    doped_marker = 'o'
    
    Tc = 50
    cu8p_tmtc = (cu8p_data[:,0] - Tc)
    # ~ cu8p_tmtc = (cu8p_data[:,0]/Tc)
    Tc = 200
    pure_tmtc = (pure_data[:,0] - Tc)
    # ~ pure_tmtc = (pure_data[:,0]/Tc)
    
    fig, axs = plt.subplots(2)
    
    axs[0].scatter(pure_tmtc,pure_intensity, color = pure_color, ec = 'black', marker = pure_marker, s = scatter_marker_size, label = r'TiSe$_2$')
    axs[0].scatter(cu8p_tmtc,10*cu8p_intensity, color = doped_color, ec = 'black', marker = doped_marker, s = scatter_marker_size, label = r'Cu$_{0.08}$TiSe$_2$')
    axs[0].set_ylabel('Intensity (normalized)')
    # ~ axs[0,0].set_xlabel(r'T - T$_c$ (K)')
    
    axs[1].scatter(pure_tmtc,pure_lc, color = pure_color, ec = 'black', marker = pure_marker, s = scatter_marker_size)
    axs[1].scatter(cu8p_tmtc,cu8p_lc, color = doped_color, ec = 'black', marker = doped_marker, s = scatter_marker_size)
    # ~ axs[1,0].set_ylabel(r'$l_\text{corr}$ (nm)')
    axs[1].set_ylabel(r'$\sigma_x$ ($\mu$m$^{-1}$)')
    
    axs[1].set_xlabel(r'T - T$_c$ (K)')
    # ~ axs[1,0].set_yscale('log')
    
    
    
    axs[0].axvline(x = 0, linestyle = '--', color = 'black', zorder = -10)
    axs[1].axvline(x = 0, linestyle = '--', color = 'black', zorder = -10)
    # ~ axs[0].axvspan(-100,0, zorder = -10, color = 'blue', alpha = 0.25)
    # ~ axs[0].axvspan(0,100, zorder = -10, color = 'red', alpha = 0.25)
    # ~ axs[1].axvspan(-100,0, zorder = -10, color = 'blue', alpha = 0.25)
    # ~ axs[1].axvspan(0,100, zorder = -10, color = 'red', alpha = 0.25)
    
    
    # indicate which points we use for XPCS
    # ~ axs[1,0].scatter([cu8p_tmtc[]],pure_intensity, color = pure_color, ec = 'black', marker = pure_marker, s = scatter_marker_size)
    
    axs[0].set_xticks([])
    axs[0].set_xlim([-42,72])
    axs[1].set_xlim([-42,72])
    axs[0].legend()
    plt.show()


def go1():
    scatter_marker_size = 25
    scatter_alpha = 1
    
    fig, axs = plt.subplots(2, 2)
    
    pure_data = np.loadtxt('pure_data.csv', delimiter = ',', skiprows = 1)
    cu8p_data = np.loadtxt('cu8p_data.csv', delimiter = ',', skiprows = 1)
    
    pure_intensity = pure_data[:,1]/max(pure_data[:,1])
    cu8p_intensity = (cu8p_data[:,1] - cu8p_data[:,3])
    cu8p_intensity = cu8p_intensity/max(cu8p_intensity)
    
    # ~ pure_lc = 1e3*4*np.pi/(np.sqrt(3)*pure_data[:,2]) # in nm
    # ~ cu8p_lc = 1e3*4*np.pi/(np.sqrt(3)*cu8p_data[:,2]) # in nm
    
    pure_lc = pure_data[:,2] # in inverse microns
    cu8p_lc = cu8p_data[:,2] # in inverse microns

    cu8p_tmtc = cu8p_data[:,0]
    pure_tmtc = pure_data[:,0]
    
    # Fit the pure sample intensity vs temp
    p0 = [201, 0.5]
    popt_pure_intensity, pcov = curve_fit(f1, pure_tmtc, pure_intensity, p0 = p0)
    print('popt_pure_intensity',popt_pure_intensity)
    # ~ print(popt)
    tt_pure = np.linspace(min(pure_tmtc), max(pure_tmtc), 1000)
    intensity_fit_pure = f1(tt_pure, *popt_pure_intensity)
    
    # Fit the pure sample peak width vs temp
    p0 = [0.6, 0.63]
    popt_pure_peakwidth, pcov = curve_fit(f2, pure_tmtc, pure_lc, p0 = p0)
    print('popt_pure_peakwidth',popt_pure_peakwidth)
    tt_pure = np.linspace(min(pure_tmtc), max(pure_tmtc), 1000)
    peakwidth_fit_pure = f2(tt_pure, *popt_pure_peakwidth)
    
    # Fit the 8% sample intensity vs temp
    p0 = [73.5767000, 19.1337667, 6.38810946e-02]
    popt_cu8p_intensity, pcov = curve_fit(f3, cu8p_tmtc, cu8p_intensity, p0 = p0)
    print('popt_cu8p_intensity',popt_cu8p_intensity)
    tt_cu8p = np.linspace(min(cu8p_tmtc), max(cu8p_tmtc), 1000)
    intensity_fit_cu8p = f3(tt_cu8p, *popt_cu8p_intensity)
    tc_dist = np.exp(-0.5*((tt_cu8p-popt_cu8p_intensity[0])/popt_cu8p_intensity[1])**2)
    
    
    # Fit the 8% sample peak width vs temp
    # ~ p0 = [73.6, 19, 5, 55]
    p0 = [73.5767000, 50, 5, 55]
    # ~ popt, pcov = curve_fit(f4, cu8p_tmtc, cu8p_lc, p0 = p0)
    # ~ print(popt)
    tt_cu8p = np.linspace(min(cu8p_tmtc), max(cu8p_tmtc), 1000)
    peakwidth_fit_cu8p = f4(tt_cu8p, *p0)
    
    
    # ~ axs[0,0].axvline(x = 202, color = 'black', linestyle = '--')
    # ~ axs[1,0].axvline(x = 202, color = 'black', linestyle = '--')
    
    # ~ axs[0,1].axvline(x = 95, color = 'black', linestyle = '--')
    # ~ axs[1,1].axvline(x = 95, color = 'black', linestyle = '--')
    
    color = 'blue'
    linewidth = 3
    linewidth_fit = 3
    alpha_fit = 0.4
    alpha = 0.5
    axs[0,1].axvline(x = 75, color = color, linestyle = '-', alpha = alpha, linewidth = linewidth, zorder = -5)
    axs[1,1].axvline(x = 75, color = color, linestyle = '-', alpha = alpha, linewidth = linewidth, zorder = -5)
    axs[0,1].axvline(x = 95, color = color, linestyle = '-', alpha = alpha, linewidth = linewidth, zorder = -5)
    axs[1,1].axvline(x = 95, color = color, linestyle = '-', alpha = alpha, linewidth = linewidth, zorder = -5)
    axs[0,1].axvline(x = 100, color = color, linestyle = '-', alpha = alpha, linewidth = linewidth, zorder = -5)
    axs[1,1].axvline(x = 100, color = color, linestyle = '-', alpha = alpha, linewidth = linewidth, zorder = -5)
    
    axs[0,0].axvline(x = 202, color = color, linestyle = '-', alpha = alpha, linewidth = linewidth, zorder = -5)
    axs[1,0].axvline(x = 202, color = color, linestyle = '-', alpha = alpha, linewidth = linewidth, zorder = -5)
    axs[0,0].axvline(x = 206, color = color, linestyle = '-', alpha = alpha, linewidth = linewidth, zorder = -5)
    axs[1,0].axvline(x = 206, color = color, linestyle = '-', alpha = alpha, linewidth = linewidth, zorder = -5)
    
    axs[0,0].plot(tt_pure, intensity_fit_pure, color = 'green', linewidth = linewidth_fit, alpha = alpha_fit, zorder = -10)
    axs[0,0].scatter(pure_tmtc, pure_intensity, facecolors='none', ec = 'black', marker = 'o', s = scatter_marker_size, alpha = scatter_alpha)
    
    axs[1,0].plot(tt_pure, peakwidth_fit_pure, color = 'green', linewidth = linewidth_fit, alpha = alpha_fit, zorder = -10)
    axs[1,0].scatter(pure_tmtc, pure_lc, facecolors = 'none', ec = 'black', marker = 'o', s = scatter_marker_size, alpha = scatter_alpha)
    
    axs[0,1].plot(tt_cu8p, intensity_fit_cu8p, color = 'green', linewidth = linewidth_fit, alpha = alpha_fit, zorder = -10)
    axs[0,1].plot(tt_cu8p, tc_dist, color = 'black', linewidth = 1, linestyle = '--', alpha = 1, zorder = -10)
    axs[0,1].scatter(cu8p_tmtc, cu8p_intensity, facecolors = 'none', ec = 'black', marker = 'o', s = scatter_marker_size, alpha = scatter_alpha)
    
    # ~ axs[1,1].plot(tt_cu8p, peakwidth_fit_cu8p, color = 'black', zorder = -10)
    axs[1,1].scatter(cu8p_tmtc, cu8p_lc, color = 'white', ec = 'black', marker = 'o', s = scatter_marker_size, alpha = scatter_alpha)
    
    axs[0,0].set_xticks([])
    axs[0,1].set_xticks([])
    
    axs[1,0].set_xlabel('Temperature (K)')
    axs[1,1].set_xlabel('Temperature (K)')
    axs[1,0].set_ylabel(r'$\sigma_q$ ($\mu$m$^{-1}$)')
    axs[0,0].set_ylabel(r'Intensity (normalized)')
    
    axs[0,0].set_title(r'Pure TiSe$_2$')
    axs[0,1].set_title(r'Cu$_{0.08}$TiSe$_2$')
    plt.show()
    


def go3():
    scatter_marker_size = 50
    scatter_alpha = 1
    
    fig, axs = plt.subplots(2, 2, sharex = 'col')
    fig.set_size_inches(3.4, 5)
    
    pure_data = np.loadtxt('pure_data.csv', delimiter = ',', skiprows = 1)
    cu8p_data = np.loadtxt('cu8p_data.csv', delimiter = ',', skiprows = 1)
    
    pure_intensity = pure_data[:,1]/max(pure_data[:,1])
    cu8p_intensity = (cu8p_data[:,1] - cu8p_data[:,3])
    cu8p_intensity = cu8p_intensity/max(cu8p_intensity)
    
    # ~ pure_lc = 1e3*4*np.pi/(np.sqrt(3)*pure_data[:,2]) # in nm
    # ~ cu8p_lc = 1e3*4*np.pi/(np.sqrt(3)*cu8p_data[:,2]) # in nm
    L_det = 1.5 # in meter
    pix_size = 75e-6 # in meter
    λ = 0.96847 # in angstrom
    pure_lc = 2.355*2*np.pi/λ*(pix_size/L_det)*pure_data[:,2] # in hcut pixels
    cu8p_lc = 2.355*2*np.pi/λ*(pix_size/L_det)*cu8p_data[:,2] # in hcut pixels

    cu8p_tmtc = cu8p_data[:,0]
    pure_tmtc = pure_data[:,0]
    
    # Fit the pure sample intensity vs temp
    p0 = [201, 0.5]
    popt_pure_intensity, pcov = curve_fit(f1, pure_tmtc, pure_intensity, p0 = p0)
    print('popt_pure_intensity',popt_pure_intensity)
    # ~ print(popt)
    tt_pure = np.linspace(min(pure_tmtc), max(pure_tmtc), 1000)
    intensity_fit_pure = f1(tt_pure, *popt_pure_intensity)
    
    # Fit the pure sample peak width vs temp
    p0 = [0.6, 0.63]
    popt_pure_peakwidth, pcov = curve_fit(f2, pure_tmtc, pure_lc, p0 = p0)
    print('popt_pure_peakwidth',popt_pure_peakwidth)
    tt_pure = np.linspace(min(pure_tmtc), max(pure_tmtc), 1000)
    peakwidth_fit_pure = f2(tt_pure, *popt_pure_peakwidth)
    
    # Fit the 8% sample peak width vs temp
    p0 = [73.5767000, 19.1, 30, 55]
    popt, pcov = curve_fit(f8, cu8p_tmtc, cu8p_lc, p0 = p0)
    print(popt)
    tt_cu8p = np.linspace(min(cu8p_tmtc), max(cu8p_tmtc), 1000)
    peakwidth_fit_cu8p = f8(tt_cu8p, *popt)
    
    # Fit the 8% sample intensity vs temp
    p0 = [73.5767000, 19.1337667, 6.38810946e-02]
    popt_cu8p_intensity, pcov = curve_fit(f3, cu8p_tmtc, cu8p_intensity, p0 = p0)
    print('popt_cu8p_intensity',popt_cu8p_intensity)
    tt_cu8p = np.linspace(min(cu8p_tmtc), max(cu8p_tmtc), 1000)
    intensity_fit_cu8p = f3(tt_cu8p, *popt_cu8p_intensity)
    tc_dist = np.exp(-0.5*((tt_cu8p-popt_cu8p_intensity[0])/popt_cu8p_intensity[1])**2)
    

    
    # ~ axs[0,0].axvline(x = 202, color = 'black', linestyle = '--')
    # ~ axs[1,0].axvline(x = 202, color = 'black', linestyle = '--')
    
    # ~ axs[0,1].axvline(x = 95, color = 'black', linestyle = '--')
    # ~ axs[1,1].axvline(x = 95, color = 'black', linestyle = '--')
    
    # ~ color_axvline = '#2b00ff'
    color_axvline = 'black'
    linewidth_axvline = 1
    linewidth_fit = 5
    alpha_fit = 0.4
    alpha = 0.4
    # ~ axs[0,1].axvline(x = 75, color = color_axvline, linestyle = '-', alpha = alpha, linewidth = linewidth_axvline, zorder = -15)
    # ~ axs[1,1].axvline(x = 75, color = color_axvline, linestyle = '-', alpha = alpha, linewidth = linewidth_axvline, zorder = -15)
    # ~ axs[0,1].axvline(x = 95, color = color_axvline, linestyle = '-', alpha = alpha, linewidth = linewidth_axvline, zorder = -15)
    # ~ axs[1,1].axvline(x = 95, color = color_axvline, linestyle = '-', alpha = alpha, linewidth = linewidth_axvline, zorder = -15)
    # ~ axs[0,1].axvline(x = 100, color = color_axvline, linestyle = '-', alpha = alpha, linewidth = linewidth_axvline, zorder = -15)
    # ~ axs[1,1].axvline(x = 100, color = color_axvline, linestyle = '-', alpha = alpha, linewidth = linewidth_axvline, zorder = -15)
    
    # ~ axs[0,0].axvline(x = 200, color = color_axvline, linestyle = '-', alpha = alpha, linewidth = linewidth_axvline, zorder = -15)
    # ~ axs[1,0].axvline(x = 200, color = color_axvline, linestyle = '-', alpha = alpha, linewidth = linewidth_axvline, zorder = -15)
    # ~ axs[0,0].axvline(x = 204, color = color_axvline, linestyle = '-', alpha = alpha, linewidth = linewidth_axvline, zorder = -15)
    # ~ axs[1,0].axvline(x = 204, color = color_axvline, linestyle = '-', alpha = alpha, linewidth = linewidth_axvline, zorder = -15)
    
    # ~ axs[1,0].axvspan(198,200, alpha = 0.3, color = 'red')
    # ~ axs[0,1].plot([75,75],[0,1])
    
    
    special_color = 'black' #'#0f8281' #'#42b9bd' #'#68edec'
    special_color_2 = '#7a7b7a'
    
    special_indices = [1,3]
    index_number = len(pure_tmtc)
    normal_indices = []
    for i in range(index_number):
        if not i in special_indices:
            normal_indices.append(i)
    
    # Normal stuff
    axs[0,0].plot(tt_pure, intensity_fit_pure, color = 'green', linewidth = linewidth_fit, alpha = alpha_fit, zorder = -5)
    axs[0,0].scatter(pure_tmtc[normal_indices], pure_intensity[normal_indices], facecolors='none', ec = 'black', marker = 'o', s = scatter_marker_size, alpha = scatter_alpha)
    
    axs[1,0].plot(tt_pure, peakwidth_fit_pure, color = 'green', linewidth = linewidth_fit, alpha = alpha_fit, zorder = -5)
    axs[1,0].scatter(pure_tmtc[normal_indices], pure_lc[normal_indices], facecolors = 'none', ec = 'black', marker = 's', s = scatter_marker_size, alpha = scatter_alpha)
    
    # Special stuff
    
    axs[0,0].scatter(pure_tmtc[special_indices], pure_intensity[special_indices], facecolors=special_color, ec = special_color_2, marker = 'o', s = scatter_marker_size, alpha = scatter_alpha)
    axs[1,0].scatter(pure_tmtc[special_indices], pure_lc[special_indices], facecolors = special_color, ec = special_color_2, marker = 's', s = scatter_marker_size, alpha = scatter_alpha)
    
    special_indices = [9,13,14]
    index_number = len(cu8p_tmtc)
    normal_indices = []
    for i in range(index_number):
        if not i in special_indices:
            normal_indices.append(i)
    
    # Normal stuff
    axs[0,1].plot(tt_cu8p, intensity_fit_cu8p, color = 'blue', linewidth = linewidth_fit, alpha = alpha_fit, zorder = -5)
    axs[0,1].plot(tt_cu8p, tc_dist, color = 'black', linewidth = 1, linestyle = '--', alpha = 1, zorder = -5)
    axs[0,1].scatter(cu8p_tmtc[normal_indices], cu8p_intensity[normal_indices], facecolors = 'none', ec = 'black', marker = 'o', s = scatter_marker_size, alpha = scatter_alpha)
    
    axs[1,1].plot(tt_cu8p, peakwidth_fit_cu8p, color = 'blue', linewidth = linewidth_fit, alpha = alpha_fit, zorder = -5)
    axs[1,1].scatter(cu8p_tmtc[normal_indices], cu8p_lc[normal_indices], facecolors = 'none', ec = 'black', marker = 's', s = scatter_marker_size, alpha = scatter_alpha)
    
    # Special Stuff
    axs[1,1].scatter(cu8p_tmtc[special_indices], cu8p_lc[special_indices], facecolors = special_color, ec = special_color_2, marker = 's', s = scatter_marker_size, alpha = scatter_alpha)
    axs[0,1].scatter(cu8p_tmtc[special_indices], cu8p_intensity[special_indices], facecolors = special_color, ec = special_color_2, marker = 'o', s = scatter_marker_size, alpha = scatter_alpha)
    
    
    #axs[0,0].set_xticks([])
    #axs[0,1].set_xticks([])
    
    axs[1,0].set_xlabel('Temperature (K)')
    axs[1,1].set_xlabel('Temperature (K)')
    axs[1,0].set_ylabel(r'$\sigma_q$ ($\AA^{-1}$)')
    axs[0,0].set_ylabel(r'Intensity (normalized)')
    
    axs[0,0].set_title(r'Pure TiSe$_2$')
    axs[0,1].set_title(r'Cu$_{0.08}$TiSe$_2$')
    
    fig.subplots_adjust(left = 0.16, bottom = 0.11, right = 0.964, top = 0.938, wspace = 0.355, hspace = 0.2)
    plt.show()
    
    
# ~ go1()
go3()
