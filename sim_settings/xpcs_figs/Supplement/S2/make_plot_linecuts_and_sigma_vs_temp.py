import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from matplotlib import cm
import tifffile
import os

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

def compute_moments(im, offset = 0):
    im = im - offset
    im[im<0] = 1e-3
    
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



def go1():
    scatter_marker_size = 40
    scatter_alpha = 1
    
    root_dir = r'C:\Users\thoma\OneDrive - UCLA IT Services\Desktop\OneDrive - UCLA IT Services\Research\TiSe2_XPCS\exhibits\Supplement\S2\simulated_diffraction_images'
    file_list = os.listdir(root_dir)
    # ~ file_list = file_list[::3]
    # ~ print(file_list)
    temps = [] # empty list that will hold the temperatures of the data
    intensities = [] # empty list that will hold the peak intensities of the data
    sigmas = [] # empty list that will hold the peak widths of the data
    
    cmap = cm.plasma
    
    fig, axs = plt.subplots(1, 2)
    
    index = 0
    for i in range(len(file_list)):
        if file_list[i].find('.tif') == -1:
            continue
            
        # ~ if i > 5:
            # ~ break
        
        temps.append(float(file_list[i][:file_list[i].find('.tif')]))
        im = tifffile.imread(f'{root_dir}//{file_list[i]}')
        im_crop = im[35:45,35:45]
        im_bg = im[0:10,0:10]
        
        
        intensities.append(np.mean(im_crop))
        
        I0, I1, I2 = compute_moments(im_crop)
        var_1, var_2 = diag_rank2(I2)
        sigma_eff = (var_1*var_2)**(1/4)
        sigmas.append(sigma_eff)
        x = np.linspace(-50,50,im.shape[0])
        
        if temps[i] == 2 or temps[i] == 2.5 or temps[i] == 3 or temps[i] == 3.5 or temps[i] == 4:
            axs[0].plot(x,im[im.shape[0]//2,:], label = temps[i], color = cmap(index/4))
            index += 1
        
    axs[0].set_xlabel('$q$ (a.u.)')
    axs[0].set_ylabel('Linecut intensity (a.u.)')
    axs[0].set_yscale('log')
    axs[0].legend()
    
    axs[1].scatter(temps, sigmas, marker = 'o', facecolor = 'none', edgecolor = 'k')
    axs[1].set_xlabel('Temperature (a.u.)')
    axs[1].set_ylabel('Peak Width (a.u.)')
    
    data_to_save = np.array([temps, sigmas]).transpose()
    np.savetxt('peakwidth_vs_temp.csv',data_to_save, delimiter = ',')
    
    
    plt.show()

    # ~ plt.scatter(temps, intensities)
    # ~ plt.xlabel('Temp')
    # ~ plt.ylabel('Intensity')
    # ~ plt.show()
    # ~ images = tifffile.imread(r"C:\Users\thoma\OneDrive - UCLA IT Services\Desktop\OneDrive - UCLA IT Services\Research\TiSe2_XPCS\exhibits\Supplement\S2\diffraction_images.tiff")
    # ~ plt.imshow(images[0])
    # ~ plt.show()
    
    
def go2():
    # temp vs. tau for a single core of size 7 on a 30 x 30 lattice with mass strength of 0.4
    temps = [2.4,2.5,2.6,2.7,2.8,2.9]
    # taus from fit with beta = 1
    tau = [1.83E+04,1.03E+04,4.04E+03,2.16E+03,1.00E+03,553.575155]
    
    plt.scatter(temps, tau)
    plt.show()
    
    # Q vs. tau plot at a temp of 2.4
    R = np.array([7,6.75,6.5,6.25,6,5.5,5,4.5,4,3.5,3.25,3])
    Q = 1/R
    tau = [2.43E+04,2.05E+04,1.66E+04,1.24E+04,9.14E+03,6.38E+03,4.28E+03,2.53E+03,1.31E+03,899.7685104,678.5192032,646.4915234]
    
    plt.scatter(Q, tau)
    plt.show()

    
go1()
# ~ go2()
