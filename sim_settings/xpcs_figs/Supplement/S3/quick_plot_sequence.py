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



images = tifffile.imread(r"diffraction_images.tiff")
rs_1 = tifffile.imread(r"1e5.tiff")
rs_2 = tifffile.imread(r"2e5.tiff")
rs_3 = tifffile.imread(r"3e5.tiff")

fig, axs = plt.subplots(2,3)
axs[0,0].imshow(images[0])
axs[0,1].imshow(images[1])
axs[0,2].imshow(images[2])

axs[1,0].imshow(rs_1, cmap = 'bwr', vmax = 255, vmin = 0)
axs[1,1].imshow(rs_2, cmap = 'bwr', vmax = 255, vmin = 0)
axs[1,2].imshow(rs_3, cmap = 'bwr', vmax = 255, vmin = 0)

axs[0,0].set_title('Step 0')
axs[0,1].set_title(r'Step $1 \times 10^7$')
axs[0,2].set_title(r'Step $2 \times 10^7$')

for ax in axs:
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

plt.show()
