import numpy as np
from numba import jit
from scipy.ndimage import gaussian_filter
from scipy import optimize as opt

def getAnnulusMask(image,xc,yc,R,n,m):
    #set labeled_roi_arry to 1 in ROI
    print(np.shape(image))
    ylim, xlim = np.shape(image)
    x = np.arange(0,xlim,1)
    y = np.arange(0,ylim,1)
    X, Y = np.meshgrid(x, y)
    # 1's for greater than n and less than m

    mask1 = np.array(((X-xc))**2 + ((Y-yc))**2 <= (R*m)**2, dtype=int)
    mask2 = np.array(((X-xc))**2 + ((Y-yc))**2 > (R*n)**2, dtype=int)
    
    mask = mask1*mask2
    
    #plt.imshow(labeled_roi_array)
    return np.array(mask, dtype = int)

# Compute spatial correlation function for 2D images
@jit(nopython = True)
def compute_c_2D(im):
    c = np.zeros(im.shape)
    norm = np.sum(im**2)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            for i1 in range(c.shape[0]):
                for j1 in range(c.shape[1]):
                    c[i,j] += im[i1,j1]*im[(i+i1)%c.shape[0],(j+j1)%c.shape[1]]
    c = c/norm
    return c
    
# Compute spatial correlation function for 3D images
@jit(nopython = True)
def compute_c_3D(im):
    c = np.zeros(im.shape)
    norm = np.sum(im**2)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            for k in range(c.shape[2]):
                for i1 in range(c.shape[0]):
                    for j1 in range(c.shape[1]):
                        for k1 in range(c.shape[2]):
                            c[i, j, k] += im[i1, j1, k1]*im[(i + i1)%c.shape[0], (j + j1)%c.shape[1], (k + k1)%c.shape[2]]
    c = c/norm
    return c
    
def gaussian(x, mu, sigma, A, O):
    return A*np.exp(-0.5*((x - mu)/sigma)**2) + O

def stretched_exp(t, tau, beta):
    # ~ beta = 1
    return np.exp(-((t-1)/tau)**beta)
    
def simple_exp(t, tau):
    return np.exp(-((t-1)/tau))
