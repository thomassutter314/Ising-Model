import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
import tifffile

from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy import optimize as opt

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
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=14)


def go1():
    # Citation for data https://www.nature.com/articles/nphys360
    
    data = np.loadtxt(r'extracted_data_morrison_cdw.txt')
    res_x_cdw = data[:7, 0]*100
    res_T_cdw = data[:7, 1]
    
    data = np.loadtxt(r'extracted_data_morrison_sc.txt')
    res_x_sc = data[:, 0]*100
    res_T_sc = data[:, 1]
    
    data = np.loadtxt(r'extracted_data_anshul.txt')
    xrd_x_cdw = data[:3,0]*100
    xrd_T_cdw = data[:3, 1]
    xrd_x_icdw = data[3:,0]*100
    xrd_T_icdw = data[3:, 1]
    
    plt.scatter(res_x_cdw, res_T_cdw, facecolor = '#1de35d', edgecolor = 'black', label = 'Resistivity (Morosan et al. 2006)')
    plt.scatter(res_x_sc, 5*res_T_sc, facecolor = '#fa5dfe', edgecolor = 'black')
    
    plt.scatter(xrd_x_cdw, xrd_T_cdw, facecolor = '#1de35d', edgecolor = 'black', marker = 's', label = 'XRD (Kogar et al. 2017)')
    plt.scatter(xrd_x_icdw, xrd_T_icdw, facecolor = '#3f67fd', edgecolor = 'black', marker = 's')
    
    plt.scatter([0],[200.3], facecolor = '#1de35d', edgecolor = 'black', marker = '^', label = 'XRD (Current Study)')
    plt.scatter([8],[73.6], facecolor = '#3f67fd', edgecolor = 'black', marker = '^')
    
    # Make dome for cdw
    # ~ x = np.array([0,0.0193,0.05, 0.0631, 0.0802, 0.0944])*100
    # ~ y = [201, 165, 88.9, 75.6, 66.7, 58.2]
    x = np.array([0,2,3,4,5,6,7,8,9,10,11])
    y = np.array([215.8,187.4,158.2,136.2,126,93.0,73.5,72.7,64.2,60.8,60.8])
    
    xx = np.linspace(-1,11,100)
    deg = 5
    popt = np.polyfit(x, y, deg = deg)
    yy = np.zeros(len(xx))
    yy_floor = np.zeros(len(xx))
    for i in range(deg + 1):
        yy += popt[i]*xx**(deg-i)
        
    plt.plot(xx, yy, 'k--', zorder = 0)
    # ~ plt.fill_between(xx, yy, 0, zorder = -2, alpha = 0.5, color = 'green')
    
    # Make a custom colormap
    colors = ["#87a0fe", "#7defa2"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
    
    
    polygon = plt.fill_between(xx, yy, yy_floor, lw=0, color='none', zorder = -10)
    xlim = plt.xlim()
    ylim = plt.ylim()
    verts = np.vstack([p.vertices for p in polygon.get_paths()])
    # ~ gradient = plt.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap='turbo', aspect='auto',
                          # ~ extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    tt = np.linspace(0, 1, 256)
    color_function = 1/(1+np.exp(40*(tt - 0.45)))
    gradient = plt.imshow(color_function.reshape(1, -1), cmap=cmap1, aspect='auto',
                          extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()], zorder = -10, alpha = 0.7)
                          
    gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)
    plt.xlim(xlim)
    plt.ylim(ylim)
    # ~ plt.fill_between(xx, 216, yy, color = '#ffe6e4', zorder = -1)
    
    # Make dome for sc
    x = np.array([0.043,0.06,0.08,0.1])*100
    y = [0, 2.41, 4.18,2.73]
    xx = np.linspace(4.3,11,100)
    popt = np.polyfit(x, y, deg = 2)
    yy = popt[0]*xx**2 + popt[1]*xx + popt[2]
    plt.plot(xx, 5*yy, 'k--', zorder = -1)
    plt.fill_between(xx, 5*yy, 0, zorder = -2, alpha = 0.7, color = '#fca4fe')
    
    
    plt.text(x = 0.1, y = 80, s = 'Commensurate CDW')
    plt.text(x = 7, y = 40, s = 'Disordered CDW')
    plt.text(x = 7.5, y = 3, s = r'SC ($\times$ 5)')
    plt.text(x = 7.5, y = 130, s = r'Metallic')
    
    plt.ylabel('Temperature (K)')
    plt.xlabel('Cu Intercalation (%)')
    plt.legend()
    
    # ~ fig = plt.gcf()
    # ~ fig.set_size_inches(3.4, 5)
    
    plt.show()
    
    
go1()

