import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
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
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

offset = 0
step = 0.22
cmap = cm.viridis_r
colors_for_qs = [cmap(offset),cmap(step+offset),cmap(2*step+offset),cmap(3*step+offset),cmap(4*step+offset)]
markers_for_qs = ['s','o','^', 'D', 'P']

# ~ data = np.loadtxt('data_beta.csv', delimiter = ',', skiprows = 1)
data = np.loadtxt('sim_tau_q.csv', delimiter = ',', skiprows = 1)
R = data[:, 0]
L = 50
tau = L**2*data[:, 3]

data = np.loadtxt('peakwidth_vs_temp.csv', delimiter = ',', skiprows = 0)
temps = data[:, 0]
peakwidths = data[:, 1]


fig, axs = plt.subplots(1,2)

axs[0].scatter(temps, peakwidths, marker = 'o', facecolor = 'none', edgecolor = 'k')
axs[0].set_ylabel(r'$\sigma$ (cells)')
axs[0].set_xlabel(r'Temperature(a.u.)')

axs[1].scatter(R, 1e-6*tau, marker = 'o', facecolor = 'none', edgecolor = 'k')
axs[1].set_ylabel(r'$\langle T \rangle$ ($10^6$ steps)')
axs[1].set_xlabel(r'$R$ (cells)')
# ~ axs[1].set_yscale('log')
# ~ axs[1].set_yticks([1e-7,2e-7])
plt.show()
# ~ indices = np.argsort(temps)
# ~ temps = temps[indices]
# ~ data = data[indices, :]
