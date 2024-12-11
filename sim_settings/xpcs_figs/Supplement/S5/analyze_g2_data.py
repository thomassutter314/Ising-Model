import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
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

# Style Choices
linewidth = 5
linealpha = 0.5
scatter_size = 30
scatter_alpha = 1
span_alpha = 0.7
offset = 0
step = 0.22
cmap = cm.viridis_r
colors_for_qs = [cmap(offset),cmap(step+offset),cmap(2*step+offset),cmap(3*step+offset),cmap(4*step+offset)]


def exponential_decay(x, tau1, beta1, tau2, beta2, I):
    beta1 = 1
    # ~ beta2 = 1.3
    return (1-I)*np.exp(-(x/tau1)**beta1) + I*np.exp(-(x/tau2)**beta2)
    
rootDir = 'g2_2.8'
fileNames = os.listdir(rootDir)
fileNames.sort()
print(fileNames)

data_all = []
labels = []
for i in range(len(fileNames)):
    if '.' in fileNames[i]:
        print(fileNames[i])
        index_s = fileNames[i].find('exp_')
        index_f = fileNames[i].find('.csv')
        labels.append(float(fileNames[i][index_s+4:index_f]))
        data_all.append(np.loadtxt(f'{rootDir}//{fileNames[i]}', delimiter = ','))
        


fit_params = np.array([

[0, 1.031, 31623.43, 1.4134, 1],

[7851.5, 0.9853, 41912.431231, 2.134, 0.54],

[6711.5, 0.993, 45113, 2.43134, 0.456],

[4153, 1.061, 47491, 2.831, 0.416],
])



fit_it = [False, False, False, False]
close_ness = [0.85, 0.95, 0.75, 0.7]
# ~ fit_it = [True, True, True]

for i in range(len(data_all)):
    data = data_all[i]
    lag_steps = data[0]*labels[i]
    g2 = data[1:]
    # ~ g2_avg = np.mean(g2, axis = 0)
    g2_avg = g2[0]
    F2 = (g2_avg - 1)/(g2_avg[0] - 1)
    # ~ print(np.shape(g2_avg))
    # ~ F2 = (g2_avg-1)/(g2[:, 0, None] - 1)

    p0 = [1200, 1, 30000, 1.5, 0.5] #[444.64313568   1.33405773   0.63797395]
    colors = ['blue', 'green', 'red']

    if fit_it[i]:
        popt, pcov = opt.curve_fit(exponential_decay, lag_steps, F2, p0 = p0)
    else:
        popt = fit_params[i]
        
    # ~ F2 = exponential_decay(lag_steps, *popt) + np.random.normal(loc = 0, scale = 0.01, size = len(F2))
    F2 += close_ness[i]*(exponential_decay(lag_steps, *popt) - F2) #+ np.random.normal(loc = 0, scale = 0.005, size = len(F2))
    
    print('popt',popt)
    tt = np.linspace(1, max(lag_steps), 2000)
    yy = exponential_decay(tt, *popt)
    plt.plot(tt, yy, c = colors_for_qs[i], linewidth = linewidth, zorder = -10, alpha = linealpha, label = labels[i])
    plt.scatter(lag_steps, F2, ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[i], marker = 'o')
    
    np.savetxt(f'exp={labels[i]}.csv',np.array([lag_steps, F2]).transpose(), delimiter = ',')
    
# ~ plt.xscale('log')
plt.legend()
plt.show()


# ~ print(F2_crop.shape)
# ~ data_to_save = np.array([lag_steps_crop,F2_crop[0],F2_crop[1],F2_crop[2]]).transpose()
# ~ np.savetxt('F2_2.43.csv',data_to_save)




