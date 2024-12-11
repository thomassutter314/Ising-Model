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
markers_for_qs = ['o','o','o', 'D', 'P']


def exponential_decay(x, tau, beta):
    # ~ beta = 1
    return np.exp(-(x/tau)**beta)
    

fit_in = 0
fit_out = 35

temp = '2.43'
# ~ temp = '1.9'
rootDir = '2.43'
fileNames = os.listdir(rootDir)
print(fileNames)

data_all = []
for i in range(len(fileNames)):
    if f'g2_{temp}' in fileNames[i]:
        print(fileNames[i])
        data_all.append(np.loadtxt(f'{rootDir}//{fileNames[i]}', delimiter = ','))
        
data = np.zeros(data_all[0].shape)

weights = np.ones(len(data_all))

for j in range(len(weights)):
    w = weights[j]
    data += w*data_all[j]/(np.sum(weights))
    
lag_steps = data[0]
g2 = data[1:]
F2 = (g2-1)/(g2[:, 0, None] - 1)

# ~ print('F2.shape',F2.shape)
# Remove some of the data
lag_steps_crop = 100*(lag_steps[fit_in:] - lag_steps[fit_in])
# ~ lag_steps_crop = lag_steps_crop - 99

# ~ print('F2[:, fit_in, None]',F2[:, fit_in, None])
# ~ normalization = np.array([0.81867587,0.6950753,0.54176673])
F2_crop = F2[:, fit_in : F2.shape[1]]/F2[:, fit_in+1, None]
# ~ F2_crop = F2[:, fit_in : F2.shape[1] - fit_out]/normalization[:, None]

# ~ X = np.linspace(1,0,len(F2_crop[0]))
# ~ F2_crop[0] += 0.15*(1-F2_crop[0])*(X)**(0.5)
# ~ F2_crop[1] += 0.13*(1-F2_crop[1])*(X)**(0.5)
# ~ F2_crop[2] -= 0.26*(1-F2_crop[2])*(X)**2

p0 = [2600, 1]
colors = ['blue', 'green', 'red']
labels = ['q0', 'q1', 'q2']
for i in range(len(F2)):
    try:
        popt, pcov = opt.curve_fit(exponential_decay, lag_steps_crop[:fit_out], F2_crop[i, :fit_out], p0 = p0)
        # ~ print('F2_crop[i]', F2_crop[i])
    except:
        popt = p0
        print(f'ERROR IN FIT {i}')
    
    print('popt',popt)
    # ~ print(f'{labels[i]}: \n tau = {popt[0]} pm {np.sqrt(pcov[0,0])} \n beta = {popt[1]} pm {np.sqrt(pcov[1,1])} \n I0 = {popt[2]} pm {np.sqrt(pcov[2,2])}')
    
    # ~ print(i)
    color = colors[i]
    tt = np.linspace(min(lag_steps_crop[1:]), max(lag_steps_crop), 2000)
    yy = exponential_decay(tt, *popt)
    # ~ print(yy)
    plt.plot(tt, yy, c = colors_for_qs[i], linewidth = linewidth, zorder = -10, alpha = linealpha)
    plt.scatter(lag_steps_crop[1:], F2_crop[i,1:], ec = 'black', s = scatter_size, alpha = scatter_alpha, fc = colors_for_qs[i], marker = markers_for_qs[i])
  
plt.xscale('log')
plt.show()

print(F2_crop.shape)
data_to_save = np.array([lag_steps_crop,F2_crop[0],F2_crop[1],F2_crop[2]]).transpose()
np.savetxt('F2_2.43.csv',data_to_save)




