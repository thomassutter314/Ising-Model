import numpy as np
import matplotlib.pyplot as plt
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


def gaussian(x, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*(x/sigma)**2)

def lorentzian(x, sigma):
    return 1/(np.pi*sigma)*1/(1+(x/sigma)**2)

fig, axs = plt.subplots(2)

rise = 0.1
qq = np.linspace(-3,3,1000)

yy_1 = lorentzian(qq, 2)

yy_2 = lorentzian(qq, 1.5) + rise
yy_2_lro = lorentzian(qq, 1.5) + 0.03*lorentzian(qq, 0.2) + rise

yy_3 = lorentzian(qq, 1) + 2*rise
yy_3_lro = lorentzian(qq, 1) + 0.05*lorentzian(qq, 0.05) + 2*rise

color_diffuse = 'red'
color_lro = 'blue'
color_mix = 'purple'

lw_diffuse = 1
lw_lro = 1

axs[0].plot(qq,yy_1, c = color_diffuse, linewidth = lw_diffuse)

axs[0].plot(qq,yy_2_lro, c = color_lro, linewidth = lw_lro)
axs[0].plot(qq,yy_2, c = color_diffuse, linewidth = lw_diffuse)

axs[0].plot(qq,yy_3_lro, c = color_lro, linewidth = lw_lro)
axs[0].plot(qq,yy_3, c = color_diffuse, linewidth = lw_diffuse)

height = 0.9
axs[0].set_ylim(0,height)


#axs[0].set_xticks([], [])
axs[0].tick_params(labelbottom=False)
axs[0].tick_params(axis="x", direction="in")
axs[0].set_yticks([], [])
axs[0].set_xticks([], [])

# Eliminate upper and right axes
# ~ axs[0].spines['left'].set_color('none')
# ~ axs[0].spines['right'].set_color('none')
# ~ axs[0].spines['top'].set_color('none')


yy_1 = lorentzian(qq, 2)

yy_2 = lorentzian(qq, 1.5) + rise
yy_2_lro = lorentzian(qq, 1.5) + 0.1*lorentzian(qq, 0.8) + rise

yy_3 = lorentzian(qq, 1) + 2*rise
yy_3_lro = lorentzian(qq, 1) + 0.15*lorentzian(qq, 0.4) + 2*rise


axs[1].plot(qq,yy_1, c = color_mix, linewidth = lw_diffuse)

#axs[1].plot(qq,yy_3_lro, c = color_lro, linewidth = lw_lro)
axs[1].plot(qq,yy_2_lro, c = color_mix, linewidth = lw_diffuse)

#axs[1].plot(qq,yy_4_lro, c = color_lro, linewidth = lw_lro)
axs[1].plot(qq,yy_3_lro, c = color_mix, linewidth = lw_diffuse)

axs[1].set_ylim(0,height)

#axs[1].set_xticks([], [])
axs[1].tick_params(labelbottom=False)
#axs[1].tick_params(axis="x", direction="in")
axs[1].set_yticks([], [])
axs[1].set_xticks([], [])

# Eliminate upper and right axes
# ~ axs[1].spines['left'].set_color('none')
# ~ axs[1].spines['right'].set_color('none')
# ~ axs[1].spines['top'].set_color('none')

axs[1].set_xlabel('Wave Vector')

plt.show()
