import matplotlib.pyplot as plt
from matplotlib import cm
import pyvista as pv
import numpy as np
from scipy import optimize as opt
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator
import os
import tifffile
import pandas as pd
import time

import utils


def box_plot_3d(data, ax, vmin = 0, vmax = 255):
    Nx, Ny, Nz = data.shape
    print(Nx, Ny, Nz)
    X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), -np.arange(Nz))
    
    
    kw = {
        'vmin': vmin,
        'vmax': vmax,
        'levels': np.linspace(data.min(), data.max(), 20),
    }
    
    # Plot contour surfaces
    _ = ax.contourf(
        X[:, :, 0], Y[:, :, 0], data[:, :, 0],
        zdir='z', offset=0, cmap = cm.bwr, **kw
    )
    _ = ax.contourf(
        X[0, :, :], data[0, :, :], Z[0, :, :],
        zdir='y', offset=0, cmap = cm.bwr, **kw
    )
    C = ax.contourf(
        data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
        zdir='x', offset=X.max(), cmap = cm.bwr, **kw
    )
    
    
    # Set limits of the plot from coord limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    
    # Plot edges
    edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
    ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
    ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
    
    
    # Set zoom and angle view
    ax.view_init(25, -40, 0)
    # ~ ax.view_init(22, -13, 0)
    ax.set_box_aspect((1, 1, 1), zoom=1)

def diag_box_plot_3d(data, ax, vmin = 0, vmax = 255,
                     mag_data_factor = 3):
    
    # tranpose to choose what cut of the data we want
    data = np.transpose(data, (2, 1, 0)) 
    data = np.roll(data, 0, axis = 0)
    data = np.roll(data, 0, axis = 1)
    data = np.roll(data, 15, axis = 2)
    
    # Up convert the size of the data for a nice, smooth looking plot
    # Define the original grid coordinates
    x = np.linspace(0, data.shape[0] - 1, data.shape[0])  # Original grid along x-axis
    y = np.linspace(0, data.shape[1] - 1, data.shape[1])  # Original grid along y-axis
    z = np.linspace(0, data.shape[2] - 1, data.shape[2])  # Original grid along z-axis
    interpolator = RegularGridInterpolator((x, y, z), data)
    
    # Define new points for interpolation
    new_x = np.linspace(0, data.shape[0] - 1, mag_data_factor*data.shape[0])  # Interpolating to a finer grid
    new_y = np.linspace(0, data.shape[1] - 1, mag_data_factor*data.shape[1])
    new_z = np.linspace(0, data.shape[2] - 1, mag_data_factor*data.shape[2])
    new_points = np.array(np.meshgrid(new_x, new_y, new_z, indexing='ij')).reshape(3, -1).T
    
    data = interpolator(new_points).reshape(mag_data_factor*data.shape[0], mag_data_factor*data.shape[1], mag_data_factor*data.shape[2])
    
    # get voxel image shape
    Nx, Ny, Nz = data.shape
    
    # Create a grid for sampling along the diagonal plane
    x = np.linspace(0, Nx, Nx)
    z = np.linspace(0, Nz, Nz)
    xv, zv = np.meshgrid(x, z)
    yv = xv  # Diagonal plane
    
    # Flatten the coordinates for interpolation
    coords = np.array([xv.ravel(), yv.ravel(), zv.ravel()])
    
    # Sample the voxel image along the diagonal plane
    diagonal_plane = map_coordinates(data, coords, order=1).reshape(Nx, Nz)
    
    # Normalize the voxel intensity for color mapping
    # ~ norm_diagonal_plane = (diagonal_plane - np.min(diagonal_plane)) / (np.max(diagonal_plane) - np.min(diagonal_plane))
    norm_diagonal_plane = diagonal_plane/255
        
    # Plot the diagonal cut using a surface
    ax.plot_surface(
        xv,  # X coordinates
        yv,  # Y coordinates
        zv,  # Z coordinates
        facecolors=plt.cm.bwr(norm_diagonal_plane),
        rstride=1,
        cstride=1,
        shade=False
        )
    
    # Create a grid for sampling along the top plane
    x = np.linspace(0, Nx, Nx)
    y = np.linspace(0, Ny, Ny)
    xv, yv = np.meshgrid(x, y)
    zv = Nz*np.ones(xv.shape)
    
    top_plane = data[:, :, 0]
    diagonal_cut = lambda x: x
    # Mask the data inside the triangle
    inside_triangle = (
        (yv >= diagonal_cut(xv)) # Above the diagonal cut
    )
    
    # Mask the data
    masked_top_plane = np.where(inside_triangle, top_plane, np.nan)
    normed_masked_top_plane = masked_top_plane/255
    
    ax.plot_surface(
        xv,
        yv,
        zv,
        facecolors=plt.cm.bwr(normed_masked_top_plane),
        rstride=1,
        cstride=1,
        shade = False
        )
    
    
    kw = {
        'vmin': vmin,
        'vmax': vmax,
        'levels': np.linspace(data.min(), data.max(), 50),
    }
    

    # Set limits of the plot from coord limits
    xmin, xmax = 0, Nx
    ymin, ymax = 0, Ny
    zmin, zmax = 0, Nz
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    
    
    removeAutoPlotWindow = True
    if removeAutoPlotWindow:
        # Plot new edges of the frame
        edges_kw = dict(color='0', linewidth=1, zorder=1e3)
        ax.plot([xmin, xmax], [ymin, ymax], zmax, **edges_kw)
        ax.plot([xmin, xmax], [ymin, ymax], zmin, **edges_kw)
        ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **edges_kw)
        ax.plot([xmax, xmax], [ymax, ymax], [zmin, zmax], **edges_kw)
        ax.plot([xmin, xmin], [ymin, ymax], [zmax, zmax], **edges_kw)
        ax.plot([xmax, xmin], [ymax, ymax], [zmax, zmax], **edges_kw)
        
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Remove gridlines and panes
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # ~ # Remove pane lines
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Makes the X-axis spine invisible
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Makes the Y-axis spine invisible
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Makes the Z-axis spine invisible
    
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    
    # Set zoom and angle view
    # ~ ax.view_init(25, -40, 0)
    ax.view_init(22, -13, 0)
    ax.set_box_aspect((1, 1, 1), zoom=1)
   
def set_box_plots_3d(file_dir, N = 5, vmin = 0, vmax = 255, diagF = True):
    files_in_dir = os.listdir(file_dir)
    files_in_dir = sorted(files_in_dir, key = lambda x: int(os.path.splitext(x)[0]))
    
    if len(files_in_dir)%N == 0:
        step = int(len(files_in_dir)/N)
    else:
        step = int(len(files_in_dir)/N) + 1
        
    print('step', step)
    # Create a figure with 3D ax
    fig = plt.figure(figsize=(9, 2.5))
    
    for i in range(N):
        ax = fig.add_subplot(1, N, i + 1, projection='3d')
        ax.set_title(f'{files_in_dir[i*step]}')
        
        data = tifffile.imread(f'{file_dir}//{files_in_dir[i*step]}')
        if diagF:
            diag_box_plot_3d(data, ax, vmin, vmax)
        else:
            box_plot_3d(data, ax, vmin, vmax)
        
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
    plt.show()

def make_cheese(data_dir, cheese_dir, vmin = 0, vmax = 255):
    files_in_dir = os.listdir(data_dir)
    files_in_dir = sorted(files_in_dir, key = lambda x: int(os.path.splitext(x)[0]))
    
    for i in range(len(files_in_dir)):
        print(i)
        voxel_im = tifffile.imread(data_dir + '//' + files_in_dir[i])
        
        fig = plt.figure(figsize=(2.5, 2.5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        diag_box_plot_3d(voxel_im, ax, vmin, vmax)
        
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig(f'{cheese_dir}//{i}.png')
        plt.clf()
                
def plot_levi_cevita():
    def explode(data):
        if len(data.shape) == 4:
            data_e = np.zeros([2*data.shape[0], 2*data.shape[1], 2*data.shape[2], data.shape[3]], dtype=data.dtype)
            data_e[::2, ::2, ::2, :] = data
        if len(data.shape) == 3:
            data_e = np.zeros([2*data.shape[0], 2*data.shape[1], 2*data.shape[2]], dtype=data.dtype)
            data_e[::2, ::2, ::2] = data
            
        return data_e
    
    # build up the numpy logo
    # 0.5 represents 0
    n_voxels = np.zeros((3, 3, 3)) + 0.5
    
    # 1 represents 1
    n_voxels[0, 1, 2] = 1
    n_voxels[2, 0, 1] = 1
    n_voxels[1, 2, 0] = 1
    
    # 0 represents -1
    n_voxels[1, 0, 2] = 0
    n_voxels[2, 1, 0] = 0
    n_voxels[0, 2, 1] = 0
    
    
    # ~ n_voxels = tifffile.imread(r"C:\Users\kogar\OneDrive\Documents\GitHub\Ising-Model\results\default\time_sequence\0.tiff")
    # ~ print(n_voxels.dtype)
    
    # ~ facecolors = np.where(n_voxels, '#FFD65DC0', '#7A88CCC0')
    # ~ print(facecolors.shape)
    facecolors = cm.bwr(n_voxels)
    
    filled = np.ones(n_voxels.shape)
    
    # upscale the above voxel image, leaving gaps
    filled_2 = explode(filled)
    fcolors_2 = explode(facecolors)
    
    print('filled', filled.shape)
    print('fcolors_1', facecolors.shape)
    print('filled_2', filled_2.shape)
    print('fcolors_2', fcolors_2.shape)
    
    # Shrink the gaps
    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float)
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 111 means 1 row, 1 column, 1 plot
    
    ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors = 'black', alpha = 0.3)
    # ~ ax.voxels(x, y, z, filled, facecolors=facecolors, edgecolors = 'black', alpha = 0.25)
    
    plt.show()

def combine_plots(maindir, dirs, label, cmap = cm.plasma):
    # Load the 1st piece of data to get the shapes
    data_0 = np.loadtxt(f'{maindir}//{dirs[0]}//g2.csv', delimiter = ',')
    lag_steps = data_0[0, :]
    g2_0 = data_0[1:, :]
    
    # Make a g2 array to hold all the g2's
    g2_s = np.zeros([len(dirs), g2_0.shape[0], g2_0.shape[1]])
    g2_s[0] = g2_0
    
    for i in range(len(dirs) - 1):
        data = np.loadtxt(f'{maindir}//{dirs[i + 1]}//g2.csv', delimiter = ',')
        
        # Makes sure that all the data are for the same set of lagsteps
        if (lag_steps != data[0, :]).any():
            print('Lag steps are unequal, cannot proceed')
            return
        
        # Put the new data in the g2_s array    
        g2_s[i + 1] = data[1:, :]
        
    print(f'Shape of g2_s = {g2_s.shape}, (number of dirs, number of rois, number of steps)')
    
    # Create an average of the g2_s, average over axis 0 because that is the axis for specifying the dir
    g2_avg = np.mean(g2_s, axis = 0)
    
    frame0 = 0
    F2_avg = ((g2_avg-1)/(g2_avg[:, frame0, None] - 1))
    
    if True:
        print(F2_avg.shape)
        data_to_save = np.array([lag_steps,F2_avg[0],F2_avg[1],F2_avg[2]]).transpose()
        np.savetxt('F2.csv',data_to_save)
        # ~ np.savetxt('test.csv', F2_avg.transpose(), delimiter = ',')
        # ~ np.savetxt('test_1.csv',lag_steps.transpose(), delimiter = ',')
    
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Separate runs plotted separately')
    # Make a scatter plot of the separate dirs data
    for j in range(g2_s.shape[1]):
        for i in range(g2_s.shape[0]):
            axs[j].scatter(lag_steps, g2_s[i, j, :], label = f'{dirs[i]}')
        
        axs[j].set_xlabel('lag steps')
        axs[j].set_title(f'ROI {j}')
        axs[j].set_xscale('log')
        axs[j].legend()
    
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Dir Average')
    
    p0 = [100, 1] # guess for fit
    tt = np.linspace(min(lag_steps), max(lag_steps), 1000) # x-vals for plotting the fit curve
    taus = []
    for i in range(g2_avg.shape[0]):
        color = cmap(i/(F2_avg.shape[0] - 1)) # color code for rois
        
        axs[0].scatter(lag_steps, g2_avg[i], color = color)
        axs[1].scatter(lag_steps, F2_avg[i], color = color)
        try:
            popt, pcov = opt.curve_fit(utils.stretched_exp, lag_steps, F2_avg[i], p0 = p0)
            # ~ 4//fi
            print('popt', popt)
        except:
            popt = p0
            print(f'Fit {i} failed, using guess vals as popt')
            
        yy = utils.stretched_exp(tt, *popt)
        
        taus.append(popt[0])
        axs[1].plot(tt, yy, c = color, label = f'tau = {round(popt[0], 3)}')
        
    # Save the fit results 
    record_fit(maindir, label, taus)
    
    for ax in axs:
        ax.set_xscale('log')
        
    axs[1].legend()
    plt.show()
   
def record_fit(maindir, label, taus):
    # Create an empty DataFrame with specific column names
    # ~ df = pd.DataFrame(columns=['temp', 'tau_1', 'tau_2', 'tau_3'])
    
    df = pd.read_csv(f'{maindir}//tau_vs_temp_data.csv')
    
    # Check if temp is already there
    is_present = label in df['temp'].values
    
    if is_present:
        new_data = {'tau_1': taus[0], 'tau_2': taus[1], 'tau_3': taus[2]}
        df.loc[df['temp'] == label, ['tau_1', 'tau_2', 'tau_3']] = new_data.values()
    else:
        # New rows as a DataFrame
        new_rows = pd.DataFrame([
            [label, taus[0], taus[1], taus[2]],
            ], columns=['temp', 'tau_1', 'tau_2', 'tau_3'])
        
        # Append rows
        df = pd.concat([df, new_rows], ignore_index=True)
        
    df.to_csv(f'{maindir}//tau_vs_temp_data.csv', index=False)
        
def plot_spatial_corr(fileName):
    image = tifffile.imread(fileName)
    c = utils.compute_c_3D(image)
    for i in range(3):
        c = np.roll(c, c.shape[i]//2, axis = i)
    print(c.shape)
    

    c = np.fft.fftshift(compute_c(mass))
    plt.imshow(c)
    plt.show()
    
    y = c[c.shape[0]//2, :]
    x = range(len(y))
    p0 = [x[len(x)//2], len(x)//4, max(y)-min(y), min(y)]
    popt, pcov = opt.curve_fit(gaussian, x, y, p0 = p0)
    print(popt)
    xx = np.linspace(min(x), max(x), 1000)
    yy = gaussian(xx, *popt)
    plt.scatter(np.array(x) - x[len(x)//2], y)
    plt.plot(xx - x[len(x)//2], yy, 'g-')
    plt.show()

def all_plot(mainDir = 'results'):
    df = pd.read_csv(f'{mainDir}//tau_vs_temp_data.csv')
    
    temps = np.array(df['temp'].tolist())
    tau_1 = np.array(df['tau_1'].tolist())
    tau_2 = np.array(df['tau_2'].tolist())
    tau_3 = np.array(df['tau_3'].tolist())
    
    cmap = cm.viridis_r
    offset = 0
    step = 0.22
    colors_for_qs = [cmap(offset),cmap(step+offset),cmap(2*step+offset)]
    
    plt.scatter(temps, tau_1, fc = colors_for_qs[0], ec = 'black', label = 'roi 0')
    plt.scatter(temps, tau_2, fc = colors_for_qs[1], ec = 'black', label = 'roi 1')
    plt.scatter(temps, tau_3, fc = colors_for_qs[2], ec = 'black', label = 'roi 2')
    
    def fit_func_1(x, A):
        # ~ x0, gamma =  1.20382101e+02, -7.66184239e-02
        x0, gamma =  5.3, -10
        return A*np.exp(gamma * (x - x0)) + 5.6
        
    # add the fit funcs
    p0_1 = [50]
    p0_2 = [20]
    p0_3 = [10]

    
    linewidth = 5
    linealpha = 0.5
    popt_1, pcov_1 = opt.curve_fit(fit_func_1, temps, tau_1, p0 = p0_1)
    popt_2, pcov_2 = opt.curve_fit(fit_func_1, temps, tau_2, p0 = p0_2)
    popt_3, pcov_3 = opt.curve_fit(fit_func_1, temps, tau_3, p0 = p0_3)
    # ~ popt_1 = p0_1
    # ~ popt_2 = p0_2
    # ~ popt_3 = p0_3
    
    tt = np.linspace(min(temps), max(temps), 1000)
    yy = fit_func_1(tt, *popt_1)
    # ~ plt.plot(tt, yy, c = colors_for_qs[0], linewidth = linewidth, zorder = -10, alpha = linealpha)
    yy = fit_func_1(tt, *popt_2)
    # ~ plt.plot(tt, yy, c = colors_for_qs[1], linewidth = linewidth, zorder = -10, alpha = linealpha)
    yy = fit_func_1(tt, *popt_3)
    # ~ plt.plot(tt, yy, c = colors_for_qs[2], linewidth = linewidth, zorder = -10, alpha = linealpha)
        
    
    plt.ylabel('tau')
    plt.xlabel('temp')
    plt.yscale('log')
    
    plt.legend()
    plt.show()

def compare_all_plot(dir1, dir2, dir3):
    fig, ax = plt.subplots()
    
    if True:
        df = pd.read_csv(f'{dir1}//tau_vs_temp_data.csv')
        
        temps = np.array(df['temp'].tolist())
        Tc = 4.52
        temps = (temps - Tc)/Tc
        tau_1 = np.array(df['tau_1'].tolist())
        tau_2 = np.array(df['tau_2'].tolist())
        tau_3 = np.array(df['tau_3'].tolist())
        
        cmap = cm.viridis_r
        offset = 0
        step = 0.22
        colors_for_qs = [cmap(offset),cmap(step+offset),cmap(2*step+offset)]
        
        ax.scatter(temps, tau_1, fc = colors_for_qs[0], ec = 'black', label = 'field disorder: roi 0')
        # ~ ax.scatter(temps, tau_2, fc = colors_for_qs[1], ec = 'black', label = 'roi 1')
        # ~ ax.scatter(temps, tau_3, fc = colors_for_qs[2], ec = 'black', label = 'roi 2')
        
        def fit_func_1(x, A):
            # ~ x0, gamma =  1.20382101e+02, -7.66184239e-02
            x0, gamma =  5.3, -10
            return A*np.exp(gamma * (x - x0)) + 5.6
            
        # add the fit funcs
        p0_1 = [50]
        p0_2 = [20]
        p0_3 = [10]
    
        
        linewidth = 5
        linealpha = 0.5
        popt_1, pcov_1 = opt.curve_fit(fit_func_1, temps, tau_1, p0 = p0_1)
        popt_2, pcov_2 = opt.curve_fit(fit_func_1, temps, tau_2, p0 = p0_2)
        popt_3, pcov_3 = opt.curve_fit(fit_func_1, temps, tau_3, p0 = p0_3)
        # ~ popt_1 = p0_1
        # ~ popt_2 = p0_2
        # ~ popt_3 = p0_3
        
        tt = np.linspace(min(temps), max(temps), 1000)
        yy = fit_func_1(tt, *popt_1)
        # ~ plt.plot(tt, yy, c = colors_for_qs[0], linewidth = linewidth, zorder = -10, alpha = linealpha)
        yy = fit_func_1(tt, *popt_2)
        # ~ plt.plot(tt, yy, c = colors_for_qs[1], linewidth = linewidth, zorder = -10, alpha = linealpha)
        yy = fit_func_1(tt, *popt_3)
        # ~ plt.plot(tt, yy, c = colors_for_qs[2], linewidth = linewidth, zorder = -10, alpha = linealpha)
    
    if True:
        df = pd.read_csv(f'{dir2}//tau_vs_temp_data.csv')
        
        temps = np.array(df['temp'].tolist())
        Tc = 4.8
        temps = (temps - Tc)/Tc
        tau_1 = np.array(df['tau_1'].tolist())
        tau_2 = np.array(df['tau_2'].tolist())
        tau_3 = np.array(df['tau_3'].tolist())
        
        cmap = cm.viridis_r
        offset = 0
        step = 0.22
        colors_for_qs = [cmap(offset),cmap(step+offset),cmap(2*step+offset)]
        
        ax.scatter(temps, tau_1, fc = colors_for_qs[0], ec = 'black', label = 'mass disorder: roi 0', marker = '^')
        ax.scatter(temps, tau_2, fc = colors_for_qs[1], ec = 'black', label = 'roi 1', marker = '^')
        ax.scatter(temps, tau_3, fc = colors_for_qs[2], ec = 'black', label = 'roi 2', marker = '^')
        
        def fit_func_1(x, A):
            # ~ x0, gamma =  1.20382101e+02, -7.66184239e-02
            x0, gamma =  5.3, -10
            return A*np.exp(gamma * (x - x0)) + 5.6
            
        # add the fit funcs
        p0_1 = [50]
        p0_2 = [20]
        p0_3 = [10]
    
        
        linewidth = 5
        linealpha = 0.5
        popt_1, pcov_1 = opt.curve_fit(fit_func_1, temps, tau_1, p0 = p0_1)
        popt_2, pcov_2 = opt.curve_fit(fit_func_1, temps, tau_2, p0 = p0_2)
        popt_3, pcov_3 = opt.curve_fit(fit_func_1, temps, tau_3, p0 = p0_3)
        # ~ popt_1 = p0_1
        # ~ popt_2 = p0_2
        # ~ popt_3 = p0_3
        
        tt = np.linspace(min(temps), max(temps), 1000)
        yy = fit_func_1(tt, *popt_1)
        # ~ plt.plot(tt, yy, c = colors_for_qs[0], linewidth = linewidth, zorder = -10, alpha = linealpha)
        yy = fit_func_1(tt, *popt_2)
        # ~ plt.plot(tt, yy, c = colors_for_qs[1], linewidth = linewidth, zorder = -10, alpha = linealpha)
        yy = fit_func_1(tt, *popt_3)
        # ~ plt.plot(tt, yy, c = colors_for_qs[2], linewidth = linewidth, zorder = -10, alpha = linealpha)
        
    if True:
        df = pd.read_csv(f'{dir3}//tau_vs_temp_data.csv')
        
        temps = np.array(df['temp'].tolist())
        Tc = 4.5
        temps = (temps - Tc)/Tc
        tau_1 = np.array(df['tau_1'].tolist())
        tau_2 = np.array(df['tau_2'].tolist())
        tau_3 = np.array(df['tau_3'].tolist())
        
        cmap = cm.viridis_r
        offset = 0
        step = 0.22
        colors_for_qs = [cmap(offset),cmap(step+offset),cmap(2*step+offset)]
        
        ax.scatter(temps, tau_1, fc = colors_for_qs[0], ec = 'black', label = 'pristine: roi 0', marker = 's')
        ax.scatter(temps, tau_2, fc = colors_for_qs[1], ec = 'black', label = 'roi 1', marker = 's')
        ax.scatter(temps, tau_3, fc = colors_for_qs[2], ec = 'black', label = 'roi 2', marker = 's')
        
        def fit_func_1(x, A):
            # ~ x0, gamma =  1.20382101e+02, -7.66184239e-02
            x0, gamma =  5.3, -10
            return A*np.exp(gamma * (x - x0)) + 5.6
            
        # add the fit funcs
        p0_1 = [50]
        p0_2 = [20]
        p0_3 = [10]
    
        
        linewidth = 5
        linealpha = 0.5
        popt_1, pcov_1 = opt.curve_fit(fit_func_1, temps, tau_1, p0 = p0_1)
        popt_2, pcov_2 = opt.curve_fit(fit_func_1, temps, tau_2, p0 = p0_2)
        popt_3, pcov_3 = opt.curve_fit(fit_func_1, temps, tau_3, p0 = p0_3)
        # ~ popt_1 = p0_1
        # ~ popt_2 = p0_2
        # ~ popt_3 = p0_3
        
        tt = np.linspace(min(temps), max(temps), 1000)
        yy = fit_func_1(tt, *popt_1)
        # ~ plt.plot(tt, yy, c = colors_for_qs[0], linewidth = linewidth, zorder = -10, alpha = linealpha)
        yy = fit_func_1(tt, *popt_2)
        # ~ plt.plot(tt, yy, c = colors_for_qs[1], linewidth = linewidth, zorder = -10, alpha = linealpha)
        yy = fit_func_1(tt, *popt_3)
        # ~ plt.plot(tt, yy, c = colors_for_qs[2], linewidth = linewidth, zorder = -10, alpha = linealpha)
    
    ax.set_ylabel('tau')
    ax.set_xlabel('(T-Tc)/Tc')
    ax.legend()
    ax.set_yscale('log')
    
    # ~ .legend()
    plt.show()
    
def voxel_plot_test():

    for i in range(42):
        data = tifffile.imread(r"C:\Users\kogar\OneDrive\Documents\GitHub\Ising-Model\results\default\time_sequence" + f"//{i}.tiff")
        # ~ data = tifffile.imread(r"C:\Users\kogar\OneDrive\Documents\GitHub\Ising-Model\results\domain_walls\random_field\time_sequence_2_c\7.tiff")
        # ~ data = tifffile.imread(r"C:\Users\kogar\OneDrive\Documents\GitHub\Ising-Model\results\random_mass\T5.0_1\time_sequence\10.tiff")
        
        
        # Create a UniformGrid (Voxel structure)
        grid = pv.ImageData()
        grid.dimensions = np.array(data.shape) + 1  # PyVista uses dimensions one more than the shape
        grid.origin = (0, 0, 0)  # Set the origin
        grid.spacing = (1, 1, 1)  # Set voxel spacing
        grid.cell_data["values"] = data.flatten(order="F")  # Flatten in Fortran order
        
    
        opacity = 256*(np.linspace(-1, 1, 256))**2
        
        # ~ opacity = 256*np.abs(np.linspace(-1, 1, 256))
        # ~ opacity = 256*np.linspace(0, 1, 256)**3
        
        # Create a PyVista plotter and add the voxel grid
        plotter = pv.Plotter(window_size=[200, 200], off_screen=True)
        plotter.add_volume(grid, cmap="bwr", opacity=opacity, show_scalar_bar = False)  # Adjust colormap and opacity
        
        # Create an outline of the grid (the cube)
        outline = grid.outline()  # Creates an outline around the grid
        
        # Add the outline to the plotter as a solid line
        plotter.add_mesh(outline, color="black", line_width=3)  # Customize color and line width
        
        # setting the camera position
        plotter.camera_position = 'iso'
        plotter.camera.azimuth = 10
        
        # saving the plot as an image
        plotter.screenshot(r"C:\Users\kogar\OneDrive\Documents\GitHub\Ising-Model\results\default\voxel_images" + f'//{i}.png')
        
        print(f"Finished Image {i}")
        
        # Show the plot
        plotter.show()
    
    
    
    
if __name__ == '__main__':
    
    mainDir = "results//random_mass"
    combine_plots(mainDir, ['T5.0_1'], label = 5.01)

    
