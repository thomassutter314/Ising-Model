
def getstatefft(self, ):
        fft_im = np.array(np.abs(np.fft.fftshift(np.fft.fft2(im)))**2, dtype = np.float32)




# ~ For trying out different neighbor cutoffs. We could do nearest neighbor or next nearest neighbor and so on.
# ~ It's fun to also do something like, only the 3rd neighbor affects you.        

# ~ nn = (np.roll(self.config, 1, axis = 0) + np.roll(self.config, -1, axis = 0) + np.roll(self.config, 1, axis = 1) + np.roll(self.config, -1, axis = 1)) + \
                # ~ (1/np.sqrt(2))**3*(np.roll(self.config, (1,1), axis = (0,1)) + np.roll(self.config, (-1,1), axis = (0,1)) + np.roll(self.config, (1,-1), axis = (0,1)) + np.roll(self.config, (-1,-1), axis = (0,1))) + \
                # ~ (1/2)**3*(np.roll(self.config, 2, axis = 0) + np.roll(self.config, -2, axis = 0) + np.roll(self.config, 2, axis = 1) + np.roll(self.config, -2, axis = 1)) + \
                # ~ (1/np.sqrt(5))**3*(np.roll(self.config, (2,1), axis = (0,1)) + np.roll(self.config, (2,-1), axis = (0,1)) + np.roll(self.config, (-2,1), axis = (0,1)) + np.roll(self.config, (-2,-1), axis = (0,1))) + \
                # ~ (1/np.sqrt(5))**3*(np.roll(self.config, (1,2), axis = (0,1)) + np.roll(self.config, (-1,2), axis = (0,1)) + np.roll(self.config, (1,-2), axis = (0,1)) + np.roll(self.config, (-1,-2), axis = (0,1))) + \
                # ~ (0.5/np.sqrt(2))**3*(np.roll(self.config, (2,2), axis = (0,1)) + np.roll(self.config, (-2,2), axis = (0,1)) + np.roll(self.config, (2,-2), axis = (0,1)) + np.roll(self.config, (-2,-2), axis = (0,1)))
                
        # ~ nn = \
            # ~ (1/np.sqrt(5))**3*(np.roll(self.config, (2,1), axis = (0,1)) + np.roll(self.config, (2,-1), axis = (0,1)) + np.roll(self.config, (-2,1), axis = (0,1)) + np.roll(self.config, (-2,-1), axis = (0,1))) + \
            # ~ (1/np.sqrt(5))**3*(np.roll(self.config, (1,2), axis = (0,1)) + np.roll(self.config, (-1,2), axis = (0,1)) + np.roll(self.config, (1,-2), axis = (0,1)) + np.roll(self.config, (-1,-2), axis = (0,1))) + \
            # ~ (0.5/np.sqrt(2))**3*(np.roll(self.config, (2,2), axis = (0,1)) + np.roll(self.config, (-2,2), axis = (0,1)) + np.roll(self.config, (2,-2), axis = (0,1)) + np.roll(self.config, (-2,-2), axis = (0,1)))
        
        # ~ nn = -(1/np.sqrt(5))**3*(np.roll(self.config, (2,1), axis = (0,1)) + np.roll(self.config, (2,-1), axis = (0,1)) + np.roll(self.config, (-2,1), axis = (0,1)) + np.roll(self.config, (-2,-1), axis = (0,1))) + \
            # ~ (1/np.sqrt(2))**3*(np.roll(self.config, (1,1), 
# ~ axis = (0,1)) + np.roll(self.config, (-1,1), axis = (0,1)) + np.roll(self.config, (1,-1), axis = (0,1)) + np.roll(self.config, (-1,-1), axis = (0,1)))


    
def go_g2():
    N_images = 1001
    im_0 = tifffile.imread(r"C:\Users\thoma\OneDrive - UCLA IT Services\Desktop\OneDrive - UCLA IT Services\Research\TiSe2_XPCS\ising_model\reciprocal_space" + f"//0.tiff")
    Icrop = np.empty([N_images,im_0.shape[0],im_0.shape[1]])
    
    N_remove = 1
    Icrop = Icrop[N_remove:]
    N_images += -N_remove
    
    for i in range(N_images):
        if i%10 == 0:
            print(f'{i}/{N_images}')
        Icrop[i] = tifffile.imread(r"C:\Users\thoma\OneDrive - UCLA IT Services\Desktop\OneDrive - UCLA IT Services\Research\TiSe2_XPCS\ising_model\reciprocal_space" + f"//{i}.tiff")
    
    # Gaussian filter the images
    Icrop_avg_gf = gaussian_filter(np.mean(Icrop, axis = 0), 1)
    
    Icrop_flat = Icrop/Icrop_avg_gf
    
    # Now compute the g2
    q_radius = 3
    N = 3

    xcentroid = im_0.shape[1]//2
    ycentroid = im_0.shape[0]//2

    g2_all = []
    colormap = np.zeros(Icrop.shape[1:3], dtype = np.int64)

    for i in range(N):
        
        #data = Icrop.copy()/(np.sum(Icrop, axis = (1,2))[:,None,None]) # Normalized images
        data = Icrop_flat.copy()/(np.sum(Icrop_flat, axis = (1,2))[:,None,None]) # Normalized images
        #data = Icrop_flat.copy()
        
        labeled_roi_array = getAnnulusMask(data[0],xcentroid,ycentroid,q_radius,i,i+1)

        colormap = colormap + labeled_roi_array*(i+1)
        
        plt.figure()
        plt.title(f'q_radius = {i*q_radius}')
        plt.imshow(np.mean(Icrop, axis = 0))
        plt.imshow(labeled_roi_array, alpha=0.5,cmap=cm.gray)
        
        #XPCS parameters and run calculation
        num_levels = 10
        num_bufs = 12
        g2n, lag_steps = corr.multi_tau_auto_corr(num_levels, num_bufs, labeled_roi_array, data[1:])
        
        g2_all.append(g2n[1:])

    # Convert g2_all to a sensible shape and structure
    g2_all = np.array(g2_all)[:,:,0]
        
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(np.mean(Icrop, axis = 0))
    axs[0].set_title('CDW Peak')
    axs[1].imshow(colormap, cmap = cm.jet)
    axs[1].set_title('ROI Regions')
    #plt.imshow(labeled_roi_array,alpha=0.2,cmap=cm.gray)


    #################################
    fig, ax = plt.subplots()
    print(np.shape(g2_all))
    #F = np.sqrt((g2_all-1)/(g2_all[:,:2].mean(axis = 1)[:, None] - 1))
    F = np.sqrt((g2_all-1)/(g2_all[:,0][:, None] - 1))

    for i in range(F.shape[0]):
        ax.scatter(lag_steps[1:], F[i], label = f'q_radius = {i*q_radius}')
        #plt.semilogx(lag_steps[1:], g2_all[i], label = f'q_radius = {i*q_radius}')
        
    ax.set_xscale('log')
    ax.set_xlim([0,100])
    ax.set_xlabel(r'$\tau$ (s)')
    ax.set_ylabel(r'$F$')
    plt.legend()
    plt.show()
    
    
def goIsing():
    rm = Ising()
    rm.simulate()
    plt.show()

    for i in range(1001):
        im = tifffile.imread(r"C:\Users\thoma\OneDrive - UCLA IT Services\Desktop\OneDrive - UCLA IT Services\Research\TiSe2_XPCS\ising_model\real_space" + f"//{i}.tiff")
        fft_im = np.array(np.abs(np.fft.fftshift(np.fft.fft2(im)))**2, dtype = np.float32)
        tifffile.imwrite(r"C:\Users\thoma\OneDrive - UCLA IT Services\Desktop\OneDrive - UCLA IT Services\Research\TiSe2_XPCS\ising_model\reciprocal_space" + f"//{i}.tiff", fft_im)











    def measure_g2(self, plot_fft = True, plot_real = True):
        # Get the times of the images from the file names
        times = os.listdir(f'{self.save_loc}\\time_sequence')
        for i in range(len(times)):
            times[i] = int(times[i][:-5])
        times.sort()
        
        realFrames = np.empty([len(times), imshape[0], imshape[1]])
        
        for i in range(len(times)):
            realFrames[i] = 2*tifffile.imread(f'{self.save_loc}\\time_sequence\\{times[i]}.tiff') - 1
            Iframes[i] = np.array(np.abs(np.fft.fftshift(np.fft.fft2(realFrames[i])))**2, dtype = np.float32) # Compute the fft
            # ~ Iframes[i] = np.array(np.abs((np.fft.fft2(realFrames[i])))**2, dtype = np.float32) # Compute the fft
        
        if plot_real:
            fig_real, ax_real = plt.subplots()
            
            real_display_obj = ax_real.imshow(realFrames[0],cmap = 'grey', vmin = -1, vmax = 1)
            plt.imshow(self.C_mask, cmap = 'cool', alpha = np.abs(self.C_mask))
            
            # Build sliders for adjusting the brightness and contrast of the image
            ax_real_setVals = [plt.axes([0.15, 0.02, 0.5, 0.02])]
            
            imagesMin, imagesMax = 0, np.max(Iframes[:,imshape[0]//2+2,imshape[1]//2+2])
            slider_image_real = matplotlib.widgets.Slider(ax_real_setVals[0], r'image #', 0, len(times)-1, valinit=0, valfmt="%i")
                    
            def sliderUpdateImage_real(val):
                real_display_obj.set_data(realFrames[int(val)])
                fig.canvas.draw()
                
            slider_image_real.on_changed(sliderUpdateImage_real)
        
        if plot_fft:
            fig_fft, ax_fft = plt.subplots()
            
            fft_display_obj = ax_fft.imshow(Iframes[0])
            
            # Build sliders for adjusting the brightness and contrast of the image
            ax_fft_setVals = [plt.axes([0.15, 0.1, 0.5, 0.02]), plt.axes([0.15, 0.06, 0.5, 0.02]), plt.axes([0.15, 0.02, 0.5, 0.02])]
            
            imagesMin, imagesMax = 0, np.max(Iframes[:,imshape[0]//2+2,imshape[1]//2+2])
            slider_vmax_fft = matplotlib.widgets.Slider(ax_fft_setVals[0], r'$v_{max}$', imagesMin, imagesMax, valinit=imagesMax)
            slider_vmin_fft = matplotlib.widgets.Slider(ax_fft_setVals[1], r'$v_{min}$', imagesMin, imagesMax, valinit=imagesMin)
            slider_image_fft = matplotlib.widgets.Slider(ax_fft_setVals[2], r'image #', 0, len(times)-1, valinit=0, valfmt="%i")
            
            def sliderUpdateVmax_fft(val):
                if val > fft_display_obj.get_clim()[0]:
                    fft_display_obj.set_clim(fft_display_obj.get_clim()[0], val)
                else:
                    slider_vmax_fft.set_val(fft_display_obj.get_clim()[0]+1)
            def sliderUpdateVmin_fft(val):
                if val < fft_display_obj.get_clim()[1]:
                    fft_display_obj.set_clim(val, fft_display_obj.get_clim()[1])
                else:
                    slider_vmin_fft.set_val(fft_display_obj.get_clim()[1]-1)
                    
            def sliderUpdateImage_fft(val):
                fft_display_obj.set_data(Iframes[int(val)])
                fig_fft.canvas.draw()
                
            slider_vmax_fft.on_changed(sliderUpdateVmax_fft)
            slider_vmin_fft.on_changed(sliderUpdateVmin_fft)
            slider_image_fft.on_changed(sliderUpdateImage_fft)
        
        magnetization = np.average(realFrames, axis = (1,2))
        fig, ax = plt.subplots()
        ax.plot(magnetization)
        ax.set_ylabel('Magnetization')
        ax.set_xlabel('Time')
        
        # Gaussian filter the images
        Iframes_avg_gf = gaussian_filter(np.mean(Iframes, axis = 0), 1)
        # ~ Iframes_flat = Iframes/Iframes_avg_gf
        Iframes_flat = Iframes
        
        # Now compute the g2
        q_radius = 3
        N = 3
        xcentroid = im_0.shape[1]//2
        ycentroid = im_0.shape[0]//2

        g2_all = []
        colormap = np.zeros(Iframes.shape[1:3], dtype = np.int64)
        
        for i in range(N):
            #data = Icrop.copy()/(np.sum(Icrop, axis = (1,2))[:,None,None]) # Normalized images
            data = Iframes_flat.copy()/(np.sum(Iframes_flat, axis = (1,2))[:,None,None]) # Normalized images
            #data = Icrop_flat.copy()
            
            labeled_roi_array = getAnnulusMask(data[0],xcentroid,ycentroid,q_radius,i,i+1)
    
            colormap = colormap + labeled_roi_array*(i+1)
            
            #XPCS parameters and run calculation
            num_levels = 10
            num_bufs = 12
            g2n, lag_steps = corr.multi_tau_auto_corr(num_levels, num_bufs, labeled_roi_array, data[1:])
            
            g2_all.append(g2n[1:])
    
        # Convert g2_all to a sensible shape and structure
        g2_all = np.array(g2_all)[:,:,0]
        
        data_to_save = np.empty([g2_all.shape[0] + 1, g2_all.shape[1]])
        data_to_save[0,:] = lag_steps[1:]
        data_to_save[1:,:] = g2_all
        # Save the g2 values from the run
        np.savetxt(f'{self.save_loc}\\g2.csv', data_to_save, delimiter = ',', header = f'q_radius = {q_radius}')
            
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(np.log(np.mean(Iframes, axis = 0) + 1))
        axs[0].set_title('Peak, Log Scale')
        axs[1].imshow(colormap, cmap = cm.jet)
        axs[1].set_title('ROI Regions')
        #plt.imshow(labeled_roi_array,alpha=0.2,cmap=cm.gray)
    
    
        #################################
        fig, ax = plt.subplots(2)
        print(np.shape(g2_all))
        #F = np.sqrt((g2_all-1)/(g2_all[:,:2].mean(axis = 1)[:, None] - 1))
        F = np.sqrt((g2_all-1)/(g2_all[:,0][:, None] - 1))
    
        for i in range(F.shape[0]):
            ax[0].scatter(lag_steps[1:], F[i], label = f'q_radius = {i*q_radius}')
            ax[1].scatter(lag_steps[1:], g2_all[i], label = f'q_radius = {i*q_radius}')
            #plt.semilogx(lag_steps[1:], g2_all[i], label = f'q_radius = {i*q_radius}')
        
            # ~ popt, pcov = opt.curve_fit(sc_exponential, lag_steps[1:], F[i])
            # ~ tt = np.linspace(min(lag_steps[1:]), max(lag_steps[1:],1000))
            # ~ yy = sc_exponential(tt, *popt)
            # ~ ax.plot(tt, yy)
        
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        # ~ ax.set_xlim([0,100])
        ax[0].set_xlabel(r'$\tau$ (s)')
        ax[0].set_ylabel(r'$F$')
        plt.legend()
        plt.show()







       # Get a sample image
        im_0 = tifffile.imread(f'{self.save_loc}\\time_sequence\\{times[0]}.tiff')
        imshape = im_0.shape
        
        # Array holding all the "diffraction" images
        Iframes = np.empty([len(times), imshape[0], imshape[1]])
        realFrames = np.empty([len(times), imshape[0], imshape[1]])
        
        for i in range(len(times)):
            realFrames[i] = 2*tifffile.imread(f'{self.save_loc}\\time_sequence\\{times[i]}.tiff') - 1
            Iframes[i] = np.array(np.abs(np.fft.fftshift(np.fft.fft2(realFrames[i])))**2, dtype = np.float32) # Compute the fft
            # ~ Iframes[i] = np.array(np.abs((np.fft.fft2(realFrames[i])))**2, dtype = np.float32) # Compute the fft
        
        if plot_real:
            fig_real, ax_real = plt.subplots()
            
            real_display_obj = ax_real.imshow(realFrames[0],cmap = 'grey', vmin = -1, vmax = 1)
            plt.imshow(self.C_mask, cmap = 'cool', alpha = np.abs(self.C_mask))
            
            # Build sliders for adjusting the brightness and contrast of the image
            ax_real_setVals = [plt.axes([0.15, 0.02, 0.5, 0.02])]
            
            imagesMin, imagesMax = 0, np.max(Iframes[:,imshape[0]//2+2,imshape[1]//2+2])
            slider_image_real = matplotlib.widgets.Slider(ax_real_setVals[0], r'image #', 0, len(times)-1, valinit=0, valfmt="%i")
                    
            def sliderUpdateImage_real(val):
                real_display_obj.set_data(realFrames[int(val)])
                fig.canvas.draw()
                
            slider_image_real.on_changed(sliderUpdateImage_real)
        
        if plot_fft:
            fig_fft, ax_fft = plt.subplots()
            
            fft_display_obj = ax_fft.imshow(Iframes[0])
            
            # Build sliders for adjusting the brightness and contrast of the image
            ax_fft_setVals = [plt.axes([0.15, 0.1, 0.5, 0.02]), plt.axes([0.15, 0.06, 0.5, 0.02]), plt.axes([0.15, 0.02, 0.5, 0.02])]
            
            imagesMin, imagesMax = 0, np.max(Iframes[:,imshape[0]//2+2,imshape[1]//2+2])
            slider_vmax_fft = matplotlib.widgets.Slider(ax_fft_setVals[0], r'$v_{max}$', imagesMin, imagesMax, valinit=imagesMax)
            slider_vmin_fft = matplotlib.widgets.Slider(ax_fft_setVals[1], r'$v_{min}$', imagesMin, imagesMax, valinit=imagesMin)
            slider_image_fft = matplotlib.widgets.Slider(ax_fft_setVals[2], r'image #', 0, len(times)-1, valinit=0, valfmt="%i")
            
            def sliderUpdateVmax_fft(val):
                if val > fft_display_obj.get_clim()[0]:
                    fft_display_obj.set_clim(fft_display_obj.get_clim()[0], val)
                else:
                    slider_vmax_fft.set_val(fft_display_obj.get_clim()[0]+1)
            def sliderUpdateVmin_fft(val):
                if val < fft_display_obj.get_clim()[1]:
                    fft_display_obj.set_clim(val, fft_display_obj.get_clim()[1])
                else:
                    slider_vmin_fft.set_val(fft_display_obj.get_clim()[1]-1)
                    
            def sliderUpdateImage_fft(val):
                fft_display_obj.set_data(Iframes[int(val)])
                fig_fft.canvas.draw()
                
            slider_vmax_fft.on_changed(sliderUpdateVmax_fft)
            slider_vmin_fft.on_changed(sliderUpdateVmin_fft)
            slider_image_fft.on_changed(sliderUpdateImage_fft)
        
        magnetization = np.average(realFrames, axis = (1,2))
        fig, ax = plt.subplots()
        ax.plot(magnetization)
        ax.set_ylabel('Magnetization')
        ax.set_xlabel('Time')
        
        # Gaussian filter the images
        Iframes_avg_gf = gaussian_filter(np.mean(Iframes, axis = 0), 1)
        # ~ Iframes_flat = Iframes/Iframes_avg_gf
        Iframes_flat = Iframes
        
        # Now compute the g2
        q_radius = 3
        N = 3
        xcentroid = im_0.shape[1]//2
        ycentroid = im_0.shape[0]//2

        g2_all = []
        colormap = np.zeros(Iframes.shape[1:3], dtype = np.int64)
        
        for i in range(N):
            #data = Icrop.copy()/(np.sum(Icrop, axis = (1,2))[:,None,None]) # Normalized images
            data = Iframes_flat.copy()/(np.sum(Iframes_flat, axis = (1,2))[:,None,None]) # Normalized images
            #data = Icrop_flat.copy()
            
            labeled_roi_array = getAnnulusMask(data[0],xcentroid,ycentroid,q_radius,i,i+1)
    
            colormap = colormap + labeled_roi_array*(i+1)
            
            #XPCS parameters and run calculation
            num_levels = 10
            num_bufs = 12
            g2n, lag_steps = corr.multi_tau_auto_corr(num_levels, num_bufs, labeled_roi_array, data[1:])
            
            g2_all.append(g2n[1:])
    
        # Convert g2_all to a sensible shape and structure
        g2_all = np.array(g2_all)[:,:,0]
        
        data_to_save = np.empty([g2_all.shape[0] + 1, g2_all.shape[1]])
        data_to_save[0,:] = lag_steps[1:]
        data_to_save[1:,:] = g2_all
        # Save the g2 values from the run
        np.savetxt(f'{self.save_loc}\\g2.csv', data_to_save, delimiter = ',', header = f'q_radius = {q_radius}')
            
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(np.log(np.mean(Iframes, axis = 0) + 1))
        axs[0].set_title('Peak, Log Scale')
        axs[1].imshow(colormap, cmap = cm.jet)
        axs[1].set_title('ROI Regions')
        #plt.imshow(labeled_roi_array,alpha=0.2,cmap=cm.gray)
    
    
        #################################
        fig, ax = plt.subplots(2)
        print(np.shape(g2_all))
        #F = np.sqrt((g2_all-1)/(g2_all[:,:2].mean(axis = 1)[:, None] - 1))
        F = np.sqrt((g2_all-1)/(g2_all[:,0][:, None] - 1))
    
        for i in range(F.shape[0]):
            ax[0].scatter(lag_steps[1:], F[i], label = f'q_radius = {i*q_radius}')
            ax[1].scatter(lag_steps[1:], g2_all[i], label = f'q_radius = {i*q_radius}')
            #plt.semilogx(lag_steps[1:], g2_all[i], label = f'q_radius = {i*q_radius}')
        
            # ~ popt, pcov = opt.curve_fit(sc_exponential, lag_steps[1:], F[i])
            # ~ tt = np.linspace(min(lag_steps[1:]), max(lag_steps[1:],1000))
            # ~ yy = sc_exponential(tt, *popt)
            # ~ ax.plot(tt, yy)
        
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        # ~ ax.set_xlim([0,100])
        ax[0].set_xlabel(r'$\tau$ (s)')
        ax[0].set_ylabel(r'$F$')
        plt.legend()
        plt.show()
    


import skbeam
