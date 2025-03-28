Monte Carlo simulation of a 3D quenched disorder Ising Model using Glauber Dynamics.

To get started with a simulation, you must have a field disorder and mass disorder (i.e. Tc disorder) distribution saved in the sim_settings directory.
These are 32 bit voxel images stored as .tiff files. There is an additional "mask" file that can be just left as an image filled with "1s".
So the files in sim_settings should be field.tiff, mass.tiff, and mask.tiff

All these voxel images must be the same size. Their size determines the size of the simulation grid.

If the field and mass voxel images are fully zeros, then the system is a pristine Ising model.
For the field distribution, + or - values pin + and - magnetizations respectively through a linear coupling to psi.
For the mass distribution, + or - values raise and lower the local interaction strength respectively (A higher interaction strength increases the local Tc)

To generate new mass and field distributions, one can run the simulate_lattice_gas method in lattice_gas.py
This randomly distributes particles on a 3D lattice and then time evolves interactions between them to form clumps. The final set of particle positions is saved as pars.npy.

The methods load_and_construct_mass and load_and_construct_field then read this pars.npy file and constuct a smooth map of mass and field disorder respectively.
Multiple particle distributions can be combined with load_pars_and_combine.

Once you have the mass.tiff, field.tiff, and mask.tiff in the settings file, you can begin a simulation with the following code:

    field = tifffile.imread(f'sim_settings//field.tiff')
    mass = tifffile.imread(f'sim_settings//mass.tiff')
    
    model = Model_3d(temperature = 5.5, cooling_rate = 0,
                  step_number = 200000, initial_step_number = 5000, exposure_time = 100,
                  real_space_images_to_save = 200,
                  mass = mass, field = field)


    plotting.set_box_plots_3d(r'results\default\time_sequence',  diagF = False)
    
    analyze_g2_curves(r'results\default\g2.csv')
        

This code reads in the mass and field files and then starts a 200000 step simulation (with 5000 initial bake in steps) at a simulation exposure time of 100 steps.
The model will save 200 real space images (equally space in time) in the results\default\time_sequence directory as it runs.
Finally, when the simulation is concluded, the code will plot sove of the voxel images saved in results\default\time_sequence. Then, it will analyze the g2 data.

The reciprocal space ROIs for the computed g2 values are specified in the method measure_g2 of the class Model_3d.
