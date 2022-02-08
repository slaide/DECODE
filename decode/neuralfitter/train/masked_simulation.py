import decode.simulation
import decode.utils


def setup_masked_simulation(param):
    """
        Setup the actual simulation

        0. Define PSF function (load the calibration)
        1. Define our struture from which we sample (random prior in 3D) and its photophysics
        2. Define background and noise
        3. Setup simulation and datasets
        """

    """ frame ranges """

    if not param.Simulation.full_frame_psf:
        param.Simulation.psf_extent[0][1]=param.Simulation.img_size[0]
        param.Simulation.psf_extent[1][1]=param.Simulation.img_size[1]
        print("using frame-wise psf simulation")
    else:
        print("using full-frame psf simulation")

    if param.Simulation.mode in ("acquisition", "apriori"):
        frame_range_train = (0, param.HyperParameter.pseudo_ds_size) # sample this many frames
    else:
        raise ValueError
        
    frame_range_test = (0, param.TestSet.test_size)

    """ Structure Prior """

    prior_struct = decode.simulation.structure_prior.CellMaskStructure.parse(param)

    """ emitter generators from structure prior"""

    prior_test = decode.simulation.emitter_generator.MaskedEmitterSampler.parse(
        param, 
        structure=prior_struct, 
        num_frames=frame_range_test[1])

    prior_train = decode.simulation.emitter_generator.MaskedEmitterSampler.parse(
        param, 
        structure=prior_struct, 
        num_frames=frame_range_train[1])

    """ background and noise model """

    bg = decode.simulation.background.MaskedBackground.parse(param)

    if param.CameraPreset is not None:
        raise NotImplementedError
    else:
        noise = decode.simulation.camera.Photon2Camera.parse(param)

    """ setup psf from PSF calibration file """

    psf = decode.utils.calibration_io.SMAPSplineCoefficient(
        calib_file=param.InOut.calibration_file
    ).init_spline(
        xextent=param.Simulation.psf_extent[0],
        yextent=param.Simulation.psf_extent[1],
        img_shape=(param.Simulation.psf_extent[0][1],param.Simulation.psf_extent[1][1]) if param.Simulation.full_frame_psf else param.Simulation.img_size,
        device=param.Hardware.device_simulation,
        roi_size=param.Simulation.roi_size,
        roi_auto_center=param.Simulation.roi_auto_center
    )

    """ setup simulation for training"""

    simulation_train = decode.simulation.simulator.MaskedSimulation(
        root_experiments_folder=param.InOut.root_experiments_folder, 
        psf=psf, 
        em_sampler=prior_train, 
        background=bg, 
        noise=noise, 
        num_frames=frame_range_train[1],
        frame_size=param.Simulation.img_size,
        full_frame_psf=param.Simulation.full_frame_psf,
        also_yield_fluorescence=param.Simulation.also_yield_fluorescence)

    """ setup simulation for testing """

    simulation_test = decode.simulation.simulator.MaskedSimulation(
        root_experiments_folder=param.InOut.root_experiments_folder, 
        psf=psf, 
        em_sampler=prior_test, 
        background=bg, 
        noise=noise,
        num_frames=frame_range_test[1],
        frame_size=param.Simulation.img_size,
        full_frame_psf=param.Simulation.full_frame_psf,
        also_yield_fluorescence=param.Simulation.also_yield_fluorescence)

    return simulation_train, simulation_test
