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


    if not param.Simulation.full_frame_psf: # simplifies params.yaml file slightly
        param.Simulation.psf_extent[0][1]=param.Simulation.img_size[0]
        param.Simulation.psf_extent[1][1]=param.Simulation.img_size[1]
        print("using frame-wise psf simulation")
    else:
        print("using full-frame psf simulation")

    if param.TestSet.frame_extent is None: # must be None or tuple of tuples
        param.TestSet.frame_extent=((0,param.Simulation.img_size[0]),(0,param.Simulation.img_size[1]))

    if param.TestSet.img_size is None: # must be none or tuple (tuple must be equal to param.Simulation.img_size)
        param.TestSet.img_size=param.Simulation.img_size

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
        num_frames=frame_range_test[1],
        noise=None)#noise)

    prior_train = decode.simulation.emitter_generator.MaskedEmitterSampler.parse(
        param, 
        structure=prior_struct, 
        num_frames=frame_range_train[1],
        noise=None)#noise)

    """ background """

    if param.Simulation.background.type=="uniform":
        print(f"using {param.Simulation.background.type} background distribution")

        param.Simulation.bg_uniform=param.Simulation.background.range

        bg=decode.simulation.background.UniformBackground.parse(param)

    elif param.Simulation.background.type=="masked":
        print(f"using {param.Simulation.background.type} background distribution")

        bg_lower_bound=param.Simulation.background.flowcell if isinstance(param.Simulation.background.flowcell,(int,float)) else param.Simulation.background.flowcell[0]
        bg_upper_bound=param.Simulation.background.cell if isinstance(param.Simulation.background.cell,(int,float)) else param.Simulation.background.cell[1]
        param.Simulation.bg_uniform=(bg_lower_bound,bg_upper_bound+bg_lower_bound)# add for real upper bound because of MaskedBackground internals

        bg=decode.simulation.background.MaskedBackground.parse(param)

    elif param.Simulation.background.type=="discrete":
        print(f"using {param.Simulation.background.type} background distribution")
        
        bg=decode.simulation.background.DiscreteBackground.parse(param)

        param.Simulation.bg_uniform=bg._bg_uniform()
    else:
        raise ValueError("param.simulation,background.type must be in [uniform, masked, discrete]")

    """ noise model """

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
        img_shape=(param.Simulation.psf_extent[0][1],param.Simulation.psf_extent[1][1]), # img_shape and psf_extent are always equal
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
        also_yield_fluorescence=param.Simulation.also_yield_fluorescence,
        device=param.Hardware.device_simulation)

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
        also_yield_fluorescence=param.Simulation.also_yield_fluorescence,
        device=param.Hardware.device_simulation)

    param.Simulation.psf_extent[0][1]=param.Simulation.img_size[0]
    param.Simulation.psf_extent[1][1]=param.Simulation.img_size[1]

    frame_size_fraction=(param.Simulation.img_size[0]*param.Simulation.img_size[1])/(40*40) # rescale <brightness threshold, other things> to match new image size (more precisely, brightness seems to be distributed across all emitters, not across the whole frame, so this parameter should better be rescaled with the [average] number of emitters per frame, though this number is proportional to the frame size)
    param.PostProcessingParam.raw_th/=frame_size_fraction
    param.HyperParameter.max_number_targets=int(param.HyperParameter.max_number_targets*frame_size_fraction)
    param.Simulation.emitter_av*=frame_size_fraction # probably not required

    return simulation_train, simulation_test
