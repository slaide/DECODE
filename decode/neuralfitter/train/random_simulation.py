import decode.simulation
import decode.utils


def setup_random_simulation(param):
    """
        Setup the actual simulation

        0. Define PSF function (load the calibration)
        1. Define our struture from which we sample (random prior in 3D) and its photophysics
        2. Define background and noise
        3. Setup simulation and datasets
        """

    psf = decode.utils.calibration_io.SMAPSplineCoefficient(
        calib_file=param.InOut.calibration_file).init_spline(
        xextent=param.Simulation.psf_extent[0],
        yextent=param.Simulation.psf_extent[1],
        img_shape=param.Simulation.img_size,
        device=param.Hardware.device_simulation,
        roi_size=param.Simulation.roi_size,
        roi_auto_center=param.Simulation.roi_auto_center
    )

    """Structure Prior"""
    prior_struct = decode.simulation.structure_prior.RandomStructure.parse(param) # TODO modify this to allow for custom structures to sample emitter positions

    if param.Simulation.mode in ('acquisition', 'apriori'): # TODO look into this frame range specification more (since we probably only want to simulate/predict a single frame)
        frame_range_train = (0, param.HyperParameter.pseudo_ds_size)

    elif param.Simulation.mode == 'samples':
        frame_range_train = (-((param.HyperParameter.channels_in - 1) // 2),
                             (param.HyperParameter.channels_in - 1) // 2)
    else:
        raise ValueError

    prior_train = decode.simulation.emitter_generator.EmitterSamplerBlinking.parse( # TODO implement a non-blinking emitter (since we do not have blinking emitters in our experiments)
        param, structure=prior_struct, frames=frame_range_train)

    """Define our background and noise model."""
    bg = decode.simulation.background.UniformBackground.parse(param) # TODO implement a background simulation that samples two different background noise models, based on a cell segmentation mask

    if param.CameraPreset == 'Perfect':
        noise = decode.simulation.camera.PerfectCamera.parse(param)
    elif param.CameraPreset is not None:
        raise NotImplementedError
    else:
        noise = decode.simulation.camera.Photon2Camera.parse(param)

    simulation_train = decode.simulation.simulator.Simulation(psf=psf, em_sampler=prior_train, background=bg,
                                                              noise=noise, frame_range=frame_range_train)

    frame_range_test = (0, param.TestSet.test_size)

    prior_test = decode.simulation.emitter_generator.EmitterSamplerBlinking.parse( # TODO change this to use the same emitter sampler used above (for training data generation)
        param, structure=prior_struct, frames=frame_range_test)

    simulation_test = decode.simulation.simulator.Simulation(psf=psf, em_sampler=prior_test, background=bg, noise=noise,
                                                             frame_range=frame_range_test)

    return simulation_train, simulation_test
