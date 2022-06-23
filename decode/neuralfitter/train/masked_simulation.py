import decode.simulation
import decode.utils
from decode.utils.types import RecursiveNamespace
import decode.utils.read_params

def setup_masked_simulation(param):
    """
        Setup the actual simulation

        0. Define PSF function (load the calibration)
        1. Define our struture from which we sample (random prior in 3D) and its photophysics
        2. Define background and noise
        3. Setup simulation and datasets
        """

    """ frame ranges """

    if not isinstance(param,RecursiveNamespace):
        param=read_params(param)

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

    """ background """

    bg=decode.simulation.background.DiscreteBackground.parse(param)

    # TODO rescale input how?
    param.Simulation.bg_uniform=bg._bg_uniform()

    """ noise model """

    if param.CameraPreset is not None:
        raise NotImplementedError
    else:
        noise = decode.simulation.camera.Photon2Camera.parse(param)

    """ setup psf from PSF calibration file """

    psf = decode.utils.calibration_io.SMAPSplineCoefficient(
        calib_file=param.InOut.calibration_file
    ).init_spline(
        xextent=param.Simulation.psf_extent_img[0],
        yextent=param.Simulation.psf_extent_img[1],
        img_shape=(param.Simulation.psf_extent_img[0][1],param.Simulation.psf_extent_img[1][1]), # img_shape and psf_extent are always equal
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
        device=param.Hardware.device_simulation)

    frame_size_fraction=(param.Simulation.img_size[0]*param.Simulation.img_size[1])/(40*40) # rescale <brightness threshold, other things> to match new image size (more precisely, brightness seems to be distributed across all emitters, not across the whole frame, so this parameter should better be rescaled with the [average] number of emitters per frame, though this number is proportional to the frame size)
    param.PostProcessingParam.raw_th/=frame_size_fraction
    param.Simulation.emitter_av*=frame_size_fraction # probably not required
    param.HyperParameter.max_number_targets=int(param.HyperParameter.max_number_targets*param.Simulation.emitter_av)

    return simulation_train, simulation_test
