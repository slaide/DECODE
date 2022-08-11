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

    if param.Simulation.mode in ("acquisition", "parallel"):
        frame_range_train = (0, param.HyperParameter.pseudo_ds_size) # sample this many frames
    else:
        raise ValueError
        
    frame_range_test = (0, param.TestSet.test_size)

    """ emitter generators from structure prior"""

    prior_test = decode.simulation.emitter_generator.MaskedEmitterSampler.parse(
        param, 
        num_frames=frame_range_test[1])

    prior_train = decode.simulation.emitter_generator.MaskedEmitterSampler.parse(
        param, 
        num_frames=frame_range_train[1])

    """ noise model """

    if param.CameraPreset is not None:
        raise NotImplementedError
    else:
        noise = decode.simulation.camera.Photon2Camera.parse(param)

    """ setup psf from PSF calibration file """

    psfs=[decode.utils.calibration_io.SMAPSplineCoefficient(
        params=psf
    ).init_spline(
        xextent=param.Simulation.psf_extent_img[0],
        yextent=param.Simulation.psf_extent_img[1],
        img_shape=(param.Simulation.psf_extent_img[0][1],param.Simulation.psf_extent_img[1][1]), # img_shape and psf_extent are always equal
        device=param.Hardware.device_simulation,
        roi_size=param.Simulation.roi_size,
        roi_auto_center=param.Simulation.roi_auto_center
    ) for psf in param.InOut.psfs]

    """ setup simulation for training"""

    simulation_train = decode.simulation.simulator.MaskedSimulation(
        segmentation_masks_glob=param.InOut.segmentation_masks, 
        psf=psfs, 
        em_sampler=prior_train, 
        noise=noise, 
        num_frames=frame_range_train[1],
        frame_size=param.Simulation.img_size,
        device=param.Hardware.device_simulation,
        background_args=param.Simulation.background)

    """ setup simulation for testing """

    simulation_test = decode.simulation.simulator.MaskedSimulation(
        segmentation_masks_glob=param.InOut.segmentation_masks, 
        psf=psfs, 
        em_sampler=prior_test, 
        noise=noise,
        num_frames=frame_range_test[1],
        frame_size=param.Simulation.img_size,
        device=param.Hardware.device_simulation,
        background_args=param.Simulation.background)

    return simulation_train, simulation_test
