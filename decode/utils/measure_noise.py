import matplotlib.pyplot as plt
from pathlib import Path
import decode
from decode.utils.img_file_io import read_img, write_img
import numpy
import edt
import torch
import skimage.io
from scipy import stats

import pickle

# calculate and save to file, the background noise as measured from real reference images
def measure_noise(
    as_photons:bool=True, # save result in photon units (not camera units), so that it can be sampled directly
    normalize:bool=True, # normalize area below curve to 1 for plotting
    bin_width:float=0.3, # bin values for discretization (from arguably already discrete brightness values measured..)
    max_x:float=200.0, # max brightness of range that is saved to disk (is asserted to actually be at least as large as real max brightness)
    eval_580:bool=False, #cherry (otherwise venus/515),
    i_display_threshold:int=5
):
    # a set of images from a brightness reference measurement experiment
    labeled_mask:numpy.ndarray=read_img(f"/mnt/big_data/code/test_decode/brightness_reference/bg_only_mock/Pos6/warped_dist_masks_{'580' if eval_580 else '515'}/dist_mask_00000000.tiff",from_dtype="u8",to_dtype="u8")
    fluor_img:numpy.ndarray=read_img(f"/mnt/big_data/code/test_decode/brightness_reference/bg_only_mock/Pos6/fluor_cropped_{'580' if eval_580 else '515'}/img_000000000.tiff",from_dtype="u16",to_dtype="u16")

    # a set of images from a real (?) experiment, including dots (probably not useful for background brightness measurement)
    #labeled_mask:numpy.ndarray=read_img(f"experiments/experiment_00000002/Pos006/warped_dist_masks{'_580' if eval_580 else ''}/dist_mask_00000043.tiff",from_dtype="u8",to_dtype="u8")
    #fluor_img:numpy.ndarray=read_img(f"experiments/experiment_00000002/Pos006/fluor_cropped{'_580' if eval_580 else ''}/img_000000043.tiff",from_dtype="u16",to_dtype="u16")

    assert labeled_mask.shape==fluor_img.shape

    labeled_mask=labeled_mask>0

    dist_mask = edt.edt(labeled_mask, order='C', parallel=10)
    dist_mask=dist_mask.astype(dtype=numpy.uint8)

    if as_photons:
        param = decode.utils.param_io.ParamHandling().load_params("/mnt/big_data/code/test_decode/konrads_params.yaml")
        camera = decode.simulation.camera.Photon2Camera.parse(param)
        fluor_img=camera.backward(torch.from_numpy(fluor_img.astype(dtype=numpy.int32)), device='cpu').numpy()
        # rescale x-axis appropriately
        max_x=camera.backward(torch.tensor([max_x])).cpu().item()

    fmin=fluor_img.min()
    fmean=fluor_img.mean()
    fmax=fluor_img.max()

    assert fmax<=max_x,f"{fmax} - {max_x}"

    #print(f"min,mean,max: {fmin, fmean, fmax}")
    print(f"bg fraction {fluor_img[dist_mask==0].reshape((-1,)).shape[0]/(labeled_mask.shape[0]*labeled_mask.shape[1])}")

    background_distribution={'as_photons':as_photons,'max_x':max_x}

    plt.figure(figsize=(20,10))
    colors=["red","blue","pink","purple","black","green","magenta"]
    for i in range(dist_mask.min(),dist_mask.max()+1):
        pixels_at_dist=fluor_img[dist_mask==i].reshape((-1,))

        dist_fmin=pixels_at_dist.min()
        dist_fmean=pixels_at_dist.mean()
        dist_fmax=pixels_at_dist.max()

        values=pixels_at_dist
        #test_statistic=stats.kstest(values,cdf='poisson',args=(dist_fmean,))
        #print(f"poisson distribution is {'rejected' if test_statistic.statistic>0.05 else 'accepted'} ; mean = {dist_fmean} ; test = {test_statistic}") # could use thise to compare distribution of real/simulated background noise?
        
        print(f"λ ({i}) = {dist_fmean}")
        #plt.axvline(dist_fmean,color=colors[i],linestyle="dashed")

        random_photons=numpy.random.poisson(lam=dist_fmean,size=pixels_at_dist.shape[0])
        random_photons=random_photons+numpy.random.normal(loc=0,scale=1.5,size=random_photons.shape) # read noise

        bins=numpy.arange(0,max_x,bin_width)
        hist=numpy.histogram(values,bins=bins,density=normalize)[0]
        hist_random=numpy.histogram(random_photons,bins=bins,density=normalize)[0]

        hist_not_zero=hist>0

        background_distribution[i]=hist[hist_not_zero].copy()
        background_distribution[f"{i}_mean"]=float(pixels_at_dist.mean())

        bins=bins[:-1]
        hist=hist[hist_not_zero]

        hist_random=hist_random/hist_random.max()*hist.max()

        if normalize and i>i_display_threshold: # do not display data for indices that are known to have few values (which looks weird on the graph when normalized)
            continue
        else:
            plt.plot(bins[hist_not_zero],hist,color=colors[i%len(colors)],label=f"inside cell ({i} to outline)" if i>0 else "flowcell")
            if False:
                plt.plot(bins[hist_random>0],hist_random[hist_random>0],color=colors[i%len(colors)],label=f"inside cell ({i} to outline)" if i>0 else "flowcell",linestyle="dashed")
            else:
                plt.axvline(dist_fmean,color=colors[i%len(colors)],label=f"inside cell ({i}) - mean" if i>0 else "flowcell - mean",linestyle="dashed")

            
    plt.title("histogram of (fluorescence) pixel values on flowcell and per distance to cell outline")
    plt.xlabel(f"pixel value ({'photons' if as_photons else 'ADU'})")
    plt.xlim(0)
    plt.ylabel("PDF")
    plt.ylim(0)
    plt.legend()
    plt.tight_layout()
    plt.show()

    file_name=f"/mnt/big_data/code/test_decode/background_distribution_{'580' if eval_580 else '515'}.pickle"
    with open(file_name,"wb") as pickle_file:
        pickle.dump(background_distribution,pickle_file)

if __name__=="__main__":
    # measure noise for venus first, then for cherry
    measure_noise(
        as_photons=True,
        normalize=True,
        bin_width=0.3,
        max_x=400.0,
        eval_580=False, # False => venus
    )
    measure_noise(
        as_photons=True,
        normalize=True,
        bin_width=0.3,
        max_x=400.0,
        eval_580=True, # True => cherry
    )