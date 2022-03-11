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

if __name__=="__main__":
    as_photons:bool=True
    show_bg:bool=True
    normalize:bool=True
    bin_width:float=0.3
    max_x:float=200.0
    binary_mask:bool=False
    eval_580:bool=False #cherry (otherwise venus/515)
    show_alignment:bool=False
    i_threshold:int=5

    if eval_580:
        labeled_mask:numpy.ndarray=read_img("brightness_reference/bg_only/Pos6/warped_dist_masks_580/dist_mask_00000043.tiff",from_dtype="u8",to_dtype="u8")
        fluor_img:numpy.ndarray=read_img("brightness_reference/bg_only/Pos6/fluor_cropped_580/img_000000043.tiff",from_dtype="u16",to_dtype="u16")
    else:
        labeled_mask:numpy.ndarray=read_img("brightness_reference/bg_only/Pos6/warped_dist_masks/dist_mask_00000043.tiff",from_dtype="u8",to_dtype="u8")
        fluor_img:numpy.ndarray=read_img("brightness_reference/bg_only/Pos6/fluor_cropped/img_000000043.tiff",from_dtype="u16",to_dtype="u16")

    if show_alignment:
        fluor_img[labeled_mask>0]=0

        plt.imshow(fluor_img,cmap="gist_gray")
        plt.show()

        raise Error

    assert labeled_mask.shape==fluor_img.shape

    labeled_mask=labeled_mask>0

    if binary_mask:
        dist_mask=labeled_mask
    else:
        dist_mask = edt.edt(labeled_mask, order='C', parallel=10)
    dist_mask=dist_mask.astype(dtype=numpy.uint8)

    if as_photons:
        param = decode.utils.param_io.ParamHandling().load_params("konrads_params.yaml")
        camera = decode.simulation.camera.Photon2Camera.parse(param)
        fluor_img=camera.backward(torch.from_numpy(fluor_img.astype(dtype=numpy.int32)), device='cpu').numpy()
        # rescale x-axis appropriately
        max_x=camera.backward(torch.tensor([max_x])).cpu().item()

    fmin=fluor_img.min()
    fmean=fluor_img.mean()
    fmax=fluor_img.max()

    #print(f"min,mean,max: {fmin, fmean, fmax}")
    print(f"bg fraction {fluor_img[dist_mask==0].reshape((-1,)).shape[0]/(labeled_mask.shape[0]*labeled_mask.shape[1])}")

    background_distribution={'as_photons':as_photons,'max_x':max_x}

    plt.figure(figsize=(20,10))
    colors=["red","blue","pink","purple","black","green","magenta"]
    for i in range(dist_mask.min(),dist_mask.max()+1):
        if i==0 and not show_bg:
            continue
        
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

        if normalize and i>i_threshold: # remove indices that are known to have few values (which looks weird on the graph when normalized)
            continue

        bins=numpy.arange(0,max_x,bin_width)
        hist=numpy.histogram(values,bins=bins,density=normalize)[0]
        hist_random=numpy.histogram(random_photons,bins=bins,density=normalize)[0]

        hist_not_zero=hist>0

        background_distribution[i]=hist[hist_not_zero].copy()
        background_distribution[f"{i}_mean"]=float(pixels_at_dist.mean())

        bins=bins[:-1]
        hist=hist[hist_not_zero]

        hist_random=hist_random/hist_random.max()*hist.max()

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

    file_name=f"background_distribution_{'580' if eval_580 else '515'}.pickle"
    with open(file_name,"wb") as pickle_file:
        pickle.dump(background_distribution,pickle_file)

    # for each level of edt, calc histogram over all fluorescence image pixel values (probably remove top and bottom 0.1 - 1%): min, max, avg -> correlation of avg edt ~ edt?