import skimage
import skimage.io
from skimage.measure import regionprops

import numpy
from numpy.random import default_rng

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision # for torchvision.transforms.Compose

import decode
from decode.utils.narsil.utils.transforms import resizeOneImage, tensorizeOneImage
from decode.utils.narsil.segmentation.run import loadNet

import edt
from pathlib import Path

from scipy.io import loadmat # loads a .mat file (matlab binary data)
import scipy.ndimage
import cv2 as cv # for cv.warpPerspective

from typing import Tuple, List, Optional

from time import perf_counter
from tqdm import tqdm

import sys
import glob

import matplotlib.pyplot as plt

numpy_rng=default_rng(seed=3339337589)

""" iterable structure to load data into a network """
class SegmentDirectory(Dataset):
    """ normalize and load all image files in directory """
    def __init__(self, directory, transform = None, flip = False, out_dir=None):
        self.directory = Path(directory)
        self.transform = transform
        self.out_dir=out_dir
        self.indices = [
            (i_batch,filename)
            for i_batch,filename 
            in enumerate(sorted(self.directory.iterdir()))
            if not filename.name.startswith(".")
            and (filename.name.endswith(".tif") or filename.name.endswith(".tiff"))
            and (True if out_dir is None else not (out_dir/f"dist_mask_{i_batch:08}.tiff").exists())
        ]
        self.n_images = len(self.indices)
        self.flip = flip

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i_batch,phase_img_path=self.indices[idx]

        phase_img = skimage.io.imread(str(phase_img_path),as_gray=True).astype(numpy.float32)/2**16
        phase_img=phase_img+numpy_rng.normal(scale=0.03,size=phase_img.shape).astype(dtype=numpy.float32)
        phase_img=phase_img.clip(0.0,1.0) # clip to [0.0,1.0] range because added noise might exceed 1.0

        phase_img_normalized = (phase_img - phase_img.mean()) / phase_img.std()

        # flips both axis! (rotates by 180 degrees)
        if self.flip:
            phase_img_normalized = numpy.flip(phase_img_normalized)
        
        if self.transform:
            phase_img_normalized = self.transform(phase_img_normalized)

        #print(phase_img_normalized.shape)

        return i_batch,phase_img_normalized

""" utility class to load an image inside a dataloader pipeline """
class rotateImageBy90Degrees(object):
    def __init__(self):
        pass
    def __call__(self, phase_image):
        return numpy.rot90(phase_image)

""" generate cell masks for a series of experiments containing pairs of phase contrast images and fluorescence images """
# this data cannot be generated on the fly because the essential part of generating the masks is an AI, which takes up a lot of vram
# the gpu does not have enough vram to run two UNet derivatives simultaneously (or to keep them in memory at the same time)
def generate_cell_masks(
        directory_list:List[str],
        fluo_roi:Tuple[Tuple[float,float],Tuple[float,float]],
        fluor_dir_name:str="fluor515",
        dir_phase:str="phase",

        fluor_cropped_out_dir_name:Optional[str]="fluor_cropped", # also crop and save the fluor image regions that were cropped for the masks
        dir_warped_dist_masks:Tuple[bool,str]=(
            False, # output folder is global target, i.e. all dist masks of all folder will be written into the same target folder (if False, output folder will be relative to folder in 'directory_list')
            "warped_dist_masks"
        ),
        warped_dist_masks_prefix:str="warped_dist_mask_",

        transmat_file_name:str="transMatV_3D.mat",

        threshold:float=0.9,
        device:str="cuda:0",
        model_path:str="mixed10epochs_betterscale_contrastAdjusted1.pth",
    ):

    y_min=fluo_roi[1][0]
    y_max=fluo_roi[1][1]
    x_min=fluo_roi[0][0]
    x_max=fluo_roi[0][1]

    net=loadNet(model_path,device)
    
    """
    create noisy phase contrast images and cell segmentation masks in advance
    
    for each experiment/directory:
        generate cell segmentation masks for all phase contrast images
        warp the fluorescence images into the shape and position of the phase contrast images
    """

    """ generate dataloader transform """
    transform = torchvision.transforms.Compose([
        rotateImageBy90Degrees(),
        tensorizeOneImage(1)])

    """ post-process segmentation mask generation network """
    def process_cell_mask(res):
        mask_pred = net_result[ds_index]>=threshold
        mask_pred = mask_pred.to("cpu").numpy().squeeze(0)

        mask_pred_labeled = skimage.measure.label(mask_pred)
        
        """ setup some parameters for mask clean-up """
        # area of one pixel in image
        pixel_area=(65e-3)**2
        #min area: 1µm (*0.7 for safety)
        min_area=0.7/pixel_area
        #max area: 3.5µm (+1µm for safety)
        max_area=4.5/pixel_area #actually currently not enforced
        #hole area threshold (upper limit of hole size that will be filled): 0.4µm
        hole_area_threshold=0.4/pixel_area
            
        mask_cleaned = skimage.morphology.remove_small_objects(mask_pred_labeled, min_size=min_area)
        mask_cleaned = skimage.morphology.remove_small_holes(mask_cleaned > 0, area_threshold=hole_area_threshold)

        """ for each region (area of pixels with value 'True' (1, or 1.0)), calculate the shortest distance to the area's border (outer cell wall) with another region. then normalize the distances, then transform the linear [0;1] space to the cell thickness at each pixel, approximated by modeling the cell as perfectly round"""
        dists = edt.edt(mask_cleaned, order='C', parallel=0)
        dists_zero_mask=dists==0

        regions=regionprops(skimage.measure.label(mask_cleaned),intensity_image=dists, cache=True)
        for cell in regions:
            minr, minc, maxr, maxc = cell.bbox

            radius=cell.intensity_image.max()
            cell_dists_zero_mask=dists_zero_mask[minr:maxr,minc:maxc]
            dists[minr:maxr,minc:maxc][cell_dists_zero_mask]=(numpy.sqrt(radius**2-(radius-cell.intensity_image)**2)*radius)[cell_dists_zero_mask]

        assert dists.dtype==numpy.float32
        
        # calc average of local neighborhood as better approximation of cell volume
        dists_smoothed=scipy.ndimage.convolve(dists,numpy.ones((3,3))/9)
        # remove cell volume that 'leaked' out of cell by calculating of local mean for pixel outsid
        dists_smoothed[dists_zero_mask]=0

        cell_mask_img=numpy.rot90(dists_smoothed,k=-1)
        assert cell_mask_img.sum()>0

        return cell_mask_img

    """ load phase contrast/fluorescence alignment transformation matrix """
    # transpose because opencv coordinate system works different from matlab
    transformation_matrix=loadmat(transmat_file_name)["transformationMatrix"].T
    transformation_matrix_inv=numpy.linalg.inv(transformation_matrix)
    
    directory_list=[Path(directory) for directory in sorted(directory_list)]
    experiment_directory_list=[directory for directory in directory_list if directory.is_dir() and not directory.name.startswith(".")]

    dir_warped_dist_mask_is_global=dir_warped_dist_masks[0]
    if dir_warped_dist_mask_is_global:
        num_warped_dist_mask=0

    segmentation_task_list=[]

    for directory in tqdm(experiment_directory_list,desc="dir",unit="dir"):
        if not fluor_cropped_out_dir_name is None:
            cropped_fluor_dir=directory/fluor_cropped_out_dir_name # is output
            if not cropped_fluor_dir.exists():
                cropped_fluor_dir.mkdir()

        if dir_warped_dist_mask_is_global:
            warped_masks_dir=Path(dir_warped_dist_masks[1]) # is output
        else:
            warped_masks_dir=directory/dir_warped_dist_masks[1] # is output
        if not warped_masks_dir.exists():
            warped_masks_dir.mkdir()
            
        phase_dir=directory/dir_phase # is input
        phase_images=[x for x in sorted(phase_dir.iterdir()) if x.is_file() and not x.name.startswith(".")]
        num_phase_images=len(phase_images)
        
        fluor_dir=directory/fluor_dir_name # is input
        fluor_images=[x for x in sorted(fluor_dir.iterdir()) if x.is_file() and not x.name.startswith(".")]
        num_fluor_images=len(fluor_images)
        
        assert num_phase_images>0,"no phase contrast images found. either folder is empty, or something went wrong"
        assert num_fluor_images>0,"no fluorescence images found. either folder is empty, or something went wrong"

        assert num_fluor_images>=num_phase_images,"there must be at least one fluorescence image per phase contrast image"
        
        # there can be a multiple of fluor images per phase contrast image (i.e. take fluor image every minute, but phase contrast only every 2 minutes)
        fluorescence_per_phase=num_fluor_images//num_phase_images
        
        assert num_fluor_images%num_phase_images==0,f"{str(directory)} {num_fluor_images} {num_phase_images}"

        """ apply segmentation network to phase contrast images """

        dataset = SegmentDirectory(phase_dir, 
            transform=transform, flip=True,
            out_dir=warped_masks_dir if not dir_warped_dist_mask_is_global else None) # this is used to skip files that have already been processed (not possible with global output folder)

        batch_size=1 # thats all that fits into 11GB vram
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        for i_batch, data in tqdm(dataloader,desc="dist+crop",leave=False,unit="img"):
            with torch.no_grad():
                phase = data.to(device)
                
                net_result=net(phase)

            for ds_index in range(batch_size):
                cell_mask_img=process_cell_mask(net_result[ds_index])

                """ align segmentation mask with fluorescence image, and crop both to the relevant ROI """
                if dir_warped_dist_mask_is_global:
                    cropped_warped_mask_path=warped_masks_dir/f"{warped_dist_masks_prefix}{num_warped_dist_mask:08}.tiff"
                    num_warped_dist_mask+=1
                else:
                    cropped_warped_mask_path=warped_masks_dir/f"{warped_dist_masks_prefix}{i_batch[ds_index]:08}.tiff"
                fluor_image_path=fluor_images[i_batch[ds_index]*fluorescence_per_phase]
                if not fluor_cropped_out_dir_name is None:
                    cropped_fluor_path=cropped_fluor_dir/fluor_image_path.name

                cropped_fluor_gen_required=(not fluor_cropped_out_dir_name is None) and (not cropped_fluor_path.exists())
                if cropped_fluor_gen_required or not cropped_warped_mask_path.exists():
                    # transform phase contrast image to be aligned with the fluorescence image
                    fluor_image=skimage.io.imread(fluor_image_path,as_gray=True)

                    if not cropped_warped_mask_path.exists():
                        warped_mask=numpy.zeros_like(fluor_image,dtype=numpy.float32)
                        warped_mask=cv.warpPerspective(cell_mask_img,transformation_matrix_inv,(fluor_image.shape[1],fluor_image.shape[0]),dst=warped_mask)
                        cropped_warped_mask=warped_mask[y_min:y_max,x_min:x_max]

                        #assert cropped_warped_mask.any(), "warping mask image resulted in black image (all pixel values 0)"
                        
                        # zlib compression is losless, and can reduce the file size significantly (e.g. 2.4MB -> 330kB)
                        skimage.io.imsave(cropped_warped_mask_path, cropped_warped_mask, plugin="tifffile",compression="zlib")
                    
                    if cropped_fluor_gen_required:
                        # write cropped fluorescence that corresponds to the image region the phase contrast image was mapped onto
                        cropped_fluor=fluor_image[y_min:y_max,x_min:x_max]
                        skimage.io.imsave(cropped_fluor_path, cropped_fluor, plugin="tifffile",compression="zlib")

    # clear vram
    del net
    torch.cuda.empty_cache()