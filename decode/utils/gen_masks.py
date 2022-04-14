import skimage
import skimage.io
from skimage.measure import regionprops

import numpy
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision # for torchvision.transforms.Compose

from narsil.utils.transforms import resizeOneImage, tensorizeOneImage
from narsil.segmentation.run import loadNet

import edt
from pathlib import Path

from scipy.io import loadmat # loads a .mat file (matlab binary data)
import cv2 as cv # for cv.warpPerspective

from typing import Tuple

from decode.utils.img_file_io import read_img, write_img

from time import perf_counter
from tqdm import tqdm

""" iterable structure to load data into a network """
class SegmentDirectory(Dataset):
    """ normalize and load all image files in directory """
    def __init__(self, directory, transform = None, flip = False, save_dir=None, overwrite_data=False):
        self.directory = Path(directory)
        self.transform = transform
        self.save_dir=save_dir
        self.indices = [
            (i_batch,filename)
            for i_batch,filename 
            in enumerate(sorted(self.directory.iterdir()))
            if not filename.name.startswith(".")
            and (filename.name.endswith(".tif") or filename.name.endswith(".tiff"))
            and (True if save_dir is None else overwrite_data or not (save_dir/f"dist_mask_{i_batch:08}.tiff").exists())
        ]
        self.n_images = len(self.indices)
        self.flip = flip

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i_batch,phase_img_path=self.indices[idx]

        phase_img = read_img(str(phase_img_path),from_dtype="u16",to_dtype="float32")

        phase_img_normalized = (phase_img - phase_img.mean()) / phase_img.std()

        # flips both axis! (rotates by 180 degrees)
        if self.flip:
            phase_img_normalized = numpy.flip(phase_img_normalized)
        
        if self.transform:
            phase_img_normalized = self.transform(phase_img_normalized)

        #print(phase_img_normalized.shape)

        return i_batch,phase_img_normalized

""" apply the segmentation net to all files in a directory """
def generate_cell_segmentation_masks(phase_dir, save_dir, overwrite_data:bool=False, transform=None, device:str="cuda:0", threshold:float=0.9, remove_small_objects=None, model_path:str="mixed10epochs_betterscale_contrastAdjusted1.pth", net=None):
    if net is None:
        net=loadNet(model_path,device)
    
    saveDir=Path(save_dir)
    if not saveDir.exists():
        saveDir.mkdir()

    dataset = SegmentDirectory(phase_dir, transform=transform, flip=False, save_dir=save_dir, overwrite_data=overwrite_data)

    if len(dataset)==0:
        return

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=6)
    dataloader=tqdm(dataloader,desc="gen dist mask",leave=False)# if len(dataset)>1 else dataloader

    # zero the paramter gradients
    # forward + backward + optimize
    with torch.no_grad():
        for i_batch, data in dataloader:
            img_path=save_dir/f"dist_mask_{i_batch.item():08}.tiff"
            if not img_path.exists() or overwrite_data: # this check should be redundant
                phase = data.to(device)
                
                mask_pred = net(phase)

                # set sigmoid if the net gives # ?
                #mask_pred = torch.sigmoid(mask_pred).to("cpu").numpy().squeeze(0).squeeze(0)
                mask_pred=mask_pred.to("cpu").numpy().squeeze(0).squeeze(0)

                mask_pred = mask_pred >= threshold

                mask_pred_labeled = skimage.measure.label(mask_pred)
                
                if not remove_small_objects is None:
                    min_area=remove_small_objects["min_area"]
                    max_area=remove_small_objects["max_area"]
                    hole_area_threshold=remove_small_objects["hole_area_threshold"]
                else:
                    pixel_area=(65e-3)**2
                    #min area: 1µm (*0.7 for safety)
                    min_area=0.7/pixel_area
                    #max area: 3.5µm (+1µm for safety)
                    max_area=4.5/pixel_area #actually currently not enforced
                    #hole area threshold (upper limit of hole size that will be filled): 0.4µm
                    hole_area_threshold=0.4/pixel_area
                    
                mask_cleaned = skimage.morphology.remove_small_objects(mask_pred_labeled, min_size=min_area)
                regions=regionprops(mask_cleaned,cache=True)
                #mask_cleaned = skimage.morphology.remove_small_holes(mask_cleaned > 0, area_threshold=hole_area_threshold)
                mask_pred = mask_cleaned

                """ for each region (area of pixels with value 'True' (1, or 1.0)), calculate the shortest distance to the area's border (outer cell wall) with another region. then normalize the distances, then transform the linear [0;1] space to the cell thickness at each pixel, approximated by modeling the cell as perfectly round"""
                dists = edt.edt(mask_pred, order='C', parallel=0)
                assert (dists>0).sum()==(mask_pred>0).sum()
                # assert dists.dtype==numpy.float32
                max_dist=dists.max()

                dists[dists>0]-=0.5
                dists/=max_dist
                dists=numpy.sin(numpy.pi/2*dists)
                dists=numpy.sqrt(dists)
                assert (dists.max()-1.0)<1.0e-6
                dists*=max_dist

                dists=dists.astype(dtype=numpy.uint8)
                dists[(dists==0) & (mask_pred>0)]=1 # make sure every pixel that was part of a cell previously is one now (clamp inner-cell values to [1;max_dist])
                
                dists=numpy.rot90(dists,k=-1)

                assert (dists>0).sum()==(mask_pred>0).sum()

                write_img(img_path, dists, from_dtype="u8", as_dtype="u8")

""" utility class to load an image inside a dataloader pipeline """
class rotateImageBy90Degrees(object):
    def __init__(self):
        pass
    def __call__(self, phase_image):
        return numpy.rot90(phase_image)

""" generate cell masks for a series of experiments containing pairs of phase contrast images and fluorescence images """
# this data cannot be generated on the fly because the essential part of generating the masks is an AI, which takes up a lot of vram
# the gpu does not have enough vram to run two UNet derivatives simultaneously (or to keep them in memory at the same time)
def generate_cell_masks(root_folder:str,fluo_roi:Tuple[Tuple[float,float],Tuple[float,float]],device:str="cuda:0",
        overwrite_data:bool=False,
        threshold:float=0.9,
        fluor_dir_name:str="fluor515",
        fluor_cropped_out_dir_name:str="fluor_cropped",
        transmat_file_name:str="transMatV_3D.mat",
        dir_phase_noisy:str="phase_noisy",
        dir_phase:str="phase",
        dir_dist_masks:str="dist_masks",
        dir_warped_dist_masks:str="warped_dist_masks",
        model_path:str="mixed10epochs_betterscale_contrastAdjusted1.pth"
    ):

    experiments_folder=Path(root_folder)
    assert experiments_folder.exists()

    y_min=fluo_roi[1][0]
    y_max=fluo_roi[1][1]
    x_min=fluo_roi[0][0]
    x_max=fluo_roi[0][1]

    net=loadNet(model_path,device)
    
    """
    create noisy phase contrast images and cell segmentation masks in advance
    
    for each experiment:
        for each position:
            generate cell segmentation masks for all phase contrast images
            warp the fluorescence images into the shape and position of the phase contrast images
    """
    experiment_folder_list=[folder for folder in sorted(experiments_folder.iterdir()) if folder.is_dir() and not folder.name.startswith(".")]
    experiment_folder_list=tqdm(experiment_folder_list,desc="experiment")# if len(experiment_folder_list)>1 else experiment_folder_list
    for experiment in experiment_folder_list:
        """ generate dataloader transform once per experiment (for absolutely no reason) """
        transform = torchvision.transforms.Compose([
            rotateImageBy90Degrees(),
            tensorizeOneImage(1)])
            
        """ load possible experiment-specific phase contrast/fluorescence alignment transformation matrix """
        transformation_matrix=loadmat(str(experiment/transmat_file_name))["transformationMatrix"].T # transpose because opencv coordinate system works different from matlab
        transformation_matrix_inv=numpy.linalg.inv(transformation_matrix)
        
        total_exp_pos_time=0.0
        exp_pos_start=perf_counter()

        positions=[position for position in sorted(experiment.iterdir()) if position.is_dir() and not position.name.startswith(".")]
        positions=tqdm(positions,leave=False,desc="position")# if len(positions)>1 else positions
        for pos_index,position in enumerate(positions,start=1):
            #print(f"pos: {position}")
            
            noisy_phase_contrast_dir=position/dir_phase_noisy # contains phase contrast images, with some noise added on top to make the segmentation net work better. needs to be saved to disk because of how the dataloader currently works
            if not noisy_phase_contrast_dir.exists():
                noisy_phase_contrast_dir.mkdir()
                
            cropped_fluor_dir=position/fluor_cropped_out_dir_name
            if not cropped_fluor_dir.exists():
                cropped_fluor_dir.mkdir()
                
            phase_dir=position/dir_phase # is input
            phase_images=[x for x in sorted(phase_dir.iterdir()) if x.is_file() and not x.name.startswith(".")]
            num_phase_images=len(phase_images)
            
            fluor_dir=position/fluor_dir_name # is input
            fluor_images=[x for x in sorted(fluor_dir.iterdir()) if x.is_file() and not x.name.startswith(".")]
            num_fluor_images=len(fluor_images)
            
            assert num_phase_images>0,"no phase contrast images found. either folder is empty, or something went wrong"
            assert num_fluor_images>0,"no fluorescence images found. either folder is empty, or something went wrong"
            
            fluorescence_per_phase=num_fluor_images//num_phase_images
            
            assert num_fluor_images%num_phase_images==0,f"{str(position)} {num_fluor_images} {num_phase_images}"
            
            phase_images=tqdm(phase_images,leave=False,desc="gen fuzzy phase") if len(phase_images)>1 else phase_images
            for (phase_index,phase_contrast_image_path) in enumerate(phase_images):
                #print(f"img: {phase_contrast_image_path}")
                    
                phase_contrast_image_noisy_path=noisy_phase_contrast_dir/phase_contrast_image_path.name
                if not phase_contrast_image_noisy_path.exists() or overwrite_data:
                    phase_contrast_image=read_img(phase_contrast_image_path,from_dtype="u16",to_dtype="float32")
                    
                    # add a small bit of noise on top of the phase contrast image to avoid multiple issues with the segmentation mask (holes, dents, splits)
                    phase_contrast_image_noisy=phase_contrast_image+numpy.random.normal(scale=0.03,size=phase_contrast_image.shape).astype(dtype=numpy.float32)
                    phase_contrast_image_noisy=phase_contrast_image_noisy.clip(0.0,1.0) # clip to [0.0,1.0] range because added noise might exceed 1.0
                    
                    write_img(phase_contrast_image_noisy_path,phase_contrast_image_noisy,from_dtype="float32",as_dtype="u16")

            cell_mask_dir=position/dir_dist_masks
            if not cell_mask_dir.exists():
                cell_mask_dir.mkdir()
            
            generate_cell_segmentation_masks(
                phase_dir=noisy_phase_contrast_dir,
                save_dir=cell_mask_dir,
                overwrite_data=overwrite_data,
                transform=transform,
                threshold=threshold,
                device=device,
                net=net)

            warped_masks_dir=position/dir_warped_dist_masks
            if not warped_masks_dir.exists():
                warped_masks_dir.mkdir()
                        
            cell_masks=[x for x in sorted(cell_mask_dir.iterdir()) if x.is_file()]
                    
            cell_masks=tqdm(cell_masks,leave=False,desc="warp dist mask, crop fluo") if len(cell_masks)>1 else cell_masks
            for (mask_index,cell_mask_path) in enumerate(cell_masks):
                #print(f"cmp: {cell_mask_path}")
                
                cropped_warped_mask_path=warped_masks_dir/cell_mask_path.name
                fluor_image_path=fluor_images[mask_index*fluorescence_per_phase]
                cropped_fluor_path=cropped_fluor_dir/fluor_image_path.name

                if not cropped_warped_mask_path.exists() or not cropped_fluor_path.exists() or overwrite_data:
                    # transform phase contrast image to be aligned with the fluorescence image
                    fluor_image=read_img(fluor_image_path,from_dtype="u12",to_dtype="u12")
                    assert fluor_image.dtype==numpy.uint16

                    if not cropped_warped_mask_path.exists() or overwrite_data:
                        cell_mask_img=read_img(cell_mask_path, from_dtype="u8", to_dtype="u8")
                        warped_mask=cv.warpPerspective(cell_mask_img,transformation_matrix_inv,(fluor_image.shape[1],fluor_image.shape[0]))
                        cropped_warped_mask=warped_mask[y_min:y_max,x_min:x_max]

                        assert cropped_warped_mask.sum()!=0, "warping mask image resulted in black image (all pixel values 0)"
                        
                        write_img(cropped_warped_mask_path, cropped_warped_mask, from_dtype="u8", as_dtype="u8")
                    
                    if not cropped_fluor_path.exists() or overwrite_data:
                        # write cropped fluorescence that corresponds to the image region the phase contrast image was mapped onto
                        cropped_fluor=fluor_image[y_min:y_max,x_min:x_max]
                        write_img(cropped_fluor_path, cropped_fluor, from_dtype="u12", as_dtype="u12")

            #print(f"pos: {pos_index:5} /{len(positions):5} ( {(pos_index/len(positions)*100):6.2f}% ) [ dir: {position.stem:16}, approx. time left: {((perf_counter()-exp_pos_start)/pos_index*(len(positions)-pos_index)):6.2f}s ]")

    # clear vram
    del net
    torch.cuda.empty_cache()
