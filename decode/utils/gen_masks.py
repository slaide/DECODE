import skimage
import skimage.io

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

""" iterable structure to load data into a network """
class SegmentDirectory(Dataset):
    """ normalize and load all image files in directory """
    def __init__(self, directory, transform = None, flip = False):
        self.directory = Path(directory)
        self.transform = transform
        self.indices = [
            filename
            for filename 
            in self.directory.iterdir()
            if not filename.name.startswith(".")
            and (filename.name.endswith(".tif") or filename.name.endswith(".tiff"))
        ]
        self.n_images = len(self.indices)
        self.flip = flip

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #print(self.indices[idx])
        phase_img = read_img(str(self.indices[idx]))
        phase_img = skimage.img_as_float32(phase_img)

        phase_img_normalized = (phase_img - phase_img.mean()) / phase_img.std()

        # flips both axis! (rotates by 180 degrees)
        if self.flip:
            phase_img_normalized = numpy.flip(phase_img_normalized)
        
        if self.transform:
            phase_img_normalized = self.transform(phase_img_normalized)

        return phase_img_normalized

""" apply the segmentation net to all files in a directory """
def generate_cell_segmentation_masks(phase_dir, save_dir, overwrite_data:bool=False, transform=None, device:str="cuda:0", threshold:float=0.9, remove_small_objects=None, model_path:str="mixed10epochs_betterscale_contrastAdjusted1.pth", net=None):
    if net is None:
        net=loadNet(model_path,device)
    
    saveDir=Path(save_dir)
    if not saveDir.exists():
        saveDir.mkdir()

    dataset = SegmentDirectory(phase_dir, transform=transform, flip=False)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=6)

    # zero the paramter gradients
    # forward + backward + optimize
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):
            phase = data.to(device)
            
            mask_pred = net(phase)

            # set sigmoid if the net gives # ?
            #mask_pred = torch.sigmoid(mask_pred).to("cpu").numpy().squeeze(0).squeeze(0)
            mask_pred=mask_pred.to("cpu").numpy().squeeze(0).squeeze(0)

            mask_pred = mask_pred >= threshold

            mask_pred_labeled = skimage.measure.label(mask_pred)
            
            if remove_small_objects:
                min_area=remove_small_objects["min_area"]
                max_area=remove_small_objects["max_area"]
                hole_area_threshold=remove_small_objects["hole_area_threshold"]
            else:
                pixel_area=(65e-3)**2
                #min area: 1µm
                min_area=1/pixel_area
                #max area: 3.5µm
                max_area=3.5/pixel_area
                #hole area threshold (upper limit of hole size that will be filled): 0.4µm
                hole_area_threshold=0.4/pixel_area
                
            mask_cleaned = skimage.morphology.remove_small_objects(mask_pred_labeled, min_size=min_area)
            mask_cleaned = skimage.morphology.remove_small_holes(mask_cleaned > 0, area_threshold=hole_area_threshold)
            mask_pred = mask_cleaned

            """ for each region (area of pixels with value 'True' (1, or 1.0)), calculate the shortest distance to the area's border (outer cell wall) with another region. then normalize the distances, then transform the linear [0;1] space to the cell thickness at each pixel, approximated by modeling the cell as perfectly round"""
            dists = edt.edt(mask_pred, order='C', parallel=4)
            dists/=dists.max()
            dists=numpy.sin(numpy.pi/2*dists)
            dists=numpy.sqrt(dists)
            dists=skimage.img_as_ubyte(dists)
            
            dists=numpy.rot90(dists,k=3)

            img_path=save_dir/f"dist_mask_{i_batch:08}.tiff"

            if not img_path.exists() or overwrite_data:
                write_img(img_path, dists)

""" [currently unused] generator that yields cell mask snippets (and if requested, the corresponding phase contrast snippets) """
def sample_cell_masks(root_folder,side_length=40,also_yield_fluorescence=False):
    experiments_folder=Path(root_folder)
    assert experiments_folder.exists()
    
    """
    for each experiment:
        for each position:
            for each pair of cell segmenation mask and fluorescence images:
                cut the same region from both images
                yield tuple of mask[/fluorescence snippets]
    
    """
    for experiment in experiments_folder.iterdir():
        if not experiment.is_dir() or experiment.name.startswith("."):
            continue
            
        for position in experiment.iterdir():
            if not position.is_dir() or position.name.startswith("."):
                continue
                
            cell_mask_dir=position/"warped_dist_masks"
            fluorescence_dir=position/"fluor_cropped"
                
            # TODO : the image lists need to be sorted by index! (by the index number that is contained in their filenames. should be last 8 letters/digits in the filename, excl. filetype)
                
            cell_masks=[x for x in cell_mask_dir.iterdir() if not x.name.startswith(".")]
            fluorescence=[x for x in fluorescence_dir.iterdir() if not x.name.startswith(".")]
                
            num_fluorescence_images=len(fluorescence)
            num_cell_masks=len(cell_masks)
            
            assert num_fluorescence_images>0,"no noise-added phase contrast images found. either folder is empty, or something went wrong"
            
            fluorescence_per_cell_mask=num_fluorescence_images//num_cell_masks
            
            assert num_fluorescence_images%num_cell_masks==0,f"{num_cell_masks} {num_fluorescence_images}"
            
            for image_index in range(0,num_cell_masks):
                cell_mask_image=read_img(cell_masks[image_index*fluorescence_per_cell_mask])
                
                if also_yield_fluorescence:
                    fluorescence_image=read_img(fluorescence[image_index])
                
                    assert fluorescence_image.shape==cell_mask_image.shape,f"{fluorescence_image.shape} {cell_mask_image.shape}"
                
                for i in np.arange(0,cell_mask_image.shape[0],side_length):
                    for j in np.arange(0,cell_mask_image.shape[1],side_length):
                        cell_mask_image_snippet=cell_mask_image[j:j+side_length,i:i+side_length]
                        
                        if also_yield_fluorescence:
                            fluorescence_image_snippet=fluorescence_image[j:j+side_length,i:i+side_length]
                        
                            yield (fluorescence_image_snippet,cell_mask_image_snippet)
                        else:
                            yield cell_mask_image_snippet

""" utility class to load an image inside a dataloader pipeline """
class rotateImageBy90Degrees(object):
    def __init__(self):
        pass
    def __call__(self, phase_image):
        return numpy.rot90(phase_image)

""" generate cell masks for a series of experiments containing pairs of phase contrast images and fluorescence images """
# this data cannot be generated on the fly because the essential part of generating the masks is an AI, which takes up a lot of vram
# the gpu does not have enough vram to run two UNet derivatives simultaneously (or to keep them in memory at the same time)
def generate_cell_masks(root_folder:str,fluo_roi:Tuple[Tuple[float,float],Tuple[float,float]],device:str="cuda:0",overwrite_data:bool=False,threshold:float=0.9,fluor_dir_name:str="fluor515",model_path:str="mixed10epochs_betterscale_contrastAdjusted1.pth"):
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
    for experiment in experiments_folder.iterdir():
        if not experiment.is_dir() or experiment.name.startswith("."):
            continue
            
        print(f"exp: {experiment}")
            
        """ generate dataloader transform once per experiment (for absolutely no reason) """
        transform = torchvision.transforms.Compose([
            rotateImageBy90Degrees(),
            tensorizeOneImage(1)])
            
        """ load possible experiment-specific phase contrast/fluorescence alignment transformation matrix """
        transformation_matrix=loadmat(experiment/"transMatV.mat")["transformationMatrix"].T # transpose because opencv coordinate system works different from matlab
        transformation_matrix_inv=numpy.linalg.inv(transformation_matrix)
        
        total_exp_pos_time=0.0
        exp_pos_start=perf_counter()

        positions=[position for position in experiment.iterdir() if position.is_dir() and not position.name.startswith(".")]
        for pos_index,position in enumerate(positions,start=1):
            if not position.is_dir() or position.name.startswith("."):
                continue
                
            #print(f"pos: {position}")
            
            noisy_phase_contrast_dir=position/"phase_noisy" # contains phase contrast images, with some noise added on top to make the segmentation net work better. needs to be saved to disk because of how the dataloader currently works
            if not noisy_phase_contrast_dir.exists():
                noisy_phase_contrast_dir.mkdir()
                
            cropped_fluor_dir=position/"fluor_cropped"
            if not cropped_fluor_dir.exists():
                cropped_fluor_dir.mkdir()
            
            # TODO : sort phase/fluoresence image lists by number at the end of the filename!
                
            phase_dir=position/"phase" # is input
            phase_images=[x for x in phase_dir.iterdir()]
            num_phase_images=len(phase_images)
            
            fluor_dir=position/fluor_dir_name # is input
            fluor_images=[x for x in fluor_dir.iterdir()]
            num_fluor_images=len(fluor_images)
            
            assert num_phase_images>0,"no phase contrast images found. either folder is empty, or something went wrong"
            assert num_fluor_images>0,"no fluorescence images found. either folder is empty, or something went wrong"
            
            fluorescence_per_phase=num_fluor_images//num_phase_images
            
            assert num_fluor_images%num_phase_images==0,f"{num_fluor_images} {num_phase_images}"
            
            for (phase_index,phase_contrast_image_path) in enumerate(phase_images):
                if not phase_contrast_image_path.is_file():
                    continue

                #print(f"img: {phase_contrast_image_path}")
                    
                phase_contrast_image=read_img(phase_contrast_image_path)
                
                """ add a small bit of noise on top of the phase contrast image to avoid multiple issues with the segmentation mask (holes, dents, splits) """
                phase_contrast_image_noisy=phase_contrast_image+skimage.img_as_ubyte(numpy.random.normal(scale=0.03,size=phase_contrast_image.shape))
                
                phase_contrast_image_noisy_path=noisy_phase_contrast_dir/phase_contrast_image_path.name
                if not phase_contrast_image_noisy_path.exists() or overwrite_data:
                    write_img(phase_contrast_image_noisy_path,phase_contrast_image_noisy)
                    
            
            cell_mask_dir=position/"dist_masks"
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
                
            warped_masks_dir=position/"warped_dist_masks"
            if not warped_masks_dir.exists():
                warped_masks_dir.mkdir()
                        
            cell_masks=[x for x in cell_mask_dir.iterdir()]
                    
            for (mask_index,cell_mask_path) in enumerate(cell_masks):
                if not cell_mask_path.is_file():
                    continue

                #print(f"cmp: {cell_mask_path}")
                    
                cell_mask_img=read_img(cell_mask_path)
                    
                """ transform phase contrast image to be aligned with the fluorescence image """
                fluor_image_path=fluor_images[mask_index*fluorescence_per_phase]
                fluor_image=read_img(fluor_image_path)
                
                warped_mask=cv.warpPerspective(cell_mask_img,transformation_matrix_inv,(fluor_image.shape[1],fluor_image.shape[0]))
                cropped_warped_mask=warped_mask[y_min:y_max,x_min:x_max]

                assert cropped_warped_mask.sum()!=0, "warping mask image resulted in black image (all pixel values 0)"
                
                cropped_warped_mask_path=warped_masks_dir/cell_mask_path.name
                if not cropped_warped_mask_path.exists() or overwrite_data:
                    write_img(cropped_warped_mask_path, cropped_warped_mask)
                
                """ write cropped fluorescence that corresponds to the image region the phase contrast image was mapped onto """
                #cropped_fluor=fluor_image[1130:2800,350:1200]
                cropped_fluor=fluor_image[y_min:y_max,x_min:x_max]
                
                cropped_fluor_path=cropped_fluor_dir/fluor_image_path.name
                if not cropped_fluor_path.exists() or overwrite_data:
                    write_img(cropped_fluor_path, cropped_fluor)

            print(f"pos: {pos_index:5} /{len(positions):5} ( {(pos_index/len(positions)*100):6.2f}% ) [ dir: {position.stem:16}, approx. time left: {((perf_counter()-exp_pos_start)/pos_index*(len(positions)-pos_index)):6.2f}s ]")

    # clear vram
    del net
    torch.cuda.empty_cache()