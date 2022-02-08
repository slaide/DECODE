import torch
from typing import Tuple, Union

from ..generic import EmitterSet
from . import psf_kernel


class Simulation:
    """
    A simulation class that holds the necessary modules, i.e. an emitter source (either a static EmitterSet or
    a function from which we can sample emitters), a psf background and noise. You may also specify the desired frame
    range, i.e. the indices of the frames you want to have as output. If they are not specified, they are automatically
    determined but may vary with new sampled emittersets.

    Attributes:
        em (EmitterSet): Static EmitterSet
        em_sampler: instance with 'sample()' method to sample EmitterSets from
        frame_range: frame indices between which to compute the frames. If None they will be
        auto-determined by the psf implementation.
        psf: psf model with forward method
        background (Background): background implementation
        noise (Noise): noise implementation
    """

    def __init__(self, psf: psf_kernel.PSF, em_sampler=None, background=None, noise=None,
                 frame_range: Tuple[int, int] = None):
        """
        Init Simulation.

        Args:
            psf: point spread function instance
            em_sampler: callable that returns an EmitterSet upon call
            background: background instance
            noise: noise instance
            frame_range: limit frames to static range
        """

        self.em_sampler = em_sampler
        self.frame_range = frame_range if frame_range is not None else (None, None)

        self.psf = psf
        self.background = background
        self.noise = noise

    def sample(self):
        """
        Sample a new set of emitters and forward them through the simulation pipeline.

        Returns:
            EmitterSet: sampled emitters
            torch.Tensor: simulated frames
            torch.Tensor: background frames
        """

        emitter = self.em_sampler()
        frames, bg = self.forward(emitter)

        return emitter, frames, bg

    def forward(self, em: EmitterSet, ix_low: Union[None, int] = None, ix_high: Union[None, int] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Forward an EmitterSet through the simulation pipeline. 
        Setting ix_low or ix_high overwrites the frame range specified in the init.

        Args:
            em (EmitterSet): Emitter Set
            ix_low: lower frame index
            ix_high: upper frame index (inclusive)

        Returns:
            torch.Tensor: simulated frames
            torch.Tensor: background frames (e.g. to predict the bg seperately)
        """

        if ix_low is None:
            ix_low = self.frame_range[0]

        if ix_high is None:
            ix_high = self.frame_range[1]

        frames = self.psf.forward(em.xyz_px, em.phot, em.frame_ix,
                                  ix_low=ix_low, ix_high=ix_high)

        """
        Add background. This needs to happen here and not on a single frame, since background may be correlated.
        The difference between background and noise is, that background is assumed to be independent of the 
        emitter position / signal.
        """
        if self.background is not None:
            frames, bg_frames = self.background.forward(frames)
        else:
            bg_frames = None

        if self.noise is not None:
            frames = self.noise.forward(frames)

        return frames, bg_frames


from pathlib import Path

import skimage
import skimage.io
from pathlib import Path
import numpy
import numpy as np

def read_img(path):
    img_data=skimage.io.imread(str(path), as_gray = True)
    img_data=skimage.img_as_ubyte(img_data)
    return img_data

""" generator that yields cell mask snippets (and if requested, the corresponding phase contrast snippets) """
def sample_cell_masks(root_folder,side_length=40,also_yield_fluorescence=False): # -> Generator[Union[Tuple[numpy.array,numpy.array],numpy.array]]
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
            
        #print(f"{str(experiment)}")
        position_list=[p for p in experiment.iterdir()]
        #print(f"num positions: {len(position_list)}")
        for position in position_list:
            if not position.is_dir() or position.name.startswith("."):
                continue
                
            cell_mask_dir=position/"warped_dist_masks"
            fluorescence_dir=position/"fluor_cropped"
            assert cell_mask_dir.exists() and fluorescence_dir.exists(),"real data not properly pre-processed"
                
            # TODO : the image lists need to be sorted by index! (by the index number that is contained in their filenames. should be last 8 letters/digits in the filename, excl. filetype)
                
            cell_masks=[x for x in cell_mask_dir.iterdir() if not x.name.startswith(".")]
            fluorescence=[x for x in fluorescence_dir.iterdir() if not x.name.startswith(".")]
                
            num_fluorescence_images=len(fluorescence)
            num_cell_masks=len(cell_masks)
            
            assert num_fluorescence_images>0,"no noise-added phase contrast images found. either folder is empty, or something went wrong"
            
            fluorescence_per_cell_mask=num_fluorescence_images//num_cell_masks
            
            assert num_fluorescence_images%num_cell_masks==0,f"{num_cell_masks} {num_fluorescence_images}"

            #print(f"num cell masks: {num_cell_masks}")
            
            for image_index in range(0,num_cell_masks):
                cell_mask_image=read_img(cell_masks[image_index*fluorescence_per_cell_mask])
                
                if also_yield_fluorescence:
                    fluorescence_image=read_img(fluorescence[image_index])
                
                    assert fluorescence_image.shape==cell_mask_image.shape,f"{fluorescence_image.shape} {cell_mask_image.shape}"

                    yield (cell_mask_image,fluorescence_image)
                else:
                    yield cell_mask_image
                
                #for i in np.arange(0,cell_mask_image.shape[0],side_length):
                #    for j in np.arange(0,cell_mask_image.shape[1],side_length):
                #        cell_mask_image_snippet=cell_mask_image[j:j+side_length,i:i+side_length]
                #        
                #        if also_yield_fluorescence:
                #            fluorescence_image_snippet=fluorescence_image[j:j+side_length,i:i+side_length]
                #        
                #            yield (fluorescence_image_snippet,cell_mask_image_snippet)
                #        else:
                #            yield cell_mask_image_snippet

from time import perf_counter
from decode.generic import EmitterSet
import matplotlib.pyplot as plt
class MaskedSimulation:
    """
    A simulation class that holds the necessary modules, i.e. an emitter source (either a static EmitterSet or
    a function from which we can sample emitters), a psf background and noise. You may also specify the desired frame
    range, i.e. the indices of the frames you want to have as output. If they are not specified, they are automatically
    determined but may vary with new sampled emittersets.

    Attributes:
        em (EmitterSet): Static EmitterSet
        em_sampler: instance with 'sample()' method to sample EmitterSets from
        num_frames: number of frames to sample emitters for
        psf: psf model with forward method
        background (Background): background implementation
        noise (Noise): noise implementation
    """

    def __init__(self, root_experiments_folder:Union[str,Path], psf: psf_kernel.PSF, em_sampler, background, num_frames: int, frame_size:Tuple[int,int], noise, full_frame_psf:bool=False, also_yield_fluorescence=False):
        """
        Init Simulation.

        Args:
            psf: point spread function instance
            em_sampler: callable that returns an EmitterSet upon call
            background: background instance
            noise: noise instance
            num_frames: number of frames in dataset
        """

        self.em_sampler = em_sampler
        self.num_frames = num_frames

        self.psf = psf
        self.background = background
        self.noise = noise

        self.also_yield_fluorescence=also_yield_fluorescence

        self.root_experiments_folder=root_experiments_folder if isinstance(root_experiments_folder,Path) else Path(root_experiments_folder)
        self.frame_size=frame_size
        #print(f"frame_size={self.frame_size}")
        self.mask_sampler=sample_cell_masks(self.root_experiments_folder,also_yield_fluorescence=self.also_yield_fluorescence)

        self.snippet_buffer=None
        self.snippet_buffer_bg=None

        self.full_frame_psf=full_frame_psf

    def sample(self):
        """
        Sample a new set of emitters and forward them through the simulation pipeline.

        (sample for the whole frame range!)

        Returns:
            EmitterSet: sampled emitters
            torch.Tensor: simulated frames
            torch.Tensor: background frames
        """

        sample_fn_start=perf_counter()

        snippets_returned=0

        snippets=torch.zeros(2,self.num_frames,self.frame_size[0],self.frame_size[1]) # will be returned from this function

        num_frames_sampled=0
        snippets_generated_total=0

        emitter_set=None # will be returned from this function

        sample_counter=0.0
        snippetization_counter=0.0
        snippet_copy_counter=0.0

        while True:
            if False and not self.snippet_buffer is None:
                raise NotImplementedError("return snippets from buffer/last frame")

            sample_start=perf_counter()

            # try sampling cell mask. if (internal) iteration stops, restart
            try:
                mask=self.mask_sampler.__next__()
            except StopIteration:
                self.mask_sampler=sample_cell_masks(self.root_experiments_folder,also_yield_fluorescence=self.also_yield_fluorescence)
                continue

            num_frames_sampled+=1

            # sample emitters per frame
            single_frame_emitter_set = self.em_sampler.sample(mask)

            # forward emitter position through image simulation pipeline
            if self.full_frame_psf:
                frames, frames_bg = self.forward(mask, single_frame_emitter_set)

            sample_counter+=perf_counter()-sample_start

            dim_0_snippet_count=mask.shape[0]//self.frame_size[0]
            dim_1_snippet_count=mask.shape[1]//self.frame_size[1]

            snippets_per_full_frame=dim_1_snippet_count*dim_0_snippet_count

            snippetization_start=perf_counter()

            snippets_generated_in_frame=0

            #assert frames.shape[1]==mask.shape[0]
            #assert frames.shape[2]==mask.shape[1]

            for i_i,i in enumerate(range(0,mask.shape[0],self.frame_size[0])):
                if snippets_returned==self.num_frames:
                    break

                for j_i,j in enumerate(range(0,mask.shape[1],self.frame_size[1])):
                    if snippets_returned==self.num_frames:
                        break

                    current_total_snippet_index=i_i*dim_1_snippet_count+j_i

                    if i+self.frame_size[0] <= mask.shape[0] and j+self.frame_size[1] <= mask.shape[1]:
                        single_frame_emitter_subset=single_frame_emitter_set.emitters_in_region(ax0=(i,i+self.frame_size[0]),ax1=(j,j+self.frame_size[1])).clone()

                        if len(single_frame_emitter_subset)>0:
                            single_frame_emitter_subset.xyz_px[:,0]-=i
                            single_frame_emitter_subset.xyz_px[:,1]-=j

                            # forward emitter position through image simulation pipeline
                            if self.full_frame_psf:
                                snippets[:,snippets_returned,:,:]=frames[:,i:i+self.frame_size[0],j:j+self.frame_size[1]]
                            else:
                                frames, frames_bg = self.forward(mask[i:i+self.frame_size[0],j:j+self.frame_size[1]], single_frame_emitter_subset)
                                snippets[:,snippets_returned,:,:]=frames

                            single_frame_emitter_subset.frame_ix[:]=snippets_returned

                            if not emitter_set is None:
                                emitter_set+=single_frame_emitter_subset
                            else:
                                emitter_set=single_frame_emitter_subset

                            snippets_returned+=1
                    
                        snippets_generated_in_frame+=1
                        snippets_generated_total+=1

            snippetization_counter+=perf_counter()-snippetization_start

            if snippets_returned==self.num_frames:
                break

        frames=snippets[0,:,:,:]
        frames_bg=snippets[1,:,:,:]

        assert len(emitter_set)>=self.num_frames,"there is not at least one emitter per frame. this is a bug"

        print(f"[perf] sample fn total: {(perf_counter()-sample_fn_start):5.3f}s [ sample: {sample_counter:5.3f}s, snippetization: {snippetization_counter:5.3f}s ]")

        return emitter_set, frames, frames_bg

    def forward(self, mask:torch.Tensor, em: EmitterSet) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward an EmitterSet through the simulation pipeline.

        Args:
            mask: cell (thickness) mask
            em (EmitterSet): Emitter Set

        Returns:
            torch.Tensor: simulated frames
            torch.Tensor: background frames (e.g. to predict the bg seperately) # this assumes that background bg_return is set to 'tuple'
        """

        frames = self.psf.forward(em.xyz_px, em.phot, em.frame_ix)

        """ Add background. The difference between background and noise is, that background is assumed to be independent of the emitter position / signal. """
        
        bg_frames = self.background.sample(mask=torch.tensor(mask))

        """
        Add background. This needs to happen here and not on a single frame, since background may be correlated.
        The difference between background and noise is, that background is assumed to be independent of the 
        emitter position / signal.
        """

        frames+=bg_frames

        frames = self.noise.forward(frames)

        return frames, bg_frames
