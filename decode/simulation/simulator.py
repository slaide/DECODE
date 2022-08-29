import torch
from typing import Tuple, Union, Generator, Optional, List

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

    def forward(self, em: EmitterSet, ix_low: Optional[int] = None, ix_high: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        ix_low_arg:int=ix_low or  self.frame_range[0]
        ix_high_arg:int=ix_high or  self.frame_range[1]

        frames = self.psf.forward(em.xyz_px, em.phot, em.frame_ix, ix_low=ix_low_arg, ix_high=ix_high_arg)

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
import numpy
import edt

import skimage.io

""" generator that yields cell mask snippets (and if requested, the corresponding phase contrast snippets) """
def sample_cell_masks(mask_list) -> Generator[numpy.array,None,None]:
    assert len(mask_list)>0
    while True:
        for mask in mask_list:
            cell_mask_image=skimage.io.imread(mask,as_gray=True).astype(numpy.float32)

            yield cell_mask_image

from time import perf_counter
from decode.generic import EmitterSet
import decode
import glob
from skimage.measure import label, regionprops
from skimage import filters
import time
import threading as td
import multiprocessing as mp
import multiprocessing.shared_memory
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# rotation code from https://stackoverflow.com/questions/64197754/how-do-i-rotate-a-pytorch-image-tensor-around-its-center-in-a-way-that-supports
def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, theta):
    rot_mat = get_rot_mat(theta)[None, ...].float().repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size(),align_corners=False).float()
    x = F.grid_sample(x, grid,align_corners=False)
    return x

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

    def __init__(self, segmentation_masks_glob:str, 
        psf: List[psf_kernel.PSF], 
        em_sampler, 
        num_frames: int, 
        frame_size:Tuple[int,int], 
        noise, # camera
        device:Union[str,torch.device]="cpu", 
        background_args=None,
        augment_rotation:bool=False
    ):
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
        self.last_psf_index=0

        self.noise = noise

        self.frame_size=frame_size
        self.mask_sampler=sample_cell_masks(glob.glob(segmentation_masks_glob))

        self.snippet_buffer=None
        self.snippet_buffer_bg=None

        self.device=device

        self.environmental_background=background_args.environmental_background
        self.mean_brightness_per_volume=background_args.mean_brightness_per_volume
        self.gaussian_width=background_args.gaussian_width

        self.augment_rotation=augment_rotation

    def sample_background(self):
        if isinstance(self.environmental_background,float):
            return self.environmental_background
        else:
            assert len(self.environmental_background)==2
            return torch.distributions.uniform.Uniform(self.environmental_background[0],self.environmental_background[1]).sample((1,)).item()

    def sample_full_frame(self,fraction_emitters_above_zero:float=0.5,override_probs=None):

        # try sampling cell mask. if (internal) iteration stops, restart
        mask=self.mask_sampler.__next__()
        #mask*=self.mean_brightness_per_volume

        # forward emitter position through image simulation pipeline
        single_frame_emitter_set, frames, bg_frames = self.forward(mask, device=self.device)

        return single_frame_emitter_set,mask,frames,bg_frames

    def sample_parallel(self):
        # queue to send/receive sampled data
        queue=mp.Queue()
        # lock as synchronization primitive to ensure exactly one set of data is sampled per sample request
        lock=mp.Lock()
        lock.acquire()

        def sample_in_parallel():
            while True:
                # sample data, then wait until the data was received to continue loop
                queue.put(self.sample())
                lock.acquire()
                lock.release()

        proc=mp.Process(target=sample_in_parallel)
        proc.daemon=True
        proc.start()

        try:
            while True:
                lock.release()

                emitter_set, frames, frames_bg=queue.get()

                # clone from memory of other process into this
                emitter_set, frames, frames_bg = emitter_set.clone(), frames.detach().clone(), frames_bg.detach().clone()

                yield emitter_set, frames, frames_bg

                lock.acquire()

        except Exception as e:
            proc.terminate()
            proc.join()

            raise e

    def sample(self):
        """
        Sample a new set of emitters and forward them through the simulation pipeline.

        (sample for the whole frame range!)

        Returns:
            EmitterSet: sampled emitters
            torch.Tensor: simulated frames
            torch.Tensor: background frames
        """

        #start=perf_counter()

        snippets_returned=0

        snippets=torch.zeros(2,self.num_frames,self.frame_size[0],self.frame_size[1]) # will be returned from this function

        emitter_set=None # will be returned from this function

        time_passed=0.0

        while snippets_returned<self.num_frames:
            if False and not self.snippet_buffer is None:
                raise NotImplementedError("return snippets from buffer/last frame")

            # sample emitter data, generated frame and background, also also get mask that was used to generate the other data
            single_frame_emitter_set, mask, frames, frames_bg = self.sample_full_frame()

            dim_0_snippet_count=mask.shape[0]//self.frame_size[0]
            dim_1_snippet_count=mask.shape[1]//self.frame_size[1]

            snippets_per_full_frame=dim_1_snippet_count*dim_0_snippet_count

            accept_empty_frames=True
            accept_partial_frames=True

            indices_i=numpy.arange(0,mask.shape[0],self.frame_size[0])
            indices_j=numpy.arange(0,mask.shape[1],self.frame_size[1])

            indices=numpy.zeros((indices_i.shape[0]*indices_j.shape[0],2),dtype=numpy.int)

            indices[:,0]=numpy.repeat(indices_i,indices_j.shape[0])
            indices[:,1]=numpy.repeat([indices_j],indices_i.shape[0],axis=0).flatten()

            xyz_coords=single_frame_emitter_set.xyz_px # x is large axis, y is short axis, in mask axis 0 is the large one, and axis 1 is the short one, too

            digits_i=numpy.digitize(xyz_coords[:,0],numpy.arange(0,mask.shape[0]+self.frame_size[0],self.frame_size[0],dtype=numpy.float32))-1
            digits_j=numpy.digitize(xyz_coords[:,1],numpy.arange(0,mask.shape[1]+self.frame_size[1],self.frame_size[1],dtype=numpy.float32))-1

            frame_indices=digits_i*indices_j.shape[0]+digits_j

            single_frame_emitter_set.xyz_px[:,:2]-=indices[frame_indices].astype(numpy.float32)

            # assert single_frame_emitter_set.xyz_px[:,0].min()>=0
            # assert single_frame_emitter_set.xyz_px[:,1].min()>=0
            # assert single_frame_emitter_set.xyz_px[:,0].max()<=self.frame_size[0]
            # assert single_frame_emitter_set.xyz_px[:,1].max()<=self.frame_size[1]

            hist,_bins=numpy.histogram(frame_indices,bins=numpy.arange(0,indices.shape[0]+1))
            hist_mask=hist==0
            offset=numpy.add.accumulate(hist_mask)
            frame_indices_adjusted=frame_indices if accept_empty_frames else frame_indices-offset[frame_indices] # for f in sorted(unique(frame_indices)): s=sum(sorted(unique(frame_indices))<f) ; frame_indices[frame_indices==f]-=s; endif

            single_frame_emitter_set.frame_ix=torch.from_numpy(frame_indices_adjusted+snippets_returned)

            if emitter_set is None:
                emitter_set=single_frame_emitter_set
            else:
                emitter_set+=single_frame_emitter_set

            for f,h in enumerate(hist):
                if accept_empty_frames or h>0:
                    i,j=indices[f]
                    next_snippet=frames[:,i:i+self.frame_size[0],j:j+self.frame_size[1]]
                    next_snippet_bg=frames_bg[:,i:i+self.frame_size[0],j:j+self.frame_size[1]]

                    if accept_partial_frames or (next_snippet.shape[1]==self.frame_size[0] and next_snippet.shape[2]==self.frame_size[1]):
                        snippets[0,snippets_returned,:next_snippet.shape[1],:next_snippet.shape[2]]=next_snippet
                        snippets[1,snippets_returned,:next_snippet.shape[1],:next_snippet.shape[2]]=next_snippet_bg

                        snippets_returned+=1

                        if snippets_returned>=self.num_frames:
                            break
                    elif not accept_partial_frames:
                        raise ValueError("frame index adjustment for discarded partial frames in simulation unimplemented")

        frames=snippets[0,:,:,:]
        frames_bg=snippets[1,:,:,:]

        return emitter_set, frames, frames_bg

    def forward(self, 
        mask:numpy.ndarray, 
        device:Union[str,torch.device]="cpu",
        fraction_emitters_above_zero:float=0.5,
        override_probs=None
    ) -> Tuple[decode.generic.emitter.EmitterSet, torch.Tensor, torch.Tensor]:
        """
        Forward an EmitterSet through the simulation pipeline.

        Args:
            mask: cell (thickness) mask

        Returns: # this assumes that background bg_return is set to 'tuple'
            EmitterSet: sampled emitters
            torch.Tensor: simulated frames
            torch.Tensor: background frames (e.g. to predict the bg seperately)
        """

        class Timer:
            def __init__(self,s=""):
                self.msg=s
            def __enter__(self):
                self.timer=time.perf_counter()
            def __exit__(self,a,b,c):
                print(f"timer {self.msg}: {(time.perf_counter()-self.timer):.3f}")

        # sample emitters per frame
        # incl. transform mask to correct cell background with included emitters
        single_frame_emitter_set = self.em_sampler.sample(mask,fraction_emitters_above_zero=fraction_emitters_above_zero,override_probs=override_probs)

        binary_mask=mask>0
        cell_regions=regionprops(label(binary_mask))

        # remove noise from psf that smap could not have known to not actually be part of the psf
        # this requires simulating the dots seperately, since there is some thresholding involved which does not work with overlapping dots
        # this is ~20% slower than simulation without psf noise removal
        em_xyz_px=single_frame_emitter_set.xyz_px
        em_phot=single_frame_emitter_set.phot
        em_frame_ix=single_frame_emitter_set.frame_ix

        em_xyz_px_int=em_xyz_px.numpy().astype(dtype=int)

        # use max size of any of the PSFs currently in use
        psf_width=numpy.max([psf.size for psf in self.psf])

        extent_x0_list=numpy.clip(em_xyz_px_int[:,0] - (psf_width//2 + 2), a_min=0, a_max=None)
        extent_x1_list=numpy.clip(em_xyz_px_int[:,0] + (psf_width//2 + 2), a_min=None, a_max=mask.shape[0])
        extent_y0_list=numpy.clip(em_xyz_px_int[:,1] - (psf_width//2 + 2), a_min=0, a_max=None)
        extent_y1_list=numpy.clip(em_xyz_px_int[:,1] + (psf_width//2 + 2), a_min=None, a_max=mask.shape[1])

        # alternate between different PSFs that were provided
        if self.last_psf_index==len(self.psf)-1:
            self.last_psf_index=0
        else:
            self.last_psf_index+=1

        psf=self.psf[self.last_psf_index]

        # scale a threshold to fit current brightness (threshold was manully determined to be 0.4 for an emitter brightness of 1e3)
        psf_background_list=psf.background_offset*em_phot.numpy()
        # (this is a different threshold value that was manually determined)
        psf_threshold=0.0

        # rotate each snippet by a random small angle
        if self.augment_rotation:
            # arbitrarily chosen range "range of small rotation values"
            angle_min,angle_max=-4,4
            random_angles=numpy.random.uniform(angle_min,angle_max,(em_xyz_px.shape[0],))
            random_angles_sin=numpy.sin(random_angles)
            random_angles_cos=numpy.cos(random_angles)

        for i in range(0,em_xyz_px.shape[0]):
            new_frame=psf.forward(em_xyz_px[i:i+1], em_phot[i:i+1], em_frame_ix[i:i+1], ix_low=None, ix_high=None).numpy()

            extent_x0=extent_x0_list[i]
            extent_x1=extent_x1_list[i]
            extent_y0=extent_y0_list[i]
            extent_y1=extent_y1_list[i]

            # only operate on dot region (+small buffer) for better performance
            snippet=new_frame[0,extent_x0:extent_x1,extent_y0:extent_y1]
            # remove some manually determined background noise from partially wrongly approximated psf
            snippet-=psf_background_list[i]
            # remove some additional noise by thresholding
            snippet[snippet<psf.background_threshold]=0
            # scale to match expected brightness of emitter
            snippet*=em_phot[i].item()/snippet.sum()

            if self.augment_rotation:
                # rotate psf slightly (around center of snippet)
                theta=numpy.pi/180*random_angles[i]
                snippet=rot_img(torch.from_numpy(snippet).unsqueeze(0).unsqueeze(0), theta).squeeze(0).squeeze(0).numpy()

                # adjust emitter coordinates to match rotated psf
                dx=(extent_x1-extent_x0)/2+extent_x0
                dy=(extent_y1-extent_y0)/2+extent_y0

                x=em_xyz_px[i,0]-dx
                y=em_xyz_px[i,1]-dy
                # manual multiplication of emitter coords with 'rotation matrix'
                em_xyz_px[i,0]=x*random_angles_cos[i]-y*random_angles_sin[i]+dx
                em_xyz_px[i,1]=x*random_angles_cos[i]+y*random_angles_sin[i]+dy

            if i==0:
                emitter_frames:torch.Tensor = new_frame
            else:
                emitter_frames[0,extent_x0:extent_x1,extent_y0:extent_y1] += snippet

        bg_frames=filters.gaussian(mask,self.gaussian_width)+self.sample_background()
        bg_frames=torch.from_numpy(bg_frames).unsqueeze(0).to(device)
        
        # in decode, poisson noise is applied to background AND emitters
        # seems counter-intuitive, but that actually looks more realistic
        frames=self.noise.forward(bg_frames+emitter_frames)

        # return background _before_ 'recorded' by camera sensor
        return single_frame_emitter_set, frames, bg_frames
