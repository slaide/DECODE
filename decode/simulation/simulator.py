import torch
from typing import Tuple, Union, Generator, Optional

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

from decode.utils.img_file_io import read_img

""" generator that yields cell mask snippets (and if requested, the corresponding phase contrast snippets) """
def sample_cell_masks(root_folder,side_length=40,also_yield_fluorescence=False) -> Generator[Optional[numpy.array],None,None] :
    experiments_folder=Path(root_folder)
    assert experiments_folder.exists()
    
    """
    for each experiment:
        for each position:
            for each pair of cell segmenation mask and fluorescence images:
                cut the same region from both images
                yield tuple of mask[/fluorescence snippets]
    
    """
    while True:
        for experiment in experiments_folder.iterdir():
            if not experiment.is_dir() or experiment.name.startswith("."):
                continue
                
            position_list=[p for p in (experiment/"Run").iterdir()]
            for position in position_list:
                if not position.is_dir() or position.name.startswith("."):
                    continue
                    
                cell_mask_dir=position/"warped_dist_masks_580"
                fluorescence_dir=position/"fluor_cropped_580"
                assert cell_mask_dir.exists() and fluorescence_dir.exists(),"real data not properly pre-processed"
                    
                # TODO : the image lists need to be sorted by index! (by the index number that is contained in their filenames. should be last 8 letters/digits in the filename, excl. filetype)
                    
                cell_masks=[x for x in cell_mask_dir.iterdir() if not x.name.startswith(".")]
                fluorescence=[x for x in fluorescence_dir.iterdir() if not x.name.startswith(".")]
                    
                num_fluorescence_images=len(fluorescence)
                num_cell_masks=len(cell_masks)
                
                assert num_fluorescence_images>0,"no noise-added phase contrast images found. either folder is empty, or something went wrong"
                
                fluorescence_per_cell_mask=num_fluorescence_images//num_cell_masks
                
                assert num_fluorescence_images%num_cell_masks==0,f"{num_cell_masks} {num_fluorescence_images}"
                
                for image_index in range(0,num_cell_masks):
                    cell_mask_image=read_img(cell_masks[image_index*fluorescence_per_cell_mask],from_dtype="u8",to_dtype="u8")

                    # TODO this is a very temporary solution!
                    cell_mask_image=edt.edt(cell_mask_image, order='C', parallel=0).astype(dtype=numpy.uint8)

                    yield cell_mask_image

from time import perf_counter
from decode.generic import EmitterSet
import matplotlib.pyplot as plt
import decode

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

    def __init__(self, root_experiments_folder:Union[str,Path], psf: psf_kernel.PSF, em_sampler, background, num_frames: int, frame_size:Tuple[int,int], noise, device:Union[str,torch.device]="cpu"):
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

        self.root_experiments_folder=root_experiments_folder if isinstance(root_experiments_folder,Path) else Path(root_experiments_folder)
        self.frame_size=frame_size
        self.mask_sampler=sample_cell_masks(self.root_experiments_folder,also_yield_fluorescence=False)

        self.snippet_buffer=None
        self.snippet_buffer_bg=None

        self.device=device

    def sample_full_frame(self,fraction_emitters_above_zero:float=0.5,override_probs=None):

        # try sampling cell mask. if (internal) iteration stops, restart
        mask_sampler_result=self.mask_sampler.__next__()

        if mask_sampler_result is None:
            self.mask_sampler=sample_cell_masks(self.root_experiments_folder,also_yield_fluorescence=False)
            return self.sample_full_frame()
        else:
            mask=mask_sampler_result

        # sample emitters per frame
        single_frame_emitter_set = self.em_sampler.sample(mask,fraction_emitters_above_zero=fraction_emitters_above_zero,override_probs=override_probs)

        # forward emitter position through image simulation pipeline
        frames, frames_bg = self.forward(mask, single_frame_emitter_set, device=self.device)
        #print(frames_bg.min(),frames_bg.mean())
        #print(frames.min(),frames.mean())

        return single_frame_emitter_set,mask,frames,frames_bg

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

        while snippets_returned!=self.num_frames:
            if False and not self.snippet_buffer is None:
                raise NotImplementedError("return snippets from buffer/last frame")

            # try sampling cell mask. if (internal) iteration stops, restart
            mask=self.mask_sampler.__next__()

            # sample emitters per frame # this takes 1/3 of the total time
            single_frame_emitter_set = self.em_sampler.sample(mask)

            # forward emitter position through image simulation pipeline
            frames, frames_bg = self.forward(mask, single_frame_emitter_set, device=self.device)

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

            assert single_frame_emitter_set.xyz_px[:,0].min()>=0
            assert single_frame_emitter_set.xyz_px[:,1].min()>=0
            assert single_frame_emitter_set.xyz_px[:,0].max()<=self.frame_size[0]
            assert single_frame_emitter_set.xyz_px[:,1].max()<=self.frame_size[1]

            hist,_bins=numpy.histogram(frame_indices,bins=numpy.arange(0,indices.shape[0]+1))
            mask=hist==0
            offset=numpy.add.accumulate(mask)
            frame_indices_adjusted=frame_indices if accept_empty_frames else frame_indices-offset[frame_indices] # for f in sorted(unique(frame_indices)): s=sum(sorted(unique(frame_indices))<f) ; frame_indices[frame_indices==f]-=s; endif

            single_frame_emitter_set.frame_ix=torch.from_numpy(frame_indices_adjusted+snippets_returned)

            if emitter_set is None:
                emitter_set=single_frame_emitter_set
            else:
                emitter_set+=single_frame_emitter_set

            # this could probably be made even faster by broadcasting the frame, but i cannot be asked (also, may increase ram usage by several megabytes.. (~1MB per frame * 15*10 snippets per frame -> ~1.5GB))
            for f,h in enumerate(hist):
                if accept_empty_frames or h>0:
                    i,j=indices[f]
                    next_snippet=frames[:,i:i+self.frame_size[0],j:j+self.frame_size[1]]

                    if accept_partial_frames or (next_snippet.shape[1]==self.frame_size[0] and next_snippet.shape[2]==self.frame_size[1]):
                        snippets[:,snippets_returned,:next_snippet.shape[1],:next_snippet.shape[2]]=next_snippet
                        snippets_returned+=1

                        if snippets_returned>=self.num_frames:
                            break
                    elif not accept_partial_frames:
                        raise ValueError("unimplemented") # need to adjust frame_ix

        frames=snippets[0,:,:,:]
        frames_bg=snippets[1,:,:,:]

        #print(f"time passed {time_passed:.2f}")

        #assert len(emitter_set)>=self.num_frames,"there is not at least one emitter per frame. this is a bug"

        return emitter_set, frames, frames_bg

    # static
    brightness_tracking_on=False
    num_emitters_brightness_tracked=100
    brightness_tracker=torch.zeros((num_emitters_brightness_tracked,)) if brightness_tracking_on else None
    num_tracked_brightness_values=0

    def forward(self, mask:numpy.ndarray, em: EmitterSet, device:Union[str,torch.device]="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward an EmitterSet through the simulation pipeline.

        Args:
            mask: cell (thickness) mask
            em (EmitterSet): Emitter Set

        Returns: # this assumes that background bg_return is set to 'tuple'
            torch.Tensor: simulated frames
            torch.Tensor: background frames (e.g. to predict the bg seperately)
        """


        emitter_frames:torch.Tensor = self.psf.forward(em.xyz_px, em.phot, em.frame_ix, ix_low=None, ix_high=None)

        if MaskedSimulation.brightness_tracking_on and MaskedSimulation.num_tracked_brightness_values < MaskedSimulation.num_emitters_brightness_tracked:
            s1=emitter_frames.sum().item()
            s2=em.phot.sum().item()

            MaskedSimulation.brightness_tracker[MaskedSimulation.num_tracked_brightness_values]=s1/s2
            MaskedSimulation.num_tracked_brightness_values+=1

            if MaskedSimulation.num_tracked_brightness_values==MaskedSimulation.num_emitters_brightness_tracked:
                tracked_brightness_mean=MaskedSimulation.brightness_tracker.mean().item()
                error_estimate=((MaskedSimulation.brightness_tracker-tracked_brightness_mean)**2/MaskedSimulation.num_emitters_brightness_tracked).sum().sqrt().item()
                assert not numpy.isnan(error_estimate)
                print(f"emitter brightness fraction mean (std dev): {tracked_brightness_mean:5.4f} ({error_estimate:5.4f})") # should be pretty close to 1, but is not. no idea why

        # bg_frames do not include camera noise ()
        bg_frames=self.background.sample(mask=mask,device=device)
        
        # in decode, poisson noise is applied to background AND emitters
        if False:
            frames=self.noise.forward(self.noise.poisson.forward(bg_frames)+emitter_frames,sample_photons=False)
        else:
            frames=self.noise.forward(bg_frames+emitter_frames)

        return frames, bg_frames
