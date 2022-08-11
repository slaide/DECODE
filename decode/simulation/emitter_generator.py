from typing import Tuple, Union
import time

import torch
from torch.distributions.exponential import Exponential

import numpy
from scipy.io import loadmat
from skimage.measure import label, regionprops

from decode.generic.emitter import EmitterSet

class MaskedEmitterSampler:
    emitters_sampled=0

    """
    static emitters, frame dependent
    """
    def __init__(self, *, 
            num_frames: int, xy_unit: str, px_size: Tuple[float, float], 
            z_range:tuple, min_brightness:float, max_brightness:float, em_avg:float):
        """

        Args:
            xy_unit:
            px_size:
            num_frames:

        """

        self.num_frames = num_frames

        self.zextent=z_range

        if not xy_unit in ("px","nm"):
            raise ValueError()

        self.xy_unit=xy_unit
        self.px_size=px_size

        self.em_avg=em_avg

        self.min_brightness=min_brightness
        self.max_brightness=max_brightness

    def __call__(self) -> EmitterSet:
        raise NotImplementedError
        #return self.sample()

    def sample(self, mask:numpy.ndarray, frame_index_override:int=None, fraction_emitters_above_zero:float=0.5, override_probs=None) -> EmitterSet:
        """
        Return sampled EmitterSet in the specified frame range.

        Returns:
            EmitterSet

        """
        
        # make (possibly) distance labeled mask binary
        label_mask=mask>0
        # label for regionprops (get bounding box for each cell in the mask)
        label_mask=label(label_mask)
        regions=regionprops(label_mask,intensity_image=mask, cache=True)

        pixel_area=(65e-3)**2 # currently hardcoded # TODO needs to be changed to take value from parameter file
        #max_area=3.5/pixel_area # currently unused (expected to be not actually an issue, currently sometimes is one due to slightly unfit segmentation mask)
        min_area=1/pixel_area
        # mask should have filtered areas with size smaller than min, but somehow this is not actually the case, so double check here
        cell_regions=[cell for cell in regions if cell.area>=min_area]

        # sample number of emitters per cell from experimental data
        # values taken from a graph konrad gave me once (representing a single dataset! best data i have currently)
        if override_probs is None:
            probs=torch.tensor([0.0,1.8,1.7,0.0,0.1])
            #probs=torch.tensor([0.1,5.0,3.0,0.5,1.0,0.5,0.5,0.5,0.4,0.4,0.4,0.4])
        else:
            probs=torch.tensor(override_probs)

        if probs.sum()>0:
            num_emitters:torch.Tensor=torch.multinomial(probs,num_samples=len(cell_regions),replacement=True)
        else:
            num_emitters=torch.zeros((len(cell_regions),))
            assert False, f"{num_emitters.shape=} {type(num_emitters[0])=}"

        total_num_emitters:int=num_emitters.sum().item()

        
        # sample emitter intensity (will be scaled appropriately few lines further down)
        # from this beta distribution
        beta_param0,beta_param1=7.5,3.5 # manually explored with https://keisan.casio.com/exec/system/1180573226
        #intensity_sampler=torch.distributions.beta.Beta(beta_param0,beta_param1)
        # or from this uniform distribution
        intensity_sampler=torch.distributions.uniform.Uniform(0.0,1.0)

        # (median) photon count per emitter
        #intensity:torch.Tensor = torch.clamp(self.intensity_dist.sample((total_num_emitters,)), min=self.intensity_th)
        #intensity:torch.Tensor = torch.zeros((total_num_emitters,))
        intensity:torch.Tensor = intensity_sampler.sample((total_num_emitters,))*(self.max_brightness-self.min_brightness)+self.min_brightness

        # convert mask from numpy to torch for later functions that require torch tensor
        torch_mask=torch.from_numpy(mask)

        # will be returned (list of all 3d emitter coordinates)
        total_emitter_coordinates=torch.zeros([total_num_emitters,3])

        assert self.zextent[0]<self.zextent[1] # should be symmetric, but is asserted nowhere. we make a math assumption below that requires at least this to hold
        
        #zrange=self.zextent[1]-self.zextent[0]
        z_sampler=torch.distributions.uniform.Uniform(self.zextent[0],self.zextent[1]) #torch.distributions.beta.Beta(5,5) is centered better, but unsure if this is what we actually want

        already_positioned_emitters:int=0

        # sample individual cell brightness (from a wide range for wider variety)
        cell_brightnesses=torch.distributions.uniform.Uniform(0.1,7).sample((len(cell_regions),))

        # for each cell/region in the mask:
        # this takes ~4/24 seconds of the simulation
        for region_id,region in enumerate(cell_regions):
            minr, minc, maxr, maxc = region.bbox
            #bx = (minc, maxc, maxc, minc, minc)
            #by = (minr, minr, maxr, maxr, minr)

            num_emitters_in_this_cell:int=num_emitters[region_id].item()

            if num_emitters_in_this_cell>0:
                mask_snippet=torch_mask[minr:maxr,minc:maxc] # _may_ include overlapping very close cells -> masking used further down
                mask_volume_snippet=mask[minr:maxr,minc:maxc]
                # apply cell background brightness to cell (individually sampled)
                mask_volume_snippet*=cell_brightnesses[region_id].item()

                # remove emitter brightness from cell background (more realistic model)
                adjust_cell_background_brightness=False
                if adjust_cell_background_brightness:
                    # calculate brightness of cell background (as if no dots were present)
                    cell_volume_total_brightness=mask_volume_snippet.sum().item()

                    # sample emitter brightness
                    cell_emitter_brightness=intensity[already_positioned_emitters:already_positioned_emitters+num_emitters_in_this_cell]

                    total_cell_emitter_brightness=cell_emitter_brightness.sum().item()

                    # rescale cell background brightness to remove emitters that bind to target sites
                    cell_background_scale=total_cell_emitter_brightness/cell_volume_total_brightness
                    if cell_background_scale>0.95:
                        cell_emitter_brightness=cell_emitter_brightness*0.95/cell_background_scale
                        cell_background_scale=0.95

                    mask_volume_snippet[mask_volume_snippet>0]*=1-cell_background_scale

                    # save those emitter brightness values to the emitter data
                    # and split emitter brightness evenly across all emitters inside the cell 
                    # (all bind to the same region on the dna, assumed to have identical reaction dynamics so that an emitter binding to any of these sites is equally likely)
                    intensity[already_positioned_emitters:already_positioned_emitters+num_emitters_in_this_cell]=cell_emitter_brightness

                # get coordinates of all pixels that are inside a cell
                cell_mask_nonzero_vector=mask_snippet.nonzero(as_tuple=False)
                # get value of the pixels that are inside cells
                cell_mask_nonzero_tuple=(cell_mask_nonzero_vector[:,0],cell_mask_nonzero_vector[:,1])
                cell_mask_weights=mask_snippet[cell_mask_nonzero_tuple]

                emitter_coordinates=total_emitter_coordinates[already_positioned_emitters:already_positioned_emitters+num_emitters_in_this_cell]

                # sample x/y coordinates from list of pixels inside cells, using the cell depth penetration value (term up for debate) as weight
                try:
                    emitter_coordinates[:,:2]=cell_mask_nonzero_vector[torch.multinomial(cell_mask_weights,num_samples=num_emitters_in_this_cell,replacement=True)]
                except RuntimeError:
                    print(f"{cell_mask_weights=} {cell_mask_nonzero_tuple=}")
                    raise RuntimeError()

                # offset coordinates for global (frame) placement (before this, the coordinates are in cell-local space)
                emitter_coordinates[:,0]+=minr
                emitter_coordinates[:,1]+=minc

                already_positioned_emitters+=num_emitters_in_this_cell

        assert already_positioned_emitters==total_num_emitters

        if total_num_emitters>0:
            # sample z coordinate from some non-experimental distribution (because we dont have better data here)
            # could multiply this with mask[emitter_coordinates] to actually sample from within the estimated cell body, though this might overfit the ai on data we _expect_ to be realistic (which e.g. also expects the segmentation mask to be perfect)
            # better leave it as is to increase the variety of the training data
            #total_emitter_coordinates[:,2]=z_sampler.sample((total_num_emitters,))

            # heavily focused around z=0
            #total_emitter_coordinates[:,2]=(torch.distributions.beta.Beta(5,5).sample((total_num_emitters,)))*(self.zextent[1]-self.zextent[0])-self.zextent[1]
            # distributed across whole range, with z=0 about twice as likely as z=+-500
            total_emitter_coordinates[:,2]=(torch.distributions.beta.Beta(1.2,1.2).sample((total_num_emitters,)))*(self.zextent[1]-self.zextent[0])-self.zextent[1]

            # add a random sub-pixel offset
            subpixel_offset_dist=torch.distributions.uniform.Uniform(-0.5,0.5)
            
            total_emitter_coordinates[:,:2]+=subpixel_offset_dist.sample((total_emitter_coordinates.shape[0],2))

            # remove the emitters that have been sampled outside the image frame for some reason (should not be more than 1-2 per frame)
            emitter_mask_outside_frame=(total_emitter_coordinates[:,0]>=0) & (total_emitter_coordinates[:,0]<mask.shape[0]) & (total_emitter_coordinates[:,1]>=0) & (total_emitter_coordinates[:,1]<mask.shape[1])
            total_emitter_coordinates=total_emitter_coordinates[emitter_mask_outside_frame]
            intensity=intensity[emitter_mask_outside_frame]

            total_num_emitters=emitter_mask_outside_frame.sum()

            num_emitters=total_emitter_coordinates.shape[0]
            z_above_zero_mask=total_emitter_coordinates[:,2]>0

            fraction_above_zero=z_above_zero_mask.sum()/num_emitters

            target_fraction=fraction_emitters_above_zero
            missing_fraction=target_fraction-fraction_above_zero

            if missing_fraction>0:
                indices_below_zero=numpy.arange(num_emitters)[~z_above_zero_mask]
                if missing_fraction>=0.5:
                    to_be_flipped_indices=indices_below_zero
                else:
                    to_be_flipped_indices=numpy.random.choice(indices_below_zero,size=int(missing_fraction*num_emitters),replace=False)
            else:#missing_fraction<0
                indices_above_zero=numpy.arange(num_emitters)[z_above_zero_mask]
                if missing_fraction<=-0.5:
                    to_be_flipped_indices=indices_above_zero
                else:
                    to_be_flipped_indices=numpy.random.choice(indices_above_zero,size=int(-missing_fraction*num_emitters),replace=False)

            total_emitter_coordinates[to_be_flipped_indices,2]*=-1.0

        self.em_avg=total_num_emitters # required by neuralfitter.train.live_engine_setup

        # replacing the realistic distribution above with something more uniform to possibly iron out some of the issues present at low emitter brightness value may look like the line below:
        # intensity:torch.Tensor = torch.distributions.uniform.Uniform(0.,1e5).sample((total_num_emitters,))
        
        # frame index of emitters
        frame_ix_:torch.Tensor=torch.zeros((total_num_emitters,))
        # id of emitters
        id_:torch.Tensor=torch.arange(total_num_emitters) # just give every emitter a unique id. does not matter for us since emitters exist for one frame only (at least for now where all frames are completely independent)

        id_+=MaskedEmitterSampler.emitters_sampled
        MaskedEmitterSampler.emitters_sampled+=total_num_emitters

        if not frame_index_override is None:
            frame_ix_+=frame_index_override

        return EmitterSet(total_emitter_coordinates, intensity, frame_ix_.long(), id_.long(), xy_unit=self.xy_unit, px_size=self.px_size) # px_size is not well documented. _should_ be nanometer per side?

    @classmethod
    def parse(cls, param, num_frames: int):
        return cls(xy_unit="px",
                   px_size=param.Camera.px_size,
                   num_frames=num_frames,
                   z_range=param.Simulation.emitter_extent[2],
                   min_brightness=param.Scaling.phot_min,
                   max_brightness=param.Scaling.phot_max,
                   em_avg=param.Simulation.emitter_av)