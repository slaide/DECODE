from abc import ABC, abstractmethod  # abstract class
from deprecated import deprecated

import numpy as np
import torch
from torch.distributions.exponential import Exponential

import decode.generic.emitter
from . import structure_prior

from typing import Tuple, Union

class EmitterSampler(ABC):
    """
    Abstract emitter sampler. All implementations / children must implement a sample method.
    """

    def __init__(self, structure: structure_prior.StructurePrior, xy_unit: str, px_size: tuple):

        super().__init__()

        self.structure = structure
        self.px_size = px_size
        self.xy_unit = xy_unit

    def __call__(self) -> decode.generic.emitter.EmitterSet:
        return self.sample()

    @abstractmethod
    def sample(self) -> decode.generic.emitter.EmitterSet:
        raise NotImplementedError


class EmitterSamplerFrameIndependent(EmitterSampler):
    """
    Simple Emitter sampler. Samples emitters from a structure and puts them all on the same frame, i.e. their
    blinking model is not modelled.

    """

    def __init__(self, *, structure: structure_prior.StructurePrior, photon_range: tuple,
                 density: float = None, em_avg: float = None, xy_unit: str, px_size: tuple):
        """

        Args:
            structure: structure to sample from
            photon_range: range of photon value to sample from (uniformly)
            density: target emitter density (exactly only when em_avg is None)
            em_avg: target emitter average (exactly only when density is None)
            xy_unit: emitter xy unit
            px_size: emitter pixel size

        """

        super().__init__(structure=structure, xy_unit=xy_unit, px_size=px_size)

        self._density = density
        self.photon_range = photon_range

        """
        Sanity Checks.
        U shall not pa(rse)! (Emitter Average and Density at the same time!
        """
        if (density is None and em_avg is None) or (density is not None and em_avg is not None):
            raise ValueError("You must XOR parse either density or emitter average. Not both or none.")

        self.area = self.structure.area

        if em_avg is not None:
            self._em_avg = em_avg
        else:
            self._em_avg = self._density * self.area

    @property
    def em_avg(self) -> float:
        return self._em_avg

    def sample(self) -> decode.generic.emitter.EmitterSet:
        """
        Sample an EmitterSet.

        Returns:
            EmitterSet:

        """
        n = np.random.poisson(lam=self._em_avg)

        return self.sample_n(n=n)

    def sample_n(self, n: int) -> decode.generic.emitter.EmitterSet:
        """
        Sample 'n' emitters, i.e. the number of emitters is given and is not immediately sampled from the Poisson dist.

        Args:
            n: number of emitters

        """

        if n < 0:
            raise ValueError("Negative number of samples is not well-defined.")

        xyz = self.structure.sample(n)
        phot = torch.randint(*self.photon_range, (n,))

        return decode.generic.emitter.EmitterSet(xyz=xyz, phot=phot,
                                                 frame_ix=torch.zeros_like(phot).long(),
                                                 id=torch.arange(n).long(),
                                                 xy_unit=self.xy_unit,
                                                 px_size=self.px_size)

class EmitterSamplerBlinking(EmitterSamplerFrameIndependent):
    def __init__(self, *, structure: structure_prior.StructurePrior, intensity_mu_sig: tuple, lifetime: float,
                 frame_range: tuple, xy_unit: str, px_size: tuple, density=None, em_avg=None, intensity_th=None):
        """

        Args:
            structure:
            intensity_mu_sig:
            lifetime:
            xy_unit:
            px_size:
            frame_range: specifies the frame range
            density:
            em_avg:
            intensity_th:

        """
        super().__init__(structure=structure,
                         photon_range=None,
                         xy_unit=xy_unit,
                         px_size=px_size,
                         density=density,
                         em_avg=em_avg)

        self.n_sampler = np.random.poisson
        self.frame_range = frame_range
        self.intensity_mu_sig = intensity_mu_sig
        self.intensity_dist = torch.distributions.normal.Normal(self.intensity_mu_sig[0],
                                                                self.intensity_mu_sig[1])
        self.intensity_th = intensity_th if intensity_th is not None else 1e-8
        self.lifetime_avg = lifetime
        self.lifetime_dist = Exponential(1 / self.lifetime_avg)  # parse the rate not the scale ...

        self.t0_dist = torch.distributions.uniform.Uniform(*self._frame_range_plus)

        """
        Determine the total number of emitters. Depends on lifetime, frames and emitters.
        (lifetime + 1) because of binning effect.
        """
        self._emitter_av_total = self._em_avg * self._num_frames_plus / (self.lifetime_avg + 1)

    @property
    def num_frames(self):
        return self.frame_range[1] - self.frame_range[0] + 1

    @property
    def _frame_range_plus(self):
        """
        Frame range including buffer in front and end to account for build up effects.

        """
        return self.frame_range[0] - 3 * self.lifetime_avg, self.frame_range[1] + 3 * self.lifetime_avg

    @property
    def _num_frames_plus(self):
        return self._frame_range_plus[1] - self._frame_range_plus[0] + 1

    def sample(self):
        """
        Return sampled EmitterSet in the specified frame range.

        Returns:
            EmitterSet

        """

        n = self.n_sampler(self._emitter_av_total)

        loose_em = self.sample_loose_emitter(n=n)
        em = loose_em.return_emitterset()
        em = em.get_subset_frame(*self.frame_range)  # because the simulated frame range is larger

        return em

    def sample_n(self, *args, **kwargs):
        raise NotImplementedError

    def sample_loose_emitter(self, n) -> decode.generic.emitter.LooseEmitterSet:
        """
        Generate loose EmitterSet. Loose emitters are emitters that are not yet binned to frames.

        Args:
            n: number of 'loose' emitters

        Returns:
            LooseEmitterSet

        """

        xyz = self.structure.sample(n)

        """Draw from intensity distribution but clamp the value so as not to fall below 0."""
        intensity = torch.clamp(self.intensity_dist.sample((n,)), self.intensity_th)

        """Distribute emitters in time. Increase the range a bit."""
        t0 = self.t0_dist.sample((n,))
        ontime = self.lifetime_dist.rsample((n,))

        return decode.generic.emitter.LooseEmitterSet(xyz, intensity, ontime, t0, id=torch.arange(n).long(),
                                                      xy_unit=self.xy_unit, px_size=self.px_size)

    @classmethod
    def parse(cls, param, structure, frames: tuple):
        return cls(structure=structure,
                   intensity_mu_sig=param.Simulation.intensity_mu_sig,
                   lifetime=param.Simulation.lifetime_avg,
                   xy_unit=param.Simulation.xy_unit,
                   px_size=param.Camera.px_size,
                   frame_range=frames,
                   density=param.Simulation.density,
                   em_avg=param.Simulation.emitter_av,
                   intensity_th=param.Simulation.intensity_th)

from decode.generic.emitter import EmitterSet

import matplotlib.pyplot as plt

import numpy
from scipy.io import loadmat
from skimage.measure import label, regionprops
import time
class DiscreteEmitterIntensitySampler:
    def __init__(self,
        path:str,
        p:str
    ):
        self.path=path

        mat=loadmat(path)

        if p not in ("c","v"):
            raise ValueError(f"DiscreteEmitterIntensiySampler.sample:p must be in (v,c) but is {p}")

        temp_intensity_scale_factor:float=1.0

        if p=="c":
            photonCounts=mat["photonCountsC"]
        else: # if p=="v":
            photonCounts=mat["photonCountsV"]

        self.max_brightness=10000 # manually filtered by konrad
        self.max_brightness=self.max_brightness
        num_bins=20000

        self.bin_max=self.max_brightness
        self.bin_min=0
        self.num_bins=num_bins
        self.bin_step=self.max_brightness/num_bins

        self.max_phot_count=photonCounts.max()
        self.mean=photonCounts.mean()

        assert self.max_phot_count<=self.max_brightness

        bins=numpy.arange(self.bin_min,self.bin_max,self.bin_step)

        binned_phot_count,_bins_c=numpy.histogram(photonCounts,bins)

        binned_phot_count=binned_phot_count.astype(numpy.float32)#/binned_phot_count.sum()

        self.intensity_dist=torch.from_numpy(binned_phot_count)

    def sample(self,shape:tuple):
        assert len(shape)==1
        if shape[0]>0:
            intensities=torch.multinomial(self.intensity_dist,num_samples=shape[0],replacement=True)+self.bin_min
            intensities=intensities*self.bin_step
        else:
            intensities=torch.zeros((shape[0],))

        return intensities

class MaskedEmitterSampler:
    emitters_sampled=0

    """
    static emitters, frame dependent
    """
    def __init__(self, *, intensity_mu_sig: Union[Tuple[float,float],Tuple[str,str]],
                num_frames: int, xy_unit: str, px_size: Tuple[float, float], z_range:tuple, min_brightness:float, max_brightness:float, intensity_th:float = 1e-8):
        """

        Args:
            structure:
            intensity_mu_sig:
            xy_unit:
            px_size:
            num_frames:
            intensity_th:

        """

        self.num_frames = num_frames
        #self.intensity_mu_sig = intensity_mu_sig
        #if isinstance(self.intensity_mu_sig[0],float):
        #    assert False

        #    assert isinstance(self.intensity_mu_sig[1],float)
        #    self.intensity_dist_type="normal"
        #    self.intensity_dist = torch.distributions.normal.Normal(self.intensity_mu_sig[0],
        #                                                            self.intensity_mu_sig[1])
        #else:
        #    assert isinstance(self.intensity_mu_sig[0],str)
        #    assert isinstance(self.intensity_mu_sig[1],str)
        #    self.intensity_dist_type="discrete"
        #    self.intensity_dist=DiscreteEmitterIntensitySampler(self.intensity_mu_sig[0],
        #                                                        self.intensity_mu_sig[1])

        self.intensity_th = intensity_th
        self.zextent=z_range

        if not xy_unit in ("px","nm"):
            raise ValueError()

        self.xy_unit=xy_unit
        self.px_size=px_size

        self.em_avg=None

        self.min_brightness=min_brightness
        self.max_brightness=max_brightness

    def __call__(self) -> decode.generic.emitter.EmitterSet:
        raise NotImplementedError
        #return self.sample()

    def sample(self, mask:numpy.ndarray, frame_index_override:int=None, fraction_emitters_above_zero:float=0.5, override_probs=None) -> decode.generic.emitter.EmitterSet:
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
        return cls(intensity_mu_sig=param.Simulation.intensity_mu_sig,
                   xy_unit="px",
                   px_size=param.Camera.px_size,
                   num_frames=num_frames,
                   z_range=param.Simulation.emitter_extent[2],
                   min_brightness=param.Scaling.phot_min,
                   max_brightness=param.Scaling.phot_max,
                   intensity_th=param.Simulation.intensity_th or 1e-8)

    # fake a normal distribution for network input scaling
    def _intensity_mu_sig(self):
        if self.intensity_dist_type=="discrete":
            mu=float(self.intensity_dist.mean)
            sig=float(self.intensity_dist.mean)/2
            return (mu,sig)
        else:
            raise RuntimeError()