import torch

from abc import ABC, abstractmethod
from typing import Tuple

import decode.utils

import matplotlib.pyplot as pyplt
from torch.nn.functional import normalize

import skimage
import skimage.io

def read_img(path):
    img_data=skimage.io.imread(str(path), as_gray = True)
    img_data=skimage.img_as_ubyte(img_data)
    return img_data
def write_img(path,img_data):
    img_data=skimage.img_as_ubyte(img_data)
    skimage.io.imsave(str(path), img_data, plugin='tifffile', compress = 6, check_contrast=False)

class StructurePrior(ABC):
    """
    Abstract structure which can be sampled from. All implementation / childs must define a 'pop' method and an area
    property that describes the area the structure occupies.

    """

    @property
    @abstractmethod
    def area(self) -> float:
        """
        Calculate the area which is occupied by the structure. This is useful to later calculate the density,
        and the effective number of emitters). This is the 2D projection. Not the volume.

        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """
        Sample n samples from structure.

        Args:
            n: number of samples

        """
        raise NotImplementedError


class RandomStructure(StructurePrior):
    """
    Random uniform 3D / 2D structure. As the name suggests, sampling from this structure gives samples from a 3D / 2D
    volume that origin from a uniform distribution.

    """

    def __init__(self, xextent: Tuple[float, float], yextent: Tuple[float, float], zextent: Tuple[float, float]):
        """

        Args:
            xextent: extent in x
            yextent: extent in y
            zextent: extent in z, set (0., 0.) for a 2D structure

        Example:
            The following initialises this class in a range of 32 x 32 px in x and y and +/- 750nm in z.
            >>> prior_struct = RandomStructure(xextent=(-0.5, 31.5), yextent=(-0.5, 31.5), zextent=(-750., 750.))

        """

        super().__init__()

        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent

        self.scale = torch.tensor([(self.xextent[1] - self.xextent[0]),
                                   (self.yextent[1] - self.yextent[0]),
                                   (self.zextent[1] - self.zextent[0])])

        self.shift = torch.tensor([self.xextent[0],
                                   self.yextent[0],
                                   self.zextent[0]])

    @property
    def area(self) -> float:
        return (self.xextent[1] - self.xextent[0]) * (self.yextent[1] - self.yextent[0])

    def sample(self, n: int) -> torch.Tensor:
        xyz = torch.rand((n, 3)) * self.scale + self.shift
        return xyz

    @classmethod
    def parse(cls, param):
        return cls(xextent=param.Simulation.emitter_extent[0],
                   yextent=param.Simulation.emitter_extent[1],
                   zextent=param.Simulation.emitter_extent[2])

import torch
import numpy

import matplotlib.pyplot as plt

from skimage.measure import label, regionprops, regionprops_table
import skimage.morphology

class CellMaskStructure:
    """
    sample emitters based on pixel values in a (cell) segmentation mask
    , where px(i,j)>0 represents a (part of a) cell at that pixel
    """
    def __init__(self, zextent: Tuple[float, float]):
        """
        Args:
            zextent: extent in z in nm (lower_limit,upper_limit)

        """

        super().__init__()

        self.zextent = zextent

    def sample(self, *, mask: numpy.ndarray, per_cell:bool=False, fraction_emitters_above_zero:float=0.5,override_probs=None) -> torch.Tensor:
        """
        Returns:
            tensor: list of emitter positions
        
        Arguments:
            per_cell : return emitters in a list that contains a list of emitters for each cell (if set to True), else the function returns a list of emitters in the frame (does not differentiate between cells)
            
        """
        
        # make (possibly) distance labeled mask binary
        label_mask=mask>0
        # label for regionprops (get bounding box for each cell in the mask)
        label_mask=label(label_mask)
        regions=regionprops(label_mask,cache=True)

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
            num_emitters=torch.multinomial(probs,num_samples=len(cell_regions),replacement=True)
            #print(num_emitters)
        else:
            num_emitters=torch.zeros((len(cell_regions),))
            assert False, f"{num_emitters.shape=} {type(num_emitters[0])=}"

        # convert mask from numpy to torch for later functions that require torch tensor
        mask=skimage.img_as_float(mask)
        torch_mask=torch.from_numpy(mask)

        # will be returned (list of all emitter coordinates in 3d)
        # size is random # TODO slight performance improvement because size is random, but known at this point: num_emitters has already been sampled
        total_emitter_coordinates=torch.tensor([])

        assert self.zextent[0]<self.zextent[1] # should be symmetric, but is asserted nowhere. we make a math assumption below that requires at least this to hold
        
        #zrange=self.zextent[1]-self.zextent[0]
        z_sampler=torch.distributions.uniform.Uniform(self.zextent[0],self.zextent[1]) #torch.distributions.beta.Beta(5,5) is centered better, but unsure if this is what we actually want

        #plt.figure(figsize=(15,16))
        #plt.imshow(mask)
        # for each cell/region in the mask:
        for region_id,region in enumerate(cell_regions):
            minr, minc, maxr, maxc = region.bbox
            #bx = (minc, maxc, maxc, minc, minc)
            #by = (minr, minr, maxr, maxr, minr)
            #plt.plot(bx,by)

            mask_snippet=torch_mask[minr:maxr,minc:maxc] # _may_ include overlapping very close cells (so far, has not been an issue)

            # get coordinates of all pixels that are inside a cell
            cell_mask_nonzero_vector=mask_snippet.nonzero(as_tuple=False)
            # get value of the pixels that are inside cells
            cell_mask_nonzero_tuple=(cell_mask_nonzero_vector[:,0],cell_mask_nonzero_vector[:,1])
            cell_mask_weights=mask_snippet[cell_mask_nonzero_tuple]

            num_emitters_in_this_cell=num_emitters[region_id].item()
            if num_emitters_in_this_cell>0:
                emitter_coordinates=torch.zeros((num_emitters_in_this_cell,3))
                # sample x/y coordinates from list of pixels inside cells, using the cell depth penetration value (term up for debate) as weight
                emitter_coordinates[:,:2]=cell_mask_nonzero_vector[torch.multinomial(cell_mask_weights,num_samples=num_emitters_in_this_cell,replacement=False)]

                # sample z coordinate from some non-experimental distribution (because we dont have better data here)
                # could multiply this with mask[emitter_coordinates] to actually sample from within the estimated cell body, though this might overfit the ai on data we _expect_ to be realistic (which e.g. also expects the segmentation mask to be perfect)
                # better leave it as is to increase the variety of the training data
                emitter_coordinates[:,2]=z_sampler.sample((num_emitters_in_this_cell,))

                # offset coordinates for global (frame) placement (before this, the coordinates are in cell-local space)
                emitter_coordinates[:,0]+=minr
                emitter_coordinates[:,1]+=minc

                # if the coordinates are supposed to be returned grouped per cell, do so
                if per_cell:
                    total_emitter_coordinates=torch.cat((total_emitter_coordinates,[emitter_coordinates]),0)
                else:
                    total_emitter_coordinates=torch.cat((total_emitter_coordinates,emitter_coordinates),0)

        if len(total_emitter_coordinates)>0:
            # add a random sub-pixel offset
            subpixel_offset_dist=torch.distributions.uniform.Uniform(-0.5,0.5)
            
            total_emitter_coordinates[:,:2]+=subpixel_offset_dist.sample((total_emitter_coordinates.shape[0],2))

            # remove the emitters that have been sampled outside the image frame for some reason (should not be more than 1-2 per frame)
            emitter_mask_outside_frame=(total_emitter_coordinates[:,0]>=0) & (total_emitter_coordinates[:,0]<mask.shape[0]) & (total_emitter_coordinates[:,1]>=0) & (total_emitter_coordinates[:,1]<mask.shape[1])
            total_emitter_coordinates=total_emitter_coordinates[emitter_mask_outside_frame]

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

        return total_emitter_coordinates

    @classmethod
    def parse(cls, param):
        return cls(zextent=param.Simulation.emitter_extent[2])