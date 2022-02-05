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
    , where px(i,j)=1 represents a cell at that pixel
    """
    def __init__(self, zextent: Tuple[float, float]):
        """
        Args:
            zextent: extent in z (lower_limit,upper_limit)

        Example:
            The following initialises this class in a range of 32 x 32 px in x and y and +/- 750nm in z.
            >>> prior_struct = RandomStructure(xextent=(-0.5, 31.5), yextent=(-0.5, 31.5), zextent=(-750., 750.))

        """

        super().__init__()

        self.zextent = zextent

    def sample(self, *, mask: numpy.ndarray, per_cell:bool=False) -> torch.Tensor:
        """
        Returns:
            tensor: list of emitter positions
        
        Arguments:
            per_cell : return emitters in a list that contains a list of emitters for each cell (if set to True), else the function returns a list of emitters in the frame (does not differentiate between cells)
            
        """
        
        label_mask=mask.copy()
        label_mask[mask>0]=255
        label_mask=label(label_mask)
        regions=regionprops(label_mask,cache=True)

        #pixel_area=(65e-3)**2
        #max_area=3.5/pixel_area
        cell_areas=torch.tensor([cell.area for cell in regions])

        probs=torch.tensor([0.0,1.8,1.7,0.0,0.1]) # values taken from a graph konrad gave me once (representing a single dataset! best data i have currently)
        num_emitters=torch.multinomial(probs,num_samples=cell_areas.shape[0],replacement=True)
        #print(num_emitters)

        mask=skimage.img_as_float(mask)
        torch_mask=torch.from_numpy(mask)

        total_emitter_coordinates=torch.tensor([])

        #plt.figure(figsize=(15,16))
        #plt.imshow(mask)
        for region_id,region in enumerate(regions):
            minr, minc, maxr, maxc = region.bbox
            #bx = (minc, maxc, maxc, minc, minc)
            #by = (minr, minr, maxr, maxr, minr)
            #plt.plot(bx,by)

            mask_snippet=torch_mask[minr:maxr,minc:maxc]

            cell_mask_nonzero_vector=mask_snippet.nonzero(as_tuple=False)

            cell_mask_nonzero_tuple=(cell_mask_nonzero_vector[:,0],cell_mask_nonzero_vector[:,1])

            cell_mask_weights=mask_snippet[cell_mask_nonzero_tuple]

            num_emitters_in_this_cell=num_emitters[region_id]

            emitter_coordinates=torch.zeros((num_emitters_in_this_cell,3))
            # TODO sample emitter positions outside pixel center (maybe add uniformly distributed offset within pixel area? should be good enough)
            emitter_coordinates[:,:2]=cell_mask_nonzero_vector[torch.multinomial(cell_mask_weights,num_samples=num_emitters_in_this_cell,replacement=False)]

            assert self.zextent[0]<self.zextent[1]
            
            zrange=self.zextent[1]-self.zextent[0]
            #emitters_z=torch.distributions.beta.Beta(5,5).sample((num_emitters_in_this_cell,)) * zrange # * mask[emitter_coordinates]
            emitters_z=torch.distributions.uniform.Uniform(0,zrange).sample((num_emitters_in_this_cell,)) # * mask[emitter_coordinates]
            emitter_coordinates[:,2]=emitters_z - self.zextent[1]

            emitter_coordinates[:,0]+=minr
            emitter_coordinates[:,1]+=minc

            if per_cell:
                total_emitter_coordinates=torch.cat((total_emitter_coordinates,[emitter_coordinates]),0)
            else:
                total_emitter_coordinates=torch.cat((total_emitter_coordinates,emitter_coordinates),0)
        
        return total_emitter_coordinates

    @classmethod
    def parse(cls, param):
        return cls(zextent=param.Simulation.emitter_extent[2])