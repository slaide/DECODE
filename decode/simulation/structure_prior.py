import torch

from abc import ABC, abstractmethod
from typing import Tuple

import decode.utils

import matplotlib.pyplot as pyplt
from torch.nn.functional import normalize

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

class CellMaskStructure(StructurePrior):
    """
    sample emitters based on pixel values in a (cell) segmentation mask
    , where px(i,j)=1 represents a cell at that pixel
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

        # use exemplary cell mask for now # TODO implement different solution for cell masks
        self.cell_mask=decode.utils.frames_io.load_tif("flowcell_40px_mask.tif").cpu()[:,:,0] # for some reason, this image has 3 color channels with the same value instead of being greyscale, therefore remove all but one channel
        # clamp the values to 1
        self.cell_mask/=self.cell_mask.max()
        # then invert the values because the (current, exemplary) cell segmentation mask is inverted.. because someone did not pay attention
        self.cell_mask=1-self.cell_mask
        # rotate image by 90 degrees because axis are 'inverted'? (inverted axis and rotation by 90 degress are not the same, but good enough for me, as long as it is consistent)
        self.cell_mask=torch.rot90(self.cell_mask)

        # get indexable array (matrix) of positions in cell mask with non-zero entries
        self.cell_mask_nonzero_vector=self.cell_mask.nonzero(as_tuple=False)
        # get tuple of positions in cell mask with non-zero entries, which can be used to index an array
        self.cell_mask_nonzero_tuple=self.cell_mask.nonzero(as_tuple=True)
        # precompute the list of non-zero cell mask values (currently, they are all 1)
        self.cell_mask_weights=self.cell_mask[self.cell_mask_nonzero_tuple]

    @property
    def area(self) -> float:
        return (self.xextent[1] - self.xextent[0]) * (self.yextent[1] - self.yextent[0])

    def sample(self, n: int) -> torch.Tensor:
        #randomly sample n points in 3d, in image volumne
        xyz = torch.rand((n, 3)) * self.scale + self.shift

        # sample n pixel indices with non-zero value
        cell_mask_random_indices=self.cell_mask_nonzero_vector[torch.multinomial(self.cell_mask_weights,num_samples=n,replacement=True)] # TODO replacement is required here right now, because we sample for multiple images at the same time and therefore cannot respect the density per image
        # pyplt.scatter(cell_mask_random_indices[:,0],cell_mask_random_indices[:,1])

        # set xy position of randomly sampled fluorophore positions to pixels with known 1 value #TODO this centers the position on the pixels, therefore does not sample 'between' pixels!!!!!!
        xyz[:,0:2]=cell_mask_random_indices

        return xyz

    @classmethod
    def parse(cls, param):
        return cls(xextent=param.Simulation.emitter_extent[0],
                   yextent=param.Simulation.emitter_extent[1],
                   zextent=param.Simulation.emitter_extent[2])