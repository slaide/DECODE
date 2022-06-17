from abc import ABC, abstractmethod  # abstract class
from collections import namedtuple

import numpy as np
import torch

from decode.simulation import psf_kernel as psf_kernel
import decode.utils

from typing import Union, Tuple

class Background(ABC):
    """
    Abstract background class. All children must implement a sample method.
    """

    _forward_modes = ('like', 'cum', 'tuple')
    _bg_return = namedtuple('bg_return', ['xbg', 'bg'])  # return arguments, x plus bg and bg term alone

    def __init__(self, forward_return: str = None):
        """

        Args:
            forward_return: determines the return of the forward function. 'like' returns a sample of the same size
                as the input, 'cum' adds the sample to the input and 'tuple' returns both the sum and the bg component
                alone.

        """
        super().__init__()

        self.forward_return=forward_return or "tuple"

        self.sanity_check()

    def sanity_check(self):
        """
        Tests the sanity of the instance.
        """

        if self.forward_return not in self._forward_modes:
            raise ValueError(f"Forward return mode {self.forward_return} unsupported. "
                             f"Available modes are: {self._forward_modes}")

    @abstractmethod
    def sample(self, size: torch.Size, device=torch.device('cpu')) -> torch.Tensor:
        """
        Samples from background implementation in the specified size.

        Args:
            size: size of the sample
            device: where to put the data

        Returns:
            background sample

        """
        raise NotImplementedError

    def sample_like(self, x: torch.Tensor) -> torch.Tensor:
        """
        Samples background in the shape and on the device as the the input.

        Args:
            x: input

        Returns:
            background sample

        """
        return self.sample(size=x.size(), device=x.device)

    def forward(self, x: torch.Tensor):
        """
        Samples background in the same shape and on the same device as the input x.
        Depending on the 'forward_return' attribute the bg is
            - returned alone ('like')
            - added to the input ('cum')
            - is added and returned as tuple ('tuple')

        Args:
            x: input frames. Dimension :math:`(N,C,H,W)`

        Returns:
            (see above description)

        """

        bg = self.sample_like(x)

        if self.forward_return == 'like':
            return bg

        elif self.forward_return == 'cum':
            return bg + x

        elif self.forward_return == 'tuple':
            return self._bg_return(xbg=x + bg, bg=bg)

        else:
            raise ValueError


class UniformBackground(Background):
    """
    Spatially constant background (i.e. a constant offset).

    """

    def __init__(self, bg_uniform: Union[float, Tuple[float, float]] = None, bg_sampler=None, forward_return=None):
        """
        Adds spatially constant background.

        Args:
            bg_uniform (float or tuple of floats): background value or background range. If tuple (bg range) the value
                will be sampled from a random uniform.
            bg_sampler (function): a custom bg sampler function that can take a sample_shape argument

        """
        super().__init__(forward_return=forward_return)

        if (bg_uniform is not None) and (bg_sampler is not None):
            raise ValueError("You must either specify bg_uniform XOR a bg_distribution")

        if bg_sampler is None:
            if isinstance(bg_uniform, (list, tuple)):
                self._bg_distribution = torch.distributions.uniform.Uniform(*bg_uniform).sample
            else:
                self._bg_distribution = _get_delta_sampler(bg_uniform)

        else:
            self._bg_distribution = bg_sampler

    @staticmethod
    def parse(param):
        return UniformBackground(param.Simulation.bg_uniform)

    def sample(self, size:Union[torch.Tensor,list,tuple], device=torch.device('cpu')) -> torch.Tensor:

        assert len(size) in (2, 3, 4), "Not implemented size spec."

        # create as many sample as there are batch-dims
        bg = self._bg_distribution(sample_shape=[size[0]] if len(size) >= 3 else torch.Size([]))

        # unsqueeze until we have enough dimensions
        if len(size) >= 3:
            bg = bg.view(-1, *((1,) * (len(size) - 1)))

        return bg.to(device) * torch.ones(size, device=device)


def _get_delta_sampler(val: float):
    def delta_sampler(sample_shape) -> float:
        return val * torch.ones(sample_shape)

    return delta_sampler

import skimage
import numpy
import matplotlib.pyplot as plt
import pickle
import edt
import math
from skimage import morphology
from skimage import filters
from skimage.measure import label,regionprops

class DiscreteBackground:
    def __init__(self,environmental_background,mean_brightness_per_volume,gaussian_width):
        self.environmental_background=environmental_background
        self.mean_brightness_per_volume=mean_brightness_per_volume
        self.gaussian_width=gaussian_width

    # mask is binary
    def sample(self, *, mask:numpy.ndarray, device:Union[torch.device,str]=torch.device('cpu')) -> torch.Tensor:
        mask=morphology.remove_small_holes(mask>0,area_threshold=64) # arbitrary threshold, 64 is default
        mask_edt=edt.edt(mask.astype(numpy.float32),parallel=0)

        for cell in regionprops(label(mask)):
            minr, minc, maxr, maxc = cell.bbox

            snippet=mask_edt[minr:maxr,minc:maxc]
            radius=snippet.max()
            mask_edt[minr:maxr,minc:maxc]=numpy.sqrt(radius**2-(radius-snippet)**2)

        sampled_bg=mask_edt*self.mean_brightness_per_volume # TODO subtract the brightness from the dots from this mask (meaning multiply by a fraction so that sumOfCellBackgroundFluorescence+sumOfDotFluorescence=sumOfCellVolume*mean_brightness_per_volume, where mean_brightness_per_volume was measured in images where no fluorophore is bound to the chromosome)
        sampled_bg+=self.environmental_background

        sampled_bg=filters.gaussian(sampled_bg,self.gaussian_width)

        return torch.from_numpy(sampled_bg).to(device)

    def _bg_uniform(self):
        max_cell_radius=math.ceil(1/0.065/2)

        return max_cell_radius*self.mean_brightness_per_volume

    @staticmethod
    def parse(param):
        return DiscreteBackground(
            param.Simulation.background.environmental_background,
            param.Simulation.background.mean_brightness_per_volume,
            param.Simulation.background.gaussian_width
        )