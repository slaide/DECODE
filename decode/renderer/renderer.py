import math
from abc import ABC

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from matplotlib.colors import rgb_to_hsv
import PIL

from ..generic import emitter


class Renderer(ABC):
    """
    Renderer. Takes emitters and outputs a rendered image.

    """

    def __init__(self, plot_axis: tuple, xextent: tuple, yextent: tuple, px_size: float, abs_clip, perc_clip, contrast):
        super().__init__()

        self.xextent = xextent
        self.yextent = yextent
        self.px_size = px_size
        self.plot_axis = plot_axis
        
        self.abs_clip = abs_clip
        self.perc_clip = perc_clip
        
        self.contrast = contrast
        
        assert self.abs_clip is None or self.perc_clip is None, "Define either an absolute or a percentage value for clipping, but not both"

    @property
    def _npx_x(self):
        return math.ceil(self)
  
    def forward(self, em: emitter.EmitterSet) -> torch.Tensor:
        """
        Forward emitterset through rendering and output rendered data.

        Args:
            em: emitter set

        """
        raise NotImplementedError

    def render(self, em: emitter.EmitterSet, ax=None):
        """
        Render emitters

        Args:
            em: emitter set
            ax: plot axis

        Returns:

        """
        raise NotImplementedError

class Renderer2D(Renderer):
    """
    2D Renderer with constant gaussian.

    """

    def __init__(self, px_size, sigma_blur, plot_axis=(0, 1), xextent=None, yextent=None, abs_clip=None, perc_clip=None, contrast=1):
        super().__init__(plot_axis=plot_axis, xextent=xextent, yextent=yextent, px_size=px_size, abs_clip=abs_clip, perc_clip=perc_clip, contrast=contrast)

        self.sigma_blur = sigma_blur

    def render(self, em, ax=None, cmap: str = 'gray'):

        hist = self.forward(em).numpy()

        if ax is None:
            ax = plt.gca()

        ax = ax.imshow(np.transpose(hist), cmap=cmap)  # because imshow use different ordering
        return ax

    def forward(self, em: emitter.EmitterSet) -> torch.Tensor:

        if self.xextent is None:
            self.xextent = (em.xyz_nm[:, self.plot_axis[0]].min(), em.xyz_nm[:, self.plot_axis[0]].max())
        if self.yextent is None:
            self.yextent = (em.xyz_nm[:, self.plot_axis[1]].min(), em.xyz_nm[:, self.plot_axis[1]].max())

        hist = self._hist2d(em.xyz_nm[:, self.plot_axis].numpy(), xextent=self.xextent, yextent=self.yextent,
                            px_size=self.px_size)
        
        if self.perc_clip is not None:
            hist = np.clip(hist, 0., hist.max()*self.perc_clip)
        if self.abs_clip is not None:
            hist = np.clip(hist, 0., self.abs_clip)  
            
        if self.sigma_blur is not None:
            hist = gaussian_filter(hist, sigma=[self.sigma_blur / self.px_size, self.sigma_blur / self.px_size])
            
        hist = np.clip(hist, 0, hist.max()/self.contrast)
            
        return torch.from_numpy(hist)

    @staticmethod
    def _hist2d(xy: np.array, xextent, yextent, px_size) -> np.array:

        hist_bins_x = np.arange(xextent[0], xextent[1] + px_size, px_size)
        hist_bins_y = np.arange(yextent[0], yextent[1] + px_size, px_size)

        hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(hist_bins_x, hist_bins_y))

        return hist


class Renderer3D(Renderer):
    """
    3D Renderer with constant gaussian.

    """

    def __init__(self, px_size, sigma_blur, plot_axis=(0, 1, 2), xextent=None, yextent=None, zextent=None,
                 abs_clip=None, perc_clip=None, contrast=1):
        super().__init__(plot_axis=plot_axis, xextent=xextent, yextent=yextent, px_size=px_size, abs_clip=abs_clip, perc_clip=perc_clip, contrast=contrast)

        self.sigma_blur = sigma_blur
        self.zextent = zextent
        
        # get jet colormap
        lin_hue = np.linspace(0,1,256)
        cmap = plt.get_cmap('jet', lut=256);
        cmap = cmap(lin_hue)
        cmap_hsv = rgb_to_hsv(cmap[:,:3])
        jet_hue = cmap_hsv[:,0]
        _,b = np.unique(jet_hue, return_index=True)
        jet_hue = [jet_hue[index] for index in sorted(b)]
        self.jet_hue = np.interp(np.linspace(0,len(jet_hue),256), np.arange(len(jet_hue)), jet_hue)

    def render(self, em: emitter.EmitterSet):

        hist = self.forward(em).numpy()

        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.25, pad=-0.25)
        colb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap('jet'), values=np.linspace(0, 1., 101),
                                         norm=mpl.colors.Normalize(0., 1.))
        colb.outline.set_visible(False)

        cax.text(0.12, 0.04, f'{self.zextent[0]} nm', rotation=90, color='white', fontsize=15, transform=cax.transAxes)
        cax.text(0.12, 0.88, f'{self.zextent[1]} nm', rotation=90, color='white', fontsize=15, transform=cax.transAxes)
        cax.axis('off')

        ax = ax.imshow(np.transpose(hist, [1, 0, 2]))
        return ax

    def forward(self, em: emitter.EmitterSet) -> torch.Tensor:

        if self.xextent is None:
            self.xextent = (em.xyz_nm[:, self.plot_axis[0]].min(), em.xyz_nm[:, self.plot_axis[0]].max())
        if self.yextent is None:
            self.yextent = (em.xyz_nm[:, self.plot_axis[1]].min(), em.xyz_nm[:, self.plot_axis[1]].max())
        if self.zextent is None:
            self.zextent = (em.xyz_nm[:, self.plot_axis[2]].min(), em.xyz_nm[:, self.plot_axis[2]].max())

        int_hist, col_hist = self._hist2d(em.xyz_nm[:, self.plot_axis].numpy(), xextent=self.xextent,
                                          yextent=self.yextent, zextent=self.zextent, px_size=self.px_size)
                
        with np.errstate(divide='ignore', invalid='ignore'):
            z_avg = col_hist / int_hist
            
        if self.perc_clip is not None:
            int_hist = np.clip(int_hist*self.contrast, 0., int_hist.max()*self.perc_clip)
            val = int_hist / int_hist.max()
        elif self.abs_clip is not None:
            int_hist = np.clip(int_hist, 0., self.abs_clip) 
            val = int_hist / self.abs_clip
        else:
            val = int_hist / int_hist.max()           
            
        val *= self.contrast
            
        z_avg[np.isnan(z_avg)] = 0
        sat = np.ones(int_hist.shape)
        hue = np.interp(z_avg,np.linspace(0,1,256),self.jet_hue)
        
        HSV = np.concatenate((hue[:, :, None], sat[:, :, None], val[:, :, None]), -1)
        RGB = hsv_to_rgb(HSV)

        if self.sigma_blur:
            RGB = np.array([gaussian_filter(RGB[:, :, i], sigma=[self.sigma_blur / self.px_size,
                                                                 self.sigma_blur / self.px_size]) for i in
                            range(3)]).transpose(1, 2, 0)
    
        RGB = np.clip(RGB, 0, 1)
        return torch.from_numpy(RGB)

    @staticmethod
    def _hist2d(xyz: np.array, xextent, yextent, zextent, px_size) -> np.array:

        hist_bins_x = np.arange(xextent[0], xextent[1] + px_size, px_size)
        hist_bins_y = np.arange(yextent[0], yextent[1] + px_size, px_size)

        int_hist, _, _ = np.histogram2d(xyz[:, 0], xyz[:, 1], bins=(hist_bins_x, hist_bins_y))

        z_pos = np.clip(xyz[:, 2], zextent[0], zextent[1])
        z_weight = ((z_pos - z_pos.min()) / (z_pos.max() - z_pos.min()))

        col_hist, _, _ = np.histogram2d(xyz[:, 0], xyz[:, 1], bins=(hist_bins_x, hist_bins_y), weights=z_weight)

        return int_hist, col_hist
    
# class Renderer2D_auto_sig(Renderer2D):
    
#     def __init__(self, px_size, n_sig_bins=10, sigma_scale=1, plot_axis = (0,1,2), xextent=None, yextent=None, zextent=None, clip_percentile=100):
#         super().__init__(px_size=px_size, sigma_blur=None, plot_axis=plot_axis, xextent=xextent, yextent=yextent, zextent=None, clip_percentile=clip_percentile)
        
#         self.n_sig_bins = n_sig_bins
#         self.sigma_scale = sigma_scale
        
#     def forward(self, em: emitter.EmitterSet) -> torch.Tensor:

#         em, xyz_extent = self._apply_extents(em)

#         hist_col = []
        
#         for i in range(self.n_sig_bins):

#             sig_extent = [np.percentile(em.comb_sig.cpu().numpy(), (i/self.n_sig_bins) * 100.), np.percentile(em.comb_sig.cpu().numpy(), ((i+1)/self.n_sig_bins) * 100.)]
#             em_sub = em[(em.comb_sig > sig_extent[0]) * (em.comb_sig < sig_extent[1])]
#             sigma_blur = (em_sub.comb_sig.mean() * em.px_size.mean()).numpy()
#             sigma_blur *= self.sigma_scale
            
#             hist = self._hist2d(em_sub.xyz_nm[:, self.plot_axis].numpy(), xextent=xyz_extent[self.plot_axis[0]], yextent=xyz_extent[self.plot_axis[1]], px_size=self.px_size)   
#             hist = gaussian_filter(hist, sigma=[sigma_blur / self.px_size, sigma_blur / self.px_size])
            
#             hist_col.append(hist)
            
#         hist = np.array(hist_col).sum(0)
            
#         if self.clip_percentile is not None:
#             hist = np.clip(hist, 0., np.percentile(hist, self.clip_percentile))
            
#         return torch.from_numpy(hist)    
    
    
# class Renderer3D_auto_sig(Renderer3D):
    
#     def __init__(self, px_size, n_sig_bins=10, sigma_scale=1, plot_axis = (0,1,2), xextent=None, yextent=None, zextent=None, clip_percentile=100, contrast=1):
#         super().__init__(px_size=px_size, sigma_blur=None, plot_axis=plot_axis, xextent=xextent, yextent=yextent, zextent=zextent, clip_percentile=clip_percentile, contrast=contrast)
        
#         self.n_sig_bins = n_sig_bins
#         self.sigma_scale = sigma_scale
        
#     def forward(self, em: emitter.EmitterSet) -> torch.Tensor:

#         em, xyz_extent = self._apply_extents(em)

#         int_col = []
#         col_col = []
        
#         for i in range(self.n_sig_bins):

#             sig_extent = [np.percentile(em.comb_sig.cpu().numpy(), (i/self.n_sig_bins) * 100.), np.percentile(em.comb_sig.cpu().numpy(), ((i+1)/self.n_sig_bins) * 100.)]
#             em_sub = em[(em.comb_sig > sig_extent[0]) * (em.comb_sig < sig_extent[1])]
#             sigma_blur = (em_sub.comb_sig.mean() * em.px_size.mean()).numpy()
#             sigma_blur *= self.sigma_scale

#             int_hist, col_hist = self._hist2d(em_sub.xyz_nm[:, self.plot_axis].numpy(), xextent=xyz_extent[self.plot_axis[0]], yextent=xyz_extent[self.plot_axis[1]], zextent=xyz_extent[self.plot_axis[2]], px_size=self.px_size)
            
#             int_col.append(gaussian_filter(int_hist, sigma=[sigma_blur / self.px_size, sigma_blur / self.px_size]))
#             col_col.append(gaussian_filter(col_hist, sigma=[sigma_blur / self.px_size, sigma_blur / self.px_size]))
            
#         int_hist = np.array(int_col).sum(0)
#         col_hist = np.array(col_col).sum(0)
            
#         RGB = self._hists_to_rgb(int_hist, col_hist)

#         return torch.from_numpy(RGB)
