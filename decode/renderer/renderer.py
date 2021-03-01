from abc import ABC
import torch
from ..generic import emitter
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter


class Renderer(ABC):
    """
    Renderer. Takes emitters and outputs a rendered image.

    """

    def __init__(self, plot_axis: tuple, xextent: tuple, yextent: tuple, zextent: tuple, px_size: float):
        super().__init__()

        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent
        self.px_size = px_size
        self.plot_axis = plot_axis
        
    @property
    def _npx_x(self):
        return math.ceil(se)
    
    def _apply_extents(self, em):
        
        xextent = (em.xyz_nm[:, 0].min(), em.xyz_nm[:, 0].max()) if self.xextent is None else self.xextent
        yextent = (em.xyz_nm[:, 1].min(), em.xyz_nm[:, 1].max()) if self.yextent is None else self.yextent
        zextent = (em.xyz_nm[:, 2].min(), em.xyz_nm[:, 2].max()) if self.zextent is None else self.zextent
        
        em_sub = em.clone()
        em_sub = em_sub[(em_sub.xyz_nm[:,0]>xextent[0])*(em_sub.xyz_nm[:,0]<xextent[1])]
        em_sub = em_sub[(em_sub.xyz_nm[:,1]>yextent[0])*(em_sub.xyz_nm[:,1]<yextent[1])]
#         em_sub = em_sub[(em_sub.xyz_nm[:,2]>zextent[0])*(em_sub.xyz_nm[:,2]<zextent[1])]
        
        return em_sub, [xextent, yextent, zextent]
            
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

    def __init__(self, px_size, sigma_blur, plot_axis = (0,1), xextent=None, yextent=None, zextent=None, clip_percentile=None):
        super().__init__(plot_axis=plot_axis,xextent=xextent, yextent=yextent, zextent=None, px_size=px_size)

        self.sigma_blur = sigma_blur
        self.clip_percentile = clip_percentile

    def render(self, em, ax=None, cmap: str = 'gray'):

        hist = self.forward(em).numpy()

        if ax is None:
            ax = plt.gca()

        ax.imshow(np.transpose(hist), cmap=cmap)  # because imshow use different ordering
        return ax

    def forward(self, em: emitter.EmitterSet) -> torch.Tensor:

        em, xyz_extent = self._apply_extents(em)

        hist = self._hist2d(em.xyz_nm[:, self.plot_axis].numpy(), xextent=xyz_extent[self.plot_axis[0]], yextent=xyz_extent[self.plot_axis[1]], px_size=self.px_size)   

        if self.sigma_blur is not None:
            hist = gaussian_filter(hist, sigma=[self.sigma_blur / self.px_size, self.sigma_blur / self.px_size])
            
        if self.clip_percentile is not None:
            hist = np.clip(hist, 0., np.percentile(hist, self.clip_percentile))

        return torch.from_numpy(hist)

    @staticmethod
    def _hist2d(xy: np.array, xextent, yextent, px_size) -> np.array:

        hist_bins_x = np.arange(xextent[0], xextent[1] + px_size, px_size)
        hist_bins_y = np.arange(yextent[0], yextent[1] + px_size, px_size)

        hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(hist_bins_x, hist_bins_y))

        return hist
    
    
class Renderer2D_auto_sig(Renderer2D):
    
    def __init__(self, px_size, n_sig_bins=10, sigma_scale=1, plot_axis = (0,1,2), xextent=None, yextent=None, zextent=None, clip_percentile=100):
        super().__init__(px_size=px_size, sigma_blur=None, plot_axis=plot_axis, xextent=xextent, yextent=yextent, zextent=None, clip_percentile=clip_percentile)
        
        self.n_sig_bins = n_sig_bins
        self.sigma_scale = sigma_scale
        
    def forward(self, em: emitter.EmitterSet) -> torch.Tensor:

        em, xyz_extent = self._apply_extents(em)

        hist_col = []
        
        for i in range(self.n_sig_bins):

            sig_extent = [np.percentile(em.comb_sig.cpu().numpy(), (i/self.n_sig_bins) * 100.), np.percentile(em.comb_sig.cpu().numpy(), ((i+1)/self.n_sig_bins) * 100.)]
            em_sub = em[(em.comb_sig > sig_extent[0]) * (em.comb_sig < sig_extent[1])]
            sigma_blur = (em_sub.comb_sig.mean() * em.px_size.mean()).numpy()
            sigma_blur *= self.sigma_scale
            
            hist = self._hist2d(em_sub.xyz_nm[:, self.plot_axis].numpy(), xextent=xyz_extent[self.plot_axis[0]], yextent=xyz_extent[self.plot_axis[1]], px_size=self.px_size)   
            hist = gaussian_filter(hist, sigma=[sigma_blur / self.px_size, sigma_blur / self.px_size])
            
            hist_col.append(hist)
            
        hist = np.array(hist_col).sum(0)
            
        if self.clip_percentile is not None:
            hist = np.clip(hist, 0., np.percentile(hist, self.clip_percentile))
            
        return torch.from_numpy(hist)

    
class Renderer3D(Renderer):
    """
    3D Renderer with constant gaussian.

    """

    def __init__(self, px_size, sigma_blur, plot_axis = (0,1,2), xextent=None, yextent=None, zextent=None, clip_percentile=100, gamma=1):
        super().__init__(plot_axis=plot_axis, xextent=xextent, yextent=yextent, zextent=None, px_size=px_size)

        self.sigma_blur = sigma_blur
        self.clip_percentile = clip_percentile
        self.gamma = gamma
        self.zextent = zextent

    def render(self, em):

        hist = self.forward(em).numpy()

        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.25, pad=-0.25)
        colb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap('hsv'), values=np.linspace(0,0.7,101), norm=mpl.colors.Normalize(0.,1.))
        colb.outline.set_visible(False)
        colb.ax.invert_yaxis()

        _, xyz_extent = self._apply_extents(em)
        zextent = xyz_extent[2]
        
        cax.text(0.12, 0.04, f'{zextent[0]} nm', rotation=90, color='white', fontsize=15, transform=cax.transAxes)
        cax.text(0.12, 0.88, f'{zextent[1]} nm', rotation=90, color='white', fontsize=15, transform=cax.transAxes)
        cax.axis('off')

        ax.imshow(np.transpose(hist,[1,0,2]))
        return ax

    def forward(self, em: emitter.EmitterSet) -> torch.Tensor:

        em, xyz_extent = self._apply_extents(em)

        int_hist, col_hist = self._hist2d(em.xyz_nm[:, self.plot_axis].numpy(), xextent=xyz_extent[self.plot_axis[0]], yextent=xyz_extent[self.plot_axis[1]], zextent=xyz_extent[self.plot_axis[2]], px_size=self.px_size)
        
        if self.sigma_blur:
            int_hist = gaussian_filter(int_hist, sigma=[self.sigma_blur / self.px_size, self.sigma_blur / self.px_size])
            col_hist = gaussian_filter(col_hist, sigma=[self.sigma_blur / self.px_size, self.sigma_blur / self.px_size])
        
        RGB = self._hists_to_rgb(int_hist, col_hist)

        return torch.from_numpy(RGB)

    def _hists_to_rgb(self, int_hist, col_hist):
        
        with np.errstate(divide='ignore', invalid='ignore'):
            z_avg = col_hist / int_hist
        
        if self.clip_percentile is not None:
            int_hist = np.clip(int_hist, 0., np.percentile(int_hist, self.clip_percentile))
            
        z_avg[np.isnan(z_avg)] = 0
            
        val = (int_hist - int_hist.min()) / (int_hist.max() - int_hist.min())
        sat = np.ones(int_hist.shape)
        # Revert coloraxis to be closer to the paper figures
        hue = -(z_avg * 0.65) + 0.65

        HSV = np.concatenate((hue[:, :, None], sat[:, :, None], val[:, :, None]), -1)
        RGB = hsv_to_rgb(HSV) ** (1 / self.gamma)

        return RGB

    @staticmethod
    def _hist2d(xyz: np.array, xextent, yextent, zextent, px_size) -> np.array:
        
        hist_bins_x = np.arange(xextent[0], xextent[1] + px_size, px_size)
        hist_bins_y = np.arange(yextent[0], yextent[1] + px_size, px_size)

        int_hist, _, _ = np.histogram2d(xyz[:, 0], xyz[:, 1], bins=(hist_bins_x, hist_bins_y))
        
        z_pos = np.clip(xyz[:,2], zextent[0], zextent[1])
        z_weight = ((z_pos - zextent[0]) / (zextent[1] - zextent[0]))
        
        col_hist, _, _ = np.histogram2d(xyz[:, 0], xyz[:, 1], bins=(hist_bins_x, hist_bins_y), weights=z_weight)
        
        return int_hist, col_hist
    
class Renderer3D_auto_sig(Renderer3D):
    
    def __init__(self, px_size, n_sig_bins=10, sigma_scale=1, plot_axis = (0,1,2), xextent=None, yextent=None, zextent=None, clip_percentile=100, gamma=1):
        super().__init__(px_size=px_size, sigma_blur=None, plot_axis=plot_axis, xextent=xextent, yextent=yextent, zextent=zextent, clip_percentile=clip_percentile, gamma=gamma)
        
        self.n_sig_bins = n_sig_bins
        self.sigma_scale = sigma_scale
        
    def forward(self, em: emitter.EmitterSet) -> torch.Tensor:

        em, xyz_extent = self._apply_extents(em)

        int_col = []
        col_col = []
        
        for i in range(self.n_sig_bins):

            sig_extent = [np.percentile(em.comb_sig.cpu().numpy(), (i/self.n_sig_bins) * 100.), np.percentile(em.comb_sig.cpu().numpy(), ((i+1)/self.n_sig_bins) * 100.)]
            em_sub = em[(em.comb_sig > sig_extent[0]) * (em.comb_sig < sig_extent[1])]
            sigma_blur = (em_sub.comb_sig.mean() * em.px_size.mean()).numpy()
            sigma_blur *= self.sigma_scale

            int_hist, col_hist = self._hist2d(em_sub.xyz_nm[:, self.plot_axis].numpy(), xextent=xyz_extent[self.plot_axis[0]], yextent=xyz_extent[self.plot_axis[1]], zextent=xyz_extent[self.plot_axis[2]], px_size=self.px_size)
            
            int_col.append(gaussian_filter(int_hist, sigma=[sigma_blur / self.px_size, sigma_blur / self.px_size]))
            col_col.append(gaussian_filter(col_hist, sigma=[sigma_blur / self.px_size, sigma_blur / self.px_size]))
            
        int_hist = np.array(int_col).sum(0)
        col_hist = np.array(col_col).sum(0)
            
        RGB = self._hists_to_rgb(int_hist, col_hist)

        return torch.from_numpy(RGB)