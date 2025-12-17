# Copyright (c) 2025 Patricio Cubillos and A. G. Sreejith
# WALTzER is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'ROOT',
    'inst_convolution',
]

import os
import pyratbay.constants as pc
import numpy as np
import scipy.interpolate as si
from scipy.signal import convolve
from scipy.signal.windows import gaussian


ROOT = os.path.realpath(os.path.dirname(__file__)) + '/'


def inst_convolution(wl, spectrum, resolution, sampling_res=None, mode='same'):
    """
    Convolve a spectrum according to an instrumental resolving power

    Parameters
    ----------
    wl: 1D float array
        Spectral array (can be either wavelength or wavenumber).
    spectrum: 1D float array
        Full-resolution spectrum to be convolved.
    resolution: float
        Instrumental resolving power R = lambda/delta_lambda
        Where delta_lambda is the FHWM of the gaussian to be applied.
    sampling_red: float
        Sampling resolution of the input wl spectrum.

    Examples
    --------
    >>> import pyratbay.spectrum as ps
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> wl_min = 1.499
    >>> wl_max = 1.501
    >>> samp_resolution = 100_000
    >>> wl = ps.constant_resolution_spectrum(wl_min, wl_max, samp_resolution)
    >>> nwave = len(wl)

    >>> # A delta at wl0 ~ 1.5
    >>> spectrum = np.zeros(nwave)
    >>> spectrum[np.where(wl>1.5)[0][0]] = 1.0
    >>> wl0 = wl[np.where(wl>1.5)[0][0]]

    >>> resolution = 5_000
    >>> conv = ps.inst_convolution(wl, spectrum, resolution)

    >>> # Plot convolved line and expected FWHM
    >>> half_max = np.amax(conv)/2
    >>> hwhm = 0.5 * wl0 / resolution
    >>> plt.figure(0)
    >>> plt.clf()
    >>> plt.plot(wl, conv, color='salmon', lw=2)
    >>> plt.plot([wl0-hwhm, wl0+hwhm], [half_max,half_max], color='xkcd:blue', lw=2)
    """
    pixel_dv = pc.c / resolution / 1e5
    n_el = int(6*pixel_dv) + 1
    kernel = gaussian(n_el, std=(pixel_dv / 2.355))
    kernel /= np.sum(kernel)

    if sampling_res is None:
        dv = pc.c/1e5 * np.ediff1d(wl) / wl[:-1]
        rv_pix = np.abs(np.mean(dv))
    else:
        rv_pix = np.abs(pc.c/1e5 / sampling_res)

    n_rv0 = int(((n_el - 1) / 2) / rv_pix)
    rv_array = np.arange(-(n_el - 1) / 2, (n_el - 1) / 2 + 1, 1)
    rv_array_mod = np.linspace(-n_rv0*rv_pix, n_rv0*rv_pix, int(2*n_rv0+1))

    csscaled = si.splrep(rv_array, kernel)
    ker_conv_pix = si.splev(rv_array_mod, csscaled, der=0)
    ker_conv_pix /= sum(ker_conv_pix)
    rconv = convolve(spectrum, ker_conv_pix, mode=mode)
    return rconv


