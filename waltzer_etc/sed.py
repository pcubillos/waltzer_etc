# Copyright (c) 2025 Sreejith and Patricio
# LICENSE TBD

__all__ = [
    'normalize_vega',
    'find_closest_teff',
    'load_sed',
]

import os
import re

import astropy.units as u
import numpy as np
import pyratbay.constants as pc
import synphot as syn
import synphot.units as su
from synphot.models import Empirical1D
from .utils import ROOT


def load_vega():
    vega_file = f"{ROOT}/data/alpha_lyr_stis_010.fits"
    vega = syn.spectrum.SourceSpectrum.from_file(vega_file)
    return vega


# Make it available in this module
vega = load_vega()


def normalize_vega(wl, flux, v_mag):
    """
    Normalize a spectrum by the Johnson V-band magnitude
    to get flux measured at Earth.

    Parameters
    ----------
    wl: 1D darray
        Wavelength array in angstrom.
    flux: 1D darray
        Flux spectrum in erg s⁻¹ cm⁻² angstrom⁻¹.
    v_mag: Float
        Johnson V magnitude.

    Returns
    -------
    norm_flux: 1D np.ndarrays
        Flux at Earth normalized according to V magnitude.

    Examples
    --------
    Examples
    --------
    >>> import waltzer_etc.sed as sed

    >>> sed_wl, sed_flux = sed.load_sed(6050.0)
    >>> norm_flux = sed.normalize_vega(sed_wl, sed_flux, v_mag=7.65)

    >>> import pyratbay.constants as pc
    >>> import pyratbay.spectrum as ps

    >>> resolution = 60_000.0
    >>> wl = ps.constant_resolution_spectrum(2_450, 20_000, resolution)
    >>> flux = np.interp(wl, sed_wl, flux)
    >>> flux = ps.inst_convolution(
    >>>     wl, flux, resolution=3000, sampling_res=resolution,
    >>> )
    >>> norm_flux = sed.normalize_vega(wl, flux, v_mag=7.65)

    >>> star_R = 1.19 * pc.rsun
    >>> star_dist = 48.3016 * pc.parsec
    >>> w_flux = flux * (star_R/star_dist)**2.0

    >>> plt.figure(0, (8,4.5))
    >>> plt.clf()
    >>> plt.plot(wl, norm_flux, color='xkcd:green', label='synphot')
    >>> plt.plot(wl, w_flux, color='xkcd:blue', label='distance', alpha=0.5)
    """
    # Note flam is erg / s / cm**2 / angstrom
    sed = syn.spectrum.SourceSpectrum(
        Empirical1D,
        points=wl*u.angstrom,
        lookup_table=flux*su.FLAM,
    )

    # passband
    johnson_v_file = f'{ROOT}/data/johnson_v.dat'
    johnson_v = syn.spectrum.SpectralElement.from_file(johnson_v_file)

    # Normalization
    norm_flux = v_mag * su.VEGAMAG
    norm_sed = sed.normalize(norm_flux, band=johnson_v, vegaspec=vega)
    norm_flux = norm_sed(norm_sed.waveset, flux_unit=su.FLAM).value

    return norm_flux


def find_closest_teff(teff, base_dir=None):
    """
    Find folder in base_dir with closest temperature to teff.

    Examples
    --------
    teff = 8000.0
    """
    if base_dir is None:
        base_dir = f'{ROOT}/data/models/'
    temp_pattern = re.compile(r't(\d+)g[\d\.]+', re.IGNORECASE)
    folders = sorted([
        folder
        for folder in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, folder))
        if temp_pattern.search(folder)
    ])
    if len(folders) == 0:
        raise ValueError(
            f"No valid folders found in directory: {repr(base_dir)}"
        )

    temps = [
        folder[1+folder.index('t'):folder.index('g')]
        for folder in folders
    ]
    temps = np.array(temps, float)

    i = np.argmin(np.abs(temps - teff))
    file = os.path.join(base_dir, folders[i], 'model.flx')
    return file, temps[i]


def load_sed(teff=None, file=None):
    """
    Load an SED model from WALTzER grid.

    Parameters
    ----------
    teff: Float
        SED effective temperature.

    Examples
    --------
    >>> import waltzer_etc.sed as sed
    >>> sed_wl, sed_flux = sed.load_sed(6050.0)
    """
    # Load SED model spectrum based on temperature
    if file is None and teff is None:
        raise ValueError('one of teff or file input arguments must be set')

    if teff is not None:
        file, teff_match = find_closest_teff(teff)

    # Load SED flux
    spectrum = np.loadtxt(file, unpack=True)
    wl, flux = spectrum[0:2]

    # Convert flux from XX[?] to erg s-1 cm-2 A-1
    flux *= (pc.c / pc.A) / (wl**2) * 4*np.pi

    return wl, flux
