#! /usr/bin/env python
# Copyright (c) 2025
# LICENSE TBD
# -*- coding: utf-8 -*-

"""
@author: saickara
"""

import sys
import os
import re

import numpy as np
import pandas as pd
import scipy.constants as sc
import pyratbay.spectrum as ps

# Constants
parsec = sc.parsec * 100.0   # 1 parsec in cm
r_sun = 6.955e10  # Solar radius in cm


def gaussbroad(wl, signal, hwhm):
    """
    Smooths a spectrum by convolution with a gaussian of specified hwhm.

    Parameters
    ----------
    wl: 1D float array
        wavelength scale of spectrum to be smoothed
    s: 1D float array
        spectrum to be smoothed
    hwhm: Float
        Half width at half maximum of smoothing gaussian.
    """
    #Calculate (uniform) dispersion.
    nwave = len(wl)
    # wavelength change per pixel
    dwl = (wl[-1] - wl[0]) / nwave
    for i in range(0, len(wl)):
        # Smoothing gaussian, extend to 4 sigma.
        # 4.0 / np.sqrt(2.0*np.log(2.0)) = 3.3972872
        # sqrt(log(2.0)) = 0.83255461
        # sqrt(log(2.0)/pi)=0.46971864 (*1.0000632 to correct for >4 sigma wings)
        if(hwhm > 5*(wl[-1] - wl[0])):
            return np.full(len(wl), np.sum(signal)/nwave)

        # points in half gaussian
        nhalf = int(3.3972872*hwhm / dwl)
        # points in gaussian (odd!)
        ng = 2 * nhalf + 1
        # wavelength scale of gaussian
        wg = dwl * (np.arange(ng) - (ng-1)/2.0)
        # convenient absisca
        xg = ( (0.83255461) / hwhm) * wg
        #unit area gaussian w/ FWHM
        gpro = ( (0.46974832) * dwl / hwhm) * np.exp(-xg*xg)
        gpro = gpro/np.sum(gpro)

    # Pad spectrum ends to minimize impact of Fourier ringing.
    npad = nhalf + 2
    spad = np.concatenate((
        np.full(npad, signal[0]),
        signal,
        np.full(npad,signal[-1])
    ))

    # Convolve with gaussian
    sout = np.convolve(spad, gpro, mode='full')
    # Trim to original data/length
    sout = sout[npad:npad+nwave]
    return sout


def trapz_error(wl, flux, error):
    """
    Trapezoidal integration with error propagation.
    """
    integ = 0.0
    var = 0.0
    dwl = np.ediff1d(wl, 0, 0)
    ddwl = dwl[1:] + dwl[:-1]

    # Standard trapezoidal integration:
    integ = 0.5 * np.sum(ddwl * flux)
    var = 0.25 * np.sum((ddwl * error)**2)

    return integ, np.sqrt(var)


def find_closest_teff(base_dir, teff):
    """
    Find folder in base_dir with closest temperature to teff.

    Examples
    --------
    base_dir = './models/'
    teff = 8000.0
    """
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
    return folders[i], temps[i]



def snr_calc(
        wl, flux, effective_area, wl_min, wl_max, fwhm, exp_time, n_obs,
        photo=False,
    ):
    """
    Parameters
    ----------
    effective_area = 170.296
    wl_min = 6700
    wl_max = 7100
    fwhm = 2.0
    photo = False
    vis_snr = snr_calc(wl, flux, effective_area, wl_min, wl_max, fwhm, exp_time, n_obs)
    """
    index_band = (wl > wl_min) & (wl < wl_max)
    mean_band = np.mean(flux[index_band])
    max_band = np.max(flux[index_band])
    min_band = np.min(flux[index_band])

    # Convert flux to photons A-1 s-1
    photons = flux * 5.03e7 * wl * effective_area
    #photons = flux / (pc.c*pc.h / (wl*pc.A))

    # Photons collected per transit
    if photo:
        band_flux = photons * exp_time
        integ_flux, integ_var = trapz_error(
            wl[index_band],
            band_flux[index_band],
            band_flux[index_band]*0.0,
        )
        snr_inter = integ_flux/np.sqrt(integ_flux)
        trans_uncer = np.sqrt(2.0) / snr_inter
        return min_band, mean_band, max_band, integ_flux, snr_inter, trans_uncer

    # Spectroscopy
    w_band = np.arange(wl_min, wl_max, fwhm)
    band_flux = photons * exp_time * n_obs
    p_band = np.interp(w_band, wl, band_flux)

    # Poisson noise
    snr = p_band / np.sqrt(p_band)

    # Compute SNR and transit uncertainty
    snr_mean = np.mean(snr)
    snr_max = np.max(snr)
    trans_uncer = np.sqrt(2.0) / snr_mean
    return (
        min_band, mean_band, max_band,
        snr_mean, snr_max,
        trans_uncer,
    )


def main(
         csv_file=None,
         output_csv="waltzer_snr.csv",
         diameter=30.0,
         n_obs=10,
    ):
    """
    WALTzER Exposure time calculator

    Usage
    -----
    From the command line:
    python snr_waltzer.py target_list_20250327.csv  waltzer_snr_test.csv

    targets.csv is a CSV list of targets (e.g., downloaded from the
    NASA Exoplanet Archive).  This file must contain these headers:
        - 'pl_name'  Target's name
        - 'st_teff'  Host's effective temperature (K)
        - 'st_rad'   Host's radius (R_sun)
        - 'st_mass'  Host's mass (M_sun)
        - 'ra'       Right ascention (degrees)
        - 'dec'      Declination (degrees)
        - 'sy_dist'  Distant to target (parsec)
        - 'sy_vmag'  Host's V magnitude

    waltzer_snr.csv: Output filename where to save the resuls (as CSV)
        - planet               Target name
        - teff                 Host's effective temperature (K)
        - r_star               Host's radius (R_sun)
        - m_star               Host's mass (R_sun)
        - ra                   Right ascention (degrees)
        - dec                  Declination (degrees)
        - distance             Distant to target (parsec)
        - V_mag                Host's V magnitude
        - NUV_min_flux         Minumum flux in NUV band (erg s-1 cm-2 A-1)
        - NUV_mean_flux        Mean flux in NUV band (erg s-1 cm-2 A-1)
        - NUV_max_flux         Maximum flux in NUV band (erg s-1 cm-2 A-1)
        - NUV_mean_snr         Mean SNR in NUV band
        - NUV_max_snr          Maximum SNR in NUV band
        - NUV_transit_uncert   NUV transit uncertainty
        - VIS_min_flux         Minumum flux in VIS band (erg s-1 cm-2 A-1)
        - VIS_mean_flux        Mean flux in VIS band (erg s-1 cm-2 A-1)
        - VIS_max_flux         Maximum flux in VIS band (erg s-1 cm-2 A-1)
        - VIS_mean_snr         Mean SNR in VIS band
        - VIS_max_snr          Maximum SNR in VIS band
        - VIS_transit_uncert   VIS transit uncertainty
        - NIR_min_flux         Minumum flux in NIR band (erg s-1 cm-2 A-1)
        - NIR_mean_flux        Mean flux in NIR band (erg s-1 cm-2 A-1)
        - NIR_max_flux         Maximum flux in NIR band (erg s-1 cm-2 A-1)
        - NIR_mean_snr         Mean SNR in NIR band
        - NIR_max_snr          Maximum SNR in NIR band
        - NIR_transit_uncert   NIR transit uncertainty

    csv_file = 'target_list_20250327.csv'
    output_csv = "waltzer_snr.csv"
    diameter = 30.0
    n_obs = 10
    """
    # Effective area calculations
    primary_area = np.pi * (0.5*diameter)**2.0
    Rprim    = 0.90  # Telescope primary reflectance in %
    Rsec     = 0.90  # Telescope secondary reflectance in %
    Sec_obstr= 0.85  # Telescope secondary obstruction in % (1-Obstruction)
    R_d1     = 0.80  # Dichroic 1 Reflectance/Transmission in %
    R_d2     = 0.80  # Dichroic 2 Reflectance/Transmission in %
    R_uvfold = 1.00  # UV fold reflectance in %
    R_uvgr   = 0.90  # UV grating reflectance in %
    Uv_geff  = 0.65  # UV grating effeciency in %
    Uv_detQE = 0.55  # UV detector QE in %
    R_opfold = 0.90  # Optical fold reflectance in %
    R_opgr   = 0.90  # Optical grating reflectance in %
    Op_geff  = 0.75  # Optical grating effeciency in %
    Op_detQE = 0.90  # Optical detector QE in %
    BB_lens  = 0.90  # IR Broad band lens transmission in % IN REALITY, A FOLD
    BB_detQE = 0.85  # IR Broad band detector QE in %

    # NUV effective area in cm^2
    NUV_EFF_AREA = (
        primary_area * Rprim * Rsec * Sec_obstr *
        R_d1 * R_uvfold * R_uvgr * Uv_geff * Uv_detQE
    )
    # Optical effective area in cm^2
    VIS_EFF_AREA = (
        primary_area * Rprim * Rsec * Sec_obstr *
        R_d1 * R_d2 * R_opfold * R_opgr * Op_geff * Op_detQE
    )
    # NIR broadband effective area in cm^2
    NIR_EFF_AREA = (
        primary_area * Rprim * Rsec * Sec_obstr *
        R_d1 * R_d2 * BB_lens * BB_detQE
    )

    # Exposure time in seconds (1 average transit duration with 60% efficiency)
    efficiency = 0.6
    exp_time = (60 * 60 * 2.5) * efficiency

    # Target list file path
    data = pd.read_csv(csv_file, delimiter=',', comment='#')
    ntargets = len(data)

    # Extract required columns into individual arrays
    planet_names = data['pl_name']
    stellar_temps = data['st_teff']
    stellar_radii = data['st_rad']
    stellar_masses = data['st_mass']
    ra = data['ra']
    dec = data['dec']
    distances = data['sy_dist']
    v_mags = data['sy_vmag']

    # WALTzER wavelength grid (angstrom)
    resolution = 50_000.0
    inst_resolution = 3_000.0
    wl = ps.constant_resolution_spectrum(2_500, 20_000, resolution=resolution)

    # Read models files
    output_data = []
    cache_seds = {}
    models_loc = './models/'
    for i in range(ntargets):
        print(f'Target {i+1}/{ntargets}: {repr(planet_names[i])}')
        star_R = stellar_radii[i] * r_sun
        star_dist = distances[i] * parsec
        # Load SED model spectrum based on temperature
        teff = stellar_temps[i]
        t_model, teff_match = find_closest_teff(models_loc, teff)
        file = os.path.join(models_loc, t_model, 'model.flx')
        if file in cache_seds:
            sed_flux = cache_seds[file]
        else:
            # Load and interpolate SED flux
            spectrum = np.loadtxt(file, unpack=True)
            sed_wl, flux = spectrum[0:2]
            sed_flux = np.interp(wl, sed_wl, flux)
            cache_seds[file] = sed_flux

        conv_flux = ps.inst_convolution(
            wl, sed_flux, inst_resolution, sampling_res=resolution,
        )

        # Convert flux from XX to erg s-1 cm-2 A-1
        # integrate over steradian
        # and evaluate at Earth
        flux = conv_flux * (
            (sc.c / sc.angstrom) / (wl**2)
            * 4.0 * np.pi
            * (star_R/star_dist)**2.0
        )

        # Calculations for bands
        nuv_wl_min = 2640.0
        nuv_wl_max = 2700.0
        nuv_fwhm = 0.8
        nuv_snr = snr_calc(
            wl, flux, NUV_EFF_AREA, nuv_wl_min, nuv_wl_max, nuv_fwhm,
            exp_time, n_obs,
        )

        vis_wl_min = 6700.0
        vis_wl_max = 7100.0
        vis_fwhm = 2.0
        vis_snr = snr_calc(
            wl, flux, VIS_EFF_AREA, vis_wl_min, vis_wl_max, vis_fwhm,
            exp_time, n_obs,
        )

        nir_wl_min =  8200.0
        nir_wl_max = 10000.0
        nir_fwhm = 3.33
        nir_exp_time = 60 * 30.0
        nir_n_obs = 1
        nir_snr = snr_calc(
            wl, flux, NIR_EFF_AREA, nir_wl_min, nir_wl_max, nir_fwhm,
            nir_exp_time, nir_n_obs, photo=True,
        )

        # Save results
        output_data.append([
            planet_names[i],
            stellar_temps[i],
            stellar_radii[i],
            stellar_masses[i],
            ra[i],
            dec[i],
            distances[i],
            v_mags[i],
            *nuv_snr,
            *vis_snr,
            *nir_snr,
        ])

    # Save result as CSV file
    header = (
        "planet, teff, r_star, m_star, ra, dec, distance, V_mag, "
        "NUV_min_flux, NUV_mean_flux, NUV_max_flux,"
        "NUV_mean_snr, NUV_max_snr, NUV_transit_uncert, "
        "VIS_min_flux, VIS_mean_flux, VIS_max_flux,"
        "VIS_mean_snr, VIS_max_snr, VIS_transit_uncert, "
        "NIR_min_flux, NIR_mean_flux, NIR_max_flux,"
        "NIR_mean_snr, NIR_max_snr, NIR_transit_uncert, "
    )
    np.savetxt(output_csv, output_data, header=header, fmt="%s", delimiter=',')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError(
            'Invalid calling sequence, must be: '
            'python snr_waltzer.py targets.csv  waltzer_snr.csv'
        )
    targets_csv, output_csv = sys.argv[1:]
    main(targets_csv, output_csv)


