#! /usr/bin/env python
# Copyright (c) 2025 Sreejith and Patricio
# LICENSE TBD
# -*- coding: utf-8 -*-

__all__ = [
    'simulate_spectrum',
    'find_closest_teff',
    'photon_spectrum',
    'flux_stats',
    'snr_stats',
    'main',
]

import sys
import os
import re
import pickle
import random

import numpy as np
import pandas as pd
import pyratbay.constants as pc
import pyratbay.spectrum as ps
import scipy.interpolate as si


def find_closest_teff(teff, base_dir='./models/'):
    """
    Find folder in base_dir with closest temperature to teff.

    Examples
    --------
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
    file = os.path.join(base_dir, folders[i], 'model.flx')
    return file, temps[i]



def photon_spectrum(
        wl, flux, effective_area, wl_min, wl_max, bin_edges,
        photometry=False,
    ):
    """
    Compute the spectrum in photons per second.

    Parameters
    ----------
    wl: 1D float array
        Input wavelength array (angstrom).
    flux: 1D float array
        Input flux spectrum (erg s⁻¹ cm⁻² angstrom⁻¹).
    effective_area: float
        Effective collecting area of the instrument (cm²).
    wl_min: float
        Minimum wavelength of the range to include (angstrom).
    wl_max: float
        Maximum wavelength of the range to include (angstrom).
    bin_edges: 1D float array
        Edges of the instrumental bins (angstrom).

    Returns
    -------
    bin_wl: 1D float array
        Central wavelength of each bin (angstrom).
    bin_flux: 1D float array
        Integrated photon flux per bin (photons s⁻¹).
    half_widths: 1D float array
        Half-width of each wavelength bin (angstrom).

    Examples
    --------
    >>> resolution = 60_000.0
    >>> wl = ps.constant_resolution_spectrum(2_450, 20_000, resolution)
    >>> inst_resolution = 3_000.0
    >>> bin_edges = ps.constant_resolution_spectrum(
    >>>     2_500, 20_000, 2.0*inst_resolution,
    >>> )

    >>> # Load stellar SED
    >>> file = './models/t06000g4.4/model.flx'
    >>> sed_wl, flux, _ = np.loadtxt(file, unpack=True)
    >>> sed_flux = np.interp(wl, sed_wl, flux)

    >>> # Flux at instrumental resolution, in erg s-1 cm-2 A-1 at Earth
    >>> conv_flux = ps.inst_convolution(
    >>>     wl, sed_flux, inst_resolution, sampling_res=resolution,
    >>> )
    >>> star_R = 1.18 * pc.rsun
    >>> star_dist = 48.3 * pc.parsec
    >>> flux = conv_flux * (
    >>>     (pc.c / pc.A) / (wl**2)
    >>>     * 4.0 * np.pi
    >>>     * (star_R/star_dist)**2.0
    >>> )

    >>> effective_area = 170.296
    >>> wl_min = 6700
    >>> wl_max = 7100
    """
    # Convert flux (erg s-1 cm-2 A-1) to photons s-1 A-1
    photons = flux * wl*pc.A / (pc.c*pc.h) * effective_area

    # Integrate at each instrumental bin to get photons s-1
    if photometry:
        mask = (wl>=wl_min) & (wl<=wl_max)
        bin_wl = 0.5 * (wl_max + wl_min)
        half_widths = 0.5 * (wl_max - wl_min)
        bin_flux = np.trapezoid(photons[mask], wl[mask])
        return (
            np.array([bin_wl]),
            np.array([bin_flux]),
            np.array([half_widths])
        )

    # Spectroscopy
    i_min = np.searchsorted(bin_edges, wl_min)
    i_max = np.searchsorted(bin_edges, wl_max)
    bin_wl = 0.5 * (bin_edges[i_min+1:i_max] + bin_edges[i_min:i_max-1])
    half_widths = bin_wl - bin_edges[i_min:i_max-1]

    nbins = len(bin_wl)
    bin_flux = np.zeros(nbins)
    for i in range(nbins):
        bin_mask = (
            (wl>=bin_edges[i_min+i]) &
            (wl<=bin_edges[i_min+i+1])
        )
        bin_flux[i] = np.trapezoid(photons[bin_mask], wl[bin_mask])

    return bin_wl, bin_flux, half_widths


def flux_stats(wl, flux, wl_min, wl_max):
    """
    Compute basic flux statistics within a wavelength interval.

    Parameters
    ----------
    wl: 1D float array
        Input wavelength array (angstrom).
    flux: 1D float array
        Input flux spectrum (erg s⁻¹ cm⁻² angstrom⁻¹).
    wl_min: float
        Minimum wavelength of the range to include (angstrom).
    wl_max: float
        Maximum wavelength of the range to include (angstrom).

    Returns
    -------
    mean_flux: float
        Mean flux within the wavelength range (erg s⁻¹ cm⁻² angstrom⁻¹).
    min_flux: float
        Minimum flux within the wavelength range (erg s⁻¹ cm⁻² angstrom⁻¹).
    max_flux: float
        Maximum flux within the wavelength range (erg s⁻¹ cm⁻² angstrom⁻¹).
    """
    band = (wl > wl_min) & (wl < wl_max)
    mean_flux = np.mean(flux[band])
    max_flux = np.max(flux[band])
    return mean_flux, max_flux


def snr_stats(wl, flux, exp_time, n_obs):
    """
    Parameters
    ----------
    effective_area = 170.296
    wl_min = 6700
    wl_max = 7100
    fwhm = 2.0
    photo = False

    bin_wl, bin_flux, half_widths = photon_spectrum(
        wl, flux, effective_area, wl_min, wl_max, bin_edges,
    )
    """
    # integrate over time
    total_flux = flux * exp_time * n_obs

    # Poisson noise estimation
    snr = total_flux / np.sqrt(total_flux)

    snr_mean = np.mean(snr)
    snr_max = np.max(snr)
    # Assume t_in=t_out and flux_in approx. flux_out:
    trans_uncer = np.sqrt(2.0) / snr_mean / pc.ppm

    return snr_mean, snr_max, np.round(trans_uncer, 3)


def main(
         csv_file=None,
         output_csv="waltzer_snr.csv",
         diameter=30.0,
         efficiency=0.6,
         t_dur=2.5,
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
        - NUV_mean_flux        Mean flux in NUV band (erg s-1 cm-2 A-1)
        - NUV_max_flux         Maximum flux in NUV band (erg s-1 cm-2 A-1)
        - NUV_mean_snr         Mean SNR in NUV band
        - NUV_max_snr          Maximum SNR in NUV band
        - NUV_transit_uncert   NUV transit uncertainty
        - VIS_mean_flux        Mean flux in VIS band (erg s-1 cm-2 A-1)
        - VIS_max_flux         Maximum flux in VIS band (erg s-1 cm-2 A-1)
        - VIS_mean_snr         Mean SNR in VIS band
        - VIS_max_snr          Maximum SNR in VIS band
        - VIS_transit_uncert   VIS transit uncertainty
        - NIR_mean_flux        Mean flux in NIR band (erg s-1 cm-2 A-1)
        - NIR_max_flux         Maximum flux in NIR band (erg s-1 cm-2 A-1)
        - NIR_snr              SNR in NIR band
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

    # Effective collecing areas in cm^2
    NUV_EFF_AREA = (
        primary_area * Rprim * Rsec * Sec_obstr *
        R_d1 * R_uvfold * R_uvgr * Uv_geff * Uv_detQE
    )
    VIS_EFF_AREA = (
        primary_area * Rprim * Rsec * Sec_obstr *
        R_d1 * R_d2 * R_opfold * R_opgr * Op_geff * Op_detQE
    )
    NIR_EFF_AREA = (
        primary_area * Rprim * Rsec * Sec_obstr *
        R_d1 * R_d2 * BB_lens * BB_detQE
    )

    # Wavelength ranges per band (angstrom)
    nuv_wl_min = 2500.0
    nuv_wl_max = 3300.0

    vis_wl_min = 4160.0
    vis_wl_max = 8200.0

    nir_wl_min =  9000.0
    nir_wl_max = 16000.0


    # Target list file path
    data = pd.read_csv(csv_file, delimiter=',', comment='#')
    ntargets = len(data)

    # Extract required columns into individual arrays
    planet_names = data['pl_name']
    #transit_depth = data['pl_trandep']
    transit_dur = data['pl_trandur']
    stellar_temps = data['st_teff']
    stellar_radii = data['st_rad']
    stellar_masses = data['st_mass']
    ra = data['ra']
    dec = data['dec']
    distances = data['sy_dist']
    v_mags = data['sy_vmag']

    # Total in-transit exposure time (s) [for .csv statistics]
    if t_dur is None:
        # Use CSV data for each target
        exp_time = (transit_dur * 3600) * efficiency
    else:
        # Use fixed transit duration for all targets
        exp_time = (t_dur * 3600) * efficiency
        exp_time = np.tile(exp_time, ntargets)

    resolution = 60_000.0
    wl = ps.constant_resolution_spectrum(2_450, 20_000, resolution=resolution)
    # WALTzER resolution (FWHM) and wavelength grid (angstrom)
    inst_resolution = 3_000.0
    bin_edges = ps.constant_resolution_spectrum(
        2_500, 20_000, resolution=2.0*inst_resolution,
    )

    # Read models files
    cache_seds = {}
    output_data = []
    spectra = {}
    for i in range(ntargets):
        target = planet_names[i]
        print(f'Target {i+1}/{ntargets}: {repr(target)}')
        star_R = stellar_radii[i] * pc.rsun
        star_dist = distances[i] * pc.parsec
        # Load SED model spectrum based on temperature
        teff = stellar_temps[i]
        file, teff_match = find_closest_teff(teff)
        if file in cache_seds:
            sed_flux = cache_seds[file]
        else:
            # Load and interpolate SED flux
            spectrum = np.loadtxt(file, unpack=True)
            sed_wl, flux = spectrum[0:2]
            sed_flux = np.interp(wl, sed_wl, flux)
            conv_flux = ps.inst_convolution(
                wl, sed_flux, inst_resolution, sampling_res=resolution,
            )
            cache_seds[file] = conv_flux

        # Convert flux from XX to erg s-1 cm-2 A-1
        # integrate over steradian
        # and evaluate at Earth
        flux = conv_flux * (
            (pc.c / pc.A) / (wl**2)
            * 4.0 * np.pi
            * (star_R/star_dist)**2.0
        )

        # Calculations for bands
        nuv_wl, nuv_flux, nuv_half_width = photon_spectrum(
            wl, flux, NUV_EFF_AREA, nuv_wl_min, nuv_wl_max, bin_edges,
        )

        vis_wl, vis_flux, vis_half_width = photon_spectrum(
            wl, flux, VIS_EFF_AREA, vis_wl_min, vis_wl_max, bin_edges,
        )

        nir_wl, nir_flux, nir_half_width = photon_spectrum(
            wl, flux, NIR_EFF_AREA, nir_wl_min, nir_wl_max, bin_edges,
            photometry=True,
        )

        nuv_flux_stats = flux_stats(wl, flux, nuv_wl_min, nuv_wl_max)
        vis_flux_stats = flux_stats(wl, flux, vis_wl_min, vis_wl_max)
        nir_flux_stats = flux_stats(wl, flux, nir_wl_min, nir_wl_max)

        nuv_snr_stats = snr_stats(nuv_wl, nuv_flux, exp_time[i], n_obs)
        vis_snr_stats = snr_stats(vis_wl, vis_flux, exp_time[i], n_obs)
        nir_snr_stats = snr_stats(nir_wl, nir_flux, exp_time[i], n_obs)[1:]

        spectra[target] = {
            'wl': (nuv_wl, vis_wl, nir_wl),
            'e_flux': (nuv_flux, vis_flux, nir_flux),
            'wl_half_width': (nuv_half_width, vis_half_width, nir_half_width),
            'transit_dur': transit_dur[i],
            #'transit_depth': transit_depth[i],
        }

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
            transit_dur[i],
            *nuv_flux_stats,
            *nuv_snr_stats,
            *vis_flux_stats,
            *vis_snr_stats,
            *nir_flux_stats,
            *nir_snr_stats,
        ])

    # Save result as CSV file
    header = (
        "target, teff, r_star, m_star, ra, dec, distance, V_mag, transit_dur,"
        "NUV_mean_flux, NUV_max_flux,"
        "NUV_mean_snr, NUV_max_snr, NUV_transit_uncert, "
        "VIS_mean_flux, VIS_max_flux, "
        "VIS_mean_snr, VIS_max_snr, VIS_transit_uncert, "
        "NIR_mean_flux, NIR_max_flux, NIR_snr, NIR_transit_uncert "
    ).split(',')
    header = [h.strip() for h in header]
    units = (
        "name, K, R_sun, M_sun, deg, deg, parsec, , h,"
        "erg s-1 cm-2 A-1, erg s-1 cm-2 A-1, , , ppm,"
        "erg s-1 cm-2 A-1, erg s-1 cm-2 A-1, , , ppm,"
        "erg s-1 cm-2 A-1, erg s-1 cm-2 A-1, , ppm"
    ).split(',')

    with open(output_csv, 'w') as f:
        # Telemetry and stats info
        f.write(f"# primary diameter: {diameter} cm\n")
        f.write(f"# duty-cycle efficiency: {efficiency}\n")
        f.write(f"# instrumental resolution = lambda/FWHM = {inst_resolution}\n")

        f.write("#\n# SNR stats are per HWHM! resolving element\n")
        f.write(f"# Number of transits (for stats): {n_obs}\n")
        if t_dur is None:
            f.write(f"# total in-transit time (for stats): see 'transit_dur' column\n")
        else:
            f.write(f"# total in-transit time (for stats): {t_dur} h\n")

        f.write("#\n# Wavelength ranges per band (angstrom)\n")
        f.write(f"# NUV  {nuv_wl_min} {nuv_wl_max}\n")
        f.write(f"# VIS  {vis_wl_min} {vis_wl_max}\n")
        f.write(f"# NIR  {nir_wl_min} {nir_wl_max}\n#\n")
        f.write(','.join(header) + '\n')
        f.write(', '.join(units) + '\n')
        np.savetxt(f, output_data, delimiter=",", fmt="%s")

    p_file = output_csv.replace('.csv', '.pickle')
    with open(p_file, 'wb') as handle:
        pickle.dump(spectra, handle, protocol=4)



if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError(
            'Invalid calling sequence, must be: '
            'python snr_waltzer.py targets.csv  waltzer_snr.csv'
        )
    targets_csv, output_csv = sys.argv[1:]
    main(targets_csv, output_csv)


