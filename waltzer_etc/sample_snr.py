# Copyright (c) 2025-2026 Patricio Cubillos and A. G. Sreejith
# WALTzER is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'waltzer_sample',
]

import pickle

import numpy as np
import pandas as pd
import pyratbay.spectrum as ps
from .snr_waltzer import Detector, calc_variances
from . import sed
from .version import __version__ as waltzer_version


def waltzer_sample(
        csv_file,
        output_csv="waltzer_snr.csv",
        diameter=35.0,
        efficiency=0.6,
        t_dur=None,
        n_obs=10,
        sed_type='llmodels',
        obs_mode='transit',
        hires=48_000,
    ):
    """
    WALTzER Exposure time calculator

    Parameters
    ----------
    csv_file: String
        A .csv list of targets (e.g., downloaded from the NASA
        Exoplanet Archive).  This file must contain these headers:
        - 'pl_name'  Target's name
        - 'pl_trandur'  Transit duration (h)
        - 'st_teff'  Host's effective temperature (K)
        - 'st_rad'   Host's radius (R_sun)
        - 'st_mass'  Host's mass (M_sun)
        - 'ra'       Right ascention (degrees)
        - 'dec'      Declination (degrees)
        - 'sy_vmag'  Host's V magnitude
        An additional 'sed' column can be set to specify cusom SEDs, which
        contain paths to SED files containing two columns:
        wavelength (microns) and flux (mJy)
    output_csv: String
        Output filename where to save the resuls (as CSV)
        - planet               Target name
        - teff                 Host's effective temperature (K)
        - r_star               Host's radius (R_sun)
        - m_star               Host's mass (R_sun)
        - ra                   Right ascention (degrees)
        - dec                  Declination (degrees)
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
    TBD

    Notes
    -----
    Can also be run from the command line, e.g.:
    waltz target_list_20250327.csv waltzer_snr.csv

    Example (TBD while we build this machine)
    -------
    >>> import waltzer_etc as w
    >>> from waltzer_etc.snr_waltzer import Detector
    >>> from waltzer_etc.snr_waltzer import *
    >>> from waltzer_etc.utils import ROOT
    >>> import waltzer_etc.sed as sed
    >>> diameter = 35.0
    >>> csv_file = 'target_list_20250327.csv'
    >>> efficiency = 0.6
    >>> t_dur = 2.5
    >>> n_obs = 10
    """
    # The three amigos
    nuv = Detector('nuv', diameter, hires=hires)
    vis = Detector('vis', diameter, hires=hires)
    nir = Detector('nir', diameter, hires=hires)
    detectors = nuv, vis, nir
    bands = [det.band for det in detectors]

    # WALTzER resolution (FWHM) and wavelength grid (angstrom)
    inst_resolution = nuv.resolution

    # Higher resolution for models (will be binned down to WALTzER later)
    wl_min = nuv.hires_wl_min
    wl_max = nuv.hires_wl_max
    resolution = nuv.hires_resolution
    wl = ps.constant_resolution_spectrum(wl_min, wl_max, resolution)

    # Target list file path
    data = pd.read_csv(csv_file, delimiter=',', comment='#')
    ntargets = len(data)

    # Extract required columns into individual arrays
    planet_names = data['pl_name']
    transit_dur = data['pl_trandur']
    stellar_temps = data['st_teff']
    stellar_radii = data['st_rad']
    stellar_masses = data['st_mass']
    ra = data['ra']
    dec = data['dec']
    v_mags = data['sy_vmag']

    # Look for custom SEDs
    has_sed = np.zeros(ntargets, bool)
    if 'sed' in data:
        input_sed = data['sed']
        has_sed = [isinstance(sed, str) for sed in input_sed]
    else:
        input_sed = None

    # Total in-transit exposure time (s) [for .csv statistics]
    if t_dur is None:
        # Use CSV data for each target
        exp_time = (transit_dur * 3600) * efficiency
    else:
        # Use fixed transit duration for all targets
        exp_time = (t_dur * 3600) * efficiency
        exp_time = np.tile(exp_time, ntargets)

    # Read models files
    cache_seds = {}
    output_data = []
    spectra = {}

    if obs_mode == 'transit':
        suffix = ' error(ppm)'
    else:
        suffix = ' S/N'
    print(f'Target    Name              V_mag   Teff       NUV       VIS       NIR{suffix}')
    for i in range(ntargets):
        target = planet_names[i]
        # Load SED model spectrum based on temperature
        if has_sed[i]:
            sed_file = input_sed[i]
            teff_match = 0.0
        else:
            teff = stellar_temps[i]
            logg = 4.5
            sed_file, sed_label, teff_match, logg_match = sed.find_closest_sed(
                teff, logg, sed_type,
            )

        if sed_label in cache_seds:
            sed_flux = cache_seds[sed_label]
        else:
            # Load SED flux
            if has_sed[i]:
                sed_wl, flux = np.loadtxt(sed_file, unpack=True)
            else:
                sed_wl, flux = sed.load_sed(teff_match, logg_match, sed_type)
            # Interpolate to regular grid
            sed_flux = np.interp(wl, sed_wl, flux)
            cache_seds[sed_label] = sed_flux

        # Normalize according to Vmag
        flux = sed.normalize_vega(wl, sed_flux, v_mags[i])

        # Flux stats
        nuv_flux_stats = nuv.flux_stats(wl, flux)
        vis_flux_stats = vis.flux_stats(wl, flux)
        nir_flux_stats = nir.flux_stats(wl, flux)

        # SNR stats
        integ_time = exp_time[i] * n_obs

        # Note: this should mirror the GUI's run_waltzer() dictionary
        tso = {}
        for det in detectors:
            band = det.band
            tso[band] = det.make_tso(wl, flux)

        tso['meta'] = {
            'version': waltzer_version,
            'mirror_diameter': float(det.diameter),
            'bands': bands,
            'efficiency': efficiency,
            'n_obs': n_obs,
            'transit_dur': transit_dur[i],
            'target': target,
        }

        var_nuv = calc_variances(tso['nuv'])
        var_vis = calc_variances(tso['vis'])
        var_nir = calc_variances(tso['nir'])

        nuv_snr_stats = nuv.snr_stats(var_nuv, integ_time)
        vis_snr_stats = vis.snr_stats(var_vis, integ_time)
        nir_snr_stats = nir.snr_stats(var_nir, integ_time)[2:]

        if obs_mode == 'transit':
            number = f'{i+1}/{ntargets}'
            print(
                f'{number:>7}: {repr(target):18} '
                f'{v_mags[i]:5.2f}  {teff_match:5.0f}  '
                f'{nuv_snr_stats[-1]:8.1f}  {vis_snr_stats[-1]:8.1f}  '
                f'{nir_snr_stats[-1]:8.1f}'
            )
        elif obs_mode == 'stare':
            # Flux SNR
            source_snr = []
            for var_data in [var_nuv, var_vis, var_nir]:
                source = var_data[2]
                variance = np.sum(np.array(var_data[2:]), axis=0)
                snr = source / np.sqrt(variance) * np.sqrt(integ_time)
                source_snr.append(snr)
            median_flux_snrs = [np.median(snr) for snr in source_snr]

            number = f'{i+1}/{ntargets}'
            print(
                f'{number:>7}: {repr(target):18} '
                f'{v_mags[i]:5.2f}  {teff_match:5.0f}  '
                f'{median_flux_snrs[0]:8.1f}  '
                f'{median_flux_snrs[1]:8.1f}  '
                f'{median_flux_snrs[2]:10.1f}'
            )

        spectra[target] = tso

        # Save results
        output_data.append([
            planet_names[i],
            stellar_temps[i],
            stellar_radii[i],
            stellar_masses[i],
            ra[i],
            dec[i],
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
        "target, teff, r_star, m_star, ra, dec, V_mag, transit_dur, "
        "NUV_min_flux, NUV_median_flux, NUV_max_flux, "
        "NUV_min_snr, NUV_median_snr, NUV_max_snr, NUV_transit_uncert, "
        "VIS_min_flux, VIS_median_flux, VIS_max_flux, "
        "VIS_min_snr, VIS_median_snr, VIS_max_snr, VIS_transit_uncert, "
        "NIR_min_flux, NIR_median_flux, NIR_max_flux, "
        "NIR_snr, NIR_transit_uncert "
    ).split(',')
    header = [h.strip() for h in header]
    units = (
        "#name, K, R_sun, M_sun, deg, deg, , h,"
        "mJy, mJy, mJy, , , , ppm,"
        "mJy, mJy, mJy, , , , ppm,"
        "mJy, mJy, mJy, , ppm"
    ).split(',')

    with open(output_csv, 'w') as f:
        # Telemetry and stats info
        f.write(f"# primary diameter: {diameter} cm\n")
        f.write(f"# duty-cycle efficiency: {efficiency}\n")
        f.write(f"# instrumental resolution = lambda/FWHM = {inst_resolution}\n")

        f.write("#\n# SNR stats are per HWHM! resolving element\n")
        f.write(f"# Number of transits (for stats): {n_obs}\n")
        if t_dur is None:
            f.write("# total in-transit time (for stats): see 'transit_dur' column\n")
        else:
            f.write(f"# total in-transit time (for stats): {t_dur} h\n")

        f.write("#\n# Wavelength ranges per band (angstrom)\n")
        f.write(f"# NUV  {nuv.wl_min} {nuv.wl_max}\n")
        f.write(f"# VIS  {vis.wl_min} {vis.wl_max}\n")
        f.write(f"# NIR  {nir.wl_min} {nir.wl_max}\n#\n")
        f.write(','.join(header) + '\n')
        f.write(', '.join(units) + '\n')
        np.savetxt(f, output_data, delimiter=",", fmt="%s")

    p_file = output_csv.replace('.csv', '.pickle')
    with open(p_file, 'wb') as handle:
        pickle.dump(spectra, handle, protocol=4)
