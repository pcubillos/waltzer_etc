# Copyright (c) 2025-2026 Patricio Cubillos and A. G. Sreejith
# WALTzER is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'power_law_source',
    'display_2d_source',
    'arcsec_to_pixel',
    'dispersion_integral',
    'image_to_detector',
    'simulate_spectrum',
]

import warnings
import random

import matplotlib.pyplot as plt
import numpy as np
import pyratbay.spectrum as ps
from ..utils import (
    inst_convolution,
    SLIT_WIDTHS,
    SLIT_LENGTHS,
)


def power_law_source(radius, pow_law_index, band, slit_pos='wide', offset=None):
    """
    Generate a 2D image (arcsec by arcsec) of a circular source with
    intensity=1.0 mJy arcsec of given radius, and a power law shape
    outside.

    Parameters
    ----------
    radius: Float
        Source radius in arcsec.
    pow_law_index: Float
        Intensity power-law index of source beyond radius.
        Set pow_law_index = 0 for a flat disk source (no flux outside radius).
    band: String
        WALTzER band, select between 'nuv' or 'vis'.
    slit_pos: String
        Select from: 'wide', 'medium', or 'narrow' to place the source
        at the center of the wide, medium, or narrow end of the slit.
    offset: x0,y0 float tuple
        Offset of the center of the source relative to slit_pos (in arcsec),
        where x0 and y0 are along the cross-dispersion and dispersino axes,
        respectively.

    Returns
    -------
    source_2d: 2D float array
        2D source spatial image in cross-dispersion (arcsec) and
        dispersion (arcsec) axis. Image outside the aperture will have
        np.nan values.
    xd_arcsec: 1D float array
        Cross-dispersion array in arcsec. Origin is always defined at
        the center of the wide-end of the slit.
    disp_arcsec: 1D float array
        Dispersion array in arcsec. Origin (x=0) is always defined at
        the center of the slit.
    xd_loc: Float
        Source center location along cross-dispersion axis in arcsec.
    disp_loc: Float
        Source center location along dispersion axis in arcsec.

    Examples
    --------
    >>> import waltzer_etc.extended as wex
    >>>
    >>> radius = 4.0
    >>> pow_law_index = 0.5
    >>> band = 'vis'
    >>> slit_pos = 'narrow'
    >>> offset = (0, 0)
    >>> source_2d, xd_arcsec, d_arcsec, xd_loc, disp_loc = wex.power_law_source(
    >>>     radius, pow_law_index, band, slit_pos, offset,
    >>> )
    >>> # Show 2D source
    >>> ax = wex.display_2d_source(source_2d, xd_loc, disp_loc, radius)
    >>> plt.savefig('../../plots/power_law_source.png', dpi=300)
    """
    # Slit geometry in arcseconds
    length_wide = SLIT_LENGTHS[band]['wide']
    length_medium = SLIT_LENGTHS[band]['medium']
    length_narrow = SLIT_LENGTHS[band]['narrow']
    width_medium = SLIT_WIDTHS[band]['medium']

    # x-axis is cross-dispersion
    # y-axis is dispersion
    if offset is None:
        x0 = 0.0
        y0 = 0.0
    else:
        x0, y0 = offset

    if slit_pos == 'medium':
        x0 += 0.5 * (length_wide + length_medium)
    elif slit_pos == 'narrow':
        x0 += length_medium + 0.5 * (length_wide + length_narrow)

    slit_length = length_wide + length_medium + length_narrow
    size_scale = 20
    nx = slit_length*size_scale + 1
    ny = length_wide*size_scale + 1
    x = np.linspace(-0.5*length_wide, slit_length - 0.5*length_wide, nx)
    y = np.linspace(-0.5*length_wide, 0.5*length_wide, ny)
    x_grid, y_grid = np.meshgrid(x, y)

    radii = np.sqrt((x_grid-x0)**2 + (y_grid-y0)**2)
    if pow_law_index == 0:
        source_2d = radii * 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            source_2d = 1.0 * (radii/radius)**-pow_law_index
    source_2d[radii<=radius] = 1.0

    # Mask aperture
    mask1 = (x_grid > 0.5*length_wide) & (np.abs(y_grid) > 0.5*width_medium)
    source_2d[mask1] = np.nan
    mask2 = (
        (x_grid > 0.5*length_wide+length_medium) &
        (np.abs(y_grid) > 0.5*length_narrow)
    )
    source_2d[mask2] = np.nan

    return source_2d, x, y, x0, y0


def display_2d_source(source_2d, xd_loc=None, disp_loc=None, radius=None):
    """
    Display a 2D source image on the WALTzER slit.
    If loc and radius are provided, mark the source radius.

    Parameters
    ----------
    source_2d: 2D float array
        2D source spatial image in cross-dispersion (arcsec) and
        dispersion (arcsec) axis.
    xd_loc: Float
        Source center location along cross-dispersion axis in arcsec.
    disp_loc: Float
        Source center location along dispersion axis in arcsec.
    radius: Float
        Source radius in arcsec.

    Examples
    --------
    See waltzer_etc.extended.power_law_source() example.
    """
    fs = 12
    fig = plt.figure(1)
    fig.set_size_inches(8, 3.25)
    plt.subplots_adjust(0.1, 0.1, 0.98, 0.9, wspace=0.3)
    plt.clf()
    ax = plt.subplot(111)
    cmap = ax.imshow(
        source_2d,
        aspect='equal', origin='lower', extent=(-30, 200, -30, 30),
    )
    show_source_edge = (
        xd_loc is not None and
        disp_loc is not None and
        radius is not None
    )
    if show_source_edge:
        t = np.linspace(0, 2*np.pi, 100)
        xd_source = xd_loc + radius*np.sin(t)
        disp_source = disp_loc + radius*np.cos(t)
        ax.plot(xd_source, disp_source, color='red', lw=0.75, dashes=(6,1))
    ax.tick_params(direction='in', which='both', labelsize=fs-1)
    ax.set_xlabel('Cross-dispersion axis (arcsec)', fontsize=fs)
    ax.set_ylabel('Dispersion axis (arcsec)', fontsize=fs)
    ax.set_xlim(-30, 200)
    ax.set_ylim(-30, 30)
    cb = plt.colorbar(cmap, location='top', pad=0.1, aspect=30)
    cb.ax.tick_params(direction='in', which='both', labelsize=fs-1)
    cb.set_label(r'Intensity (mJy arcsec$^{-2}$)', fontsize=fs)
    return ax


def arcsec_to_pixel(xd_loc, det):
    """
    Find cross-dispersion pixel position given a position x0 (in arcsec)
    in the slit.

    Parameters
    ----------
    xd_loc: Float
        Source center location along cross-dispersion axis in arcsec.
    det: Detector()
        A WALTzER Detector object.
    """
    length_wide = SLIT_LENGTHS[det.band]['wide']
    pos = (xd_loc + 0.5*length_wide) / det.pix_scale - 0.5
    return pos


def dispersion_integral(source_2d, xd_arcsec, disp_arcsec, det):
    """
    Integrate a 2D source image (mJy arcsec⁻²) into the WALTzER
    1D cross-dispersion profile at each pixel.

    Parameters
    ----------
    source_2d: 2D float array
        2D source spatial image in cross-dispersion (arcsec) and
        dispersion (arcsec) axis. Image outside the aperture will have
        np.nan values.
    xd_arcsec: 1D float array
        Cross-dispersion array in arcsec. Origin is always defined at
        the center of the wide-end of the slit.
    disp_arcsec: 1D float array
        Dispersion array in arcsec. Origin (x=0) is always defined at
        the center of the slit.
    det: Detector()
        A WALTzER Detector object.

    Returns
    -------
    source_1d: 1D float array
        Integrated source flux (mJy) at each WALTzER cross-dispersion pixel.
    cross_disp: 1D float array
        Cross-dispersion array in arcsec for source_1d.
    slit_area: 1D float array
        Collecting area (arcsec²) for each source_1d bin.

    Examples
    --------
    >>> from waltzer_etc import Detector
    >>> import waltzer_etc.extended as wex
    >>>
    >>> # Create a 2D source
    >>> radius = 5.0
    >>> pow_law_index = 0.6
    >>> det = Detector('vis')
    >>> slit_pos = 'medium'
    >>> offset = (0, 0)
    >>> source_2d, xd_arcsec, disp_arcsec, xd_loc, disp_loc = wex.power_law_source(
    >>>     radius, pow_law_index, det.band, slit_pos, offset,
    >>> )
    >>> ax = wex.display_2d_source(source_2d, xd_loc, disp_loc, radius)
    >>>
    >>> # Integrate along dispersion axis and XD pixels:
    >>> source_1d, cross_disp, slit_area = wex.dispersion_integral(
    >>>     source_2d, xd_arcsec, disp_arcsec, det,
    >>> )
    >>> width0 = slit_area[np.searchsorted(cross_disp, xd_loc)]
    >>> width_scale = width0 / slit_area
    >>>
    >>> # Show 1D-integrated profile of source
    >>> plt.figure(5)
    >>> plt.clf()
    >>> plt.subplots_adjust(0.1, 0.1, 0.98, 0.98, hspace=0.18)
    >>> ax = plt.subplot(211)
    >>> plt.plot(cross_disp, source_1d, c='royalblue')
    >>> plt.axvline(xd_loc, dashes=(6,1), color='xkcd:goldenrod')
    >>> plt.ylim(bottom=0.0)
    >>> plt.ylabel('Flux (mJy)')
    >>> ax = plt.subplot(212)
    >>> plt.plot(cross_disp, source_1d * width_scale, c='xkcd:green')
    >>> plt.axvline(xd_loc, dashes=(6,1), color='xkcd:goldenrod')
    >>> plt.xlabel('Cross-dispersion (arcsec)')
    >>> plt.ylabel('Width-scaled flux (mJy)')
    """
    # Integrate 2D intensity image along dispersion axis
    dy = disp_arcsec[1] - disp_arcsec[0]
    slit_mask = np.isfinite(source_2d)
    nx = len(xd_arcsec)
    source_1D = np.zeros(nx)
    for i in range(nx):
        m = slit_mask[:,i]
        source_1D[i] = np.trapezoid(source_2d[m,i], dx=dy)
    width = (np.sum(slit_mask, axis=0)-1) * dy

    # Integrate along cross-dispersion axis into each pixel
    int_limits = np.arange(np.amin(xd_arcsec), np.amax(xd_arcsec), det.pix_scale)
    y_edges = np.interp(int_limits, xd_arcsec, source_1D)
    w_edges = np.interp(int_limits, xd_arcsec, width)

    x_ext = np.concatenate([xd_arcsec, int_limits])
    isort = np.argsort(x_ext)
    x_ext = x_ext[isort]
    y_ext = np.concatenate([source_1D, y_edges])[isort]
    w_ext = np.concatenate([width, w_edges])[isort]

    npix = len(int_limits) - 1
    source_1d = np.zeros(npix)
    slit_area = np.zeros(npix)
    for i in range(npix):
        mask = (x_ext >= int_limits[i]) & (x_ext <= int_limits[i+1])
        source_1d[i] = np.trapezoid(y_ext[mask], x_ext[mask])
        slit_area[i] = np.trapezoid(w_ext[mask], x_ext[mask])

    cross_disp = 0.5*(int_limits[1:] + int_limits[:-1])

    return source_1d, cross_disp, slit_area


def image_to_detector(source_2d, xd_arcsec, disp_arcsec, source_spectrum, det):
    r"""
    Convert 2D-spatial image to 2D-detector image (wavelength vs
    cross-dispersion) and compute variances from all noise components.

    Parameters
    ----------
    source_2d: 2D float array
        2D source spatial image in cross-dispersion (arcsec) and
        dispersion (arcsec) axis. Image outside the aperture will have
        np.nan values.
    xd_arcsec: 1D float array
        Cross-dispersion array in arcsec. Origin is always defined at
        the center of the wide-end of the slit.
    disp_arcsec: 1D float array
        Dispersion array in arcsec. Origin (x=0) is always defined at
        the center of the slit.
    source_spectrum: tuple of two 1D float arrays
        Arrays containing the source spectrum:
        - wavelength (micron)
        - intensity (mJy arcsec⁻²)
    det: Detector()
        A WALTzER Detector object.

    Returns
    -------
    variances: 3D float array
        Array of shape (ndata, xd_pix, nwave) containing the variances
        (e⁻ s⁻¹ pixel⁻¹) for the following sources:
        - var_source: Source variance.
        - var_background: Background sky variance.
        - var_dark: Dark-current variance.
        - var_read: Read-noise variance.
        - var_systematics: Variance from all other systematic noises.
    cross_disp: 1D float array
        Cross-dispersion array in arcsec for source_1d.
    slit_area: 1D float array
        Collecting area (arcsec²) for each source_1d bin.
    wl: 1D float array
        Wavelenght array (microns) of length nwave.
    half_widths: 1D float array
        Wavelength half widths for each spectral point (of shape nwave).
    throughput: 1D float array
        Detector throughput along wavelength axis (of shape nwave).

    Examples
    --------
    >>> import waltzer_etc as waltz
    >>> import waltzer_etc.extended as wex
    >>>
    >>> # Read a spectrum: wl(um) and intensity (mJy arcsec⁻²)
    >>> s_file = 'psg_rad_comet_upload.txt'
    >>> source_spectrum = np.loadtxt(s_file, unpack=True)
    >>>
    >>> # Generate spatial profile
    >>> radius = 5.0
    >>> pow_law_index = 0.6
    >>> det = waltz.Detector('vis')
    >>> slit_pos = 'medium'
    >>> source_2d, xd_arcsec, d_arcsec, xd_loc, disp_loc = wex.power_law_source(
    >>>     radius, pow_law_index, det.band, slit_pos,
    >>> )
    >>> ax = wex.display_2d_source(source_2d, xd_loc, disp_loc, radius)
    >>>
    >>> # Image and variances as obtained in the WALTzER detector
    >>> obs = wex.image_to_detector(
    >>>     source_2d, xd_arcsec, d_arcsec, source_spectrum, det,
    >>> )
    >>> # Unpack outputs
    >>> variances, cross_disp, slit_area, wl, half_widths, throughput = obs
    >>> source, background, dark, read_noise, sys_noise = variances
    >>>
    >>> # Plot source (only correct by throughput)
    >>> fs = 12
    >>> fig = plt.figure(11)
    >>> plt.clf()
    >>> fig.set_size_inches(7, 6)
    >>> plt.subplots_adjust(0.09, 0.09, 0.99, 0.95)
    >>> ax = plt.subplot(111)
    >>> cm = plt.pcolormesh(wl, cross_disp, source/throughput)
    >>> ax.set_title(f'Source: WALTzER {det.band.upper()} channel')
    >>> ax.set_xlabel(r'Wavelength ($\mathrm{\mu}$m)', fontsize=fs)
    >>> ax.set_ylabel('Cross-dispersion (arcsec)', fontsize=fs)
    >>> ax.tick_params(direction='in', which='both', labelsize=fs-1, color='w')
    >>> bx = plt.colorbar(cm, pad=0.03)
    >>> bx.ax.tick_params(direction='in', which='both', labelsize=fs-1, color='w')
    >>> bx.set_label(r'Flux (e$^{-}$ s$^{-1}$)', fontsize=fs)
    >>>
    >>> # Now plot source and sky (correcting throughput and slit width)
    >>> width0 = slit_area[np.searchsorted(cross_disp, xd_loc)]
    >>> width_scale = np.expand_dims(width0 / slit_area, axis=1)
    >>> image = (source+background)/throughput * width_scale
    >>>
    >>> fs = 12
    >>> fig = plt.figure(12)
    >>> plt.clf()
    >>> fig.set_size_inches(7, 6)
    >>> plt.subplots_adjust(0.09, 0.09, 0.99, 0.95)
    >>> ax = plt.subplot(111)
    >>> cm = plt.pcolormesh(wl, cross_disp, image)
    >>> ax.set_title(f'Source + background: WALTzER {det.band.upper()} channel')
    >>> ax.set_xlabel(r'Wavelength ($\mathrm{\mu}$m)', fontsize=fs)
    >>> ax.set_ylabel('Cross-dispersion (arcsec)', fontsize=fs)
    >>> ax.tick_params(direction='in', which='both', labelsize=fs-1, color='w')
    >>> bx = plt.colorbar(cm, pad=0.03)
    >>> bx.ax.tick_params(direction='in', which='both', labelsize=fs-1, color='w')
    >>> bx.set_label(r'Flux (e$^{-}$ s$^{-1}$)', fontsize=fs)
    """
    # WALTzER high-resolution wavelength grid (angstrom, to be binned later)
    wl_min = det.hires_wl_min
    wl_max = det.hires_wl_max
    resolution = det.hires_resolution
    wl = ps.constant_resolution_spectrum(wl_min, wl_max, resolution)

    # Interpolate spectrum to regular grid
    source_wl, source_intensity = source_spectrum
    source_flux = np.interp(wl, source_wl, source_intensity)

    # Source flux and variance in e- per second
    det.photon_spectrum(wl, source_flux)
    hr_throughput = det.throughput(det.hires_wl)

    # Convolve SED flux to WALTzER resolving power
    flux = inst_convolution(
        det.hires_wl, det.e_flux, det.resolution, det.hires_resolution,
    )

    # Background flux (photons s-1 um-1 pixel-1) to (photons s-1 um-1 arcsec-2)
    width = SLIT_WIDTHS[det.band]['medium'] * det.pix_scale
    e_background = det.e_background / width

    # Integrate wavelength at pixels into fluxes in photons per second
    binsize = det.over_sampling
    nwave = det.nwave
    bin_flux = np.zeros(nwave)
    bin_bkg = np.zeros(nwave)
    throughput = np.zeros(nwave)
    i_start = det.i_start
    for i in range(nwave):
        i1 = i_start + i*binsize
        i2 = i1 + binsize + 1
        bin_flux[i] = np.trapezoid(flux[i1:i2], det.hires_wl[i1:i2])
        bin_bkg[i] = np.trapezoid(e_background[i1:i2], det.hires_wl[i1:i2])
        throughput[i] = np.trapezoid(hr_throughput[i1:i2], det.hires_wl[i1:i2])

    i_end = i_start + (nwave+1)*binsize
    wl_edges = det.hires_wl[i_start:i_end:binsize]
    # Center and half-widths of wavelength bins:
    wl = 0.5 * (wl_edges[1:] + wl_edges[:-1])
    half_widths = 0.5 * (wl_edges[1:] - wl_edges[:-1])

    source_1d, cross_disp, slit_area = dispersion_integral(
        source_2d, xd_arcsec, disp_arcsec, det,
    )

    # Convolve along cross-dispersion axis
    # Cross-dispersion (pixels) is ~5x the FWHM
    fwhm = np.mean(det.cross_dispersion) / 5.0
    spatial_res = 1.0/fwhm
    convolved_source = inst_convolution(
        cross_disp, source_1d, spatial_res, sampling_res=1.0,
    )

    # Now construct spectra-vs-cross-dispersion 2D image(s)
    image = np.outer(convolved_source, bin_flux)
    background = np.outer(slit_area, bin_bkg)

    # Noise
    nreads = 1.0 / det.exp_time
    image_size = np.shape(image)
    # e per second per pixel
    variances = np.array([
        np.abs(image),
        background,
        np.tile(det.dark, image_size),
        np.tile(det.read_noise * nreads, image_size),
        np.tile(det.systematic_noise * nreads, image_size),
    ])

    return variances, cross_disp, slit_area, wl, half_widths, throughput


def simulate_spectrum(
        variances,
        wl,
        throughput,
        xd_arcsec,
        xd_binsize=1,
        wl_binsize=1,
        obs_dur=1.0,
        efficiency=0.6,
        noiseless=False,
    ):
    """
    obs_dur = 1.0
    efficiency = 0.6
    noiseless = False
    xd_binsize = 3
    wl_binsize = 2
    """
    dt = obs_dur * 3600 * efficiency
    signal = variances[0] * dt
    variance = np.sum(variances*dt, axis=0)
    npix, nwave = np.shape(signal)

    # Cross-dispersion binning
    bin_idx = np.arange(0, npix, xd_binsize)
    counts = np.diff(np.append(bin_idx, npix))
    bin_xd = np.add.reduceat(xd_arcsec, bin_idx) / counts
    bin_source = np.add.reduceat(signal, bin_idx, axis=0)
    bin_variance = np.add.reduceat(variance, bin_idx, axis=0)

    # Wavelength binning
    bin_idx = np.arange(0, nwave, wl_binsize)
    counts = np.diff(np.append(bin_idx, nwave))
    bin_wl = np.add.reduceat(wl, bin_idx) / counts
    #bin_widths = np.add.reduceat(half_width, bin_idx)
    bin_source = np.add.reduceat(bin_source, bin_idx, axis=1)
    bin_variance = np.add.reduceat(bin_variance, bin_idx, axis=1)
    bin_throughput = np.add.reduceat(throughput, bin_idx) / counts

    # Undo throughput scaling
    bin_flux = bin_source / bin_throughput
    bin_err = np.sqrt(bin_variance) / bin_throughput
    if not noiseless:
        np.random.seed(random.randint(0, 100_000))
        rand_noise = np.random.normal(0.0, bin_err)
        bin_flux += rand_noise

    return bin_flux, bin_err, bin_wl, bin_xd
