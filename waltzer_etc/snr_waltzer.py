# Copyright (c) 2025-2026 Patricio Cubillos and A. G. Sreejith
# WALTzER is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'Detector',
    'calc_variances',
    'bin_tso_data',
    'simulate_spectrum',
]

import configparser
import os
import random

import numpy as np
import pyratbay.constants as pc
import pyratbay.spectrum as ps
import scipy.interpolate as si

from .utils import ROOT, inst_convolution

DEFAULT_DETS = {
    'nuv': f'{ROOT}/data/detectors/waltzer_nuv.cfg',
    'vis': f'{ROOT}/data/detectors/waltzer_vis.cfg',
    'nir': f'{ROOT}/data/detectors/waltzer_nir.cfg',
}

# Slit widths in arcsec for each band/configuration
SLIT_WIDTHS = {
    'nuv': {
        'wide': 60,
        'medium': 20,
        'narrow': 10,
    },
    'vis': {
        'wide': 60,
        'medium': 30,
        'narrow': 15,
    },
}


def calc_collecting_area(diameter, band):
    """
    Calculate the effective collecting area of the telescope+detector
    (everything except the quantum efficiencies)

    No longer used, but might come back to life in the future.
    """
    primary_area = np.pi * (0.5*diameter)**2.0
    Rprim    = 0.90  # Telescope primary reflectance in %
    Rsec     = 0.90  # Telescope secondary reflectance in %
    Sec_obstr= 0.85  # Telescope secondary obstruction in % (1-Obstruction)
    R_d1     = 0.80  # Dichroic 1 Reflectance/Transmission in %
    R_d2     = 0.80  # Dichroic 2 Reflectance/Transmission in %

    R_uvfold = 0.87  # UV fold reflectance in %
    R_uvgr   = 0.87  # UV grating reflectance in %
    Uv_geff  = 0.65  # UV grating effeciency in %

    R_opfold = 0.90  # Optical fold reflectance in %
    R_opgr   = 0.90  # Optical grating reflectance in %
    Op_geff  = 0.75  # Optical grating effeciency in %

    # Effective collecing areas in cm^2
    if band == 'nuv':
        eff_area = (
            primary_area * Rprim * Rsec * Sec_obstr
            * R_d1 * R_uvfold * R_uvgr * Uv_geff
        )
        return eff_area

    if band == 'vis':
        eff_area = (
            primary_area * Rprim * Rsec * Sec_obstr
            * R_d1 * R_d2 * R_opfold**2 * R_opgr * Op_geff
        )
        return eff_area

    if band == 'nir':
        eff_area = (
            primary_area * Rprim * Rsec * Sec_obstr
            * R_d1 * R_d2 * R_opfold
        )
        return eff_area

    raise ValueError(f'Invalid band {repr(band)}')


def throughput(file, primary_area):
    """
    Load a detector throughput curve into a scipy interpolator function

    Parameters
    ----------
    file: String
        Path to a tabulated throughput response function.
        Wavelenth in first column (um), throughput in last column.
    primary_area: Float
        Area of primary mirror in cm^2.
    Returns
    -------
    min_wl: Float
    max_wl: Float
    throughput: scipy interpolator
    """
    if not os.path.exists(file):
        raise ValueError(f'Throughput file not found: {repr(file)}')
    det_response = np.loadtxt(file, unpack=True)
    wl = det_response[0]
    wl_min = np.amin(wl)
    wl_max = np.amax(wl)
    response = det_response[-1] * primary_area

    fill_value = response[0], response[-1]
    throughput = si.interp1d(
        wl, response,
        kind='slinear',
        bounds_error=False, fill_value=fill_value,
    )
    return wl_min, wl_max, throughput


class Detector():
    def __init__(self, detector_cfg, diameter=35.0, hires=48_000):
        """
        detector_cfg = 'detectors/waltzer_nuv.cfg'
        det = Detector(detector_cfg)

        Parameters
        ----------
        detector_cfg: string
            Path to a detector configuration file.
            Or 'nuv', 'vis', 'nir' to use one of the default configurations.
        diameter: float
            Telescope primary-mirror diameter (cm).
        hires: Float
            Internal high-resolution sampling. Input value will be
            adjusted to an integer factor of the WALTzER HWHM.

        Examples
        --------
        >>> import waltzer_etc as waltz
        >>> det = waltz.Detector('nuv')
        """
        if detector_cfg in DEFAULT_DETS:
            detector_cfg = DEFAULT_DETS[detector_cfg]

        config = configparser.ConfigParser()
        config.read(detector_cfg)
        det = config['detector']
        self.diameter = diameter
        self.band = det.get('band')
        self.type = det.get('mode')

        self.resolution = det.getfloat('resolution')
        self.pix_scale = det.getfloat('pix_scale')
        self.dark = det.getfloat('dark')
        self.read_noise = det.getfloat('read_noise')
        self.exp_time = det.getfloat('exp_time')
        self.aperture = det.getint('aperture')
        self.cross_dispersion = det.get('cross_dispersion')

        # Number of pixels in source and in sky-background
        aperture_radius = self.aperture // 2
        if self.type == 'photometry':
            self.npix = int(np.pi*aperture_radius**2)
            sky_in = aperture_radius
            sky_out = 2*aperture_radius
            self.nsky = int(np.pi * (sky_out**2 - sky_in**2))
        elif self.type == 'spectroscopy':
            self.npix = 2*aperture_radius
            sky_in = aperture_radius
            sky_out = 3*aperture_radius
            self.nsky = 2 * (sky_out-sky_in)

        self.primary_area = np.pi * (0.5*diameter)**2.0
        throughput_file = f'{ROOT}/data/detectors/{det.get("throughput_file")}'
        self.wl_min, self.wl_max, self.throughput = throughput(
            throughput_file, self.primary_area,
        )
        margin_left = 2*self.wl_min/self.resolution
        margin_right = 2*self.wl_max/self.resolution

        # High-resolution sampling
        # (must be an integer factor of WALTzER HWHM to simplify integrations)
        resolution = 2.0*self.resolution
        over_sampling = int(np.round(hires / resolution))
        over_sampling = np.amax([8, over_sampling])

        self.hires_resolution = resolution * over_sampling
        self.hires_wl_min = 0.23
        self.hires_wl_max = 2.0
        hires_wl = ps.constant_resolution_spectrum(
            self.hires_wl_min, self.hires_wl_max, self.hires_resolution,
        )

        wl_min = self.wl_min - margin_left
        wl_max = self.wl_max + margin_right
        self.hires_wl = hires_wl[
            (hires_wl>=wl_min) &
            (hires_wl<=wl_max)
        ]

        i_wl_ini = np.searchsorted(self.hires_wl, self.wl_min)
        i_wl_end = np.searchsorted(self.hires_wl, self.wl_max)
        wl_edges = self.hires_wl[i_wl_ini:i_wl_end:over_sampling]
        # Center and half-widths of wavelength bins (micron):
        self.wl = 0.5 * (wl_edges[1:] + wl_edges[:-1])
        self.half_widths = 0.5 * (wl_edges[1:] - wl_edges[:-1])

        # Sampling conversion
        self._wl_edges = wl_edges
        self.i_start = i_wl_ini
        self.nwave = len(wl_edges) - 1
        self.over_sampling = over_sampling

        if self.type == 'photometry':
            self.wl = np.array([0.5 * (self.wl_max + self.wl_min)])
            self.half_widths = np.array([0.5 * (self.wl_max - self.wl_min)])
        self.nwave = len(self.wl)

        # Cross-dispersion beam-size (pixels)
        if self.cross_dispersion is None:
            self.cross_dispersion = np.tile(self.npix, self.nwave)
        else:
            xfile = f'{ROOT}/data/detectors/{self.cross_dispersion}'
            wl_xd, xd_size = np.loadtxt(xfile, unpack=True)
            fill_value = np.amin(xd_size), np.amax(xd_size)
            cross_dispersion = si.interp1d(
                wl_xd, xd_size, bounds_error=False, fill_value=fill_value,
            )
            max_wl = wl_edges[:-1]
            self.cross_dispersion = np.ceil(cross_dispersion(max_wl))

        # TBD: In future this might depend on {RA,dec} of targets
        # Background flux (erg s-1 cm-2 A-1 arcsec-2)
        bkg_file = f'{ROOT}/data/background_total.dat'
        wl_bkg, sky = np.loadtxt(bkg_file, unpack=True)
        self.bkg_model = si.interp1d(
            wl_bkg, sky, kind='slinear', bounds_error=False,
        )

        # Will be set when running self.photon_spectrum()
        self.e_flux = None
        self.e_background = None


    def photon_spectrum(self, wl, flux, aperture='medium'):
        """
        Compute spectra in photons per second.

        Parameters
        ----------
        wl: 1D float array
            Input wavelength array (micron).
        flux: 1D float array
            Input flux spectrum (mJy).

        Returns
        -------
        bin_flux: 1D float array
            Integrated photon flux per bin (photons s⁻¹).
        bin_bkg: 1D float array
            Integrated background photon flux per bin (photons s⁻¹ pixel⁻¹).

        Examples
        --------
        >>> # TBD
        >>> resolution = 60_000.0
        >>> wl = ps.constant_resolution_spectrum(0.23, 2.0, resolution)
        >>> inst_resolution = 3_000.0
        >>> bin_edges = ps.constant_resolution_spectrum(
        >>>     2_400, 20_000, 2.0*inst_resolution,
        >>> )

        >>> # Load stellar SED
        >>> file = './models/t06000g4.4/model.flx'
        >>> sed_wl, flux, _ = np.loadtxt(file, unpack=True)
        >>> sed_flux = np.interp(wl, sed_wl, flux)

        >>> # Flux at instrumental resolution, in erg s-1 cm-2 A-1 at Earth
        >>> conv_flux = inst_convolution(
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
        wl_min = np.amin(self.hires_wl)
        wl_max = np.amax(self.hires_wl)
        ini = np.searchsorted(wl, wl_min)
        end = np.searchsorted(wl, wl_max) + 1

        flux = flux[ini:end]
        wl = wl[ini:end]

        throughput = self.throughput(wl)
        # Convert flux (mJy) to photons s-1 Hz-1
        photon_energy = pc.h * pc.c / (wl*pc.um)
        photons_nu = 1e-26 * flux / photon_energy * throughput
        # Now convert to photons s-1 um-1
        photons = photons_nu * pc.c / (wl*pc.um)**2.0 * pc.um

        # Background flux (erg s-1 cm-2 A-1 pixel-1)
        if self.type == 'photometry':
            bkg_flux = self.bkg_model(wl) * self.pix_scale**2.0
        else:
            slit_width = SLIT_WIDTHS[self.band][aperture]
            bkg_flux = self.bkg_model(wl) * slit_width * self.pix_scale
        # Convert flux (erg s-1 cm-2 A-1 pixel-1) to photons s-1 um-1 pixel-1
        bkg_photons = bkg_flux / photon_energy * throughput * pc.um/pc.A

        self.e_flux = photons
        self.e_background = bkg_photons
        return photons, bkg_photons


    def make_tso(self, wl, flux, aperture='medium'):
        """
        Generate a TSO dictionary summarizing the detector and scence data.
        """
        # Source flux and variance in e- per second
        self.photon_spectrum(wl, flux, aperture)
        throughput = self.throughput(self.hires_wl)

        tso = {
            'band': self.band,
            'hires_wl': self.hires_wl,
            'flux': self.e_flux,
            'background': self.e_background,
            'throughput': throughput,
            'dark': self.dark,
            'read_noise': self.read_noise,
            'det_type': self.type,
            'aperture': aperture,
            'cross_dispersion': self.cross_dispersion,
            'npix': self.npix,
            'nsky': self.nsky,
            'nwave': self.nwave,
            'i_start': self.i_start,
            'over_sampling': self.over_sampling,
            'resolution': self.resolution,
            'hires_resolution': self.hires_resolution,
            'half_widths': self.half_widths,
            'wl_min': self.wl_min,
            'wl_max': self.wl_max,
        }
        return tso


    def flux_stats(self, wl, flux):
        """
        Compute basic flux statistics within a wavelength interval.

        Parameters
        ----------
        wl: 1D float array
            Input wavelength array (microns).
        flux: 1D float array
            Input flux spectrum (mJy).

        Returns
        -------
        min_flux: float
            Minimum flux within the wavelength range (mJy).
        median_flux: float
            Mean flux within the wavelength range (mJy).
        max_flux: float
            Maximum flux within the wavelength range (mJy).
        """
        wl_min = np.amin(self.hires_wl)
        wl_max = np.amax(self.hires_wl)
        band = (wl >= wl_min) & (wl <= wl_max)

        sed_flux = inst_convolution(
            wl[band], flux[band], self.resolution, self.hires_resolution,
        )

        min_flux = np.min(sed_flux)
        median_flux = np.median(sed_flux)
        max_flux = np.max(sed_flux)
        return min_flux, median_flux, max_flux


    def snr_stats(self, var_data, integ_time):
        """
        Compute basic SNR statistics.

        Parameters
        ----------
        wl: 1D float array
            Input wavelength array (micron).
        flux: 1D float array
            Input flux spectrum (mJy).
        integ_time: Float
            In-transit exposure time (s).

        Returns
        -------
        snr_min: Float
            Min signal-to-noise of the flux measurement.
        snr_median: Float
            Mean signal-to-noise of the flux measurement.
        snr_max: Float
            Max signal-to-noise of the flux measurement.
        transit_uncert: Float
            Mean transit-depth uncertainty (ppm) assuming tdur=integ_time.
        """
        source = var_data[2]
        variance = np.sum(np.array(var_data[2:]), axis=0)

        # Signal-to-noise estimation
        snr = source / np.sqrt(variance) * np.sqrt(integ_time)
        snr_min = np.min(snr)
        snr_median = np.median(snr)
        snr_max = np.max(snr)

        # Assume t_in = t_out
        # Assume flux_in approx flux_out
        transit_uncert = np.sqrt(2.0) / snr_median / pc.ppm

        return snr_min, snr_median, snr_max, transit_uncert


def calc_variances(
        tso, readout='full_frame', aperture='medium',
        transit_flux=None, exp_time=300.0,
    ):
    """
    Compute the electrons per second signal of the source,
    background, dark, and read noise.
    This is done by convolving to WALTzER resolving power (FWHM=3000),
    and then integrating the signals over the WALTzER pixel sampling.

    Parameters
    ----------
    tso: dictionary
        A WALTzER tso output for a given target.
    readout: String
        WALTzER readout mode. Yet to be fully tested.
    aperture: String
        Slit aperture position (width): 'narrow', 'medium', 'wide'.
    transit_flux: 1D float array
        An optional second flux spectrum to compute the variances for.
        Typically, an in-transit or an out-of-eclipse spectrum.
        Must be sampled at tso['hires_wl']
    exp_time: Float
        Exposure time (s).

    Returns
    -------
    wl: 1D float array
        Wavelength array at WALTzER pixel positions.
    half_width: 1D float array
        Wavelength half width of WALTzER pixels.
    var_source: 1D float array
        Source Poisson noise variance.
    var_background: 1D float array
        Background Poisson noise variance.
    var_dark: 1D float array
        Dark-current variance.
    var_read: 1D float array
        Read-noise variance.
    var_transit: 1D float array
        Variance of transit_flux. Only returned when transit_flux is not None.
    """
    hires_wl = tso['hires_wl']
    background = tso['background']
    # For the time being, ignore the fact that nreads should be integer
    nreads = 1.0 / exp_time

    # Convolve SED flux to WALTzER resolving power
    flux = inst_convolution(
        hires_wl, tso['flux'], tso['resolution'],  tso['hires_resolution'],
    )

    has_transit = transit_flux is not None
    if has_transit:
        t_flux = inst_convolution(
            hires_wl, transit_flux, tso['resolution'],  tso['hires_resolution'],
        )

    # Integrate at each instrumental pixel to get photons s-1
    if tso['det_type'] == 'photometry':
        bin_flux = np.array([np.trapezoid(flux, hires_wl)])
        if has_transit:
            bin_t_flux = np.array([np.trapezoid(t_flux, hires_wl)])
        bin_bkg = np.array([np.trapezoid(background, hires_wl)])
        nwave = 1
        rebin = 1
        wl = [0.5 * (tso['wl_max']+tso['wl_min'])]
        half_widths = [0.5 * (tso['wl_max']-tso['wl_min'])]
        slit_scale = 1.0

    elif tso['det_type'] == 'spectroscopy':
        rebin = 1
        if readout == 'faint':
            rebin = 2
        elif readout == 'ultra_faint':
            rebin = 4
        nwave = tso['nwave'] // rebin
        binsize = tso['over_sampling'] * rebin

        band = tso['band']
        slit_width = tso['aperture']
        slit_scale = SLIT_WIDTHS[band][aperture] / SLIT_WIDTHS[band][slit_width]
        # Fluxes in photons per second
        bin_flux = np.zeros(nwave)
        bin_t_flux = np.zeros(nwave)
        bin_bkg = np.zeros(nwave)
        i_start = tso['i_start']
        for i in range(nwave):
            i1 = i_start + i*binsize
            i2 = i1 + binsize + 1
            bin_flux[i] = np.trapezoid(flux[i1:i2], hires_wl[i1:i2])
            bin_bkg[i] = np.trapezoid(background[i1:i2], hires_wl[i1:i2])
            if has_transit:
                bin_t_flux[i] = np.trapezoid(t_flux[i1:i2], hires_wl[i1:i2])

        i_end = i_start + (nwave+1)*binsize
        wl_edges = hires_wl[i_start:i_end:binsize]
        # Center and half-widths of wavelength bins:
        wl = 0.5 * (wl_edges[1:] + wl_edges[:-1])
        half_widths = 0.5 * (wl_edges[1:] - wl_edges[:-1])

    # Integrate over time
    var_source = np.abs(bin_flux)

    nsky = tso['nsky']
    # Wavelength-dependent cross dispersion size
    npix = tso['cross_dispersion'][::rebin][0:nwave]

    # Background number of photons
    var_background = npix*(1+npix/nsky) * bin_bkg * slit_scale

    # Dark number of photons
    var_dark = rebin * npix*(1+npix/nsky) * tso['dark']

    # Read-out noise
    if readout == 'full_frame' or tso['det_type'] == 'photometry':
        var_read = npix*(1+npix/nsky) * tso['read_noise'] * nreads

    elif readout in ['bright', 'faint', 'ultra_faint']:
        # Only two reads (instead of nsky) for the background:
        var_read = npix*(1+2*npix/nsky**2) * tso['read_noise'] * nreads


    if has_transit:
        var_transit = np.abs(bin_t_flux)
        return (
            wl,
            half_widths,
            var_source,
            var_background,
            var_dark,
            var_read,
            var_transit,
        )

    return (
        wl,
        half_widths,
        var_source,
        var_background,
        var_dark,
        var_read,
    )


def bin_tso_data(
        det_type, wl, half_width,
        flux_in, var_in, dt_in,
        flux_out=None, var_out=None, dt_out=0.0,
        binsize=None, resolution=None,
        short_to_long=True,
        rebin=1,
    ):
    """
    Compute binned in- and out-of-transit spectra and their variances
    for WALTzER observations.

    Parameters
    ----------
    det_type: string
        Detector type: 'spectroscopy' or 'photometry'.
    flux_in: 1D float arrays
        In-transit flux (electrons per second).
    var_in: 1D float arrays
        In-transit variance (electrons per second).
    dt_in: Float
        In-transit or stare integration time (seconds).
    flux_out: 1D float arrays
        Out-of-transit flux (electrons per second).
    var_out: 1D float arrays
        Out-of-transit variance (electrons per second).
    dt_out: Float
        Out-of-transit integration time (seconds).
    binsize: Integer
        Bin down spectrum by number of pixels.
    resolution: Float
        Alternative to binsize, bin down at specified resolution.
    short_to_long: Bool
        Binning direction (last bin can have fewer points).
    rebin: Integer
        Binning already applied due to readout mode.
    """
    # Fluxes [e- collected] in and out of transit
    # Variance estimations (e- collected) in and out of transit
    flux_in *= dt_in
    var_in *= dt_in
    if flux_out is None or var_out is None or dt_out==0:
        flux_out = flux_in * 0.0
        var_out = var_in * 0.0
    else:
        flux_out *= dt_out
        var_out *= dt_out

    if binsize is None and resolution is None:
        binsize = 1
    elif resolution == 0.0:
       binsize = 1

    if binsize is not None:
        binsize = binsize / rebin
        fraction = binsize - np.floor(binsize)
        binsize = binsize if fraction <= 0.5 else np.ceil(binsize)
        binsize = np.clip(int(binsize), 1, 120000)

    # Photometry
    no_binning = (
        det_type == 'photometry'
        or binsize == 1
    )

    if no_binning:
        return wl, flux_in, flux_out, var_in, var_out, half_width

    # Bin by binsize
    elif resolution is None:
        nwave = len(wl)
        if short_to_long:
            bin_idx = np.arange(0, nwave, binsize)
        else:
            remainder = nwave % binsize
            bin_idx = np.arange(remainder, nwave, binsize)
            if remainder != 0:
                bin_idx = np.append(0, bin_idx)
        counts = np.diff(np.append(bin_idx, nwave))

        bin_widths = np.add.reduceat(half_width, bin_idx)
        bin_wl = np.add.reduceat(wl, bin_idx) / counts

        bin_in = np.add.reduceat(flux_in, bin_idx)
        bin_var_in = np.add.reduceat(var_in, bin_idx)
        bin_out = np.add.reduceat(flux_out, bin_idx)
        bin_var_out = np.add.reduceat(var_out, bin_idx)

    # Bin by resolution
    else:
        wl_min = np.amin(wl)
        wl_max = np.amax(wl)
        if short_to_long:
            bin_edges = ps.constant_resolution_spectrum(
                wl_min, wl_max, resolution,
            )
            bin_edges = np.append(bin_edges, wl_max)
        else:
            bin_edges = 1/ps.constant_resolution_spectrum(
                1/wl_max, 1/wl_min, resolution,
            )
            bin_edges = np.append(wl_min, bin_edges[::-1])

        bin_wl = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_widths = bin_wl - bin_edges[:-1]
        nbins = len(bin_wl)
        bin_out = np.zeros(nbins)
        bin_in = np.zeros(nbins)
        bin_var_out = np.zeros(nbins)
        bin_var_in = np.zeros(nbins)
        for i in range(nbins):
            bin_mask = (wl>=bin_edges[i]) & (wl<bin_edges[i+1])
            bin_out[i] = np.sum(flux_out[bin_mask])
            bin_in[i] = np.sum(flux_in[bin_mask])
            bin_var_out[i] = np.sum(var_out[bin_mask])
            bin_var_in[i] = np.sum(var_in[bin_mask])

    return (
        bin_wl,
        bin_in,
        bin_out,
        bin_var_in,
        bin_var_out,
        bin_widths,
    )


def simulate_spectrum(
        tso, depth_model=None, obs_type='transit', n_obs=1,
        transit_dur=None, obs_dur=None,
        binsize=None, resolution=None, noiseless=False,
        readout='full_frame', aperture='medium', efficiency=None,
        ret_variances=False,
        phantom_var=0.0,
    ):
    """
    Simulate a WALTzER TSO observation, that is, a transit or eclipse
    observation, or a stare on the source.

    Parameters
    ----------
    tso: Dictionary
        A WALTzER TSO data, output from running waltz (stage 1, see example).
    depth_model: 2D float array
        Transit depth model, array of shape [2,nwave] with:
        - wavelength (micron)
        - transit depth
        (sample at R ~50000 for best results)
        Ignore for stare mode.
    obs_type: String
        The observing geometry 'transit', 'eclipse', or 'stare'.
    n_obs: Integer
        Number of transits to co-add.
    transit_dur: Float
        In-transit time (hours).
        Default, take transit duration (h) from tso dictionary.
        Ignore for stare mode.
    obs_dur: Float
        Total observation duration (h).
        If None, assume an out-of-transit duration of 2*max(t_dur, 1h).
    binsize: Integer or iterable
        Binning over spectral axis, in number of pixels.
        Output resolution will be R = 6000.0 / binsize
        If iterable, there must be one binsize value for each band.
    resolution: Float or iterable
        Alternative to binsize, set output resolution.
        Note this input take precedence over binsize.
        If iterable, there must be one resolution value for each band.
    noiseless: Bool
        If False, add scatter to simulated spectrum according to
        the signal's uncertainty.  Set to True to return the ground truth
    readout: String
        WALTzER readout mode, select from: 'full_frame', 'bright',
        'faint', and 'ultra_faint'.
    aperture: String
        Slit aperture position (width): 'narrow', 'medium', 'wide'.
    efficiency: Float
        WALTzER duty cycle efficiency. If None, take value from tso.
    ret_variances: Bool
        Flag to return in- and out-of-transit variances (see below).
    phantom_var: Integer or 1D integer iterable
        Additional variance added in quadrature to noise components.
        Set only if you know what you are doing.

    Returns
    -------
    For transit or eclipse obs_type:
        bands: 1D string list
            The WALTzER band names.
        bin_wl: List of 1D float arrays
            WALTzER wavelength array for NUV, VIS, and NIR bands (micron)
        bin_depth: List of 1D float arrays
            WALTzER transit depths for each band
        bin_error: List of 1D float arrays
            WALTzER transit-depth uncertainties for each band
        bin_widths: List of 1D float arrays
            Wavelength half-widths of WALTzER data points (micron).

    For stare obs_type:
        bands: 1D string list
            WALTzER band names.
        bin_wl: list of 1D float arrays
            Binned wavelength arrays for each band (micron).
        bin_flux: list of 1D float arrays
            Binned collected fluxes (number of electrons).
            (the detector throughputs have been corrected-out from these).
        bin_error: list of 1D float arrays
            Binned-flux uncertainties.
        bin_widths: list of 1D float arrays
            Bin-widths for each data point.

    If ret_variances is True:
        bands: 1D string list
            WALTzER band names.
        bin_wl: list of 1D float arrays
            Binned wavelength arrays for each band (micron).
        bin_in: list of 1D float arrays
            Binned in-transit collected fluxes (number of electrons).
        bin_out: list of 1D float arrays
            Binned out-of-transit collected fluxes (number of electrons).
        bin_var_in: list of 1D float arrays
            Binned in-transit variances.
        bin_var_out: list of 1D float arrays
            Binned out_of-transit variances.
        bin_widths: list of 1D float arrays
            Bin-widths for each data point.

    Examples
    --------
    >>> import pickle
    >>> import numpy as np
    >>> import pyratbay.constants as pc
    >>> import waltzer_etc as w

    >>> # Load a WALTzER SNR output pickle file
    >>> tso_file = 'waltzer_snr.pickle'
    >>> with open(tso_file, 'rb') as handle:
    >>>     spectra = pickle.load(handle)
    >>> tso = spectra['HD 209458 b']

    >>> # Load a transit-depth spectrum
    >>> tdepth_file = 'transit_saturn_1600K_clear.dat'
    >>> wl, depth = np.loadtxt(tdepth_file, unpack=True)
    >>> depth_model = wl, depth

    >>> # Simulate WALTzER observation
    >>> sim = w.simulate_spectrum(
    >>>     tso, depth_model,
    >>>     n_obs=10,
    >>>     resolution=300.0,
    >>>     noiseless=False,
    >>> )

    >>> # Show noised-up WALTzER spectrum
    >>> bands, waltzer_wl, waltzer_spec, waltzer_err, waltzer_widths = sim
    >>> fig = plt.figure(1)
    >>> plt.clf()
    >>> fig.set_size_inches(8,5)
    >>> plt.subplots_adjust(0.1,0.1,0.98,0.98, hspace=0.15)
    >>> ax = plt.subplot(3,1,(1,2))
    >>> plt.plot(depth_model[0], depth_model[1]/pc.percent, color='xkcd:blue')
    >>> bands = tso['meta']['bands']
    >>> for j,band in enumerate(bands):
    >>>     plt.errorbar(
    >>>         waltzer_wl[j], waltzer_spec[j]/pc.percent,
    >>>         waltzer_err[j]/pc.percent, xerr=waltzer_widths[j],
    >>>         fmt='o', ecolor='salmon', color='xkcd:orangered',
    >>>         mfc='w', ms=4, zorder=0,
    >>>     )
    >>> plt.xscale('log')
    >>> ax.set_xticks([0.25, 0.3, 0.4, 0.6, 0.8, 1.0, 1.6])
    >>> ax.set_xticklabels([])
    >>> ax.tick_params(which='both', direction='in')
    >>> plt.xlim(0.24, 1.7)
    >>> plt.ylim(0.99, 1.12)
    >>> ax.set_ylabel('Transit depth (%)')

    >>> ax = plt.subplot(3,1,3)
    >>> for j,band in enumerate(bands):
    >>>     ax.errorbar(
    >>>         waltzer_wl[j], waltzer_err[j]/pc.ppm, xerr=waltzer_widths[j],
    >>>         fmt='o', ecolor='salmon', color='tomato',
    >>>         mfc='w', ms=4, zorder=0,
    >>>     )
    >>> ax.set_xscale('log')
    >>> ax.set_yscale('log')
    >>> ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    >>> ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    >>> ax.set_xticks([0.25, 0.3, 0.4, 0.6, 0.8, 1.0, 1.6])
    >>> ax.set_xlim(0.24, 1.7)
    >>> ax.tick_params(which='both', direction='in')
    >>> ax.set_xlabel('Wavelength (um)')
    >>> ax.set_ylabel('Depth error (ppm)')
    """
    if depth_model is None or obs_type == 'stare':
        depth_model = 0.0
        transit_dur = 0.0

    # An interpolator to eval the depth over the WALTzER bands
    if np.isscalar(depth_model):
        wl_model = [0.23, 20.0]
        depth = np.tile(depth_model, 2)
    else:
        wl_model, depth = depth_model

    if np.isscalar(phantom_var):
        phantom_var = np.tile(phantom_var, 3)

    model = si.interp1d(
        wl_model, depth, kind='slinear',
        bounds_error=False, fill_value=np.nan,
    )

    if efficiency is None:
        efficiency = tso['meta']['efficiency']
    if n_obs is None:
        n_obs = tso['meta']['n_obs']

    if transit_dur is None:
        transit_dur = tso['meta']['transit_dur']
    if obs_dur is None:
        obs_dur = transit_dur + 2.0*np.amax([1.5*transit_dur, 1.0])

    bands = tso['meta']['bands']
    if np.isscalar(binsize) or binsize is None:
        binsize = np.tile(binsize, len(bands))
    elif len(binsize) != len(bands):
        raise ValueError('Length of binsize does not match the number of bands')

    if np.isscalar(resolution) or resolution is None:
        resolution = np.tile(resolution, len(bands))
    elif len(resolution) != len(bands):
        raise ValueError('Length of resolution does not match the number of bands')

    # Total times integrating in- and out-of-transit (in seconds)
    total_time = (obs_dur * 3600) * efficiency * n_obs
    dt_in = (transit_dur * 3600) * efficiency * n_obs
    dt_out = total_time - dt_in

    walz = dict(
        wl = [],
        depth = [],
        flux = [],
        flux_out = [],
        error = [],
        var = [],
        var_out = [],
        width = [],
    )

    for j,band in enumerate(bands):
        det = tso[band]
        # Fluxes [e- collected] in-transit or out-of-eclipse
        depth = model(det['hires_wl'])
        if obs_type == 'transit':
            transit_flux = det['flux'] * (1.0-depth)
        elif obs_type == 'eclipse':
            transit_flux = det['flux'] * (1.0+depth)
        elif obs_type == 'stare':
            transit_flux = None
            dt_in = total_time

        # Binning correction due to readout mode
        rebin = 1
        if readout == 'faint':
            rebin = 2
        elif readout == 'ultra_faint':
            rebin = 4

        # Data at WALTzER sampling and resolving power
        var_data = calc_variances(det, readout, aperture, transit_flux)
        wl = var_data[0]
        half_width = var_data[1]
        flux = var_data[2]
        variance = np.sum(np.array(var_data[2:6]), axis=0)
        variance += phantom_var[j] * rebin

        if obs_type == 'stare':
            flux_out = None
            var_out = None
        else:
            # eclipse
            flux_out = var_data[6]
            var_out = np.sum(np.array(var_data[3:7]), axis=0)
            var_out += phantom_var[j] * rebin
            if obs_type == 'transit':
                flux, flux_out = flux_out, flux
                variance, var_out = var_out, variance

        # Bin NUV from long to short, VIS from short to long
        short_to_long = band != 'nuv'
        bin_data = bin_tso_data(
            det['det_type'], wl, half_width,
            flux, variance, dt_in,
            flux_out, var_out, dt_out,
            binsize[j], resolution[j],
            short_to_long,
            rebin,
        )

        if obs_type == 'stare':
            # Undo throughput scaling
            if det['det_type'] == 'spectroscopy':
                det_wl = det['hires_wl']
                det_resp = det['throughput']
            else:
                det_wl = det['wl_min'], det['wl_max']
                det_resp = det['throughput'][0:2]
            # TBD: band-integrate instead of interpolating
            throughput = si.interp1d(
                det_wl, det_resp, kind='slinear',
                bounds_error=False, fill_value=0.0,
            )
            response = throughput(bin_data[0])
            bin_flux = bin_data[1] / response
            bin_err = np.sqrt(bin_data[3] / response**2.0)
            if not noiseless:
                np.random.seed(random.randint(0, 100_000))
                rand_noise = np.random.normal(0.0, bin_err)
                bin_flux += rand_noise
            walz['wl'].append(bin_data[0])
            walz['flux'].append(bin_flux)
            walz['error'].append(bin_err)
            walz['width'].append(bin_data[5])

        elif ret_variances:
            walz['wl'].append(bin_data[0])
            walz['flux'].append(bin_data[1])
            walz['flux_out'].append(bin_data[2])
            walz['var'].append(bin_data[3])
            walz['var_out'].append(bin_data[4])
            walz['width'].append(bin_data[5])

        elif obs_type in ['transit', 'eclipse']:
            bin_in = bin_data[1]
            bin_out = bin_data[2]
            bin_var_in = bin_data[3]
            bin_var_out = bin_data[4]
            bin_depth = 1.0 - (bin_in/dt_in) / (bin_out/dt_out)
            # Error propagation into transit-depth uncertainty
            bin_err = np.sqrt(
                (dt_out / dt_in / bin_out)**2.0 * bin_var_in +
                (bin_in / dt_in * dt_out / bin_out**2.0)**2.0 * bin_var_out
            )
            if not noiseless:
                # The numpy random system must have its seed reinitialized
                # in each sub-processes to avoid identical 'random' steps.
                # random.randomint is process- and thread-safe.
                np.random.seed(random.randint(0, 100_000))
                rand_noise = np.random.normal(0.0, bin_err)
                bin_depth += rand_noise

            walz['wl'].append(bin_data[0])
            walz['depth'].append(bin_depth)
            walz['error'].append(bin_err)
            walz['width'].append(bin_data[5])

    # Unpack returns
    rets = [
        val
        for val in walz.values()
        if len(val) > 0
    ]

    return [bands] +  rets
