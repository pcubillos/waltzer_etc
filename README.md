# :rocket: WALTzER ETC :telescope:
> An ETC for the WALTzER F-mission concept

### Index

- [Installation](#instructions)
- [Stage 1](#stage-1)

  - [Generate SNR files](#run-the-snr)
  - [Additional configuration options](#additional-configuration-options)

- [Stage 2](#stage-2)

  - [Simulate transmission spectra](#simulate-transit-depth-spectra)
  - [Simulate stare-mode spectra](#simulate-stare-mode-spectra)

- [Interactive GUI ETC](#waltzer-tso-gui)


## Instructions

### Install code

Download the code with this shell command:
```shell
git clone https://github.com/pcubillos/waltzer_etc
```

### Install requirements

Go into the ``waltzer_etc`` and execute this shell commad to install the required Python packages
```shell
pip install -e .
```

### Download auxiliary SED files

- **LLMODELS** (required):
  1. Download the stellar SED's from this link: https://drive.google.com/file/d/1pvAs8Z7RUMJrNp-JsHunZyH2vqniUnJj/view?usp=sharing
  2. Unzip the file and place the .flx files into the ``waltzer_etc/data/models/`` folder.

- **BT-Settl** (optional, 1200-3500K):
  1. Download this file: https://archive.stsci.edu/hlsps/reference-atlases/hlsp_reference-atlases_hst_multi_other-spectra_multi_v2_sed.tar
  2. Unzip the file and then move these files ``grp/redcat/trds/source/phoenixm0.0_*_5.0_2011.fits``
     into this folder ``waltzer_etc/data/bt_settl/``.

- **PHOENIX** (optional, 3500-45000K):
  1. Download this file: https://archive.stsci.edu/hlsps/reference-atlases/hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_synphot5.tar
  2. Unzip the file and then move these files ``grp/redcat/trds/grid/phoenix/phoenixm00/phoenixm00_*.fits``
     into this folder ``waltzer_etc/data/phoenix/``.

The final folder structure should look like this:
```
waltzer_etc/data/models/t*g4.4.flx
waltzer_etc/data/bt_settl/phoenixm0.0_*_5.0_2011.fits
waltzer_etc/data/phoenix/phoenixm00_*.fits
```


## Stage 1:

### Run the SNR

You will need an input .csv file with the targets to simulate (e.g., from a NASA Exoplanet Archive request). This file must contain these headers:
- `'pl_name'`
- `'pl_trandur'`
- `'st_teff'`
- `'st_rad'`
- `'st_mass'`
- `'ra'`
- `'dec'`
- `'sy_vmag'`

There is an example target list ``'target_list_20250327.csv'`` in the repository.

Run the code with this shell command:
```shell
waltz target_list.csv waltzer_snr.csv
```

This will produce an output .csv file ``waltzer_snr.csv`` with
- mean and max fluxes at each band (NUV, VIS, and NIR)
- mean and max SNR at each band
- mean transit-depth uncertainties at each band

Spectroscopic SNR values are per sampling element, i.e., per each HWHM,
given the WALTzER's resolution of R = lambda/FWHM = 3000.0
Photometric SNR values (NIR) are band-integrated.


### Additional configuration options

Run this command to see all options
```shell
waltz -h
```

**Select SED library**

Users can set the libray of SEDs to use with the `--sed` argument, for example:
```shell
waltz target_list_20250327.csv waltzer_snr.csv --sed bt_settl
```

**Custom SEDs**

Users can further set cusom SEDs for each target by adding an `sed` header to the input .csv target list. For each planet (row), users can put in the `sed` column the path to the SED file to use.  The SED files are plain-text files with two columns, the wavelength (in microns) and the flux (in either mJy or erg /s /cm^2 /Hz).

**Custom spectral resolution**

While WALTzER output spectra has a defined spectral resolution, calculation are performed internally at a high resolution, before being brought down to the WALTzER instrumental resolution. This high-resolution is by default `R=48_000`.  Users can adjust this high-resolution sampling with the ``--hires`` argument when a higher resolution is needed.  Note that higher resolutions will produce larger files.

```shell
waltz target_list_20250327.csv waltzer_snr.csv --hires 250_000
```

Also note that the ETC will adjust the input ``hires`` value to the closest integer factor of the WALTzER pixel sampling rate (R = HWHM = 6000) for technical reasons (e.g., a `hires` of 250_000 will be adjusted to 252_000).

**Stare mode statistics**

For users interested in S/N statistics on the source, rather than transit depths, use the ``--obs_mode stare`` argument to display the median S/N values on the source.  For stare observations, the `pl_trandur` column sets the total observation duration.

Note that the output pickle always has the same content, regardless of the ``--obs_mode`` argument.  This option is mostly for getting the desired screen output.


## Stage 2:

### Simulate transit depth spectra

Stage 1 will also produce a pickle file (``waltzer_snr.pickle``) with the noise estimation for
each source.  Combine this with a transit-depth model to simulate
WALTzER observations (a sample transit-depth file is provided in the repo):

```python
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyratbay.constants as pc
import waltzer_etc as w
plt.ion()

# Load a WALTzER SNR output pickle file
tso_file = 'waltzer_snr.pickle'
with open(tso_file, 'rb') as handle:
    spectra = pickle.load(handle)
tso = spectra['HD 209458 b']

# Load a transit-depth spectrum
tdepth_file = 'transit_saturn_1600K_clear.dat'
wl, depth = np.loadtxt(tdepth_file, unpack=True)
depth_model = wl, depth

# Simulate WALTzER observation
sim = w.simulate_spectrum(
    tso,
    depth_model,
    n_obs=10,
    resolution=300.0,
    noiseless=False,
)

# Show noised-up WALTzER spectrum
bands, waltzer_wl, waltzer_spec, waltzer_err, waltzer_widths = sim
fig = plt.figure(1)
plt.clf()
fig.set_size_inches(8,5)
plt.subplots_adjust(0.1,0.1,0.98,0.98, hspace=0.15)
ax = plt.subplot(3,1,(1,2))
plt.plot(depth_model[0], depth_model[1]/pc.percent, color='xkcd:blue', label='model')
bands = tso['meta']['bands']
for j,band in enumerate(bands):
    w_label = 'WALTzER simulation' if j==0 else None
    plt.errorbar(
        waltzer_wl[j], waltzer_spec[j]/pc.percent,
        waltzer_err[j]/pc.percent, xerr=waltzer_widths[j],
        fmt='o', ecolor='salmon', color='xkcd:orangered',
        mfc='w', ms=4, zorder=0, label=w_label,
    )
plt.xscale('log')
ax.set_xticks([0.25, 0.3, 0.4, 0.6, 0.8, 1.0, 1.6])
ax.set_xticklabels([])
ax.tick_params(which='both', direction='in')
ax.set_xlim(0.23, 1.7)
ax.set_ylim(0.99, 1.12)
ax.set_ylabel('Transit depth (%)')
ax.legend(loc='upper right')

ax = plt.subplot(3,1,3)
for j,band in enumerate(bands):
    ax.errorbar(
        waltzer_wl[j], waltzer_err[j]/pc.ppm, xerr=waltzer_widths[j],
        fmt='o', ecolor='salmon', color='tomato',
        mfc='w', ms=4, zorder=0,
    )
ax.set_xscale('log')
ax.set_yscale('log')
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticks([0.25, 0.3, 0.4, 0.6, 0.8, 1.0, 1.6])
ax.set_xlim(0.23, 1.7)
ax.tick_params(which='both', direction='in')
ax.set_xlabel('Wavelength (um)')
ax.set_ylabel('Depth error (ppm)')
```

<img alt="WALTzER transit" src="https://github.com/pcubillos/waltzer_etc/blob/main/waltzer_etc/data/waltzer_demo_transmission.png" width="600">

### Simulate stare-mode spectra

'Stare' observations on the source can also be simulated as in the script below. Note that the output spectra is in number of photons that arrived at Earth (instrumental throughput has been detrended).  To get the ground-truth model, users can set the ``noiseless=True`` argument in ``w.simulate_spectrum()``.

```python
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyratbay.constants as pc
import waltzer_etc as w
plt.ion()

# Load a WALTzER SNR output pickle file
tso_file = 'waltzer_snr.pickle'
with open(tso_file, 'rb') as handle:
    spectra = pickle.load(handle)

# Simulate a 1.5h WALTzER observation of the star WASP-69
tso = spectra['WASP-69 b']
sim = w.simulate_spectrum(
    tso,
    obs_type='stare',
    obs_dur=1.5,
    n_obs=1,
)
bands, wl, flux, uncert, wl_widths = flux_data

fig = plt.figure(1)
plt.clf()
fig.set_size_inches(7, 4)
plt.subplots_adjust(0.1, 0.11, 0.98, 0.98, hspace=0.2)
ax = plt.subplot(3,1,(1,2))
bx = plt.subplot(3,1,3)
bands = tso['meta']['bands']
# Show only NUV and VIS
for j,band in enumerate(bands[:-1]):
    ax.errorbar(
        wl[j],
        flux[j],
        yerr=uncert[j],
        color='xkcd:blue',
        ecolor='0.8',
    )
    bx.plot(wl[j], flux[j]/uncert[j], color='tomato')
ax.tick_params(which='both', direction='in')
ax.set_xlim(0.23, 0.825)
ax.set_ylim(bottom=0.0)
ax.set_ylabel('Source flux (# photons)')

bx.tick_params(which='both', direction='in')
bx.set_xlim(0.23, 0.825)
bx.set_ylim(bottom=0.0)
bx.set_ylabel('Source S/N')
bx.set_xlabel('Wavelength (um)')
```

<img alt="WALTzER stare" src="https://github.com/pcubillos/waltzer_etc/blob/main/waltzer_etc/data/waltzer_demo_stare.png" width="600">

## WALTzER TSO GUI

Alternativelty, there is an interactive GUI application to simulate WALTzER TSO's.  The GUI can be launch with this command:

```shell
waltz -tso
```
