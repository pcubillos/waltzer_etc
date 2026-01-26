# :rocket: WALTzER ETC :telescope:
ETC for the WALTzER F-mission concept

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


## Stage 1: run the SNR

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


### Extra

Run this command to see all options
```shell
waltz -h
```


Run this command to compute stats for a fixed time duration (h)
```shell
waltz target_list_20250327.csv waltzer_snr.csv --tdur=2.5
```


## Stage 2: simulate transit depth spectra

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
    tso, depth_model,
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
plt.plot(depth_model[0], depth_model[1]/pc.percent, color='xkcd:blue')
bands = tso['meta']['bands']
for j,band in enumerate(bands):
    plt.errorbar(
        waltzer_wl[j], waltzer_spec[j]/pc.percent,
        waltzer_err[j]/pc.percent, xerr=waltzer_widths[j],
        fmt='o', ecolor='salmon', color='xkcd:orangered',
        mfc='w', ms=4, zorder=0,
    )
plt.xscale('log')
ax.set_xticks([0.25, 0.3, 0.4, 0.6, 0.8, 1.0, 1.6])
ax.set_xticklabels([])
ax.tick_params(which='both', direction='in')
plt.xlim(0.24, 1.7)
plt.ylim(0.99, 1.12)
ax.set_ylabel('Transit depth (%)')

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
ax.set_xlim(0.24, 1.7)
ax.tick_params(which='both', direction='in')
ax.set_xlabel('Wavelength (um)')
ax.set_ylabel('Depth error (ppm)')
```

## WALTzER TSO GUI

Alternativelty, there is an interactive GUI application to simulate WALTzER TSO's.  The GUI can be launch with this command:

```shell
waltz -tso
```
