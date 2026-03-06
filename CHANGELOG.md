# CHANGELOG

## Version 0.3.7 (2026-03-06)

- Updated to telescope configuration with two NUV fold.  This brings down the efficiency for bright targets by ~4%, but increases the efficiency of faint targets by ~20-80%
- Updated dichroic efficiencies for NIR photometer (gain of ~20%)
- Fixed bug with missing PHOENIX SED models not being found
- Enabled custom binning per band for simulate_spectrum()
- In GUI, implemented 'Depth Uncertainty" TSO plot to show transit/eclipse depth uncertainties
- Changed NUV binning direction to go from long-to-short wavelengths, this prioritizes the red-end of the spectrum where there are more photons


## Version 0.3.6 (2026-03-02)

- Updated dichroic efficiencies with less conservative (but still responsible) values
- Enabled customizing the primary mirror diameter when launching the GUI (e.g., for a 45cm mirror, launch the application as: `waltz -tso --diam 45`)
- In GUI, added option to bin separately the NUV and VIS bands.

## Version 0.3.5 (2026-02-23)

- In the VIS, updated throughput with an ~8-10% increase across the band.
- In the NUV, incorporated cross-dispersion beam-size data to optimize extraction, dark and read-noise values are lower, leading to a ~25%-50% improved S/N on the transit dephs toward the blue edge.

## Version 0.3.4 (2026-02-20)

- Refactored code to perform more accurate transit-depth SNRs, performing calculations at high-resolution, then convolving to WALTzER resolving power, and lastly integrating over WALTzER pixels.
- Implemented '--hires' argument to command line call, allowing to set the internal high-resolution sampling.  Default (and minimum accepted value) is 48_000.
- Removed simulate_fluxes() function, now incorporated into simulate_spectrum().

## Version 0.3.3 (2026-02-12)

- Updated throughputs

## Version 0.3.2 (2026-02-10)

- Enabled '--obs_mode' argument to command line call, which allows to set 'stare' mode (i.e., no transits).
- Enabled custom SED inputs in command line call, by adding a column with header 'sed', for each target, users can set the path to an SED file.
- simulate_fluxes() can now produce simulated source spectra for stare mode.
- Updated primary-mirror size assumed in command line call to 35cm.

## Version 0.3.1 (2026-01-28)

- Enabled blackbody planet spectrum for eclipse geometry.
- Enabled 'Save data' button to save SED model when the display tab is at "Stellar SED"

## Version 0.3.0 (2026-01-26)

Updated instrument according to step1-revision

- Increased primary diameter from 30 to 35 cm
- NUV band pass shifted from 0.25-0.33um to 0.24-0.32um to capture additional Fe bands
- The `.csv` output from command-line call now includes the minimum fluxes and SNRs in band
- The `.csv` output from command-line call replaced mean fluxes and SNRs in band with the median
- Added quantum-efficiency throughputs as function of wavelength 
- Added PHOENIX and BT-Settl stellar SEDs
- Added option to upload custom stellar SEDs
- Noise and TSO resolution inputs now show the true resolution being used (data is binned down by # of pixels: 2, 3, 4, ...etc. So there's only a fixed number of resolutions possible, particularly at the highest resolutions).
- Updated JWST-programs catalog (there were some missing programs before)

