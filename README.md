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
pip install -r requirements.txt
```

### Download auxiliary files

Fetch stellar SED's from this link: https://drive.google.com/file/d/1Xlg6qLCSNAlZ9gXwupWl-58bkfJ-6dhR/view?usp=sharing

Unzip the file and place the ``models`` folder in the same directory where snr_waltzer.py is located.

## Run the SNR

You will need an input .csv file with the targets to simulate. This file must contain these headers:
- `'pl_name'`
- `'st_teff'`
- `'st_rad'`
- `'st_mass'`
- `'ra'`
- `'dec'`
- `'sy_dist'`
- `'sy_vmag'`

There is an example target list ``'target_list_20250327.csv'`` in the repository.

Run the code with this shell command:
```shell
python snr_waltzer.py targets.csv  waltzer_snr.csv
```

This will produce an output .csv file ``waltzer_snr.csv`` with the SNR values.
