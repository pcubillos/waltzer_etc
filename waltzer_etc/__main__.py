# Copyright (c) 2025 Sreejith and Patricio
# LICENSE TBD

import sys
import argparse
from .snr_waltzer import waltzer_snr


def is_csv_file(filename: str) -> str:
    """Validate that filename string is a .csv file."""
    if not filename.lower().endswith(".csv"):
        error = f"File '{filename}' must have .csv extension."
        raise argparse.ArgumentTypeError(error)
    return filename


def parse_args():
    """
    Command-line parser for call from the prompt.
    """
    parser = argparse.ArgumentParser(
        description="WALTzER SNR and ETC."
    )
         
    # Required positional arguments
    parser.add_argument(
        "input_file",
        type=is_csv_file,
        help="Input CSV file with target list.",
    )
    parser.add_argument(
        "output_file",
        type=is_csv_file,
        help="Output CSV file with SNR statistics.",
    )

    # Optional arguments
    parser.add_argument(
        "--nobs",
        type=int,
        default=10,
        help="Number of observations (default: 1).",
    )
    parser.add_argument(
        "--diam",
        type=float,
        default=30.0,
        help="Telescope diameter in cm (default: 30.0).",
    )
    parser.add_argument(
        "--eff",
        type=float,
        default=0.6,
        help="Telescope duty-cycle efficiency (default: 0.6).",
    )
    parser.add_argument(
        "--tdur",
        type=float,
        default=None,
        help="Transit duration in hours; if set, calculate statistics assuming a fixed transit duration for all targets. Else, take values from input .csv target list. (default: None).",
    )

    args = parser.parse_args()
    return args


def main():
    """
    WALTzER Exposure time calculator

    Usage
    -----
    From the command line, run:
    waltz target_list_20250327.csv  waltzer_snr_test.csv
    """
    args = parse_args()
    waltzer_snr(
         csv_file=args.input_file,
         output_csv=args.output_file,
         diameter=args.diam,
         efficiency=args.eff,
         t_dur=args.tdur,
         n_obs=args.nobs,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()

