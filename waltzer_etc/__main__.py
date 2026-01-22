# Copyright (c) 2025 Patricio Cubillos and A. G. Sreejith
# WALTzER is open-source software under the GPL-2.0 license (see LICENSE)

import argparse
import os
import sys

from shiny import run_app
from .__init__ import __version__
from .sample_snr import waltzer_sample


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

    parser.add_argument(
        '-v', '--version',
        action='version',
        help="Show WALTzER's version.",
        version=f'WALTzER ETC version {__version__}',
    )

    # Required positional arguments
    parser.add_argument(
        "input_file",
        type=is_csv_file,
        default=None,
        nargs="?",
        help="Input CSV file with target list.",
    )
    parser.add_argument(
        "output_file",
        type=is_csv_file,
        default=None,
        nargs="?",
        help="Output CSV file with SNR statistics.",
    )

    # GUI
    parser.add_argument(
        "-tso", "--tso",
        action='store_true',
        default=False,
        help="Launch WALTzER TSO GUI",
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="Run GUI in debug mode",
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

    if args.tso:
        reload = args.debug
        app = os.path.realpath(os.path.dirname(__file__)) + '/gui_waltzer.py'
        run_app(app, reload=reload, launch_browser=True, port=8001, dev_mode=True)
        return

    waltzer_sample(
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

