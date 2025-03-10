import argparse
from pathlib import Path
from .lsp import Lsp

VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 0
VERSION_STRING = f'xmtolsp v{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}'

DESCRIPTION = "Converts a FastTracker 2 XM module file to Amiga LSP (LightSpeed Player) format."
HELP_INPUT_FILE = "the input XM file"
HELP_OUTPUT = "specify output directory (default: current directory)"
HELP_NTSC = "use NTSC clock rate (instead of PAL) when calculating Paula period"
HELP_NO_TRIM = "do not trim any samples to their max playback position"
HELP_NO_DOWNSAMPLE = "do not downsample any samples"
HELP_MIN_PER = "specify Paula period that samples' highest note will play at (default: 124)"
HELP_OPTIMAL_PER = "specify target optimal Paula period that samples' lowest note will play at, if possible (default: 160)"
HELP_QUIET = "run silently (do not output any text to stdout)"
HELP_VERSION = "print xmtolsp version number and exit"

parser = argparse.ArgumentParser(prog='xmtolsp', description=DESCRIPTION)
parser.add_argument('input_file', type=Path, help=HELP_INPUT_FILE)
parser.add_argument('-q', '--quiet', action='store_true', help=HELP_QUIET)
parser.add_argument('-o', dest='output_dir', type=Path, metavar='DIRECTORY', default=Path.cwd(), help=HELP_OUTPUT)
parser.add_argument('--min-per', type=int, metavar='PERIOD', default=124, help=HELP_MIN_PER)
parser.add_argument('--optimal-per', type=int, metavar='PERIOD', default=160, help=HELP_OPTIMAL_PER)
parser.add_argument('--no-downsample', action='store_true', help=HELP_NO_DOWNSAMPLE)
parser.add_argument('--no-trim', action='store_true', help=HELP_NO_TRIM)
parser.add_argument('--ntsc', action='store_true', help=HELP_NTSC)
parser.add_argument('--version', action='version', version=VERSION_STRING, help=HELP_VERSION)
args = parser.parse_args()
if not args.output_dir.is_dir():
    raise ValueError(f"Output path is not a directory or does not exist: {args.o}")
if not args.quiet:
    print(VERSION_STRING)
    print(f"Processing {args.input_file.name}...")
lsp = Lsp(args.input_file,
          ntsc=args.ntsc,
          no_trim=args.no_trim,
          no_downsample=args.no_downsample,
          min_per=args.min_per)
bank, score = lsp.build()
input_stem = args.input_file.stem
with open(args.output_dir/f'{input_stem}.lsbank', 'wb') as f:
    f.write(bank)
with open(args.output_dir/f'{input_stem}.lsmusic', 'wb') as f:
    f.write(score)
if not args.quiet:
    print("Operation finished. LSP checksum: " + bank[:4].hex())
