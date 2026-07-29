#!/usr/bin/env python
"""Stage 0 — copy the input UVFITS, then apply coarse-band flagging to the copy.

**The original file is never modified.** It is copied to ``{output_dir}/UVFITS/``
first, and every later stage reads that copy via ``cfg.science_uvfits``. If anything
here goes wrong, delete the copy and run again — the input is untouched.

The MWA band is divided into coarse bands (24 x 32 channels at 768 channels). Each
has unusable channels at its edges from the polyphase filterbank rolloff, and a
corrupted centre channel from the DC term. This flags, per coarse band, the first N
channels, the centre, and the last N.

Flagging is applied by making the UVFITS weight negative, which is the convention
`myutils.tge.grid.grid` already honours (``weight <= 0`` -> sample zeroed). Data
values are left alone, so the operation is reversible in meaning and idempotent:
running it twice flags exactly the same channels.

    python pipeline/00_flag.py             # copy if needed, then flag
    python pipeline/00_flag.py --force     # re-copy from the original, then flag
    python pipeline/00_flag.py --dry-run   # report what would be flagged
"""

import argparse
import shutil
import sys
import time

import numpy as np
from astropy.io import fits

from myutils.config import load


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config')
    p.add_argument('--force', action='store_true',
                   help='re-copy from the original even if the working copy exists')
    p.add_argument('--dry-run', action='store_true',
                   help='report the channels that would be flagged, change nothing')
    return p.parse_args()


def describe(cfg, idx):
    f = cfg.get('flagging') or {}
    nchan = int(cfg.observation.nchan)
    width = int(f.get('coarse_width') or (nchan // int(f.get('n_coarse_bands', 24))))
    print(f"coarse bands  : {nchan // width} x {width} channels")
    print(f"edge N        : {f.get('coarse_edge_n')}")
    print(f"flagged       : {idx.size} of {nchan} channels "
          f"({100 * idx.size / nchan:.1f}%)")
    print(f"band 0 pattern: {idx[idx < width].tolist()}")


def main():
    args = parse_args()
    cfg = load(args.config, echo=True, stage='stage 0 · coarse-band flagging')

    if not cfg.flagging_enabled:
        print("flagging.enabled is false — nothing to do. Later stages will read "
              "the original input directly.")
        return 0

    idx = cfg.coarse_flag_channels()
    describe(cfg, idx)

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return 0

    src = cfg.paths.input_uvfits
    dst = cfg.paths.work_uvfits
    dst.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. copy, never touching the original -------------------------------
    if dst.exists() and not args.force:
        print(f"\nworking copy exists: {dst}")
        print("Pass --force to re-copy from the original and re-flag.")
    else:
        if dst.exists():
            print(f"\n--force: replacing {dst.name}")
        gb = src.stat().st_size / 1e9
        print(f"\ncopying {gb:.2f} GB", flush=True)
        print(f"  from {src}")
        print(f"  to   {dst}", flush=True)
        t0 = time.time()
        shutil.copy2(src, dst)
        dt = time.time() - t0
        print(f"copied in {dt:.1f}s ({gb / max(dt, 1e-9):.2f} GB/s)", flush=True)

    if src.samefile(dst):                      # paranoia: never edit the input
        raise RuntimeError(
            f"the working copy resolves to the input file itself ({src}). "
            f"Refusing to modify the original.")

    # ---- 2. flag the copy ----------------------------------------------------
    print(f"\nflagging {dst.name}", flush=True)
    t0 = time.time()

    # memmap=False: read the data unit once sequentially, edit in memory, write it
    # back. The strided alternative is pathological on this filesystem.
    with fits.open(dst, mode='update', memmap=False) as hdul:
        data = hdul[0].data['DATA']
        nchan_file = data.shape[4]
        keep = idx[idx < nchan_file]

        w = data[:, 0, 0, 0, keep, :, 2]
        before = int((w <= 0).sum())
        # negative weight == flagged; -abs() is idempotent and keeps the magnitude
        data[:, 0, 0, 0, keep, :, 2] = -np.abs(w)
        after = int((data[:, 0, 0, 0, keep, :, 2] <= 0).sum())

        total = data[:, 0, 0, 0, :, :, 2]
        frac = float((total <= 0).mean())

        hdul.flush()

    print(f"flagged in {time.time() - t0:.1f}s")
    print(f"  samples in flagged channels : {after:,} "
          f"({after - before:,} newly flagged)")
    print(f"  overall flagged fraction     : {100 * frac:.1f}% of all samples")
    print(f"\nlater stages will read {dst}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
