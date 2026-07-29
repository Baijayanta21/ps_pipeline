#!/usr/bin/env python
"""Measure UVFITS read strategies: memmap (lazy, strided) vs sequential.

Extracting UU/VV from a random-groups UVFITS is a strided walk over the whole file
because the group parameters are interleaved with each group's visibilities. On a
network filesystem that is far slower than simply streaming the data unit once.

Run this on a compute node, ideally one that has NOT already read the file — page
cache makes both strategies look identical:

    sbatch --partition=Main --mem=32G --time=00:30:00 \
           --output=bench_%j.log --wrap \
           "source /idia/users/schatterjee/vnv_ilifu/vnvMWA/bin/activate && \
            python pipeline/bench_read.py"

Add --file PATH to test a specific file, otherwise the config's input is used.
"""

import argparse
import sys
import time

import numpy as np
from astropy.io import fits


def timed(label, fn):
    start = time.time()
    result = fn()
    secs = time.time() - start
    print(f"  {label:<34s} {secs:8.2f}s", flush=True)
    return result, secs


def bench(path):
    size_gb = path.stat().st_size / 1e9 if hasattr(path, 'stat') else 0.0
    print(f"\nfile: {path}  ({size_gb:.2f} GB)\n")

    print("memmap=True  (astropy default — lazy, strided)")

    def lazy():
        with fits.open(path, mode='readonly', memmap=True) as h:
            uu = np.copy(h[0].data['UU'])
            vv = np.copy(h[0].data['VV'])
        return uu, vv

    (uu1, vv1), t_lazy = timed("open + read UU/VV", lazy)

    print("\nmemmap=False (sequential read of the data unit)")

    def seq():
        with fits.open(path, mode='readonly', memmap=False) as h:
            uu = np.copy(h[0].data['UU'])
            vv = np.copy(h[0].data['VV'])
        return uu, vv

    (uu2, vv2), t_seq = timed("open + read UU/VV", seq)

    same = np.array_equal(uu1, uu2) and np.array_equal(vv1, vv2)
    print(f"\n  identical results               {same}")
    print(f"  rows                            {uu1.size:,}")
    if t_seq > 0:
        print(f"  speedup (lazy / sequential)     {t_lazy / t_seq:.1f}x")
    if t_seq > 0 and size_gb:
        print(f"  sequential throughput           {size_gb / t_seq:.2f} GB/s")

    print("\nNOTE both numbers are meaningless if the file is already in page cache.")
    print("     A cold read is the one that matters — use a node that has not")
    print("     touched this file, and run the benchmark first thing.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config')
    p.add_argument('--file', help='UVFITS to benchmark; defaults to the config input')
    args = p.parse_args()

    if args.file:
        from pathlib import Path
        path = Path(args.file)
    else:
        from myutils.config import load
        path = load(args.config).paths.input_uvfits

    return bench(path)


if __name__ == '__main__':
    sys.exit(main())
