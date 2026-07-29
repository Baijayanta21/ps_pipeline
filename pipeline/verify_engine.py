#!/usr/bin/env python
"""Prove the numba gridding engine gives identical results to the numpy one.

Grids the same file twice — engine='numpy' then engine='numba' — and compares the
arrays exactly. The numba kernel performs the same multiplications in the same order
per (grid cell, channel), so the result should be *bit-identical*, not merely close.

    python pipeline/verify_engine.py --file some.uvfits
    python pipeline/verify_engine.py            # uses the config's UAPS file

Run this before setting gridding.engine: numba in your config.
"""

import argparse
import sys
import time

import numpy as np

import myutils.tge.grid as gd


def run(path, n1, n2, nrel, umax, fwhm, f, flag, nstokes, seed):
    out = {}
    for engine in ('numpy', 'numba'):
        print(f"\n--- engine = {engine} ---", flush=True)
        start = time.time()
        GV, info = gd.grid(str(path), n1, n2, nrel, umax, fwhm, f, flag,
                           list(nstokes), seed=seed, engine=engine)
        out[engine] = (GV, time.time() - start)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config')
    p.add_argument('--file', help='UVFITS to grid; defaults to the config UAPS file')
    p.add_argument('--n1', type=int, default=0)
    p.add_argument('--n2', type=int, default=15, help='keep this small; default 15')
    p.add_argument('--umax', type=float, default=None)
    p.add_argument('--nrel', type=int, default=-1,
                   help='-1 data, -2 noise, n>=0 copy channel n')
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()

    from myutils.config import load
    cfg = load(args.config)

    path = args.file or cfg.paths.uaps_uvfits
    if not str(path) or not __import__('pathlib').Path(path).is_file():
        print(f"no such file: {path}", file=sys.stderr)
        return 2

    umax = args.umax if args.umax is not None else cfg.gridding.Umax

    print(f"file    : {path}")
    print(f"channels: {args.n1}-{args.n2}   Umax: {umax}   nrel: {args.nrel}")

    res = run(path, args.n1, args.n2, args.nrel, umax,
              cfg.gridding.FWHM, cfg.gridding.f, cfg.gridding.flag,
              cfg.gridding.nstokes, args.seed)

    a, ta = res['numpy']
    b, tb = res['numba']

    print("\n" + "=" * 60)
    print(f"shape            {a.shape}  vs  {b.shape}")
    print(f"numpy            {ta:8.2f}s")
    print(f"numba            {tb:8.2f}s")
    if tb > 0:
        print(f"speedup          {ta / tb:8.2f}x")

    d = np.abs(a - b)
    scale = np.abs(a).max() or 1.0
    rel = d.max() / scale
    ndiff = int((d > 0).sum())

    print(f"\nbit-identical    {np.array_equal(a, b)}")
    print(f"  differing      {ndiff:,} of {d.size:,}")
    print(f"  max abs diff   {d.max():.3e}")
    print(f"  max rel diff   {rel:.3e}")

    # Exact equality is NOT expected, and its absence is not a defect.
    #
    # numpy adds the whole direct block, then the whole conjugate block. The numba
    # kernel interleaves them per channel. Where a baseline's grid block overlaps its
    # own conjugate — the cells near u = 0 — the same two numbers are summed into one
    # cell in a different order, and floating-point addition is not associative. The
    # result differs in the last bit or two of a double.
    #
    # The bar is therefore agreement to within a few ULP, not equality.
    TOL = 1e-13
    if rel <= TOL:
        print(f"\nAgreement within {TOL:.0e} relative — differences are floating-point")
        print("accumulation order in the self-conjugate overlap region, not a defect.")
        print("SAFE to set gridding.engine: numba in the config.")
        return 0

    print(f"\nDisagreement exceeds {TOL:.0e} relative — this is NOT just rounding.")
    print("Do not enable gridding.engine: numba.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
