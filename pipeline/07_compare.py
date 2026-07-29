#!/usr/bin/env python
"""Stage 7 — comparison figures across pipeline variants.

Finds the ``{output_dir}/with_SCF`` and ``{output_dir}/without_SCF`` directories
written by run_variants.py and builds the side-by-side figures: the layout of
``Documentation/RA_11.pdf``, where the same field appears with and without SCF.

    python pipeline/07_compare.py
    python pipeline/07_compare.py --runs with_scf=/path/a without_scf=/path/b
    python pipeline/07_compare.py --only spherical

Reads only; it never touches science products.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np                                                    # noqa: E402

from myutils.config import load                                       # noqa: E402
import myutils.plots as mp                                            # noqa: E402

FIGURES = ('spherical', 'cylindrical', 'cl_dnu', 'pk_cuts')

#: subdirectory of output_dir -> legend label
DEFAULT_VARIANTS = [('with_SCF', 'with SCF'), ('without_SCF', 'without SCF')]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config')
    p.add_argument('--runs', nargs='*', default=None, metavar='LABEL=PATH',
                   help='explicit runs to compare; default: the _scf/_noscf dirs')
    p.add_argument('--only', choices=FIGURES, action='append', default=None)
    p.add_argument('--formats', default='pdf,png')
    p.add_argument('--outdir', default=None,
                   help='where to write; default {output_dir}/plots_compare')
    return p.parse_args()


def discover(cfg, spec):
    """Resolve the runs to compare, as [(label, directory), ...]."""
    if spec:
        out = []
        for item in spec:
            if '=' not in item:
                raise ValueError(f"--runs wants LABEL=PATH, got {item!r}")
            label, path = item.split('=', 1)
            out.append((label.replace('_', ' '), Path(path)))
        return out

    base = cfg.paths.output_dir
    found = []
    for sub, label in DEFAULT_VARIANTS:
        d = base / sub
        if d.is_dir() and sorted(d.glob('ps_*.npz')):
            found.append((label, d))
    if not found and sorted(base.glob('ps_*.npz')):
        found.append((cfg.index, base))
    return found


def main():
    args = parse_args()
    cfg = load(args.config, echo=True, stage='stage 7 · variant comparison')

    runs = discover(cfg, args.runs)
    if len(runs) < 2:
        print(f"\nfound {len(runs)} run(s) with a power spectrum — nothing to compare.")
        print("Run both variants first:")
        print("  python pipeline/run_variants.py --local")
        for label, d in runs:
            print(f"    have: {label:<14s} {d}")
        return 1

    print(f"\ncomparing {len(runs)} runs:")
    loaded = []
    for label, d in runs:
        data = mp.load_run(d)
        loaded.append((label, data))
        print(f"  {label:<14s} {d}")

    out_dir = Path(args.outdir) if args.outdir else cfg.out('plots_compare')
    out_dir.mkdir(parents=True, exist_ok=True)
    mp.style()

    pl = cfg.get('plots') or {}
    cvmin = pl.get('cyl_vmin')
    cvmax = pl.get('cyl_vmax')

    wanted = args.only or list(FIGURES)
    tag = cfg.index
    made = []

    if 'spherical' in wanted:
        made += mp.compare_spherical(
            loaded, out=out_dir / f'compare_spherical_{tag}',
            title=f'{tag} — spherical power spectrum',
            reference=pl.get('reference') or None)

    if 'cylindrical' in wanted:
        # one shared colour scale, as RA_11 does; per-panel autoscaling would
        # renormalise away the suppression the comparison exists to show
        made += mp.compare_cylindrical(
            loaded, out=out_dir / f'compare_cylindrical_{tag}',
            vmin=cvmin, vmax=cvmax,
            title=f'{tag} — cylindrical power spectrum')

    if 'pk_cuts' in wanted:
        ranges = cfg.power_spectrum.get('kpara_ranges') or []
        made += mp.pk_cross_sections(
            loaded, reference=float(ranges[0]) if ranges else None,
            out=out_dir / f'compare_pk_cuts_{tag}',
            title=f'{tag} — cuts at fixed $k_\\perp$')

    if 'cl_dnu' in wanted:
        cl_runs, lval = [], None
        for label, d in runs:
            hits = sorted(Path(d).glob('cl_*.npy'))
            bins = sorted(Path(d).glob('bin_info_*.npz'))
            if hits:
                cl_runs.append((label, np.load(hits[0])))
            if bins and lval is None:
                lval = np.load(bins[0])['lval']
        if len(cl_runs) >= 2 and lval is not None:
            made += mp.cl_dnu_panels(
                cl_runs, lval, cfg.observation.dnuc_mhz,
                out=out_dir / f'compare_cl_dnu_{tag}',
                title=f'{tag} — $C_\\ell(\\Delta\\nu)$')
        else:
            print("skipping cl_dnu — need cl_*.npy in both runs plus bin_info")

    print(f"\nwrote {len(made)} file(s) to {out_dir}:")
    for p in made:
        print(f"  {p.name:<44s} {p.stat().st_size / 1024:8.1f} KB")
    return 0


if __name__ == '__main__':
    sys.exit(main())
