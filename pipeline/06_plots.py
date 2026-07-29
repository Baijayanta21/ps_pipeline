#!/usr/bin/env python
"""Stage 6 — figures.

Reads the stage-5 product and writes the standard figures to {output_dir}/plots/.
Cheap and safe to re-run: it touches no science, only reads.

    python pipeline/06_plots.py
    python pipeline/06_plots.py --only spherical
    python pipeline/06_plots.py --formats pdf
"""

import argparse
import sys

import matplotlib
matplotlib.use('Agg')                       # no display on a compute node

import numpy as np                                                    # noqa: E402

from myutils.config import load                                       # noqa: E402
import myutils.plots as mp                                            # noqa: E402

FIGURES = ('cylindrical', 'cylindrical_masked', 'xstat', 'spherical',
           'uv', 'windows', 'cl_dnu', 'cl_nunu', 'pk_cuts')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config')
    p.add_argument('--only', choices=FIGURES, action='append', default=None,
                   help='make just this figure; repeatable')
    p.add_argument('--formats', default='pdf,png',
                   help='comma-separated output formats (default pdf,png)')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load(args.config, echo=True, stage='stage 6 · figures')

    formats = tuple(f.strip() for f in args.formats.split(',') if f.strip())
    pl = cfg.get('plots') or {}
    cvmin = pl.get('cyl_vmin')
    cvmax = pl.get('cyl_vmax')

    wanted = args.only or list(FIGURES)

    plot_dir = cfg.out('plots')
    plot_dir.mkdir(parents=True, exist_ok=True)
    mp.style()

    # windows and uv need no power spectrum, so make them before requiring it
    made = []

    if 'windows' in wanted:
        sm = cfg.scf.SM_mhz
        scales = sorted({1.0, float(sm), 3.0})
        made += mp.scf_windows(plot_dir / f'scf_windows_{cfg.index}',
                               smoothing_mhz=scales, dnuc=cfg.observation.dnuc_mhz)

    if 'uv' in wanted:
        cfg.require('bininfo')
        made += mp.uv_coverage(np.load(cfg.product('bininfo')),
                               plot_dir / f'uv_coverage_{cfg.index}')

    if 'cl_dnu' in wanted:
        cfg.require('cl', 'bininfo')
        lval = np.load(cfg.product('bininfo'))['lval']
        made += mp.cl_dnu_panels([(cfg.index, np.load(cfg.product('cl')))],
                                 lval, cfg.observation.dnuc_mhz, max_dnu=10.2,
                                 out=plot_dir / f'cl_dnu_{cfg.index}')

    if 'cl_nunu' in wanted:
        p2 = cfg.product('cl2d')
        if not p2.exists():
            print(f"skipping cl_nunu — {p2.name} not found. "
                  f"Set clfuncs.keep_2d: true and re-run stage 3.")
        else:
            lval = np.load(cfg.product('bininfo'))['lval']
            cl2d = np.load(p2)
            ib = min(len(lval) - 1, 2)
            made += mp.cl_nunu(cl2d, ib, cfg.observation.nuc_mhz,
                               cfg.observation.dnuc_mhz, lval=lval,
                               symmetric=cfg.scf.enabled,
                               out=plot_dir / f'cl_nunu_{cfg.index}')

    ps_figs = {'cylindrical', 'cylindrical_masked', 'xstat', 'spherical',
               'pk_cuts'}
    if ps_figs & set(wanted):
        ps_path = cfg.product('ps')
        if not ps_path.exists():
            raise FileNotFoundError(
                f"missing {ps_path.name} — run pipeline/05_ps.py first")
        d = np.load(ps_path)

        if 'cylindrical' in wanted:
            made += mp.cylindrical_ps(d, plot_dir / f'cyl_ps_{cfg.index}',
                                      vmin=cvmin, vmax=cvmax)
        if 'cylindrical_masked' in wanted:
            made += mp.cylindrical_ps(d, plot_dir / f'cyl_ps_masked_{cfg.index}',
                                      masked=True, vmin=cvmin, vmax=cvmax)
        if 'xstat' in wanted:
            made += mp.x_statistic(d, plot_dir / f'X_{cfg.index}')
        if 'spherical' in wanted:
            made += mp.spherical_ps(
                d, plot_dir / f'spherical_ps_{cfg.index}',
                reference=pl.get('reference') or None)
        if 'pk_cuts' in wanted:
            ranges = cfg.power_spectrum.get('kpara_ranges') or []
            made += mp.pk_cross_sections(
                [(cfg.index, d)],
                reference=float(ranges[0]) if ranges else None,
                out=plot_dir / f'pk_cuts_{cfg.index}')

    print(f"\nwrote {len(made)} file(s) to {plot_dir}:")
    for p in made:
        if p.suffix.lstrip('.') in formats or True:
            print(f"  {p.name:<40s} {p.stat().st_size / 1024:8.1f} KB")
    return 0


if __name__ == '__main__':
    sys.exit(main())
