#!/usr/bin/env python
"""Stage 2 — annular binning information.

Grids the UAPS file with nrel = -1 over the realisation channels: in a 2-D simulated
file each channel IS an independent realisation, and mkbin averages V·V* along that
axis to obtain M_g. Gridding with nrel = 0 here would leave every channel identical
and M_g with nothing to average.
"""

import sys

import myutils.tge.grid as gd
from myutils.config import load
from myutils.stages import grid_pass


def main():
    cfg = load(echo=True, stage='stage 2 · binning information')

    uaps = cfg.paths.uaps_uvfits
    if not uaps.is_file():
        raise FileNotFoundError(
            f"missing UAPS file {uaps} — run pipeline/01_uaps.py first"
        )

    nrea = cfg.simulation.Nrea
    print(f"gridding {uaps.name}, channels 0-{nrea - 1} (the {nrea} realisations), "
          f"XX only, no flagging", flush=True)

    # nstokes=[0] and flag=False deliberately differ from the science passes here:
    # M_g only needs one polarisation, and the simulated file has no real flags.
    GVU, info = grid_pass(cfg, uaps, nrel=-1, n1=0, n2=nrea - 1,
                          nstokes=[0], flag=False)
    dU = info[0]
    print(f"grid {GVU.shape}, dU = {dU:.4f}", flush=True)

    b = cfg.binning
    outf = str(cfg.product('bininfo').with_suffix(''))    # mkbin appends .npz
    gd.mkbin(GVU[0], dU, cfg.gridding.Umax, cfg.gridding.FWHM, cfg.gridding.f,
             b.binUmax, b.binUmin, b.Nbin, b.Mg_min, outf=outf)

    print(f"\nwrote {cfg.product('bininfo')}")
    print("The 'data :' percentage above is the fraction of grid points surviving "
          "Mg_min and the |U| range. Near zero means those cuts are wrong for this array.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
