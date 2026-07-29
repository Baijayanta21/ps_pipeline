#!/usr/bin/env python
"""Stage 3 — data and UAPS passes, and the C_l ratio.

Both passes use identical gridding and SCF parameters, taken from the config. The
UAPS pass here uses nrel = 0, which copies realisation 0 across every channel so the
shape matches the data — unlike stage 2, which needs the realisations separate.

Writes el (data), ml (UAPS normalisation) and cl = el / ml.
"""

import sys

import numpy as np

from myutils.config import load
from myutils.stages import (correlate_pass, grid_pass, load_bininfo,
                            one_pass, scf_pass)


def main():
    cfg = load(echo=True, stage='stage 3 · data + UAPS passes')

    uaps = cfg.paths.uaps_uvfits
    if not uaps.is_file():
        raise FileNotFoundError(
            f"missing UAPS file {uaps} — run pipeline/01_uaps.py first"
        )

    Nbin, ni, mask, _, _ = load_bininfo(cfg)
    print(f"{int(mask.sum())} grid points in {Nbin} bins\n", flush=True)

    keep_2d = bool((cfg.get('clfuncs') or {}).get('keep_2d', False))

    print("--- data pass (nrel = -1) ---", flush=True)
    if keep_2d:
        # grid + SCF once, then correlate both ways so the 2-D MAPS costs no extra pass
        GV, _ = grid_pass(cfg, cfg.science_uvfits, -1)
        GV = scf_pass(cfg, GV)
        el = correlate_pass(cfg, GV, ni, mask, Nbin)
        el2d = correlate_pass(cfg, GV, ni, mask, Nbin, two_d=True)
        del GV
        np.save(cfg.product('el2d'), el2d)
        print(f"wrote {cfg.product('el2d').name}  {el2d.shape}", flush=True)
    else:
        el2d = None
        el = one_pass(cfg, cfg.science_uvfits, -1, ni, mask, Nbin)
    np.save(cfg.product('el'), el)
    print(f"wrote {cfg.product('el').name}  {el.shape}\n", flush=True)

    # The UAPS pass is deliberately NOT filtered by default: nrel = 0 copies one
    # realisation across every channel, so V_cg is constant in frequency and its
    # smooth component is itself — filtering would return identically zero.
    apply_scf = bool(cfg.scf.enabled and cfg.scf.get('apply_to_uaps', False))
    print(f"--- UAPS pass (nrel = 0, scf "
          f"{'applied' if apply_scf else 'skipped'}) ---", flush=True)

    GVU, _ = grid_pass(cfg, uaps, 0)
    if apply_scf:
        GVU = scf_pass(cfg, GVU)
    elif cfg.NW:
        # Everything is computed over the full band. SCF trims NW channels from each
        # end of the DATA, and that same trim must propagate to the normalisation:
        # otherwise el(dnu) is summed over (nchan_out - dnu) channel pairs while
        # ml(dnu) is summed over (nchan - dnu), and el/ml is biased by the mismatch
        # at every separation.
        #
        # Trimming the channel axis is NOT the same as truncating separations
        # afterwards, which is what this used to do.
        print(f"    trimming UAPS channels by NW = {cfg.NW} from each end "
              f"to match the filtered data ({GVU.shape[-1]} -> "
              f"{GVU.shape[-1] - 2 * cfg.NW})")
        GVU = GVU[..., cfg.NW:-cfg.NW]

    ml = correlate_pass(cfg, GVU, ni, mask, Nbin)
    ml2d = (correlate_pass(cfg, GVU, ni, mask, Nbin, two_d=True)
            if el2d is not None else None)
    del GVU

    # Shapes must now match by construction — the UAPS was trimmed on the channel
    # axis exactly as the data was. A mismatch here means the two passes disagree
    # about the band, which would silently bias every C_l.
    if ml.shape[-1] != el.shape[-1]:
        raise ValueError(
            f"el has {el.shape[-1]} separations but ml has {ml.shape[-1]}. "
            f"The data and UAPS passes covered different channel ranges — check "
            f"scf.enabled / scf.apply_to_uaps and the channel range n1/n2.")

    zero_frac = float(np.mean(ml == 0.0))
    if zero_frac > 0.5:
        print(f"WARNING: {100 * zero_frac:.1f}% of ml is exactly zero. If SCF was "
              f"applied to the UAPS pass, that is expected and wrong — set "
              f"scf.apply_to_uaps: false.", flush=True)

    np.save(cfg.product('ml'), ml)
    print(f"wrote {cfg.product('ml').name}  {ml.shape}\n", flush=True)

    # The 2-D MAPS needs its own normalisation. Because the UAPS was trimmed on the
    # channel axis above, el2d and ml2d already cover the same absolute channels and
    # no realignment is needed — the earlier offset-slicing is gone.
    if el2d is not None and ml2d is not None:
        if ml2d.shape != el2d.shape:
            raise ValueError(
                f"el2d {el2d.shape} and ml2d {ml2d.shape} disagree — the two passes "
                f"covered different channel ranges.")

        with np.errstate(divide='ignore', invalid='ignore'):
            cl2d = el2d / ml2d
        np.save(cfg.product('ml2d'), ml2d)
        np.save(cfg.product('cl2d'), cl2d)
        print(f"wrote {cfg.product('ml2d').name}  {ml2d.shape}")
        print(f"wrote {cfg.product('cl2d').name}  {cl2d.shape}  "
              f"(normalised MAPS)\n", flush=True)
        del el2d, ml2d, cl2d

    # The normalisation: this division is what turns a correlation into C_l.
    with np.errstate(divide='ignore', invalid='ignore'):
        cl = el / ml

    bad = ~np.isfinite(cl)
    if bad.any():
        print(f"WARNING: {bad.sum()} non-finite entries in cl "
              f"({100 * bad.mean():.2f}%) — ml has zeros where the UAPS pass had no "
              f"signal. Those bins carry no information.", flush=True)

    np.save(cfg.product('cl'), cl)
    print(f"wrote {cfg.product('cl').name}  {cl.shape}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
