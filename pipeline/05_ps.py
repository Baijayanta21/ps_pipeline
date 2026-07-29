#!/usr/bin/env python
"""Stage 5 — power spectrum.

Fast, and the stage you re-run most while tuning the mode mask and the number of k
bins. Writes ps_{index}.npz with everything needed to plot.
"""

import sys

import numpy as np

from myutils.config import load
from myutils.stages import load_bininfo
import myutils.masking as msk
import myutils.psfuncs.psestimation as pe


def build_flag_mask(cfg, kper, kpara, fac, geom=None):
    """Resolve the mode selection: cached file, or built from the config."""
    ps = cfg.power_spectrum
    path = ps.flag_mask
    if path.exists():
        fm = np.load(path)
        if fm.shape != (kper.size, kpara.size):
            raise ValueError(
                f"{path} has shape {fm.shape}, but this run needs "
                f"{(kper.size, kpara.size)}. Delete it to rebuild, or fix NE/Nbin."
            )
        print(f"mode mask   : loaded {path.name}, {int(fm.sum())} modes")
        return fm

    spec = dict(ps.get('mask') or {})
    preset = spec.get('preset')
    if not spec:
        # no mask section -> the tutorials' tabulated windows, as before
        spec = {'use_tabulated': True,
                'kpara_ranges': ps.get('kpara_ranges'),
                'kpara_start_index': ps.get('kpara_start_index')}
    elif preset is None:
        # Only fall back to the config's static bands when no preset is in play.
        # A preset carries its own, already rescaled to this band — filling them in
        # here would overwrite the scaling with the paper's 154.2 MHz numbers.
        spec.setdefault('kpara_ranges', ps.get('kpara_ranges'))
        spec.setdefault('kpara_start_index', ps.get('kpara_start_index'))

    # Expand before reporting, so what is printed is what is applied rather than
    # what was typed — including the rescaling to this run's band.
    spec = msk.resolve_spec(spec, geom, ps.get('cosmology') or 'Planck18')

    mask, parts = msk.build_mask(kper, kpara, fac, spec, geom,
                                 ps.get('cosmology') or 'Planck18')
    print("mode mask   : built from config"
          f"{f' (preset: {preset})' if preset else ''}")
    if preset and geom:
        ref = msk.PAPER_REF['nuc_mhz']
        if abs(geom['nuc_mhz'] - ref) > 0.5:
            print(f"  rescaled      : published limits calibrated at "
                  f"{ref} MHz -> this run at {geom['nuc_mhz']:.3f} MHz "
                  f"(SM = {geom['SM_mhz']:g} MHz)")
    buf = spec.get('wedge_buffer') or 0.0
    print(f"  wedge (always): excluding k_par <= {fac:.3f} * k_perp"
          f"{f' + {buf}' if buf else ''}")
    opts = [n for n, on in (('k limits', 'ranges' in parts),
                            ('tabulated windows', 'tabulated' in parts)) if on]
    print(f"  optional      : {', '.join(opts) if opts else 'none — wedge only'}")
    print(msk.describe(parts, kper, kpara, mask))

    if not mask.any():
        raise ValueError(
            "the mask selects no modes. Loosen power_spectrum.mask — a wedge cut "
            "plus a k_para floor can easily exclude everything at low k_perp."
        )
    fm = mask.astype('int')
    np.save(path, fm)
    print(f"              saved {path.name}")
    return fm


def main():
    cfg = load(echo=True, stage='stage 5 · power spectrum')
    cfg.require('cl', 'cln')

    ps = cfg.power_spectrum
    NE = cfg.NE

    _, _, _, lval, _ = load_bininfo(cfg)
    lval = lval.astype(int)

    cl_all = np.load(cfg.product('cl'))
    cln_all = np.load(cfg.product('cln'))

    # --- how many separations are actually usable? --------------------------------
    #
    # The number of contributing channel pairs falls as (nchan - dnu), so at the
    # largest separations only a handful remain and flagging can remove all of them.
    # There ml = 0, and cl = el/ml is meaningless.
    #
    # TRUNCATE rather than down-weight. func_pk inverts (A^dag N^-1 A) per ell bin;
    # zeroing rows of N^-1 makes that matrix rank-deficient and the inverse explodes —
    # measured |pk| = 2e25 instead of ~1e12. Dropping the unusable tail keeps the
    # system full rank, and costs only separations that carried no data anyway.
    ml_path = cfg.product('ml')
    usable = NE
    if ml_path.exists():
        ml = np.load(ml_path)
        scale = np.abs(ml).max(axis=-1, keepdims=True)
        good = np.abs(ml) > 1e-8 * np.where(scale > 0, scale, 1.0)
        allgood = good.all(axis=tuple(range(ml.ndim - 1)))
        usable = int(np.argmin(allgood)) if not allgood.all() else ml.shape[-1]
        if usable < NE:
            print(f"NOTE: the UAPS normalisation collapses beyond separation "
                  f"{usable} — only {ml.shape[-1] - usable} of {ml.shape[-1]} "
                  f"separations affected, each with <= {ml.shape[-1] - usable} "
                  f"contributing channel pairs.")
            print(f"      truncating NE {NE} -> {usable} so the MLE stays full rank.")
            NE = usable
    else:
        print(f"note: {ml_path.name} not found — cannot check the normalisation.")

    cl = cl_all[..., :NE]
    cln = cln_all[..., :NE]
    print(f"cl {cl.shape}, cln {cln.shape}\n", flush=True)

    if cln.shape[0] < 10:
        print(f"WARNING: only {cln.shape[0]} noise realisations — std() will be "
              f"a poor estimate and sigma_est unreliable.\n", flush=True)

    # anything still non-finite after truncation is a genuine surprise
    if not np.isfinite(cl).all() or not np.isfinite(cln).all():
        nb = int((~np.isfinite(cl)).sum())
        print(f"WARNING: {nb} non-finite entries remain in cl after truncation; "
              f"replacing with zero.")
        cl = np.nan_to_num(cl, nan=0.0, posinf=0.0, neginf=0.0)
        cln = np.nan_to_num(cln, nan=0.0, posinf=0.0, neginf=0.0)

    r, rp, fac, vfac, kper, kpara = pe.build_essential(
        cfg.observation.nuc_mhz, cfg.observation.dnuc_mhz, NE, lval,
        model=ps.cosmology)

    w = pe.window(NE)
    dcln = np.std(cln, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        covi = 1.0 / dcln ** 2
    covi = np.where(np.isfinite(covi) & (dcln > 0), covi, 0.0)

    pk = pe.func_pk(cl, w, covi, vfac)
    pkn = pe.func_pk(cln, w, covi, vfac)
    dpkn = np.std(pkn, axis=0) * ps.noise_scale
    print(f"\nnoise scale : × {ps.noise_scale:.4f}  "
          f"(n_nights = {ps.n_nights})", flush=True)

    geom = dict(r=r, rprime=rp, fac=fac,
                nuc_mhz=float(cfg.observation.nuc_mhz),
                SM_mhz=float(cfg.scf.SM_mhz))
    fm = build_flag_mask(cfg, kper, kpara, fac, geom)

    X, mu, sigma = pe.X(pk, dpkn, fm)
    print(f"X statistic : {X.size} modes, mu = {mu:.4f}, sigma = {sigma:.4f}")
    if sigma > 2 or sigma < 0.5:
        print(f"              sigma is far from 1 — the simulated noise does not "
              f"match the observed scatter.")

    kk, ppk, dppk = pe.binned_pk(kper, kpara, pk, sigma * dpkn, ps.NBin, fm)
    dk2, dpk2, snr, ul = pe.func_dT(kk, ppk, dppk)

    print(f"\n{'k [1/Mpc]':>11}  {'Delta^2 [mK^2]':>16}  {'2 sigma':>14}  {'SNR':>8}  {'upper limit':>14}")
    for i in range(len(kk)):
        print(f"{kk[i]:11.4f}  {dk2[i]:16.4e}  {dpk2[i]:14.4e}  "
              f"{snr[i]:8.3f}  {ul[i]:14.4e}")

    out = cfg.product('ps')
    np.savez(out,
             kk=kk, dk2=dk2, dpk2=dpk2, snr=snr, ul=ul,
             ppk=ppk, dppk=dppk, mu=mu, sigma=sigma,
             kper=kper, kpara=kpara, pk=pk, dpkn=dpkn, flag_mask=fm,
             r=r, rp=rp, fac=fac, vfac=vfac, NE=NE,
             nuc_mhz=cfg.observation.nuc_mhz, dnuc_mhz=cfg.observation.dnuc_mhz,
             noise_scale=ps.noise_scale, n_nights=ps.n_nights)
    print(f"\nwrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
