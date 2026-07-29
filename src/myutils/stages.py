r"""
Shared pipeline stage helpers
=============================

Thin wrappers that read every parameter from the :class:`~myutils.config.Config`
object, so the data, UAPS and noise passes cannot drift apart. That drift is the
failure mode this module exists to prevent: nothing in the underlying estimator
checks that the three passes used the same gridding, and a mismatch produces a
plausible-looking but meaningless power spectrum.

If you find yourself passing a gridding parameter explicitly, something has gone
wrong — the exceptions are stage 2, which deliberately grids one polarisation with
flagging off, and the noise pass, which varies only its seed.
"""

import numpy as np

import myutils.tge.grid as gd
import myutils.scf.scf as sf
import myutils.clfuncs.correlate as corrf
import myutils.clfuncs.clfuncs as corr2d

__all__ = ['grid_pass', 'scf_pass', 'load_bininfo', 'correlate_pass',
           'one_pass', 'check_channels']


def grid_pass(cfg, infits, nrel, seed=None, nstokes=None, flag=None,
              n1=None, n2=None):
    """Stage 1 for one pass.

    Parameters
    ----------
    cfg : myutils.config.Config
        Supplies Umax, FWHM, f, nstokes, flag and the channel range.
    infits : str or pathlib.Path
        Input UVFITS.
    nrel : int
        ``-1`` data as-is, ``-2`` Gaussian noise, ``n >= 0`` copy channel *n*.
    seed : int, optional
        Only meaningful for ``nrel = -2``.
    nstokes, flag, n1, n2 : optional
        Override the config. Used by stage 2.

    Returns
    -------
    tuple
        ``(GV, info)`` exactly as :func:`myutils.tge.grid.grid` returns them.
    """
    g, obs = cfg.gridding, cfg.observation
    memmap = bool((cfg.get('io') or {}).get('memmap', False))
    engine = str((cfg.gridding.get('engine') or 'numpy')).lower()
    return gd.grid(
        str(infits),
        obs.n1 if n1 is None else n1,
        obs.n2 if n2 is None else n2,
        nrel,
        g.Umax, g.FWHM, g.f,
        g.flag if flag is None else flag,
        list(g.nstokes) if nstokes is None else list(nstokes),
        seed=seed,
        memmap=memmap,
        engine=engine,
    )


def scf_pass(cfg, GV, per_pol=True):
    """Stage 3. A no-op when ``scf.enabled`` is false, so shapes stay consistent.

    SCF acts along the frequency axis independently for every ``(pol, u, v)`` point,
    so filtering one polarisation at a time is mathematically identical to filtering
    both at once, and cuts the peak memory by about 38%. That matters here: at MWA
    sizes — (2, 457, 457, 768) complex128, 5.1 GB — ``doscf`` internally allocates the
    smooth model, two flag masks, two convolved masks and the filtered output, and the
    FFT pads the frequency axis to 875. Measured estimates: ~40 GB for both
    polarisations in one call, ~25 GB one at a time.

    Pass ``per_pol=False`` to filter the whole array in one call.
    """
    if not cfg.scf.enabled:
        return GV

    kw = dict(window=cfg.scf.window, method=cfg.scf.method)

    if not per_pol or GV.ndim < 4 or GV.shape[0] < 2:
        return sf.doscf(GV, cfg.scf.SM_mhz, **kw)

    out = None
    for ip in range(GV.shape[0]):
        # keep the leading axis so doscf sees the same dimensionality as before
        filtered = sf.doscf(GV[ip:ip + 1], cfg.scf.SM_mhz, **kw)
        if out is None:
            out = np.empty((GV.shape[0],) + filtered.shape[1:], dtype=filtered.dtype)
        out[ip] = filtered[0]
        del filtered
    return out


def load_bininfo(cfg):
    """Return ``(Nbin, ni, mask, lval, dU)`` from the stage-2 product."""
    cfg.require('bininfo')
    BIN = np.load(cfg.product('bininfo'))
    return (int(BIN['Nbin']),
            BIN['ni'],
            BIN['NI'] >= 0,
            BIN['lval'],
            float(BIN['dU']))


def correlate_pass(cfg, GV, ni, mask, Nbin, two_d=False):
    """Cross-TGE over the selected grid points.

    ``two_d=False`` (default) accumulates straight into frequency *separation*,
    giving ``(Nbin, NC)`` — all the power spectrum needs. ``two_d=True`` keeps the
    full ``(Nbin, NC, NC)`` MAPS in (nu_a, nu_b), which is what the
    C_l(nu_a, nu_b) figures in Documentation/RA_11.pdf are drawn from. Same compute,
    ~94 MB more memory at Nbin=20, NC=768.
    """
    sub = GV[:, mask]
    if two_d:
        return corr2d.correlate(sub, ni, Nbin)
    return corrf.correlate(sub, ni, Nbin)


def one_pass(cfg, infits, nrel, ni, mask, Nbin, seed=None):
    """Stages 1, 3 and 4 for a single pass.

    Frees the gridded array as soon as the correlation is in hand — it is ~5 GB at
    MWA sizes, and holding two at once is the usual cause of a MemoryError here.
    """
    GV, _ = grid_pass(cfg, infits, nrel, seed=seed)
    GV = scf_pass(cfg, GV)
    corr = correlate_pass(cfg, GV, ni, mask, Nbin)
    del GV
    return check_channels(cfg, corr)


def check_channels(cfg, corr):
    """Warn if a correlation came back with an unexpected channel count."""
    got = corr.shape[-1]
    if got != cfg.nchan_out:
        print(f"WARNING: correlation has {got} channels, config expects "
              f"{cfg.nchan_out}. Check observation.n1/n2 and scf.SM_mhz.", flush=True)
    return corr
