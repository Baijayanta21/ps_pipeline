r"""
Mode selection in the :math:`(k_\perp, k_\parallel)` plane
=========================================================

Which modes enter the spherical average. One mandatory constraint, two optional
ones on top of it.

**Wedge — always applied, not switchable.** Foreground emission from a source at
angle :math:`\theta` leaks up to a delay set by the geometry, so smooth-spectrum
foregrounds occupy a wedge below

.. math::

    [k_\parallel]_H = \frac{r}{r'\nu_c}\, k_\perp = \mathrm{fac}\cdot k_\perp

(the *horizon* line — ``fac`` is what :func:`myutils.psfuncs.psestimation.build_essential`
returns and what the dashed line in the cylindrical figures marks). Modes **below**
that line are foreground-dominated and are always excluded. There is deliberately no
switch for this: a spherical average that includes the wedge is not a 21-cm
measurement, it is a foreground measurement, and making that a toggle invites it
being left off by accident.

Only ``wedge_buffer`` is tunable — extra :math:`k_\parallel` above the horizon line.
The wedge edge is soft, because the instrument's frequency response smears it, so a
buffer is standard practice.

**Rectangular ranges — optional,** gated by ``use_limits``. Simple :math:`k_\perp`
and :math:`k_\parallel` limits: a :math:`k_\parallel` floor removes modes where
filtering leaves residuals, a :math:`k_\perp` ceiling drops the sparsely-sampled long
baselines.

**Tabulated windows — optional,** gated by ``use_tabulated``. A per-:math:`k_\perp`
list of :math:`k_\parallel` bands. These are not arbitrary: they are the published
mode selection of TTGE III (see :data:`PAPER`), carried over from the tutorials.

Every function returns a boolean array of shape ``(len(kper), len(kpara))`` that is
**True where the mode is used**.
"""

import numpy as np

__all__ = ['wedge_mask', 'range_mask', 'tabulated_mask', 'build_mask',
           'describe', 'recompute', 'exclude_boxes', 'for_run',
           'PAPER', 'PRESETS', 'resolve_spec']


# ------------------------------------------------------------------- presets

#: The mode selection published in **TTGE III** (Sarkar et al. 2026,
#: arXiv:2604.24144, in ``Documentation/``), Section 4 and its Figure 5.
#:
#: Three cuts, in the paper's own words:
#:
#: * :math:`k_\perp \le 0.045\ \mathrm{Mpc}^{-1}` — beyond that the wedge boundary
#:   rises above the SCF smoothing scale, so the long baselines cannot be cleaned
#:   by filtering alone ("at large baselines, the visibilities ... exceeds the
#:   smoothing scale of :math:`k_\parallel = 0.135`").
#: * :math:`k_\parallel \ge 0.135\ \mathrm{Mpc}^{-1}` — below this SCF itself
#:   removes power, so the paper avoids those modes rather than trusting them.
#:   This number is tied to the **2 MHz** smoothing scale (``N = 50`` channels),
#:   which is exactly what ``scf.SM_mhz: 2.0`` gives here, so it transfers
#:   unchanged to this pipeline.
#: * :math:`k_\parallel \le 1.399\ \mathrm{Mpc}^{-1}` — above that the estimate is
#:   noise-dominated.
#:
#: Inside that box the paper further masks contaminated modes — the periodic
#: horizontal streaks at :math:`\Delta k_\parallel = 0.290\ \mathrm{Mpc}^{-1}` —
#: which is what the tabulated bands encode. The six-entry ``kpara_start_index``
#: also enforces the :math:`k_\perp` ceiling on its own, because the tabulation
#: only covers the first six :math:`k_\perp` bins.
#:
#: The paper's :math:`k_\perp` grid (0.007-0.146, 20 bins) is the same as this
#: pipeline's default, so the bands land on the same bins.
PAPER = dict(
    wedge_buffer=0.0,        # the box below is what does the work; see the note
    use_limits=True,
    kperp_min=0.007, kperp_max=0.045,
    kpara_min=0.135, kpara_max=1.399,
    use_tabulated=True,
    kpara_ranges=[0.135, 0.228, 0.36, 0.5, 0.72, 0.8, 0.92, 1.09, 1.17, 1.39],
    kpara_start_index=[0, 0, 1, 2, 2, 2],
)

#: Named specs usable as ``mask: {preset: paper}`` in the config, or as the
#: ``spec`` argument anywhere a dict is accepted. Every one of them excludes the
#: wedge — that is not a property of the preset, it is unconditional.
PRESETS = {
    'paper': PAPER,
    #: The box without the streak tabulation — the paper's region of interest,
    #: every mode in it kept. More modes, less hand-tuning.
    'paper_box': dict(PAPER, use_tabulated=False),
    #: Nothing but the wedge. The widest selection available.
    'wedge_only': dict(use_limits=False, use_tabulated=False),
}

#: Old name for ``wedge_only``. It used to mean "no selection at all", which is no
#: longer possible now that the wedge is mandatory.
PRESETS['none'] = PRESETS['wedge_only']


def resolve_spec(spec):
    """Expand a preset name, or a dict carrying ``preset:``, into a plain spec.

    ``spec`` may be ``None``, a preset name, or a dict. Keys given alongside
    ``preset`` override the preset, so ``{'preset': 'paper', 'wedge_buffer': 0.05}``
    is the paper's selection with a buffer added.

    Raises
    ------
    ValueError
        If ``use_wedge: false`` is given. The wedge cut is no longer optional, and
        honouring the key would be impossible while quietly dropping it would run a
        different analysis than the config asks for.
    """
    if spec is None:
        return {}
    if isinstance(spec, str):
        try:
            return dict(PRESETS[spec])
        except KeyError:
            raise KeyError(f"unknown mask preset {spec!r}. "
                           f"Available: {', '.join(sorted(PRESETS))}") from None
    spec = dict(spec)

    # use_wedge was a switch before the wedge became mandatory.
    if 'use_wedge' in spec:
        if spec.pop('use_wedge') is False:
            raise ValueError(
                "mask.use_wedge: false is no longer supported — the foreground "
                "wedge is always excluded. A spherical average that includes the "
                "wedge measures foregrounds, not the 21-cm signal. Remove the key; "
                "to keep more modes near the boundary set wedge_buffer: 0.0, and "
                "to widen the selection use preset: wedge_only.")

    name = spec.pop('preset', None)
    if name is None:
        return spec
    base = resolve_spec(name)
    base.update({k: v for k, v in spec.items() if v is not None})
    return base


def wedge_mask(kper, kpara, fac, buffer=0.0):
    r"""True outside the foreground wedge, i.e. where
    :math:`k_\parallel > \mathrm{fac}\,k_\perp + \mathrm{buffer}`.

    Parameters
    ----------
    kper, kpara : array_like
        The two axes, 1-D.
    fac : float
        Wedge slope :math:`r/(r'\nu_c)`.
    buffer : float
        Extra :math:`k_\parallel` in Mpc\ :sup:`-1` above the horizon line. The wedge
        edge is not sharp, so a buffer of a few hundredths is common.
    """
    kper = np.asarray(kper)[:, None]
    kpara = np.asarray(kpara)[None, :]
    return kpara > (fac * kper + float(buffer))


def range_mask(kper, kpara, kperp_min=None, kperp_max=None,
               kpara_min=None, kpara_max=None):
    """True inside the given rectangular limits. ``None`` means unbounded."""
    kper = np.asarray(kper)[:, None]
    kpara = np.asarray(kpara)[None, :]
    m = np.ones((kper.size, kpara.size), dtype=bool)
    if kperp_min is not None:
        m &= kper >= float(kperp_min)
    if kperp_max is not None:
        m &= kper <= float(kperp_max)
    if kpara_min is not None:
        m &= kpara >= float(kpara_min)
    if kpara_max is not None:
        m &= kpara <= float(kpara_max)
    return m


def tabulated_mask(kper, kpara, ranges, start_index):
    r"""The tutorials' hand-picked :math:`k_\parallel` windows per :math:`k_\perp`.

    ``ranges`` is a flat list of band edges and ``start_index[i]`` is the first band
    used for the i-th :math:`k_\perp`. Reproduces the original ``flag_mask``
    construction exactly.
    """
    ks = np.asarray(ranges, dtype=float)
    ksv = np.asarray(start_index, dtype=int)
    kpara = np.asarray(kpara)
    nband = len(ks) // 2                 # from the edge list, not from start_index
    m = np.zeros((len(kper), kpara.size), dtype=bool)
    for i in range(len(ksv)):
        if i >= len(kper):
            break
        for j in range(ksv[i], nband):
            m[i] |= (kpara >= ks[2 * j]) & (kpara <= ks[2 * j + 1])
    return m


def build_mask(kper, kpara, fac, spec=None):
    r"""Combine the active constraints into one selection.

    ``spec`` is the ``power_spectrum.mask`` config section::

        preset: paper            start from a named selection (see PRESETS)
        wedge_buffer: 0.0        Mpc^-1 added above the horizon line
        use_limits: true         apply the rectangular k limits below
        kperp_min / kperp_max    null for unbounded
        kpara_min / kpara_max
        use_tabulated: false     intersect with the published k_para windows

    It may also be a bare preset name. Keys given next to ``preset`` override it.

    **The wedge is always excluded**; there is no switch for it, and an empty spec
    still gives a wedge cut. The other two constraints are opt-in and are ANDed on
    top: a mode survives only if every active constraint keeps it.

    Returns
    -------
    mask : ndarray of bool
        ``(len(kper), len(kpara))``, True where the mode is used.
    parts : dict
        Per-constraint survivor counts, for reporting.
    """
    spec = resolve_spec(spec)
    parts = {'total': len(kper) * len(kpara)}

    # The wedge is mandatory. Only the buffer is tunable.
    buf = spec.get('wedge_buffer') or 0.0
    mask = wedge_mask(kper, kpara, fac, buf)
    parts['wedge'] = int(mask.sum())
    parts['wedge_buffer'] = float(buf)

    lims = {k: spec.get(k) for k in
            ('kperp_min', 'kperp_max', 'kpara_min', 'kpara_max')}
    # Default on when limits are actually given, so a spec that predates the
    # use_limits switch behaves as it always did.
    if spec.get('use_limits', any(v is not None for v in lims.values())):
        r = range_mask(kper, kpara, **lims)
        parts['ranges'] = int(r.sum())
        mask = mask & r

    if spec.get('use_tabulated', False):
        t = tabulated_mask(kper, kpara, spec.get('kpara_ranges') or [],
                           spec.get('kpara_start_index') or [])
        parts['tabulated'] = int(t.sum())
        mask = mask & t

    parts['selected'] = int(mask.sum())
    return mask, parts


def describe(parts, kper=None, kpara=None, mask=None):
    """Human-readable summary of what each constraint kept."""
    buf = parts.get('wedge_buffer') or 0.0
    w = 28
    lines = [f"  {'modes available':<{w}s} {parts['total']:,}"]
    for key, label in (
            ('wedge', f"outside the wedge{f' (+{buf:g})' if buf else ''} [always]"),
            ('ranges', 'inside k limits'),
            ('tabulated', 'in tabulated windows')):
        if key in parts:
            lines.append(f"  {label:<{w}s} {parts[key]:,}")
    lines.append(f"  {'selected (all combined)':<{w}s} {parts['selected']:,} "
                 f"({100 * parts['selected'] / max(parts['total'], 1):.1f}%)")
    if mask is not None and kper is not None and kpara is not None and mask.any():
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        lines.append(f"  {'k_perp spanned':<{w}s} {kper[rows[0]]:.4f} - "
                     f"{kper[rows[-1]]:.4f} Mpc^-1")
        lines.append(f"  {'k_para spanned':<{w}s} {kpara[cols[0]]:.4f} - "
                     f"{kpara[cols[-1]]:.4f} Mpc^-1")
    return '\n'.join(lines)


def recompute(d, mask, nbin=8, noise_scale=None):
    r"""Re-derive the spherical power spectrum under a different mode mask.

    Everything downstream of the mask — the X statistic, ``sigma``, the binning and
    :math:`\Delta^2(k)` — depends on which modes are selected, so changing the mask
    means redoing all of it. None of it needs the gridding or correlation again, so
    this is instant and can be driven interactively.

    Parameters
    ----------
    d : mapping
        A loaded ``ps_*.npz``. Uses ``pk``, ``dpkn``, ``kper``, ``kpara``.
    mask : ndarray of bool or int
        ``(len(kper), len(kpara))``, True/1 where the mode is used.
    nbin : int
        Number of logarithmic k bins.
    noise_scale : float, optional
        Override the stored scaling. ``None`` keeps whatever produced ``dpkn``.

    Returns
    -------
    dict
        ``kk, dk2, dpk2, snr, ul, mu, sigma, nmodes`` — the same quantities stage 5
        writes, for this mask.
    """
    import myutils.psfuncs.psestimation as pe

    fm = np.asarray(mask).astype(int)
    if not fm.any():
        raise ValueError("the mask selects no modes")

    pk = d['pk']
    dpkn = d['dpkn']
    if noise_scale is not None:
        stored = float(d['noise_scale']) if 'noise_scale' in getattr(d, 'files', []) else 1.0
        dpkn = dpkn * (noise_scale / stored)

    X, mu, sigma = pe.X(pk, dpkn, fm)
    kk, ppk, dppk = pe.binned_pk(d['kper'], d['kpara'], pk, sigma * dpkn, nbin, fm)
    dk2, dpk2, snr, ul = pe.func_dT(kk, ppk, dppk)
    return {'kk': kk, 'dk2': dk2, 'dpk2': dpk2, 'snr': snr, 'ul': ul,
            'mu': float(mu), 'sigma': float(sigma), 'nmodes': int(fm.sum())}


def exclude_boxes(kper, kpara, boxes):
    """False inside each ``(kperp_lo, kperp_hi, kpara_lo, kpara_hi)`` box.

    For carving specific contaminated regions out of an existing mask::

        m &= exclude_boxes(kper, kpara, [(0.0, 0.02, 0.0, 0.3)])

    Use ``None`` for an open edge.
    """
    kp = np.asarray(kper)[:, None]
    ka = np.asarray(kpara)[None, :]
    keep = np.ones((kp.size, ka.size), dtype=bool)
    for lo_p, hi_p, lo_a, hi_a in boxes:
        inside = np.ones_like(keep)
        if lo_p is not None:
            inside &= kp >= lo_p
        if hi_p is not None:
            inside &= kp <= hi_p
        if lo_a is not None:
            inside &= ka >= lo_a
        if hi_a is not None:
            inside &= ka <= hi_a
        keep &= ~inside
    return keep


def for_run(d, spec=None, boxes=()):
    r"""Build a mask for one loaded run from a shared specification.

    Masks are **per grid**: two runs with different ``NE`` have different
    :math:`k_\parallel` axes, so a mask array built for one cannot be applied to the
    other. Share the *spec* between runs and build the array separately for each —
    that is what this does.

    Parameters
    ----------
    d : mapping
        A loaded ``ps_*.npz``; supplies ``kper``, ``kpara`` and ``fac``.
    spec : dict, optional
        As :func:`build_mask` — ``use_wedge``, ``wedge_buffer``, ``kpara_min`` etc.
    boxes : sequence, optional
        ``(kperp_lo, kperp_hi, kpara_lo, kpara_hi)`` regions to exclude.

    Returns
    -------
    mask, parts
    """
    kper, kpara, fac = d['kper'], d['kpara'], float(d['fac'])
    mask, parts = build_mask(kper, kpara, fac, spec)
    if boxes:
        before = int(mask.sum())
        mask = mask & exclude_boxes(kper, kpara, boxes)
        parts['boxes_removed'] = before - int(mask.sum())
        parts['selected'] = int(mask.sum())
    return mask, parts
