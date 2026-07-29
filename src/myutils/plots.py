r"""
Publication figures for the pipeline products
=============================================

Reads ``ps_{index}.npz`` (and ``bin_info_{index}.npz``) and writes the standard set
of figures. Everything is driven by the config, so nothing here hardcodes a path.

The tutorial notebooks contain equivalent plotting code, but it does not survive
being moved off its author's machine:

* ``plt.style.use('/home/cts23ph/SP_FINAL/mplstyle')`` — an absolute path to another
  user's file. :func:`style` sets the same kind of defaults inline instead.
* ``plt.show()`` after every figure — a no-op in a batch job, and it keeps figures
  alive in memory.
* ``r'\bf Cylindrical PS'`` — requires ``text.usetex`` and a full LaTeX install;
  raises otherwise. Matplotlib's own mathtext renders ``$...$`` with no LaTeX.
* filenames written to the current directory, so products land wherever you happened
  to launch from.

Colour choices
--------------
**Sequential (magnitude):** ``viridis``. The tutorials use ``rainbow``, which is
perceptually non-uniform — it manufactures banding where the data is smooth and
hides real structure where it is not — and is unreadable with colour vision
deficiency. ``viridis`` is monotonic in lightness, so it also survives greyscale
printing.

**Categorical (window functions):** ``#0072B2, #009E73, #D55E00, #8B5FBF``, assigned
in fixed order. Validated for colour-vision-deficiency separation (worst adjacent
pair ΔE 11.0 deutan), lightness band, chroma floor and ≥3:1 contrast against the
figure surface. Line style varies with colour as a secondary encoding.
"""

import numpy as np

__all__ = ['style', 'PALETTE', 'CYLINDRICAL_CMAP', 'cylindrical_ps', 'compare_cylindrical',
           'cl_dnu_panels', 'cl_nunu', 'pk_cross_sections', 'load_cl', 'x_statistic', 'spherical_ps',
           'uv_coverage', 'scf_windows', 'compare_spherical', 'compare_cl_nunu',
           'unflagged_channels', 'select_bins', 'mask_overlay', 'save', 'load_run',
           'draw_reference', 'REFERENCE_STYLES']

#: fixed categorical order — never cycled, never reordered per figure
PALETTE = ['#0072B2', '#009E73', '#D55E00', '#8B5FBF']

#: sequential ramp for magnitude (uv coverage, C_l maps)
SEQUENTIAL = 'viridis'

#: Colour map for the cylindrical power spectrum.
#:
#: ``rainbow`` matches Documentation/RA_11.pdf, where |P(k_per,k_par)| runs purple at
#: 1e7 through cyan/green/yellow to red at 1e15. Keeping the same ramp means figures
#: from this pipeline can be dropped straight into that deck.
#:
#: Rainbow is not perceptually uniform — equal steps in value are not equal steps in
#: apparent colour, so it can suggest edges where the data is smooth — and it does not
#: survive greyscale printing. Set ``CYLINDRICAL_CMAP = 'viridis'`` or pass
#: ``cmap='viridis'`` if you ever want the perceptually uniform version.
CYLINDRICAL_CMAP = 'rainbow'

_INK = '#1a1a1a'
_MUTED = '#6b6b6b'
_GRID = '#d9d9d9'

#: how far an upper-limit arrow drops, as a factor on the log axis
ARROW_DROP = 2.2


def style():
    """Set self-contained rcParams. No external style file, no LaTeX required."""
    import matplotlib as mpl
    mpl.rcParams.update({
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'axes.edgecolor': _MUTED,          # recessive axes
        'axes.linewidth': 0.8,
        'axes.labelcolor': _INK,
        'axes.grid': True,
        'grid.color': _GRID,               # recessive grid
        'grid.linewidth': 0.6,
        'grid.alpha': 0.7,
        'axes.axisbelow': True,            # data over grid, never under
        'xtick.color': _MUTED,
        'ytick.color': _MUTED,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'text.color': _INK,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': _GRID,
        'legend.fontsize': 9,
        'axes.prop_cycle': mpl.cycler(color=PALETTE),
        'text.usetex': False,              # mathtext only — no LaTeX dependency
    })


def _finish(fig, out):
    """Return the figure when *out* is None (notebooks), else save and close it."""
    if out is None:
        return fig
    return save(fig, out)


def save(fig, out, formats=('pdf', 'png')):
    """Write *fig* to *out* (a path without suffix) in each format, then close it."""
    import matplotlib.pyplot as plt
    written = []
    for ext in formats:
        path = out.with_suffix(f'.{ext}')
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return written



def _log_k_axis(ax, values):
    """Label a log x-axis that spans less than a couple of decades.

    Matplotlib's default log locator puts a label only at exact powers of ten, so a
    range like k = 0.16 to 1.28 renders with a single tick. Force minor ticks with
    plain decimal labels instead.
    """
    from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
    lo, hi = float(np.min(values)), float(np.max(values))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=(0.2, 0.3, 0.5, 0.7), numticks=12))
    fmt = FuncFormatter(lambda v, _: f'{v:g}' if lo / 3 <= v <= hi * 3 else '')
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(fmt)
    ax.tick_params(axis='x', which='minor', labelsize=8)

# ---------------------------------------------------------------- cylindrical

def cylindrical_ps(d, out=None, masked=False, title=None, cmap=None,
                   vmin=None, vmax=None, ax=None):
    r"""Cylindrical power spectrum :math:`|P(k_\perp, k_\parallel)|`.

    Parameters
    ----------
    d : mapping
        Loaded ``ps_{index}.npz``.
    out : pathlib.Path, optional
        Output path without suffix. ``None`` returns the figure for inline display.
    masked : bool
        If True, show only the modes selected by ``flag_mask`` and zoom to them.
    cmap : str, optional
        Defaults to :data:`CYLINDRICAL_CMAP` (``rainbow``, matching RA_11.pdf).
    vmin, vmax : float, optional
        Fix the colour limits. **Set these when comparing panels** — RA_11 uses one
        scale across before/after SCF so the panels can be read against each other.
        Autoscaling each panel independently would hide exactly the change you are
        looking for. Default: autoscale to the finite positive values.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axis, for building side-by-side comparison figures.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    kper, kpara, pk, fac = d['kper'], d['kpara'], d['pk'], float(d['fac'])
    fm = d['flag_mask']

    values = np.abs(pk)
    if masked:
        values = np.ma.masked_array(values, mask=~fm.astype(bool))

    if vmin is None or vmax is None:
        finite = values.compressed() if masked else values[np.isfinite(values)]
        finite = finite[finite > 0]
        if finite.size == 0:
            raise ValueError("no positive finite values to plot")
        vmin = finite.min() if vmin is None else vmin
        vmax = finite.max() if vmax is None else vmax

    # pcolormesh wants cell edges; kpara starts at 0 which log axes reject
    y, x = np.meshgrid(kpara + (kpara[1] - kpara[0]), kper)

    cm = plt.get_cmap(cmap or CYLINDRICAL_CMAP).copy()
    cm.set_bad('#e8e8e8')                  # excluded modes read as absent, not as data

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        standalone = True
    else:
        fig, standalone = ax.figure, False

    mesh = ax.pcolormesh(x, y, values, norm=LogNorm(vmin, vmax),
                         shading='auto', cmap=cm, rasterized=True)

    # the foreground wedge: k_par = (r / r' nu_c) k_perp
    ax.plot(kper, fac * kper, color=_INK, ls='--', lw=1.2,
            label='foreground wedge')
    ax.legend(loc='lower right')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$k_{\perp}\ \ [\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$k_{\parallel}\ \ [\mathrm{Mpc}^{-1}]$')
    ax.set_title(title or ('Cylindrical power spectrum — modes used' if masked
                           else 'Cylindrical power spectrum'))
    ax.grid(False)                         # a grid over a heatmap is noise

    if masked:
        cols = np.where(fm.any(axis=1))[0]
        rows = np.where(fm.any(axis=0))[0]
        if cols.size and rows.size:
            ax.set_xlim(kper[cols[0]], kper[cols[-1]])
            ax.set_ylim(max(kpara[rows[0]], kpara[1]), kpara[rows[-1]])

    if not standalone:
        return mesh                        # caller owns the colorbar and layout

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, aspect=40)
    cbar.set_label(r'$|P(k_{\perp},k_{\parallel})|\ \ [\mathrm{mK}^2\,\mathrm{Mpc}^3]$')
    cbar.outline.set_edgecolor(_MUTED)

    return _finish(fig, out)


# ---------------------------------------------------------------- X statistic

def x_statistic(d, out=None, nbins=25):
    r"""Histogram of :math:`X = P/\delta P_N` with a Student-t fit."""
    import matplotlib.pyplot as plt
    import scipy.stats

    pk, dpkn, fm = d['pk'], d['dpkn'], d['flag_mask']
    X = (pk / dpkn)[fm == 1]
    mu, sigma = float(d['mu']), float(d['sigma'])

    counts, edges = np.histogram(X, bins=nbins, range=(X.min(), X.max()))
    width = edges[1] - edges[0]
    N = X.size

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.stairs(counts / (N * width), edges, color=PALETTE[0], lw=1.6,
              label='measured')

    grid = np.linspace(X.min(), X.max(), 500)
    df, loc, scale = scipy.stats.t.fit(X)
    ax.plot(grid, scipy.stats.t.pdf(grid, df, loc, scale), ls='--', lw=1.6,
            color=PALETTE[2], label=f'Student $t$ fit ($\\nu={df:.1f}$)')

    ax.set_yscale('log')
    ax.set_xlabel('$X$')
    ax.set_ylabel('PDF')
    ax.set_title('$X$ statistic')
    ax.legend(loc='upper right')
    ax.annotate(f'$\\mu = {mu:.3f}$\n$\\sigma = {sigma:.3f}$\n$N = {N}$',
                xy=(0.03, 0.95), xycoords='axes fraction',
                va='top', ha='left', fontsize=9, color=_MUTED)

    return _finish(fig, out)


# ---------------------------------------------------------------- spherical

#: Marker and dash pattern per published reference. Reference curves deliberately
#: do **not** take categorical hues: those identify *our* runs, and a run must never
#: be repainted because a comparison was added. References are neutral ink and are
#: told apart by marker and dash, so identity survives greyscale and CVD alike.
#:
#: These markers also avoid ``D`` and ``v``, which this module already spends on
#: "detection" and "noise dominated". A hollow ``v`` for a published limit next to a
#: hollow ``v`` for one of our own noise-dominated points is indistinguishable in the
#: legend key, which is exactly the confusion the overlay must not create.
REFERENCE_STYLES = [
    ('s', (0, (5, 2))),
    ('o', (0, (1, 1.6))),
    ('X', (0, (6, 2, 1, 2))),
    ('P', (0, (3, 1, 1, 1, 1, 1))),
    ('*', (0, (4, 1, 1, 1))),
]

#: Below this many points a connecting line implies a measured curve where there is
#: only a quoted number or two, so markers are drawn bare.
_REF_LINE_MIN = 3


def _legend_headroom(ax, n_entries=None):
    """Extend a log y-axis upward so an inset legend does not sit on top of data.

    Matplotlib places the legend inside the axes and will draw it over the top-left
    points without complaint. Adding room is more robust than hunting for a free
    corner, because which corner is free depends on the data.
    """
    lo, hi = ax.get_ylim()
    if not (lo > 0 and hi > lo):
        return
    if n_entries is None:
        leg = ax.get_legend()
        n_entries = len(leg.get_texts()) if leg else 0
    ax.set_ylim(lo, hi * 10 ** (0.15 * n_entries + 0.15))


def _as_reference_list(reference):
    """Normalise the ``reference`` argument to a list of dataset dicts.

    Accepts a name, a dataset dict, or a sequence of either — whichever is least
    effort at the call site.
    """
    import myutils.reference as refdata

    if reference is None:
        return []
    if isinstance(reference, str) or isinstance(reference, dict):
        reference = [reference]
    return [refdata.get(r) if isinstance(r, str) else r for r in reference]


def draw_reference(ax, reference, annotate=False, short=True):
    r"""Overlay published :math:`\Delta^2_{UL}(k)` on an existing axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A log-log axis already carrying :math:`|\Delta^2(k)|` in mK\ :sup:`2`.
    reference : str, dict or sequence
        Names from :data:`myutils.reference.DATASETS`, or dataset dicts.
    annotate : bool
        Label individual points that carry a ``note`` — useful for the two headline
        numbers, noise for a full table.
    short : bool
        Use each dataset's short legend name rather than its full label.

    Notes
    -----
    Published limits are :math:`2\sigma` **upper limits**, not measurements, so they
    are drawn with downward carets and a dashed line. Reading one as a data point
    that our curve should match is the mistake this styling exists to prevent.
    """
    import myutils.reference as refdata

    drawn = []
    for ds, (marker, dash) in zip(_as_reference_list(reference), REFERENCE_STYLES):
        k, ul = refdata.as_arrays(ds)
        label = ds.get('short' if short else 'label') or ds['label']
        if k.size >= _REF_LINE_MIN:
            ax.plot(k, ul, ls=dash, lw=1.2, color=_MUTED, zorder=2, alpha=0.9)
        ax.plot(k, ul, ls='none', marker=marker, ms=6, mfc='white',
                mec=_INK, mew=1.1, zorder=4, label=label)
        if annotate:
            for p in ds['points']:
                if p.get('note'):
                    # below-right: these points sit at the low-k, low-power corner,
                    # so the space under them is free while the space above carries
                    # our own error bars
                    ax.annotate(p['note'], xy=(p['k'], p['ul']),
                                xytext=(6, -4), textcoords='offset points',
                                va='top', ha='left',
                                fontsize=7.5, color=_MUTED)
        drawn.append(ds)
    return drawn


def spherical_ps(d, out=None, reference=None, annotate_reference=False):
    r"""Dimensionless spherical power spectrum :math:`\Delta^2(k)`.

    Positive and negative :math:`\Delta^2` are drawn with different markers as well
    as different colours — a negative value is noise-dominated and its point is an
    upper limit, which is a difference in meaning, not just in sign.

    Parameters
    ----------
    reference : str, dict or sequence, optional
        Published limits to overlay — see :func:`draw_reference` and
        :mod:`myutils.reference`. ``'ttge3_alpha11'`` is the like-for-like
        comparison for a single-pointing run.
    annotate_reference : bool
        Label reference points that carry a note.
    """
    import matplotlib.pyplot as plt

    kk, dk2, dpk2, ul = d['kk'], d['dk2'], d['dpk2'], d['ul']
    mag = np.abs(dk2)
    pos = dk2 > 0

    # On a log axis an error bar whose lower end reaches zero is drawn all the way to
    # the axis floor, which looks like a measurement rather than an artefact. Clip the
    # lower arm, and flag the points it had to be clipped for — those are consistent
    # with zero and are upper limits, not detections.
    at_zero = dpk2 >= mag
    # For an upper limit the lower arm is meaningless — the point is "no more than
    # this". Draw a fixed visual drop (a factor of ARROW_DROP on the log axis) rather
    # than 0.92 of the value, which rendered as a two-decade bar and read as a
    # colossal uncertainty.
    lower = np.where(at_zero, mag * (1.0 - 1.0 / ARROW_DROP), dpk2)
    yerr = np.vstack([lower, dpk2])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(kk, mag, color=_MUTED, lw=1.0, zorder=1)

    def _draw(sel, marker, colour, label, size=6, hollow=False):
        if not sel.any():
            return
        # a hollow marker distinguishes "consistent with zero" from a detection at a
        # glance — the arrowhead alone does not survive into the legend key
        ax.errorbar(kk[sel], mag[sel], yerr=yerr[:, sel], fmt=marker, ms=size,
                    color=colour, mec=_INK, mew=0.8, elinewidth=1.6,
                    mfc='white' if hollow else colour,
                    capsize=3, zorder=3, label=label,
                    uplims=at_zero[sel])          # arrow head when clipped

    _draw(pos & ~at_zero, 'D', PALETTE[0], r'detection, $\Delta^2 > 0$')
    _draw(pos & at_zero, 'D', PALETTE[0], r'$\Delta^2 > 0$, consistent with zero',
          hollow=True)
    _draw(~pos, 'v', PALETTE[2], r'$\Delta^2 < 0$  (noise dominated)', size=7,
          hollow=True)

    ax.plot(kk, ul, ls=':', lw=1.4, color=PALETTE[3], zorder=2,
            label=r'$2\sigma$ upper limit')

    refs = draw_reference(ax, reference, annotate=annotate_reference)
    span = [kk] + [np.asarray([p['k'] for p in ds['points']]) for ds in refs]

    ax.set_yscale('log')
    _log_k_axis(ax, np.concatenate(span))
    ax.set_xlabel(r'$k\ \ [\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$|\Delta^2(k)|\ \ [\mathrm{mK}^2]$')
    ax.set_title('Spherical power spectrum')
    ax.legend(loc='upper left')
    ax.margins(x=0.08, y=0.12)
    _legend_headroom(ax)

    return _finish(fig, out)


# ---------------------------------------------------------------- diagnostics

def uv_coverage(bin_npz, out=None):
    """Which grid points survived the binning cuts, and which annulus each is in.

    A quick sanity check on stage 2: a healthy map shows a filled annulus between
    ``binUmin`` and ``binUmax``. Large empty regions mean ``Mg_min`` or the ``|U|``
    range is wrong for this array.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    NI = bin_npz['NI']
    Nbin = int(bin_npz['Nbin'])
    dU = float(bin_npz['dU'])

    half = NI.shape[0] // 2
    extent = [-half * dU, half * dU, -half * dU, half * dU]

    shown = np.ma.masked_where(NI < 0, NI)
    cmap = plt.get_cmap(SEQUENTIAL, Nbin).copy()
    cmap.set_bad('#e8e8e8')

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(shown.T, origin='lower', extent=extent, cmap=cmap,
                   norm=BoundaryNorm(np.arange(-0.5, Nbin), Nbin),
                   interpolation='nearest')
    ax.set_xlabel(r'$u\ \ [\lambda]$')
    ax.set_ylabel(r'$v\ \ [\lambda]$')
    used = int((NI >= 0).sum())
    ax.set_title(f'Grid points used — {used:,} of {NI.size:,} '
                 f'({100 * used / NI.size:.1f}%)')
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, aspect=40,
                        ticks=np.arange(0, Nbin, max(1, Nbin // 10)))
    cbar.set_label('annular bin')
    cbar.outline.set_edgecolor(_MUTED)

    return _finish(fig, out)


def load_run(path):
    """Load a ``ps_*.npz`` from a path, a config, or an output directory.

    Accepts whatever is convenient in a notebook::

        load_run('/path/to/ps_RA39p7_R5.npz')
        load_run('/path/to/ps_run_RA39p7_R5')       # directory — finds ps_*.npz
        load_run(cfg)                                # a Config object
    """
    from pathlib import Path
    if hasattr(path, 'product'):                     # a Config
        direct = path.product('ps')
        if direct.exists():
            return np.load(direct)
        # Since run_variants.py, the products live in with_SCF/ and without_SCF/
        # rather than at the top of output_dir. Point at one explicitly.
        subs = sorted(d for d in path.paths.output_dir.iterdir()
                      if d.is_dir() and sorted(d.glob('ps_*.npz')))
        raise FileNotFoundError(
            f"no {direct.name} in {path.paths.output_dir}.\n"
            f"Variant products found in: {[d.name for d in subs] or 'none'}.\n"
            f"Load one directly, e.g. mp.load_run(cfg.paths.output_dir / "
            f"'{subs[0].name if subs else 'with_SCF'}')")
    p = Path(path)
    if p.is_dir():
        hits = sorted(p.glob('ps_*.npz'))
        if not hits:
            raise FileNotFoundError(f"no ps_*.npz in {p}")
        p = hits[0]
    return np.load(p)


def compare_spherical(runs, out=None, title='Spherical power spectrum',
                      reference=None, annotate_reference=False):
    r"""Overlay :math:`\Delta^2(k)` from several runs.

    This is the figure ``Documentation/RA_11.pdf`` is built from — the same field
    processed with and without SCF, subtracted and not, read side by side.

    Parameters
    ----------
    runs : sequence of (label, data)
        ``label`` is the legend text; ``data`` is a loaded ``ps_*.npz`` or anything
        :func:`load_run` accepts.
    out : pathlib.Path, optional
        Save target without suffix. ``None`` returns the figure for inline display.
    reference : str, dict or sequence, optional
        Published limits to overlay — see :func:`draw_reference`.
    annotate_reference : bool
        Label reference points that carry a note.

    Notes
    -----
    Colour identifies the run and is assigned in fixed order, so adding a run never
    repaints the others. Marker fill still distinguishes detections from points
    consistent with zero, exactly as in :func:`spherical_ps`. Reference curves stay
    in neutral ink for the same reason — so that overlaying one cannot recolour a run.
    """
    import matplotlib.pyplot as plt

    if len(runs) > len(PALETTE):
        raise ValueError(
            f"{len(runs)} runs but only {len(PALETTE)} categorical colours. "
            f"Split into two figures rather than inventing hues.")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    span = []

    for (label, src), colour in zip(runs, PALETTE):
        d = src if hasattr(src, 'files') else load_run(src)
        kk, dk2, dpk2 = d['kk'], d['dk2'], d['dpk2']
        span.append(np.asarray(kk))
        mag = np.abs(dk2)
        at_zero = dpk2 >= mag
        lower = np.where(at_zero, mag * (1.0 - 1.0 / ARROW_DROP), dpk2)

        ax.plot(kk, mag, color=colour, lw=1.0, alpha=0.55, zorder=1)
        for sel, hollow in ((~at_zero, False), (at_zero, True)):
            if not sel.any():
                continue
            ax.errorbar(kk[sel], mag[sel],
                        yerr=np.vstack([lower, dpk2])[:, sel],
                        fmt='D', ms=6, color=colour, mec=_INK, mew=0.8,
                        mfc='white' if hollow else colour,
                        elinewidth=1.5, capsize=3, zorder=3,
                        uplims=at_zero[sel],
                        label=label if not hollow else None)

    refs = draw_reference(ax, reference, annotate=annotate_reference)
    span += [np.asarray([p['k'] for p in ds['points']]) for ds in refs]

    ax.set_yscale('log')
    _log_k_axis(ax, np.concatenate(span))
    ax.set_xlabel(r'$k\ \ [\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$|\Delta^2(k)|\ \ [\mathrm{mK}^2]$')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.margins(x=0.08, y=0.12)
    _legend_headroom(ax)
    ax.annotate('hollow = consistent with zero', xy=(0.98, 0.03),
                xycoords='axes fraction', ha='right', va='bottom',
                fontsize=8, color=_MUTED)

    return _finish(fig, out)


def load_cl(src, kind='cl'):
    """Load a ``cl_*.npy`` / ``cl2d_*.npy`` from a path, directory or Config."""
    from pathlib import Path
    if hasattr(src, 'product'):
        return np.load(src.product(kind))
    p = Path(src)
    if p.is_dir():
        hits = sorted(p.glob(f'{kind}_*.npy'))
        if not hits:
            raise FileNotFoundError(f"no {kind}_*.npy in {p}")
        p = hits[0]
    return np.load(p)


def cl_dnu_panels(runs, lval, dnuc, ell_indices=None, n_bins=None, max_dnu=None,
                  ncols=3, out=None, title=None):
    r"""Small multiples of :math:`C_{\ell}(\Delta\nu)`, one panel per :math:`\ell`.

    This is the "All 3 cases" figure in ``Documentation/RA_11.pdf`` — the same field
    processed several ways, compared bin by bin.

    Parameters
    ----------
    runs : sequence of (label, array or path)
        Each ``array`` is ``cl`` with shape ``(Nbin, NE)``; a path or directory is
        loaded via :func:`load_cl`.
    lval : array_like
        Effective multipole per bin, from ``bin_info_{index}.npz``.
    dnuc : float
        Channel width in MHz — sets the :math:`\Delta\nu` axis.
    ell_indices : sequence of int, optional
        Explicit bin indices. Default: six spread across the available bins.
    n_bins : int, optional
        Show the first N bins instead. Ignored if ``ell_indices`` is given.
    max_dnu : float, optional
        Truncate the :math:`\Delta\nu` axis, e.g. ``10.2`` as in RA_11.

    Notes
    -----
    Every panel shares one y-scale by default only if the runs are comparable; here
    each panel autoscales, matching RA_11, because the amplitude falls steeply with
    :math:`\ell` and a shared scale would flatten the high-\ :math:`\ell` panels.
    """
    import matplotlib.pyplot as plt

    loaded = [(label, src if isinstance(src, np.ndarray) else load_cl(src))
              for label, src in runs]
    if len(loaded) > len(PALETTE):
        raise ValueError(f"{len(loaded)} runs but only {len(PALETTE)} colours")

    lval = np.asarray(lval)
    nbin, ne = loaded[0][1].shape

    if ell_indices is None and n_bins is None:
        ell_indices = np.linspace(0, nbin - 1, min(6, nbin)).astype(int)
    ell_indices = select_bins(nbin, ell_indices, n_bins)
    ell_indices = list(dict.fromkeys(ell_indices))

    # Runs may have different numbers of separations — SCF trims NW channels from each
    # end — so the Delta_nu axis is derived per run, not from the first one.
    def axis_for(cl):
        d = np.arange(cl.shape[-1]) * dnuc
        k = (slice(None) if max_dnu is None
             else slice(0, min(cl.shape[-1], int(max_dnu / dnuc) + 1)))
        return d, k

    nrows = int(np.ceil(len(ell_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows),
                             squeeze=False)
    flat = axes.ravel()

    for ax, ib in zip(flat, ell_indices):
        for (label, cl), colour in zip(loaded, PALETTE):
            dnu, keep = axis_for(cl)
            n = f" (NE={cl.shape[-1]})" if len(loaded) > 1 else ''
            ax.plot(dnu[keep], cl[ib][keep], color=colour, lw=1.5,
                    label=f'{label}{n}')
        ax.axhline(0.0, color=_MUTED, lw=0.8, ls=':')
        ax.set_title(rf'$\ell = {int(lval[ib])}$')
        ax.margins(x=0.02)

    for ax in flat[len(ell_indices):]:
        ax.set_visible(False)

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel(r'$\Delta\nu$  [MHz]')
    for row in axes:
        row[0].set_ylabel(r'$C_{\ell}(\Delta\nu)$  [mK$^2$]')

    flat[0].legend(loc='best')
    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()

    return _finish(fig, out)


def mask_overlay(d, mask, out=None, title=None, cmap=None, vmin=None, vmax=None,
                 fac=None):
    r"""The cylindrical PS with the excluded region greyed out.

    Shows exactly which part of the :math:`(k_\perp,k_\parallel)` plane a mask keeps,
    over the power it is selecting from — the picture to look at before trusting a
    mask, and the one that makes a stray exclusion box obvious.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    kper, kpara, pk = d['kper'], d['kpara'], d['pk']
    fac = float(d['fac']) if fac is None else fac
    m = np.asarray(mask).astype(bool)

    values = np.abs(pk)
    if vmin is None or vmax is None:
        f = values[np.isfinite(values) & (values > 0)]
        vmin = f.min() if vmin is None else vmin
        vmax = f.max() if vmax is None else vmax

    y, x = np.meshgrid(kpara + (kpara[1] - kpara[0]), kper)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    cm = plt.get_cmap(cmap or CYLINDRICAL_CMAP).copy()
    mesh = ax.pcolormesh(x, y, values, norm=LogNorm(vmin, vmax), shading='auto',
                         cmap=cm, rasterized=True)
    # grey veil over everything the mask throws away
    ax.pcolormesh(x, y, np.where(m, np.nan, 1.0), shading='auto',
                  cmap=plt.matplotlib.colors.ListedColormap(['#ffffff']),
                  alpha=0.72, rasterized=True)

    ax.plot(kper, fac * kper, color=_INK, ls='--', lw=1.2, label='wedge')
    ax.legend(loc='lower right')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$k_{\perp}\ \ [\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$k_{\parallel}\ \ [\mathrm{Mpc}^{-1}]$')
    ax.set_title(title or f'modes used: {int(m.sum()):,} of {m.size:,} '
                          f'({100 * m.mean():.1f}%)')
    ax.grid(False)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, aspect=40)
    cbar.set_label(r'$|P(k_{\perp},k_{\parallel})|$  [mK$^2$ Mpc$^3$]')
    cbar.outline.set_edgecolor(_MUTED)
    return _finish(fig, out)


def compare_cylindrical(runs, out=None, masked=False, cmap=None, title=None,
                        vmin=None, vmax=None):
    r"""Cylindrical power spectra side by side on **one shared colour scale**.

    The before/after-SCF layout of ``Documentation/RA_11.pdf`` page 45. A single
    colour scale across all panels is the whole point — per-panel autoscaling would
    renormalise away the suppression you are trying to show.

    Parameters
    ----------
    runs : sequence of (label, data)
        Loaded ``ps_*.npz`` or anything :func:`load_run` accepts.
    vmin, vmax : float, optional
        Shared limits. Default: the combined range over every run, so no panel is
        clipped and all are directly comparable.
    """
    import matplotlib.pyplot as plt

    loaded = [(label, src if hasattr(src, 'files') else load_run(src))
              for label, src in runs]

    if vmin is None or vmax is None:
        pool = []
        for _, d in loaded:
            v = np.abs(d['pk'])
            if masked:
                v = v[d['flag_mask'].astype(bool)]
            v = v[np.isfinite(v) & (v > 0)]
            if v.size:
                pool.append((v.min(), v.max()))
        if not pool:
            raise ValueError("no positive finite values in any run")
        vmin = min(p[0] for p in pool) if vmin is None else vmin
        vmax = max(p[1] for p in pool) if vmax is None else vmax

    n = len(loaded)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n + 1.2, 5.0), squeeze=False)
    mesh = None

    for ax, (label, d) in zip(axes[0], loaded):
        mesh = cylindrical_ps(d, masked=masked, cmap=cmap,
                              vmin=vmin, vmax=vmax, ax=ax)
        ax.set_title(label)

    for ax in axes[0][1:]:
        ax.set_ylabel('')

    cbar = fig.colorbar(mesh, ax=axes[0].tolist(), pad=0.02, aspect=40)
    cbar.set_label(r'$|P(k_{\perp},k_{\parallel})|\ \ [\mathrm{mK}^2\,\mathrm{Mpc}^3]$')
    cbar.outline.set_edgecolor(_MUTED)
    if title:
        fig.suptitle(title, y=1.00)

    return _finish(fig, out)


def pk_cross_sections(runs, kper_indices=None, max_kpara=1.5, ncols=3,
                      reference=None, out=None, title=None, shade=True):
    r"""Cuts through the cylindrical PS: :math:`P(k_\perp,k_\parallel)` vs
    :math:`k_\parallel` at fixed :math:`k_\perp`.

    These are the cross-section pages of ``Documentation/RA_11.pdf`` — one panel per
    :math:`k_\perp`, several runs overlaid, with the :math:`k_\parallel` ranges that
    enter the spherical average shaded.

    Parameters
    ----------
    runs : sequence of (label, data)
        ``data`` is a loaded ``ps_*.npz`` or anything :func:`load_run` accepts. The
        first run supplies ``kper``/``kpara``/``flag_mask``.
    kper_indices : sequence of int, optional
        Which :math:`k_\perp` bins to show. Default: the first six that carry any
        selected mode, matching RA_11.
    max_kpara : float
        Right-hand limit of the :math:`k_\parallel` axis.
    reference : float, optional
        Draw a vertical reference line, e.g. ``0.135`` — the lower edge of the first
        :math:`k_\parallel` window in RA_11.
    shade : bool
        Shade the :math:`k_\parallel` bands selected by ``flag_mask``. This is what
        makes the figure readable: it shows which features are inside the analysis
        and which are being discarded.

    Notes
    -----
    A **linear** y-scale, as in RA_11 — these cuts are signed and cross zero, so a
    log scale would drop half the information. Each panel autoscales because the
    amplitude falls by orders of magnitude with :math:`k_\perp`.
    """
    import matplotlib.pyplot as plt

    loaded = [(label, src if hasattr(src, 'files') else load_run(src))
              for label, src in runs]
    if len(loaded) > len(PALETTE):
        raise ValueError(f"{len(loaded)} runs but only {len(PALETTE)} colours")

    d0 = loaded[0][1]
    kper, kpara, fm = d0['kper'], d0['kpara'], d0['flag_mask']

    if kper_indices is None:
        have = np.where(fm.any(axis=1))[0]
        kper_indices = (have[:6] if have.size else np.arange(min(6, kper.size)))
    kper_indices = [int(i) for i in kper_indices]

    # Each run has its own k_par grid: SCF trims NW channels from each end, so a
    # filtered run has fewer frequency separations and therefore a different
    # k_par = pi m / (r' dnu (NE-1)) spacing AND a different maximum. Masking every
    # run with the first run's grid is wrong — and raises, which is how this was found.
    nrows = int(np.ceil(len(kper_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.1 * nrows),
                             squeeze=False)
    flat = axes.ravel()

    for ax, ip in zip(flat, kper_indices):
        if shade:
            # contiguous runs of selected k_par become shaded bands
            sel = fm[ip].astype(bool)
            edges = np.diff(np.concatenate([[0], sel.view(np.int8), [0]]))
            for lo, hi in zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1):
                ax.axvspan(kpara[lo], kpara[min(hi, kpara.size - 1)],
                           color='#000000', alpha=0.07, lw=0, zorder=0)

        for (label, d), colour in zip(loaded, PALETTE):
            kp = d['kpara']
            sel_k = kp <= max_kpara                 # this run's own grid
            n = f" (NE={kp.size})" if len(loaded) > 1 else ''
            ax.plot(kp[sel_k], d['pk'][ip][sel_k], color=colour, lw=1.2,
                    label=f'{label}{n}', zorder=3)

        if reference is not None:
            ax.axvline(reference, color=PALETTE[2], ls='--', lw=1.1, zorder=2)

        ax.axhline(0.0, color=_MUTED, lw=0.8, ls=':', zorder=1)
        ax.set_title(rf'$k_\perp = {kper[ip]:.3f}\ \mathrm{{Mpc}}^{{-1}}$')
        ax.set_xlim(0, max_kpara)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        # Scale to the structure ABOVE the reference, not to the foreground spike.
        # At low k_par the unfiltered power reaches ~1e15 while the filtered run and
        # the whole EoR window sit near 1e11 — autoscaling to the peak flattens
        # everything of interest into the zero line.
        lo = reference if reference is not None else 0.0
        pool = []
        for _, d in loaded:
            kp = d['kpara']
            m = (kp >= lo) & (kp <= max_kpara)
            v = d['pk'][ip][m]
            v = v[np.isfinite(v)]
            if v.size:
                pool.append(np.percentile(np.abs(v), 99))
        if pool:
            lim = max(pool) * 1.3
            if lim > 0:
                ax.set_ylim(-lim, lim)

    for ax in flat[len(kper_indices):]:
        ax.set_visible(False)

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel(r'$k_{\parallel}$  [Mpc$^{-1}$]')
    for row in axes:
        row[0].set_ylabel(r'$P(k_{\perp},k_{\parallel})$  [mK$^2$ Mpc$^3$]')

    flat[0].legend(loc='best', fontsize=8)
    if shade:
        flat[len(kper_indices) - 1].annotate(
            'shaded = modes used', xy=(0.98, 0.03), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=8, color=_MUTED)
    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()

    return _finish(fig, out)


def unflagged_channels(m):
    r"""Indices of channels that carry data in a ``C_\ell(\nu_a,\nu_b)`` matrix.

    A flagged channel is zeroed at gridding, so its whole row and column in the MAPS
    are zero — or NaN once divided by a zero normalisation.
    """
    bad = ~np.isfinite(m) | (m == 0)
    return np.where(~bad.all(axis=1))[0]


def _nunu_panel(ax, m, nu, symmetric, drop_flagged, cmap=None):
    """Draw one C_l(nu_a,nu_b) map, optionally with flagged channels removed."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if drop_flagged:
        keep = unflagged_channels(m)
        if keep.size < 2:
            ax.text(0.5, 0.5, 'no unflagged channels', ha='center', va='center',
                    transform=ax.transAxes, color=_MUTED, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            return None
        m = m[np.ix_(keep, keep)]
        # index axis: the retained channels are not contiguous in frequency, so
        # plotting against nu would reintroduce the gaps we just removed
        coord = np.arange(m.shape[0])
        tick_nu = nu[keep]
    else:
        coord = nu
        tick_nu = nu

    finite = m[np.isfinite(m)]
    if finite.size == 0:
        ax.text(0.5, 0.5, 'no finite values', ha='center', va='center',
                transform=ax.transAxes, color=_MUTED, fontsize=9)
        return None

    if symmetric:
        lim = np.percentile(np.abs(finite[finite != 0]), 99) if np.any(finite != 0) else 1.0
        mesh = ax.pcolormesh(coord, coord, m.T, cmap=cmap or 'RdBu_r',
                             norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim),
                             shading='auto', rasterized=True)
    else:
        mesh = ax.pcolormesh(coord, coord, m.T, cmap=cmap or SEQUENTIAL,
                             shading='auto', rasterized=True)

    if drop_flagged and coord.size > 1:
        # label a handful of ticks with the real frequency of the retained channel
        n = coord.size
        idx = np.linspace(0, n - 1, min(5, n)).astype(int)
        ax.set_xticks(coord[idx]); ax.set_xticklabels([f'{tick_nu[i]:.0f}' for i in idx])
        ax.set_yticks(coord[idx]); ax.set_yticklabels([f'{tick_nu[i]:.0f}' for i in idx])

    ax.set_aspect('equal')
    ax.grid(False)
    return mesh


def select_bins(nbin, bins=None, n_bins=None):
    """Resolve a bin selection to a list of indices.

    ``bins``   explicit indices, e.g. ``[0, 3, 7]`` or ``range(0, 20, 4)``
    ``n_bins`` the first N bins, e.g. ``n_bins=5``
    neither    every bin

    ``bins`` wins if both are given.
    """
    if bins is not None:
        out = [int(b) for b in bins]
    elif n_bins is not None:
        out = list(range(min(int(n_bins), nbin)))
    else:
        out = list(range(nbin))
    bad = [b for b in out if not (0 <= b < nbin)]
    if bad:
        raise ValueError(f"bin index/indices {bad} outside 0..{nbin - 1}")
    if not out:
        raise ValueError("no bins selected")
    return out


def compare_cl_nunu(runs, lval, nuc, dnuc, bins=None, n_bins=None,
                    drop_flagged=True,
                    symmetric=None, out=None, title=None, panel=2.6):
    r"""``C_\ell(\nu_a,\nu_b)`` for every :math:`\ell` bin, two columns.

    One row per :math:`\ell` bin; the columns are the two runs in the order given —
    pass ``[('without SCF', ...), ('with SCF', ...)]`` to get without on the left.

    Parameters
    ----------
    runs : sequence of (label, cl2d array or directory)
        Exactly two runs.
    lval : array_like
        Effective multipole per bin.
    bins : sequence of int, optional
        Explicit bin indices, e.g. ``[0, 3, 7]`` or ``range(0, 20, 4)``.
    n_bins : int, optional
        Show the first N bins instead. Ignored if ``bins`` is given.
        With neither, every bin is shown — 20 rows is a ~25 MB figure, so prefer a
        subset for interactive use.
    drop_flagged : bool
        Remove flagged channels entirely rather than leaving them as blank lattice
        lines. The axes then run over retained channels, tick-labelled with the real
        frequency, so the map is continuous.
    symmetric : bool or sequence of bool, optional
        Diverging scale about zero. Default: per column — off for the unfiltered run
        (one-sided) and on for the filtered one (signed residual).

    Notes
    -----
    Each panel gets its own colour scale, as in ``Documentation/RA_11.pdf``. A shared
    one would be useless here: the unfiltered MAPS is orders of magnitude larger, so
    the filtered panel would render as flat zero.
    """
    import matplotlib.pyplot as plt

    if len(runs) != 2:
        raise ValueError(f"compare_cl_nunu wants exactly 2 runs, got {len(runs)}")

    loaded = []
    for label, src in runs:
        arr = src if isinstance(src, np.ndarray) else load_cl(src, kind='cl2d')
        loaded.append((label, arr))

    nbin = loaded[0][1].shape[0]
    bins = select_bins(nbin, bins, n_bins)

    if symmetric is None:
        # unfiltered is one-sided, filtered is a signed residual
        symmetric = [False, True]
    elif isinstance(symmetric, bool):
        symmetric = [symmetric, symmetric]

    nc = loaded[0][1].shape[-1]
    nu = nuc + (np.arange(nc) - nc / 2.0) * dnuc

    fig, axes = plt.subplots(len(bins), 2,
                             figsize=(2 * panel + 2.2, panel * len(bins)),
                             squeeze=False)

    for row, ib in enumerate(bins):
        for col, (label, arr) in enumerate(loaded):
            ax = axes[row][col]
            ncc = arr.shape[-1]
            nu_c = nuc + (np.arange(ncc) - ncc / 2.0) * dnuc
            mesh = _nunu_panel(ax, np.asarray(arr[ib], dtype=float), nu_c,
                               symmetric[col], drop_flagged)
            if mesh is not None:
                cb = fig.colorbar(mesh, ax=ax, pad=0.02, aspect=30)
                cb.ax.tick_params(labelsize=7)
                cb.outline.set_edgecolor(_MUTED)
            if row == 0:
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.set_ylabel(rf'$\ell = {int(lval[ib])}$' + '\n' +
                              (r'$\nu_b$ [MHz]'), fontsize=9)
            if row == len(bins) - 1:
                ax.set_xlabel(r'$\nu_a$  [MHz]', fontsize=9)
            ax.tick_params(labelsize=7)

    if title:
        fig.suptitle(title, y=1.002)
    fig.tight_layout()
    return _finish(fig, out)


def cl_nunu(cl2d, bin_index, nuc, dnuc, lval=None, out=None, title=None,
            symmetric=False, drop_flagged=False):
    r"""The MAPS :math:`C_{\ell}(\nu_a, \nu_b)` at one :math:`\ell`.

    Reproduces the frequency-frequency maps in ``Documentation/RA_11.pdf``. Flagged
    channels show up as the empty lattice, because they are exactly zero.

    Parameters
    ----------
    cl2d : np.ndarray
        ``(Nbin, NC, NC)`` array from :func:`myutils.clfuncs.clfuncs.correlate`.
        Stage 3 only writes this when ``clfuncs.keep_2d`` is true.
    bin_index : int
        Which annular bin to show.
    nuc, dnuc : float
        Central frequency and channel width in MHz — set the axis values.
    symmetric : bool
        Use a diverging scale about zero. Appropriate after SCF, where the residual
        is signed and roughly zero-mean; the default sequential scale suits the
        unfiltered case, where the values are one-sided.
    """
    import matplotlib.pyplot as plt

    m = np.asarray(cl2d[bin_index], dtype=float)
    nc = m.shape[0]
    nu = nuc + (np.arange(nc) - nc / 2.0) * dnuc

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    mesh = _nunu_panel(ax, m, nu, symmetric, drop_flagged)
    if mesh is None:
        return _finish(fig, out)

    lab = ' (flagged channels removed)' if drop_flagged else ''
    ax.set_xlabel(r'$\nu_a$  [MHz]' + lab)
    ax.set_ylabel(r'$\nu_b$  [MHz]')
    tlab = f'$\\ell = {int(lval[bin_index])}$' if lval is not None else f'bin {bin_index}'
    ax.set_title(title or f'$C_\\ell(\\nu_a,\\nu_b)$,  {tlab}')

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, aspect=40)
    cbar.set_label(r'$C_{\ell}(\nu_a,\nu_b)$  [mK$^2$]')
    cbar.outline.set_edgecolor(_MUTED)

    return _finish(fig, out)


def scf_windows(out=None, smoothing_mhz=(1.0, 2.0, 3.0), dnuc=0.04):
    """The four SCF window functions at several smoothing scales."""
    import matplotlib.pyplot as plt

    names = ['Hann', 'Hamming', 'Blackman', r'Kaiser ($\beta=14$)']
    styles = ['-', '--', '-.', ':']        # secondary encoding beside colour

    fig, axes = plt.subplots(1, len(smoothing_mhz),
                             figsize=(4.2 * len(smoothing_mhz), 3.6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, sm in zip(axes, smoothing_mhz):
        NW = int(sm / dnuc)
        NN = 2 * NW + 1
        n = np.arange(-NW, NW + 1)
        for w, name, colour, ls in zip(
                [np.hanning(NN), np.hamming(NN), np.blackman(NN), np.kaiser(NN, 14)],
                names, PALETTE, styles):
            ax.plot(n, w / (2 * NW), ls=ls, lw=1.6, color=colour, label=name)
        ax.set_xlabel('$n$')
        ax.set_title(f'{sm} MHz  ($N={NW}$)')

    axes[0].set_ylabel('$H(n)$')
    axes[-1].legend(loc='upper right')
    fig.tight_layout()

    return _finish(fig, out)
