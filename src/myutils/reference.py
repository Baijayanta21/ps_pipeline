r"""
Published 21-cm upper limits, for plotting alongside this pipeline's results
===========================================================================

Every number here is transcribed from a paper and carries its source. Nothing is
computed and nothing is fitted — this module is data, so that a comparison drawn in
a notebook can be traced back to a table in a PDF.

The anchor is **TTGE III** (Sarkar et al. 2026, arXiv:2604.24144, in
``Documentation/``), which is the paper this pipeline implements. It uses the same
instrument, the same :math:`\nu_c = 154.2` MHz, the same 2 MHz SCF smoothing scale
and the same :math:`k_\perp` grid as the configuration here, so its numbers are
directly commensurable with ours — the same estimator applied to different
pointings of the same drift scan.

Units are :math:`\mathrm{mK}^2` for :math:`\Delta^2` and :math:`\mathrm{Mpc}^{-1}`
for :math:`k`. The papers quote upper limits as :math:`(A)^2`, so the stored value
is ``A**2``; ``amp`` keeps :math:`A` because that is the number the text uses.

Caveat worth carrying into any comparison
-----------------------------------------
TTGE III finds :math:`\alpha \lesssim 22^\circ` to be the relatively foreground-free
part of the drift scan, and its headline limits come from there. This pipeline's
default field is :math:`\alpha = 39.7^\circ`, which is outside that range and
between the paper's EoR0 (:math:`0^\circ`) and EoR1 (:math:`60^\circ`) fields — and
the paper reports EoR1 as considerably worse than EoR0/EoR2. Expect our limits to
sit above theirs; that is a property of the field, not necessarily of the run.
"""

import numpy as np

__all__ = ['TTGE3_ALPHA11', 'TTGE3_CASE_I', 'TTGE3_CASE_II', 'TTGE3_HEADLINE',
           'EXTERNAL', 'DATASETS', 'get', 'as_arrays', 'describe']


def _ul(pairs):
    """``[(k, A), ...]`` where the paper quotes the limit as ``(A)^2`` mK^2."""
    return [{'k': k, 'amp': a, 'ul': a * a} for k, a in pairs]


# ------------------------------------------------------------------ TTGE III
#
# Sarkar, Elahi, Choudhuri, Bharadwaj, Chatterjee, Bhattacharyya, Sethi & Patwa,
# "The Tracking Tapered Gridded Estimator for the 21-cm power spectrum from the
# MWA drift scan observations - III. Improved upper limits at z = 8.2 from
# multiple pointings", MNRAS 000, 1-17 (2026), arXiv:2604.24144v2.
#
# z = 8.2, nu_c = 154.2 MHz, delta = -26.7 deg.

#: Table 1, the :math:`\alpha = 11.0^\circ` column — the best *single* pointing
#: centre. This is the like-for-like comparison for a single-field run.
TTGE3_ALPHA11 = {
    'label': r'TTGE III, best single PC ($\alpha=11^\circ$)',
    'short': 'TTGE III (1 PC)',
    'source': 'Sarkar et al. 2026 (arXiv:2604.24144), Table 1',
    'z': 8.2,
    'points': _ul([(0.161, 173.13), (0.212, 346.62), (0.371, 1156.05),
                   (0.443, 867.62), (0.741, 2764.43), (0.959, 2657.21),
                   (1.271, 2952.88)]),
}

#: Table 2, Case I — 23 pointing centres combined incoherently, sorted on the
#: low-:math:`k` bin. Contains the paper's headline result.
TTGE3_CASE_I = {
    'label': 'TTGE III, Case I (23 PCs combined)',
    'short': 'TTGE III (23 PCs)',
    'source': 'Sarkar et al. 2026 (arXiv:2604.24144), Table 2',
    'z': 8.2,
    'points': _ul([(0.156, 98.67), (0.204, 148.63), (0.406, 746.52),
                   (0.482, 943.66), (0.728, 1674.52), (0.817, 1308.67),
                   (1.048, 2551.15), (1.294, 3785.13)]),
}

#: Table 2, Case II — 14 pointing centres, sorted on the :math:`k = 0.406` bin
#: instead, which tightens the high-:math:`k` end at the cost of the lowest bin.
TTGE3_CASE_II = {
    'label': 'TTGE III, Case II (14 PCs combined)',
    'short': 'TTGE III (14 PCs)',
    'source': 'Sarkar et al. 2026 (arXiv:2604.24144), Table 2',
    'z': 8.2,
    'points': _ul([(0.156, 146.45), (0.204, 164.37), (0.406, 325.32),
                   (0.482, 568.73), (0.728, 1597.98), (0.817, 950.71),
                   (1.048, 2186.37), (1.294, 3107.71)]),
}

#: The two numbers the paper's abstract leads with — the tightest single-pointing
#: and combined limits. Use this when one annotated marker says more than a curve.
TTGE3_HEADLINE = {
    'label': 'TTGE III headline limits',
    'short': 'TTGE III',
    'source': 'Sarkar et al. 2026 (arXiv:2604.24144), abstract',
    'z': 8.2,
    'points': [
        dict(k=0.161, amp=173.13, ul=173.13 ** 2,
             note=r'best single PC, $\alpha=11^\circ$'),
        dict(k=0.156, amp=98.67, ul=98.67 ** 2,
             note='23 PCs, incoherent'),
    ],
}

#: Other experiments, as quoted *by* TTGE III when it places itself in context.
#: Transcribed from its discussion of Figure 9 rather than from the original
#: papers, so cite the original before publishing.
EXTERNAL = {
    'trott2020_eor0': {
        'label': 'MWA EoR0 (Trott et al. 2020)',
        'short': 'MWA EoR0',
        'source': 'Trott et al. 2020, as quoted by arXiv:2604.24144',
        'z': 8.2,
        'points': _ul([(0.099, 223.5)]),
    },
    'trott2020_eor2': {
        'label': 'MWA EoR2 (Trott et al. 2020)',
        'short': 'MWA EoR2',
        'source': 'Trott et al. 2020, as quoted by arXiv:2604.24144',
        'z': 8.2,
        'points': _ul([(0.099, 167.7), (0.148, 376.3)]),
    },
    'lofar2025': {
        'label': 'LOFAR (Mertens et al. 2025)',
        'short': 'LOFAR',
        'source': 'Mertens et al. 2025, as quoted by arXiv:2604.24144',
        'z': 9.1,
        'points': _ul([(0.0581, 65.5)]),
    },
    'hera2023': {
        'label': 'HERA (HERA Collaboration et al. 2023)',
        'short': 'HERA',
        'source': 'HERA Collaboration et al. 2023, as quoted by arXiv:2604.24144',
        'z': 7.9,
        # quoted as k = 0.34 h Mpc^-1; 0.238 Mpc^-1 is the paper's own conversion
        'points': _ul([(0.238, 21.4)]),
    },
}

#: Everything addressable by name.
DATASETS = {
    'ttge3_alpha11': TTGE3_ALPHA11,
    'ttge3_case_i': TTGE3_CASE_I,
    'ttge3_case_ii': TTGE3_CASE_II,
    'ttge3_headline': TTGE3_HEADLINE,
    **EXTERNAL,
}


def get(name):
    """Look a dataset up by name, with the available names in the error message."""
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown reference {name!r}. "
                       f"Available: {', '.join(sorted(DATASETS))}") from None


def as_arrays(name_or_dataset):
    r"""``(k, ul)`` as float arrays, sorted by :math:`k`.

    Accepts a name from :data:`DATASETS` or a dataset dict, so a caller can pass
    either without branching.
    """
    d = get(name_or_dataset) if isinstance(name_or_dataset, str) else name_or_dataset
    pts = sorted(d['points'], key=lambda p: p['k'])
    return (np.array([p['k'] for p in pts], dtype=float),
            np.array([p['ul'] for p in pts], dtype=float))


def describe(name_or_dataset=None):
    """A readable table of one dataset, or a listing of all of them."""
    if name_or_dataset is None:
        return '\n'.join(
            f"  {n:<18s} {len(d['points']):>2d} pts   {d['source']}"
            for n, d in sorted(DATASETS.items()))
    d = get(name_or_dataset) if isinstance(name_or_dataset, str) else name_or_dataset
    lines = [d['label'], f"  {d['source']}   (z = {d['z']})",
             f"  {'k [1/Mpc]':>11}  {'Delta^2_UL [mK^2]':>19}  quoted"]
    for p in sorted(d['points'], key=lambda p: p['k']):
        note = f"   {p['note']}" if p.get('note') else ''
        lines.append(f"  {p['k']:11.4f}  {p['ul']:19.3e}  ({p['amp']:.2f})^2{note}")
    return '\n'.join(lines)
