#!/usr/bin/env python
"""Stage 4 — noise realisations.

Each realisation grids a ~5 GB array, so it is correlated and discarded immediately;
only the (Nbin, NE) result is kept.

Three modes:

* no arguments — loop over every realisation in this process, write cln directly
* ``--task N`` (or ``$SLURM_ARRAY_TASK_ID``) — do realisation N only, write a part file
* ``--merge`` — stack the part files into cln and delete them

The SLURM submitter uses the array form plus a merge job.
"""

import argparse
import os
import sys

import numpy as np

from myutils.config import load
from myutils.stages import load_bininfo, one_pass


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--task', type=int, default=None,
                   help='realisation index; defaults to $SLURM_ARRAY_TASK_ID')
    p.add_argument('--merge', action='store_true',
                   help='stack part files into cln and remove them')
    return p.parse_args()


def n_tasks(cfg):
    """Number of array tasks, given per_task realisations each."""
    n = cfg.noise.n_realisations
    per = max(1, int(cfg.noise.get('per_task', 1) or 1))
    return -(-n // per), per                       # ceil division


def task_slice(cfg, task):
    """Realisation indices handled by array task *task*."""
    n = cfg.noise.n_realisations
    ntask, per = n_tasks(cfg)
    if not (0 <= task < ntask):
        raise ValueError(f"--task {task} outside 0..{ntask - 1} "
                         f"({n} realisations at {per} per task)")
    return list(range(task * per, min((task + 1) * per, n)))


def merge(cfg):
    """Concatenate cln part files into the single cln product."""
    n = cfg.noise.n_realisations
    ntask, _ = n_tasks(cfg)
    parts = [cfg.product('cln_part', task=i) for i in range(ntask)]
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(parts)} noise parts missing, e.g. {missing[:5]}. "
            f"Re-run the failed array tasks before merging."
        )

    # Each part holds however many realisations its task ran, so concatenate rather
    # than stack — parts are (k, Nbin, NE) with k = per_task, short in the last part.
    chunks = [np.load(p) for p in parts]
    cln = np.concatenate(chunks, axis=0)

    if cln.shape[0] != n:
        raise ValueError(
            f"merged {cln.shape[0]} realisations but the config asks for {n}. "
            f"Parts are inconsistent with noise.per_task — delete the parts and re-run "
            f"stage 4."
        )

    np.save(cfg.product('cln'), cln)
    print(f"merged {len(parts)} parts -> {n} realisations -> "
          f"{cfg.product('cln').name}  {cln.shape}")

    for p in parts:
        p.unlink()
    print(f"removed {len(parts)} part files")
    return 0


def main():
    args = parse_args()
    cfg = load(args.config, echo=True, stage='stage 4 · noise realisations')

    if args.merge:
        return merge(cfg)

    task = args.task
    if task is None and os.environ.get('SLURM_ARRAY_TASK_ID'):
        task = int(os.environ['SLURM_ARRAY_TASK_ID'])

    cfg.require('ml')                      # the noise is normalised by the same ml
    ml = np.load(cfg.product('ml'))
    Nbin, ni, mask, _, _ = load_bininfo(cfg)

    n = cfg.noise.n_realisations
    if task is not None:
        todo = task_slice(cfg, task)
        ntask, per = n_tasks(cfg)
        print(f"array task {task} of {ntask}: realisations {todo[0]}-{todo[-1]} "
              f"({len(todo)} of {n}, {per} per task)\n", flush=True)
    else:
        todo = list(range(n))

    results = []
    for k, i in enumerate(todo):
        seed = cfg.noise.seed_start + i    # distinct per realisation, by construction
        print(f"--- realisation {i + 1}/{n} (task item {k + 1}/{len(todo)}), "
              f"seed {seed} ---", flush=True)
        en = one_pass(cfg, cfg.science_uvfits, -2, ni, mask, Nbin, seed=seed)
        with np.errstate(divide='ignore', invalid='ignore'):
            results.append(en / ml)
        del en

    cln = np.stack(results)                # (len(todo), Nbin, NE)

    if task is not None:
        out = cfg.product('cln_part', task=task)
        np.save(out, cln)
        print(f"\nwrote {out.name}  {cln.shape}")
    else:
        np.save(cfg.product('cln'), cln)
        print(f"\nwrote {cfg.product('cln').name}  {cln.shape}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
