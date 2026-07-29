#!/usr/bin/env python
"""Stage 1 — simulate the UAPS reference UVFITS.

This is the denominator of the C_l ratio. Without it there is no power spectrum,
only an unnormalised correlation.

By default an existing UAPS file is kept and the stage is skipped, because
regenerating it is expensive. Pass --force to delete and rebuild it:

    python pipeline/01_uaps.py --force

Doing so invalidates every downstream product (they were normalised against the old
file), so --force also lists what has gone stale, and --clean removes those too.
"""

import argparse
import sys

# simvis pulls in healpy, matplotlib, astropy and numexpr — a few thousand small files.
# On a cold or contended shared filesystem that import alone can take several minutes,
# with no output, which reads exactly like a hang. Say so before it starts.
print("importing simvis dependencies (healpy, matplotlib, astropy, numexpr) — "
      "can take minutes on a cold filesystem cache ...", flush=True)

from myutils.config import load                                       # noqa: E402
import myutils.simvis.simvis as sv                                    # noqa: E402

print("imports done.", flush=True)


#: products that are only meaningful for a given UAPS file
DOWNSTREAM = ('bininfo', 'ml', 'el', 'cl', 'cln')


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config')
    p.add_argument('-f', '--force', action='store_true',
                   help='delete an existing UAPS file and regenerate it')
    p.add_argument('--clean', action='store_true',
                   help='with --force, also delete the downstream products that the '
                        'old UAPS file normalised (bin_info, el, ml, cl, cln)')
    return p.parse_args()


def stale_products(cfg):
    """Downstream products that exist and were tied to the current UAPS file."""
    out = []
    for kind in DOWNSTREAM:
        try:
            path = cfg.product(kind)
        except KeyError:
            continue
        if path.exists():
            out.append(path)
    return out


def main():
    args = parse_args()
    cfg = load(args.config, echo=True, stage='stage 1 · UAPS simulation')

    out = cfg.paths.uaps_uvfits

    if out.exists():
        if not args.force:
            print(f"UAPS file already exists: {out}")
            print("Pass --force to delete and regenerate it, or point "
                  "paths.uaps_uvfits elsewhere.")
            return 0

        size_gb = out.stat().st_size / 1e9
        print(f"--force: deleting {out} ({size_gb:.2f} GB)", flush=True)
        out.unlink()

        stale = stale_products(cfg)
        if stale:
            if args.clean:
                print("--clean: removing downstream products normalised against it:")
                for p in stale:
                    print(f"    deleted  {p.name}")
                    p.unlink()
            else:
                print("\nWARNING: these downstream products were normalised against the "
                      "old UAPS file and are now stale:")
                for p in stale:
                    print(f"    {p.name}")
                print("Re-run stages 2-5, or pass --clean to delete them now.\n",
                      flush=True)

    # simvis reads nside / Nrea / chunk_size out of builtins
    cfg.apply_simulation_globals()

    # The GRF cache lives under the configured output directory rather than the
    # current working directory, which is where simvis would otherwise put it.
    import os
    os.chdir(cfg.paths.output_dir)

    print(f"simulating {cfg.simulation.Nrea} realisations at nside="
          f"{cfg.simulation.nside}, seed={cfg.simulation.seed}", flush=True)

    sv.sim_vis(str(cfg.science_uvfits), str(out),
               skysimtype='2D',
               seed=cfg.simulation.seed,
               apsfunc=sv.uaps)          # unit APS — this is what makes it UAPS

    print(f"\nwrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
