#!/usr/bin/env python
"""Run the pipeline twice — with and without SCF — sharing everything that is common.

Stages 0-2 do not depend on SCF at all:

* **0 flag**    copies and flags the UVFITS
* **1 uaps**    simulates the UAPS reference
* **2 bininfo** grids the UAPS to get the annular binning

so they run **once**, in the base output directory. Only stages 3-5 see the SCF
setting, and those run per variant into their own directory:

    {output_dir}/with_SCF/       scf.enabled: true
    {output_dir}/without_SCF/    scf.enabled: false

Everything shared stays at the top of {output_dir} — the flagged UVFITS, the UAPS
reference and bin_info — so there is exactly one copy of each 2.6 GB file:

    {output_dir}/
        UVFITS/{input}.fits          flagged working copy   (shared)
        {index}_uaps.fits            UAPS reference         (shared)
        bin_info_{index}.npz         binning                (shared)
        with_SCF/                    el, ml, cl, cln, ps, plots
        without_SCF/                 el, ml, cl, cln, ps, plots
        plots_compare/               the comparison figures

    python pipeline/run_variants.py --local          # run here, sequentially
    python pipeline/run_variants.py --submit         # submit both chains to SLURM
    python pipeline/run_variants.py --dry-run        # write configs, run nothing
    python pipeline/run_variants.py --variants with_SCF   # just one

Afterwards, pipeline/07_compare.py builds the comparison figures.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from myutils.config import load

PIPELINE_DIR = Path(__file__).resolve().parent

#: variant name -> (scf.enabled, trim_band)
#:
#: ``trim_band`` restricts the channel range to the same NW..nchan-NW window that SCF
#: would leave, WITHOUT filtering. That isolates the effect of the filter: with_SCF and
#: without_SCF differ in both filtering and bandwidth (668 vs 768 channels, 26.72 vs
#: 30.72 MHz, different k_par sampling and mode counts), so a difference between them is
#: not attributable to SCF alone. without_SCF_matched holds the band fixed so it is.
VARIANTS = {
    'with_SCF':            (True,  False),
    'without_SCF':         (False, False),   # full band, as requested
    'without_SCF_matched': (False, True),    # same band as with_SCF, no filter
}

#: run these unless --variants says otherwise
DEFAULT_VARIANTS = ('with_SCF', 'without_SCF')

#: stages that do not depend on SCF, run once and shared
SHARED_STAGES = (0, 1, 2)
VARIANT_STAGES = (3, 4, 5)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config')
    p.add_argument('--local', action='store_true', help='run here, sequentially')
    p.add_argument('--submit', action='store_true', help='submit to SLURM')
    p.add_argument('--dry-run', action='store_true',
                   help='write the variant configs and print the plan only')
    p.add_argument('--variants', default=','.join(DEFAULT_VARIANTS),
                   help=f"comma-separated subset of {','.join(VARIANTS)}")
    p.add_argument('--skip-shared', action='store_true',
                   help='assume stages 0-2 are already done in the base directory')
    return p.parse_args()


def write_variant_config(cfg, name, spec):
    """Derive a variant config: own output_dir, own scf setting, shared inputs."""
    enabled, trim_band = spec
    raw = yaml.safe_load(Path(cfg.source).read_text())

    base_out = Path(raw['paths']['output_dir']).expanduser()
    var_out = base_out / name                      # a subdirectory, not a sibling
    var_out.mkdir(parents=True, exist_ok=True)

    raw['paths']['output_dir'] = str(var_out)
    raw.setdefault('scf', {})['enabled'] = bool(enabled)

    if trim_band:
        # same channel window SCF would leave, but unfiltered — so this variant and
        # with_SCF share NE, the k_par grid and the mode count exactly
        nw = cfg.NW
        obs = raw.setdefault('observation', {})
        obs['n1'] = int(cfg.observation.n1 + nw)
        obs['n2'] = int(cfg.observation.n2 - nw)

    # Share the expensive inputs rather than regenerating them. The UAPS file and the
    # flagged UVFITS are identical across variants — SCF is applied downstream of both.
    raw['paths']['uaps_uvfits'] = str(cfg.paths.uaps_uvfits)
    if cfg.flagging_enabled:
        # point the variant at the already-flagged working copy
        raw.setdefault('flagging', {})['enabled'] = True
        link = var_out / cfg.paths.work_dir.name
        link.mkdir(parents=True, exist_ok=True)
        target = link / cfg.paths.work_uvfits.name
        if not target.exists():
            try:
                target.symlink_to(cfg.paths.work_uvfits)
            except OSError:
                shutil.copy2(cfg.paths.work_uvfits, target)

    # bin_info is small; copy it so each variant is self-contained
    src_bin = cfg.product('bininfo')
    if src_bin.exists():
        shutil.copy2(src_bin, var_out / src_bin.name)

    path = var_out / f'config.{name}.yaml'
    path.write_text(yaml.safe_dump(raw, sort_keys=False, width=100))
    return path, var_out


def run(cmd, dry):
    print('   ', ' '.join(str(c) for c in cmd), flush=True)
    if dry:
        return 0
    return subprocess.run(cmd).returncode


def main():
    args = parse_args()
    cfg = load(args.config, echo=True, stage='SCF comparison run')

    names = [v.strip() for v in args.variants.split(',') if v.strip()]
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        print(f"unknown variant(s): {unknown}. Known: {list(VARIANTS)}",
              file=sys.stderr)
        return 2
    if not (args.local or args.submit or args.dry_run):
        print("choose --local, --submit or --dry-run", file=sys.stderr)
        return 2

    dry = args.dry_run
    py = sys.executable

    # ---- shared stages, once ------------------------------------------------
    upstream = None                      # SLURM job the variants must wait for
    if not args.skip_shared:
        print(f"\n=== shared stages {SHARED_STAGES} (SCF-independent) "
              f"in {cfg.paths.output_dir} ===", flush=True)
        if args.submit:
            # submit them as their own chain; the variants depend on the last one.
            # Running these locally would put a multi-GB copy and the nside=512
            # simulation on whatever node you happen to be sitting on.
            cmd = [py, str(PIPELINE_DIR / 'submit.py'), '--config', str(cfg.source),
                   '--from', 'flag', '--to', 'bininfo', '--print-last-job']
            print('   ', ' '.join(cmd), flush=True)
            if not dry:
                res = subprocess.run(cmd, capture_output=True, text=True)
                sys.stdout.write(res.stdout)
                if res.returncode:
                    sys.stderr.write(res.stderr)
                    return res.returncode
                for line in res.stdout.splitlines():
                    if line.startswith('LAST_JOB='):
                        upstream = line.split('=', 1)[1].strip()
                print(f"    variants will wait for job {upstream}")
        else:
            for stage in SHARED_STAGES:
                script = {0: '00_flag.py', 1: '01_uaps.py',
                          2: '02_bininfo.py'}[stage]
                rc = run([py, str(PIPELINE_DIR / script),
                          '--config', str(cfg.source)], dry)
                if rc:
                    print(f"shared stage {stage} failed (exit {rc})", file=sys.stderr)
                    return rc
    else:
        print("\n--skip-shared: assuming stages 0-2 are already done")

    # ---- per-variant stages -------------------------------------------------
    made = []
    for name in names:
        path, out = write_variant_config(cfg, name, VARIANTS[name])
        made.append((name, path, out))
        en, trim = VARIANTS[name]
        extra = ', band trimmed to match with_SCF' if trim else ''
        print(f"\n=== variant '{name}'  (scf.enabled = {en}{extra}) ===")
        print(f"    config : {path}")
        print(f"    output : {out}", flush=True)

        if args.submit:
            cmd = [py, str(PIPELINE_DIR / 'submit.py'), '--config', str(path),
                   '--from', 'data']
            if upstream:
                cmd += ['--after', upstream]
            rc = run(cmd, dry)
        else:
            for stage in VARIANT_STAGES:
                script = {3: '03_data.py', 4: '04_noise.py', 5: '05_ps.py'}[stage]
                rc = run([py, str(PIPELINE_DIR / script), '--config', str(path)], dry)
                if rc:
                    break
        if rc:
            print(f"variant '{name}' failed (exit {rc})", file=sys.stderr)
            return rc

    print("\n=== done ===")
    for name, path, out in made:
        print(f"  {name:<8s} {out}")
    print("\nCompare them with:")
    print(f"  python pipeline/07_compare.py --config {cfg.source}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
