#!/usr/bin/env python
"""Render and submit the whole pipeline to SLURM, with dependencies.

Reads the `slurm:` section of the config, writes one sbatch script per stage into
{output_dir}/slurm/, and chains them with --dependency=afterok so each stage starts
only if the previous one succeeded. Stage 4 runs as an array job, one task per noise
realisation, followed by a merge step.

    python pipeline/submit.py --dry-run          # write the scripts, submit nothing
    python pipeline/submit.py                    # write and submit
    python pipeline/submit.py --from 3           # start at stage 3
    python pipeline/submit.py --only 5           # just one stage

Without SLURM, run the stage scripts directly in order instead.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from myutils.config import load

PIPELINE_DIR = Path(__file__).resolve().parent

#: stage number -> (key in slurm.stages, script, human name)
STAGES = {
    0: ('flag',    '00_flag.py',    'copy + coarse-band flagging'),
    1: ('uaps',    '01_uaps.py',    'UAPS simulation'),
    2: ('bininfo', '02_bininfo.py', 'binning information'),
    3: ('data',    '03_data.py',    'data + UAPS passes'),
    4: ('noise',   '04_noise.py',   'noise realisations'),
    5: ('ps',      '05_ps.py',      'power spectrum'),
}


#: stage key -> number, so stages can be named instead of numbered
STAGE_NAMES = {key: num for num, (key, _, _) in STAGES.items()}

#: convenience spellings
ALIASES = {'flagging': 'flag', 'flags': 'flag',
           'sim': 'uaps', 'simulate': 'uaps', 'simvis': 'uaps',
           'bin': 'bininfo', 'binning': 'bininfo', 'bininfo': 'bininfo',
           'cl': 'data', 'grid': 'data',
           'power': 'ps', 'powerspectrum': 'ps', 'spectrum': 'ps'}


def stage_id(value):
    """Accept a stage as a number (1-5) or a name ('data', 'noise', ...)."""
    s = str(value).strip().lower()
    if s.isdigit():
        n = int(s)
        if n in STAGES:
            return n
        raise argparse.ArgumentTypeError(
            f"stage {n} out of range; use {min(STAGES)}-{max(STAGES)}")
    s = ALIASES.get(s, s)
    if s in STAGE_NAMES:
        return STAGE_NAMES[s]
    known = ', '.join(f"{n}={k}" for n, (k, _, _) in sorted(STAGES.items()))
    raise argparse.ArgumentTypeError(f"unknown stage '{value}'. Use {known}")


def parse_args():
    known = ', '.join(f"{n}={k}" for n, (k, _, _) in sorted(STAGES.items()))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config')
    p.add_argument('--dry-run', action='store_true',
                   help='write the sbatch scripts but do not submit')
    p.add_argument('--from', dest='start', type=stage_id, default=0,
                   metavar='STAGE',
                   help=f'first stage to submit, by number or name ({known})')
    p.add_argument('--only', type=stage_id, default=None, metavar='STAGE',
                   help='submit a single stage, by number or name')
    p.add_argument('--local', action='store_true',
                   help='run the stages here, sequentially, instead of submitting to '
                        'SLURM. Respects --from/--to/--only. Stage 4 runs every '
                        'realisation in one process, so no merge step is needed.')
    p.add_argument('--to', dest='end', type=stage_id, default=None, metavar='STAGE',
                   help='last stage to run, by number or name')
    p.add_argument('--status', action='store_true',
                   help="show this pipeline's jobs and exit")
    p.add_argument('--cancel', action='store_true',
                   help="cancel this pipeline's jobs and exit")
    p.add_argument('--yes', action='store_true',
                   help='with --cancel, do not ask for confirmation')
    p.add_argument('--after', default=None, metavar='JOBID',
                   help='make the first submitted stage depend on this job id, so '
                        'separate submit.py calls can be chained')
    p.add_argument('--print-last-job', action='store_true',
                   help='print only the final job id on stdout, for scripting')
    return p.parse_args()


def my_jobs(cfg):
    """Jobs belonging to this pipeline, matched by name — never by raw id.

    Names are ``ps<N>_<stage>_<index>``, so the index scopes the match to this config.
    Anything else you have running (Jupyter, VS Code, unrelated work) cannot match.
    """
    result = subprocess.run(
        ['squeue', '--noheader', '--user', os.environ.get('USER', ''),
         '--format=%i|%j|%T|%M|%R'],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"squeue failed: {result.stderr.strip()}")

    pattern = re.compile(rf'^ps[0-5]_[a-z]+_{re.escape(str(cfg.index))}$')
    jobs = []
    for line in result.stdout.splitlines():
        parts = line.split('|')
        if len(parts) < 5:
            continue
        jid, name, state, elapsed, reason = parts[:5]
        if pattern.match(name.strip()):
            jobs.append((jid.strip(), name.strip(), state.strip(),
                         elapsed.strip(), reason.strip()))
    return jobs


def run_local(cfg, wanted):
    """Run the stages in this process's machine, one after another.

    Output is streamed to the terminal and appended to the same universal log the
    SLURM path writes, so local and batch runs leave the same trail.
    """
    log_dir = (Path(cfg.slurm['log_dir']).expanduser().resolve()
               if cfg.slurm.get('log_dir') else cfg.paths.output_dir / 'logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'pipeline_{cfg.index}.log'

    env = os.environ.copy()
    env['PS_CONFIG'] = str(cfg.source)
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('MPLBACKEND', 'Agg')                 # simvis draws healpy figures
    env.setdefault('MPLCONFIGDIR', f"/tmp/{env.get('USER', 'user')}-mpl")
    Path(env['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)

    print(f"running locally on {os.uname().nodename}")
    print(f"log: {log_path}\n")

    with open(log_path, 'a') as log:
        for stage in wanted:
            key, script, human = STAGES[stage]
            banner = (f"\n{'=' * 63}\n=== stage {stage} · {human}  (local)\n"
                      f"=== {os.uname().nodename}  "
                      f"{__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S}\n"
                      f"{'=' * 63}")
            print(banner, flush=True)
            log.write(banner + '\n')
            log.flush()

            cmd = [sys.executable, str(PIPELINE_DIR / script),
                   '--config', str(cfg.source)]
            # stage 4 with no --task loops every realisation in one process and
            # writes cln directly, so the merge step is unnecessary here
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            proc.wait()
            log.flush()

            if proc.returncode != 0:
                msg = (f"\nFAILED at stage {stage} ({key}), exit {proc.returncode}.\n"
                       f"Resume with:  python pipeline/submit.py --local "
                       f"--from {key}\n")
                print(msg)
                log.write(msg)
                return 1

    print(f"\nall requested stages finished. log: {log_path}")
    return 0


def show_status(cfg):
    jobs = my_jobs(cfg)
    if not jobs:
        print(f"no jobs running for index '{cfg.index}'")
        return 0
    print(f"jobs for index '{cfg.index}':\n")
    print(f"  {'JOBID':<16} {'NAME':<26} {'STATE':<10} {'TIME':>8}  REASON")
    for jid, name, state, elapsed, reason in jobs:
        print(f"  {jid:<16} {name:<26} {state:<10} {elapsed:>8}  {reason}")
    print(f"\nlog: {cfg.paths.output_dir / 'logs' / f'pipeline_{cfg.index}.log'}")
    return 0


def cancel_jobs(cfg, assume_yes=False):
    jobs = my_jobs(cfg)
    if not jobs:
        print(f"no jobs to cancel for index '{cfg.index}'")
        return 0

    print(f"about to cancel {len(jobs)} job(s) for index '{cfg.index}':")
    for jid, name, state, elapsed, _ in jobs:
        print(f"  {jid:<16} {name:<26} {state:<10} {elapsed}")

    if not assume_yes:
        try:
            reply = input("\ncancel these? [y/N] ").strip().lower()
        except EOFError:
            reply = ''
        if reply not in ('y', 'yes'):
            print("nothing cancelled.")
            return 1

    # base ids only — cancelling 12345 also kills 12345_0, 12345_1, ...
    ids = sorted({jid.split('_')[0] for jid, *_ in jobs})
    result = subprocess.run(['scancel', *ids], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"scancel failed: {result.stderr.strip()}", file=sys.stderr)
        return 1
    print(f"cancelled: {' '.join(ids)}")
    return 0


def stage_setting(cfg, key, name, default=None):
    """Per-stage override, else slurm.defaults, else *default*."""
    stages = cfg.slurm.get('stages') or {}
    override = (stages.get(key) or {}).get(name)
    if override is not None:
        return override
    fallback = (cfg.slurm.get('defaults') or {}).get(name)
    return fallback if fallback is not None else default


def render(cfg, stage, script_dir, log_dir, array=None, extra_args=''):
    """Write one sbatch script and return its path."""
    key, script, human = STAGES[stage]
    s = cfg.slurm
    job = f"ps{stage}_{key}_{cfg.index}"

    lines = ['#!/bin/bash', f'#SBATCH --job-name={job}']

    if array:
        lines.append(f'#SBATCH --array={array}')

    # One log for the whole pipeline by default. --open-mode=append is essential:
    # without it each job truncates the file and only the last one survives. stderr is
    # merged into the same file so the ordering of errors relative to progress is
    # preserved. Set slurm.log_mode: per_job for the old one-file-per-job behaviour.
    universal = (s.get('log_mode') or 'universal') == 'universal'
    if universal:
        lines += [f'#SBATCH --output={log_dir}/pipeline_{cfg.index}.log',
                  '#SBATCH --open-mode=append']
    elif array:
        lines += [f'#SBATCH --output={log_dir}/{job}_%A_%a.out',
                  f'#SBATCH --error={log_dir}/{job}_%A_%a.err']
    else:
        lines += [f'#SBATCH --output={log_dir}/{job}_%j.out',
                  f'#SBATCH --error={log_dir}/{job}_%j.err']

    lines += [f'#SBATCH --time={stage_setting(cfg, key, "time", "04:00:00")}',
              f'#SBATCH --mem={stage_setting(cfg, key, "mem", "32G")}',
              f'#SBATCH --cpus-per-task={stage_setting(cfg, key, "cpus_per_task", 4)}',
              '#SBATCH --ntasks=1']

    if s.get('partition'):
        lines.append(f'#SBATCH --partition={s.partition}')
    if s.get('account'):
        lines.append(f'#SBATCH --account={s.account}')
    if s.get('qos'):
        lines.append(f'#SBATCH --qos={s.qos}')
    if s.get('mail_user'):
        lines += [f'#SBATCH --mail-user={s.mail_user}',
                  f'#SBATCH --mail-type={s.get("mail_type") or "FAIL"}']

    python = s.get('python') or 'python'
    lines += ['', 'set -euo pipefail', '',
              f'# stage {stage} — {human}', '']
    for cmd in (s.get('setup') or []):
        lines.append(cmd)

    # With every job appending to one log, each block must say who wrote it —
    # otherwise concurrent array tasks are indistinguishable.
    tag = ('${SLURM_JOB_ID}'
           + ('[task ${SLURM_ARRAY_TASK_ID}]' if array else ''))
    lines += ['',
              'echo ""',
              'echo "==============================================================="',
              f'echo "=== stage {stage} · {human}"',
              f'echo "=== job {tag}  on $(hostname)  at $(date \'+%Y-%m-%d %H:%M:%S\')"',
              'echo "==============================================================="',
              '',
              f'export PS_CONFIG={cfg.source}',
              # keep numeric libraries inside the allocation
              'export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}',
              'export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}',
              'export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-1}',
              '',
              f'{python} {PIPELINE_DIR / script} --config {cfg.source} {extra_args}'.rstrip(),
              '',
              f'echo "=== stage {stage} finished at $(date \'+%Y-%m-%d %H:%M:%S\') '
              f'(job {tag}) ==="',
              '']

    suffix = '_merge' if 'merge' in extra_args else ''
    path = script_dir / f'{stage:02d}_{key}{suffix}.sbatch'
    path.write_text('\n'.join(lines))
    path.chmod(0o755)
    return path


def sbatch(path, dependency=None, dry_run=False):
    """Submit *path*, returning the job id (or a placeholder when dry-running)."""
    cmd = ['sbatch', '--parsable']
    if dependency:
        cmd.append(f'--dependency=afterok:{dependency}')
    cmd.append(str(path))

    if dry_run:
        print(f"  would run: {' '.join(cmd)}")
        return f'<{path.stem}>'

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed for {path.name}:\n{result.stderr.strip()}")
    # --parsable gives "jobid" or "jobid;cluster"
    return result.stdout.strip().split(';')[0]


def main():
    args = parse_args()

    # status and cancel are quiet — no config banner, they are used interactively
    if args.status or args.cancel:
        cfg = load(args.config)
        return show_status(cfg) if args.status else cancel_jobs(cfg, args.yes)

    cfg = load(args.config, echo=True,
               stage='local run' if args.local else 'SLURM submission')

    # stage selection is shared by the local and SLURM paths
    if args.only:
        wanted = [args.only]
    else:
        end = args.end if args.end is not None else max(STAGES)
        wanted = [s for s in sorted(STAGES) if args.start <= s <= end]
    if not wanted:
        print(f"no stages selected (from={args.start}, to={args.end})", file=sys.stderr)
        return 2

    if args.local:
        return run_local(cfg, wanted)

    if not cfg.slurm.get('enabled'):
        print("slurm.enabled is false — use --local, or run the stage scripts directly.")
        return 1

    if not args.dry_run and shutil.which('sbatch') is None:
        print("sbatch not found on PATH. Submit from a cluster login node, or use "
              "--dry-run to inspect the generated scripts.", file=sys.stderr)
        return 1

    script_dir = cfg.paths.output_dir / 'slurm'
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = (Path(cfg.slurm['log_dir']).expanduser().resolve()
               if cfg.slurm.get('log_dir') else cfg.paths.output_dir / 'logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"scripts       : {script_dir}")
    print(f"logs          : {log_dir}")
    print(f"stages        : {', '.join(str(s) for s in wanted)}")
    if args.dry_run:
        print("mode          : DRY RUN — nothing will be submitted")
    print()

    dependency = args.after
    for stage in wanted:
        key, _, human = STAGES[stage]
        print(f"stage {stage} — {human}")

        if stage == 4:
            n = cfg.noise.n_realisations
            per = max(1, int((cfg.noise.get('per_task') or 1)))
            ntask = -(-n // per)                       # ceil division
            throttle = stage_setting(cfg, 'noise', 'throttle', 10)
            array = f'0-{ntask - 1}%{throttle}'
            path = render(cfg, 4, script_dir, log_dir, array=array)
            print(f"  {n} realisations, {per} per task -> array 0-{ntask - 1}, "
                  f"at most {throttle} at once -> {path.name}")
            dependency = sbatch(path, dependency, args.dry_run)
            print(f"  job {dependency}")

            merge_path = render(cfg, 4, script_dir, log_dir, extra_args='--merge')
            print(f"  merge -> {merge_path.name}")
            dependency = sbatch(merge_path, dependency, args.dry_run)
            print(f"  job {dependency}")
        else:
            path = render(cfg, stage, script_dir, log_dir)
            print(f"  -> {path.name}")
            dependency = sbatch(path, dependency, args.dry_run)
            print(f"  job {dependency}")
        print()

    if args.dry_run:
        print(f"Inspect the scripts in {script_dir}, then re-run without --dry-run.")
    else:
        print(f"Submitted. Watch with:  squeue -u $USER")
        print(f"Logs:                   {log_dir}")
    if args.print_last_job:
        print(f"LAST_JOB={dependency}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
