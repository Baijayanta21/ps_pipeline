#!/bin/bash
# Run all five stages with config.smoke.yaml — real UVFITS, tiny settings.
# Proves the chain works before committing hours to the real run.
# THE NUMBERS ARE MEANINGLESS; only the shapes and the absence of errors matter.
#
# Needs a compute node, not the login node:
#
#     srun --mem=16G --cpus-per-task=4 --time=00:45:00 --pty bash
#     bash smoke.sh
#
# Or submit it:  sbatch --mem=16G --time=00:45:00 --wrap "bash $PWD/smoke.sh"

set -euo pipefail
cd "$(dirname "$0")"

VENV=/idia/users/schatterjee/vnv_ilifu/vnvMWA
export PS_CONFIG="$PWD/config.smoke.yaml"
export MPLBACKEND=Agg                     # simvis calls hp.mollview; no display here
export PYTHONUNBUFFERED=1                 # else stdout block-buffers when redirected to a
                                          # log and progress appears minutes late
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-mpl}"   # keep the font cache off cephfs
mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

echo "config : $PS_CONFIG"
echo

ALL=(01_uaps 02_bininfo 03_data 04_noise 05_ps)

# Stage selection — by number or by name:
#   bash smoke.sh                    all five
#   bash smoke.sh --from data        stages 3,4,5
#   bash smoke.sh --only noise       just stage 4
#   bash smoke.sh --from 2 --to 4    numbers work too
#
#   1=uaps  2=bininfo  3=data  4=noise  5=ps

stage_num() {                       # name or number -> 1..5
    case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
        1|uaps|sim|simulate|simvis) echo 1 ;;
        2|bininfo|bin|binning)      echo 2 ;;
        3|data|cl|grid)             echo 3 ;;
        4|noise)                    echo 4 ;;
        5|ps|power|powerspectrum)   echo 5 ;;
        *) echo "unknown stage '$1' — use 1..5 or uaps|bininfo|data|noise|ps" >&2
           echo "BAD" ;;
    esac
}

FROM=1; TO=5
while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM=$(stage_num "$2"); shift 2 ;;
        --to)   TO=$(stage_num "$2");   shift 2 ;;
        --only) FROM=$(stage_num "$2"); TO="$FROM"; shift 2 ;;
        -h|--help)
            echo "usage: bash smoke.sh [--from STAGE] [--to STAGE] [--only STAGE]"
            echo "  STAGE is a number or a name:"
            echo "    1 uaps      simulate the UAPS reference file"
            echo "    2 bininfo   annular binning information"
            echo "    3 data      data + UAPS passes -> cl"
            echo "    4 noise     noise realisations -> cln"
            echo "    5 ps        power spectrum"
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$FROM$TO" in *BAD*) exit 2 ;; esac

if ! [ "$FROM" -ge 1 ] 2>/dev/null || ! [ "$TO" -le 5 ] 2>/dev/null || [ "$FROM" -gt "$TO" ]; then
    echo "stage range must satisfy 1 <= from <= to <= 5 (got from=$FROM to=$TO)" >&2
    exit 2
fi

echo "stages : $FROM to $TO"
echo

for idx in $(seq "$FROM" "$TO"); do
    stage="${ALL[$((idx - 1))]}"
    echo "########################################################"
    echo "###  $stage   (stage $idx of 5)"
    echo "########################################################"
    if ! python "pipeline/${stage}.py"; then
        echo
        echo "FAILED at $stage — stopping. Fix this before the real run."
        echo "Resume with:  bash smoke.sh --from $idx"
        exit 1
    fi
    echo
done

if [ "$TO" -lt 5 ]; then
    echo "stopped after stage $TO as requested; skipping the summary."
    exit 0
fi

echo "########################################################"
echo "###  smoke test passed"
echo "########################################################"
python - <<'PY'
from myutils.config import load
import numpy as np
cfg = load()
out = cfg.paths.output_dir
print(f"products in {out}:")
for p in sorted(out.iterdir()):
    if p.is_file():
        print(f"  {p.name:32s} {p.stat().st_size / 1e6:9.2f} MB")
d = np.load(cfg.product('ps'))
print(f"\nk       = {np.array2string(d['kk'], precision=4)}")
print(f"sigma   = {float(d['sigma']):.4f}   (should be near 1 for a real run)")
print(f"SNR     = {np.array2string(d['snr'], precision=3)}")
PY

echo
echo "Now the real run:"
echo "  export PS_CONFIG=\$PWD/config.yaml"
echo "  python pipeline/submit.py --dry-run"
