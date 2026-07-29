#!/bin/bash
# Check the install and the configuration. Safe on a login node — reads the FITS
# header only, never the visibilities.
#
#     bash verify.sh                    # checks config.yaml
#     bash verify.sh config.smoke.yaml  # checks a different config

set -uo pipefail
cd "$(dirname "$0")"

VENV=/idia/users/schatterjee/vnv_ilifu/vnvMWA
CONFIG="${1:-config.yaml}"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "activating $VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

echo
echo "================ install ================"
pip show myutils 2>/dev/null | grep -E "^(Name|Version|Location|Editable)" \
    || echo "myutils NOT INSTALLED — run: pip install -e ."

echo
echo "================ packages ==============="
python - <<'PY'
import importlib, sys
print(f"{'python':10s} {sys.version.split()[0]}")
for name in ('numpy', 'scipy', 'astropy', 'healpy', 'numba', 'numexpr', 'yaml'):
    try:
        mod = importlib.import_module(name)
        print(f"{name:10s} {getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"{name:10s} MISSING")
major, minor = sys.version_info[:2]
if (major, minor) < (3, 9):
    print(f"\nWARNING: Python {major}.{minor} is too old — the pipeline needs 3.9+")
PY

echo
echo "================ pipeline modules ======="
python -c "
import myutils.config, myutils.stages
import myutils.tge.grid, myutils.scf.scf
import myutils.clfuncs.correlate, myutils.psfuncs.psestimation
print('all pipeline modules import OK')
" 2>&1 | grep -v "^Imported"

echo
echo "================ configuration =========="
PS_CONFIG="$PWD/$CONFIG" python -m myutils.config 2>&1 | grep -v "^Imported"

echo
echo "If everything above looks right:  bash smoke.sh"
