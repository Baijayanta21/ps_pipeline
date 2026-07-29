# Running the pipeline

Everything is driven by **one file: `config.yaml`**. Edit it, then run five scripts in
order — or submit the whole chain to SLURM with a single command.

Reference material is in [`PIPELINE.md`](PIPELINE.md); the diagrams are in
[`FLOWCHART.md`](FLOWCHART.md).

- [Install](#install)
- [Configure](#configure)
- [Run — SLURM](#run--slurm)
- [Run — directly](#run--directly)
- [What each stage does](#what-each-stage-does)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)

---

## Install

```bash
source /idia/users/schatterjee/vnv_ilifu/vnvMWA/bin/activate
cd /idia/users/schatterjee/MWA_work/MWA_2026_newwork/new_PS_pipeline/ps_pipeline
pip install -e .
```

`-e` (editable) matters: the bug fixes on this branch live in the working tree, so an
editable install guarantees you run the fixed code rather than a stale copy in
`site-packages`.

Verify:

```bash
python -c "
import numpy, scipy, astropy, healpy, numba, numexpr, yaml, sys
print('python  ', sys.version.split()[0])
for m in (numpy, scipy, astropy, healpy, numba, numexpr):
    print(f'{m.__name__:9s}', m.__version__)
import myutils.config, myutils.stages
print('pipeline modules OK')
"
```

Requirements that are easy to miss:

- **Python ≥ 3.9.** The `>=3.7` in `pyproject.toml` is wrong (`PIPELINE.md` §7.8).
- **`numexpr`** — imported by `simvis`, and absent from the dependency list until this branch.
- **`pyyaml`** — needed by the config loader. Astropy pulls it in transitively, but it is
  now declared explicitly.

---

## Configure

Copy the template and edit your copy:

```bash
cp config.yaml my_run.yaml
$EDITOR my_run.yaml
export PS_CONFIG=$PWD/my_run.yaml        # every script picks this up
```

The three fields you must set:

```yaml
paths:
  input_uvfits: /home/MWA_data/1000007200.fits    # your data
  output_dir:   /idia/users/schatterjee/MWA_work/ps_run_7200
slurm:
  setup:
    - source /idia/users/schatterjee/vnv_ilifu/vnvMWA/bin/activate
```

Everything else has a working default. Fields left as `null` are **derived**:

| Field | Derived from |
| --- | --- |
| `paths.index` | last 4 digits of the input filename |
| `paths.uaps_uvfits` | `{output_dir}/{index}_uaps.fits` |
| `observation.nuc_mhz` | header `CRVAL4 × 1e-6` |
| `observation.dnuc_mhz` | header `CDELT4 × 1e-6` |
| `observation.nchan` | header `NAXIS4` |
| `observation.n2` | `nchan − 1` |
| `power_spectrum.NE` | `nchan_used − 2·NW` after SCF |
| `power_spectrum.noise_scale` | `(60/√n_nights)²` — **check this against your integration time** |

Check the resolved configuration before committing hours of compute:

```bash
python -m myutils.config
```

```
====================================================================
ps_pipeline — config check
====================================================================
config        : /.../my_run.yaml
input         : /home/MWA_data/1000007200.fits
output        : /idia/.../ps_run_7200
index         : 7200
observation   : nuc = 154.2550 MHz, dnuc = 0.0400 MHz, nchan = 768 (from header: ...)
channels      : 0-767 (768 used)
gridding      : Umax = 250, FWHM = 23 deg, f = 0.6, nstokes = [0, 1], flag = True
binning       : Nbin = 20, |U| = 6.0-220, Mg_min = 0.01
scf           : SM = 2.0 MHz, window = hann, method = fft, NW = 50 -> 668 channels
noise         : 50 realisations, seeds 10000-10049
power spectrum: NE = 668, NBin = 8, cosmology = Planck18, noise_scale = 400.0000
====================================================================
```

Every stage prints this block on startup, so each log records exactly what produced it.

The loader refuses obviously wrong configurations rather than failing hours later —
a missing input file, `nstokes` that isn't `[0, 1]`, an unknown SCF window, `NE` larger
than the channels SCF leaves, `Nrea` above the channel count.

---

## Run — SLURM

```bash
python pipeline/submit.py --dry-run     # write the sbatch scripts, submit nothing
python pipeline/submit.py               # write and submit the whole chain
```

Five jobs are chained with `--dependency=afterok`, so nothing starts unless its
predecessor succeeded. Stage 4 is an array job — one task per noise realisation, throttled
by `slurm.stages.noise.throttle` — followed by a merge step.

```
ps1_uaps    ──▶ ps2_bininfo ──▶ ps3_data ──▶ ps4_noise[0-49] ──▶ ps4_merge ──▶ ps5_ps
```

Scripts land in `{output_dir}/slurm/`, logs in `{output_dir}/logs/`. Resource requests
come from `slurm.defaults` with per-stage overrides in `slurm.stages`.

```bash
python pipeline/submit.py --from 3      # resume from stage 3
python pipeline/submit.py --only 5      # re-run just the power spectrum
squeue -u $USER
```

Read the `--dry-run` output the first time. Partition names, accounts and QOS differ
between clusters, and a wrong `--partition` fails at submission.

**Do not run the stages on a login node.** Stages 1, 3 and 4 need tens of GB. At MWA
sizes — `Umax = 250`, `FWHM = 23`, `f = 0.6` → a `(2, 457, 457, 768)` complex128 array of
5.1 GB — SCF is the peak, because `doscf` holds the smooth model, two flag masks, two
convolved masks and the output simultaneously, and the FFT pads the frequency axis to 875:

| | peak |
| --- | --- |
| SCF, both polarisations in one call | ~40 GB |
| SCF, one polarisation at a time (**the default**) | ~25 GB |
| Correlation after mode selection | <1 GB |

`myutils.stages.scf_pass` filters per polarisation for exactly this reason — SCF acts along
frequency independently for every `(pol, u, v)` point, so it is numerically identical.
The configured `mem: 64G` for stages 3 and 4 leaves comfortable headroom; drop it only if
you have measured your own peak.

Only `python -m myutils.config` is safe on a login node — it reads the FITS header and
nothing else.

---

## Run — directly

No SLURM, or debugging a single stage:

```bash
python pipeline/01_uaps.py       # UAPS reference simulation
python pipeline/02_bininfo.py    # annular binning information
python pipeline/03_data.py       # data + UAPS passes, writes cl
python pipeline/04_noise.py      # all noise realisations in one process (slow)
python pipeline/05_ps.py         # power spectrum
```

Any script takes `--config PATH` if you would rather not set `PS_CONFIG`.

Stage 4 has three modes:

```bash
python pipeline/04_noise.py                # every realisation, sequentially
python pipeline/04_noise.py --task 7       # realisation 7 only -> a part file
python pipeline/04_noise.py --merge        # stack the part files into cln
```

---

## What each stage does

**Stage 1 — `01_uaps.py`.** Simulates the UAPS reference UVFITS with `apsfunc=sv.uaps`.
This is the *denominator* of the `C_l` ratio; without it there is no power spectrum, only
an unnormalised correlation. Skipped if the file already exists.

**Stage 2 — `02_bininfo.py`.** Grids the UAPS file with `nrel = -1` over channels
`0 … Nrea−1`, because in a 2-D simulated file each channel is an independent realisation
and `mkbin` averages `V·V*` along that axis to get `M_g`. Writes `bin_info_{index}.npz`.

Watch the printed `data :` percentage — the fraction of grid points surviving `Mg_min` and
the `|U|` range. Near zero means those cuts are wrong for your array, and every later stage
is starved of modes.

**Stage 3 — `03_data.py`.** The data pass (`nrel = -1`) and the UAPS pass (`nrel = 0`,
which copies one realisation across all channels so the shape matches the data), each
gridded → SCF → correlated. Writes `el`, `ml`, and `cl = el/ml`.

**Stage 4 — `04_noise.py`.** `n_realisations` passes with `nrel = -2` and seed
`seed_start + i`. Each realisation grids a ~5 GB array, so it is correlated and discarded
immediately; only `(Nbin, NE)` is kept. Writes `cln`.

**Stage 5 — `05_ps.py`.** `build_essential` → `window` → `func_pk` → `X` → `binned_pk` →
`func_dT`. Builds `flag_mask.npy` from `power_spectrum.kpara_ranges` on first run and
reuses it afterwards. Prints a table of `k`, `Δ²(k)`, `2σ`, SNR and upper limits.

**All three passes share one config**, so the gridding and SCF parameters cannot drift
apart between them. That drift is silent — nothing in the estimator checks it, and a
mismatch yields a plausible-looking but meaningless spectrum. It was the main reason for
introducing the config file.

---

## Outputs

Everything under `paths.output_dir`:

```
{index}_uaps.fits          UAPS reference visibilities
bin_info_{index}.npz       ni · NI · lval · dU · gridding parameters
el_{index}.npy             data correlation      (Nbin, NE)
ml_{index}.npy             UAPS normalisation    (Nbin, NE)
cl_{index}.npy             el / ml               (Nbin, NE)
cln_{index}.npy            noise                 (Nreal, Nbin, NE)
flag_mask.npy              mode selection        (Nbin, NE)
ps_{index}.npz             kk · dk2 · dpk2 · snr · ul · pk · dpkn · sigma · …
grf/                       cached HEALPix sky maps (large)
slurm/                     generated sbatch scripts
logs/                      job stdout and stderr
```

Plot from `ps_{index}.npz` — it carries `kper`, `kpara`, `pk` and `dpkn` alongside the
binned results, plus the cosmology and noise scaling actually used.

---

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `No configuration file found` | set `PS_CONFIG`, pass `--config`, or run from the repo root |
| `paths.input_uvfits does not exist` | wrong path in the config |
| `gridding.nstokes must be [0, 1]` | cross-TGE hard-codes `GV[0]`=XX, `GV[1]`=YY |
| `observation.dnuc_mhz resolved to 0.0` | input is a 2-D simulation (`CDELT4 = 0`) — set `dnuc_mhz` explicitly |
| `missing bin_info_… — run pipeline/02_bininfo.py first` | stage order; the loader checks upstream products |
| `ModuleNotFoundError: numexpr` / `yaml` | env predates the dependency fixes |
| `AttributeError: np.bool` | running unfixed code on numpy ≥ 1.24 |
| `IndexError` inside `doscf` | running code at `535fc80`; the fix is on this branch |
| `sbatch not found` | submit from a login node, or use `--dry-run` |
| Array tasks fail but merge succeeds | it won't — `--merge` refuses to stack an incomplete set |
| `MemoryError` | lower `simulation.chunk_size`, or narrow `observation.n1`/`n2` |
| `sigma` far from 1 | too few noise realisations, or a wrong `noise_scale` |
| Empty `kk` / `the mode mask selects nothing` | `kpara_ranges` don't overlap the `kpara` available at this `NE` |
| `cl` full of `inf`/`nan` | `ml` has zeros — UAPS pass had no signal in those bins |
