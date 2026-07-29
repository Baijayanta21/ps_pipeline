# `myutils` — MWA Power Spectrum Pipeline

End-to-end reference for estimating the 21-cm power spectrum from MWA visibility data
using the **Tapered Gridded Estimator (TGE)**.

This document describes what the pipeline *currently does*, stage by stage, with exact
array shapes and the parameters each stage consumes. Per-function mathematical
derivations live in the module docstrings and the rendered PDFs under `Documentation/`.

- [1. Installation](#1-installation)
- [2. Pipeline overview](#2-pipeline-overview)
- [3. Global parameters](#3-global-parameters)
- [4. Stage reference](#4-stage-reference)
- [5. End-to-end worked example](#5-end-to-end-worked-example)
- [6. Data products and on-disk layout](#6-data-products-and-on-disk-layout)
- [7. Known issues and gotchas](#7-known-issues-and-gotchas)

---

## 1. Installation

```bash
python3 -m venv .myutils
source .myutils/bin/activate
pip install git+https://github.com/Baijayanta21/ps_pipeline.git
```

Or from a checkout, for development:

```bash
pip install -e .
```

**Dependencies:** `numpy`, `scipy`, `astropy`, `healpy`, `numba`, `numexpr`.

The `Tutorials/*.ipynb` additionally need `matplotlib` and `jupyter`, which are not
declared — install them yourself if you want to run the notebooks.

Verified working combination: Python 3.10, numpy 1.26, scipy 1.15, numba 0.62,
healpy 1.18, numexpr 2.14. See [§7](#7-known-issues-and-gotchas) for a numpy-version
caveat affecting `psfuncs.X`.

### Import style

Sub-modules must be imported by their full path — the sub-packages are
[implicit namespace packages](https://peps.python.org/pep-0420/) and `myutils/__init__.py`
does not re-export anything usable:

```python
import myutils.simvis.simvis     as sv
import myutils.tge.grid          as gd
import myutils.scf.scf           as sf
import myutils.clfuncs.clfuncs   as cfunc   # MAPS as C_l(nu_a, nu_b)
import myutils.clfuncs.correlate as corrf   # MAPS collapsed straight to C_l(dnu)
import myutils.psfuncs.psestimation as pe
```

Every module prints an "Imported …" banner at import time.

---

## 2. Pipeline overview

```
              ┌─────────────────────────────────────────────┐
   UVFITS ───▶│ 0. simvis   (optional — simulated sky)      │──▶ UVFITS
  (real or    │    sim_vis: APS→2D maps  |  PS→3D maps      │   (simulated
   template)  └─────────────────────────────────────────────┘    visibilities)
                                    │
                                    ▼
              ┌─────────────────────────────────────────────┐
              │ 1. tge.grid.grid                            │
              │    taper + grid uv-plane                    │
              │    nrel = -1 data | -2 noise | n≥0 UAPS     │
              └─────────────────────────────────────────────┘
                       │                          │
                       │  GV (UAPS)               │  GV (data / noise)
                       ▼                          │
              ┌──────────────────────┐            │
              │ 2. tge.grid.mkbin    │            │
              │    annular binning   │            │
              │    → bin_info.npz    │            │
              └──────────────────────┘            │
                       │  ni, NI, lval            ▼
                       │              ┌─────────────────────────────────┐
                       │              │ 3. scf.doscf   (optional)       │
                       │              │    smooth-component filtering   │
                       │              └─────────────────────────────────┘
                       │                          │
                       └──────────┬───────────────┘
                                  ▼
              ┌─────────────────────────────────────────────┐
              │ 4. clfuncs — cross-TGE correlation          │
              │    corr(data) / corr(UAPS)  → C_l           │
              │    → C_l(nu_a,nu_b) → C_l(dnu_bar,dnu)      │
              │    → C_l(dnu)                               │
              └─────────────────────────────────────────────┘
                                  ▼
              ┌─────────────────────────────────────────────┐
              │ 5. psfuncs — power spectrum estimation      │
              │    build_essential → cosmology, k-vectors   │
              │    func_pk         → P(k_perp, k_para) MLE  │
              │    X               → X statistic            │
              │    binned_pk       → P(k)                   │
              │    func_dT         → Δ²(k), SNR, limits     │
              └─────────────────────────────────────────────┘
```

Three separate passes through stages 1–4 are needed for a full analysis, because the
estimator is a *ratio* and the errors come from simulations:

| Pass | `nrel` | Purpose |
| --- | --- | --- |
| **Data** | `-1` | The measurement. Numerator of the `C_l` ratio. |
| **UAPS** | `n ≥ 0` | Unit angular power spectrum. Denominator (normalisation `M_g`) of the `C_l` ratio, **and** the input to `mkbin`. |
| **Noise** | `-2` | Many realisations of unit Gaussian noise → noise covariance `covi` and `δP_N`. |

---

## 3. Global parameters

### 3.1 Gridding / TGE

| Name | Meaning | Typical MWA value |
| --- | --- | --- |
| `n1`, `n2` | First / last frequency channel (inclusive) | `0`, `767` |
| `Umax` | Baselines with \|**U**\| > `Umax` are discarded | `250` λ |
| `FWHM` | Primary-beam FWHM, **degrees** | `23` |
| `f` | Tapering parameter; `f ≤ 1` tapers harder | `0.6` |
| `Flag` | Apply the UVFITS weight-based flags (`weight ≤ 0` → 0) | `True` for data |
| `nstokes` | Polarisation indices to grid (0 = XX, 1 = YY) | `[0, 1]` |
| `nrel` | Data mode selector — see table above | `-1` / `-2` / `0` |
| `seed` | RNG seed, noise mode only | e.g. `7200` |

Derived internally (returned as `info`):

- `theta_0 = 0.6 · FWHM`, `theta_w = f · theta_0`, `theta_eff = f·theta_0/√(1+f²)`
- `dU = √(ln 2) / (π · theta_eff · sf)` with sampling factor `sf = 2`
- `Nm = 3·sf = 6` — convolution half-width in grid cells; `Ng = ⌈Umax/dU⌉ + Nm`;
  grid side `Nu = 2·Ng + 1`
- The returned array is trimmed by `2·Nm = 12` cells on each side of both uv axes,
  so the usable baseline range is slightly smaller than `Umax`.

### 3.2 Binning

| Name | Meaning | Typical |
| --- | --- | --- |
| `Nbin` | Number of annular \|**U**\| bins | `20` |
| `binUmin`, `binUmax` | \|**U**\| range used for binning (λ) | `6.0`, `220` |
| `Mg_min` | Reject grid points whose UAPS self-correlation `M_g` is below this | `0.01` |

### 3.3 SCF

| Name | Meaning | Typical |
| --- | --- | --- |
| `SM` | Smoothing scale in MHz | `1`, `2`, `3` |
| `window` | `'hann'`, `'hamming'`, `'blackman'`, `'kaiser'` (β=14) | `'hann'` |
| `method` | `'auto'`, `'direct'`, `'fft'` | `'fft'` |
| `scf.dnuc` | Module-level channel width in MHz used to convert `SM` → channels | `0.04` |

`NW = int(SM / dnuc)` channels are removed from **each** end of the frequency axis.

### 3.4 Power spectrum

| Name | Meaning | Typical |
| --- | --- | --- |
| `nuc` | Central frequency, MHz | `154.255` |
| `dnuc` | Channel separation, MHz | `0.04` |
| `NE` | Number of frequency separations used | `668` |
| `model` | `'Planck18'` or `'FlatLambdaCDM'` | `'Planck18'` |
| `NBin` | Number of logarithmic `k` bins | `5`–`8` |
| `flag_mask` | 2-D `(len(kper), len(kpara))` 0/1 mask of modes to keep | built by hand |

### 3.5 Simulation (set via `builtins`)

`simvis` reads three parameters out of the `builtins` namespace, so they must be assigned
there before calling `sim_vis`:

```python
import builtins as blt
blt.nside      = 512   # HEALPix resolution
blt.Nrea       = 100   # number of sky realisations (2D mode)
blt.chunk_size = 400   # baselines processed per matrix multiply
```

`chunk_size` controls peak RAM: the phase matrix is `12·nside²/2 × chunk_size` complex128.
Rough guidance at `nside = 512`, `Nrea = 100`: 32 GB → 300–400, 16 GB → 200, less → 100.

---

## 4. Stage reference

### Stage 0 — `myutils.simvis.simvis` (optional)

Simulates visibilities into a UVFITS file, reusing the baseline distribution, pointing
(`CRVAL6`/`CRVAL7`), central frequency (`CRVAL4`), channel count (`NAXIS4`) and channel
width (`CDELT4`) of an input UVFITS.

```python
sv.sim_vis(uvinfits, uvoutfits, skysimtype, seed=None, apsfunc=aps, psfunc=ps)
```

| `skysimtype` | Sky signal | Output layout |
| --- | --- | --- |
| `'2D'` | `Nrea` independent GRF realisations at `nuc` from an **angular** power spectrum `apsfunc(l)` | realisations written along the frequency axis; channels `Nrea+1 … Nc` untouched; `CDELT4` set to `0.0` |
| `'3D'` | one frequency-dependent GRF cube from a **3-D** power spectrum `psfunc(k)` | all `Nc` channels |

- `lmax = 3·nside − 1`; the monopole is always removed.
- `Nrea` must not exceed the channel count of the input file.
- 3-D mode caches the GRF at `./grf/grf_{nside}_{seed}.npy` and reuses it on the next run.
- The same simulated visibility is written to **every** polarisation, so XX and YY are
  identical in simulated files.
- `sim_vis_hdul(hdulist, …)` does the same work in place on an open HDU list if you want
  to post-process before writing.

Built-in models — override with your own vectorised callables:

```python
aps(l)  = 100 · l**(-2)                # mK²
uaps(l) = 1.0                          # unit APS
ps(k)   = 1.0 · k**(-2), 0 at k = 0    # mK² Mpc³
```

Useful pieces you can call directly: `grf`, `sky3dgrf`, `allskysim`, `beammwa`, `pbgen`,
`hat_n`, `dot_cal_superfast`, `calculate_phase`, `visgen_mwa_multi`. The MWA primary beam
is modelled as a product of two `sinc²` factors for a 4 m square aperture, zenith pointed,
integrated over the upper hemisphere only.

### Stage 1 — `myutils.tge.grid.grid`

```python
GV, info = gd.grid(infile, n1, n2, nrel, Umax, FWHM, f, Flag, nstokes, seed=None)
```

Convolves the visibilities with the tapering kernel
`w̃(U) = π·θ_w²·exp(−π²·θ_w²·U²)` and accumulates onto a square uv grid.

**Returns**

- `GV` — `complex128`, shape `(len(nstokes), Nu−4·Nm, Nu−4·Nm, n2−n1+1)`
  → `(pol, u, v, channel)`. For MWA with `Umax=250, FWHM=23, f=0.6`: `(2, 457, 457, 768)`.
- `info` — `(dU, Umax, FWHM, f)`.

**Behaviour notes**

- Baselines are taken at the reference frequency `CRVAL4` and held fixed across channels
  (**no baseline migration**).
- Every baseline is used twice: `V(−U) = V(U)*`. Everything is folded into the upper half
  plane before gridding.
- Each baseline contributes only to grid points inside a `(2·Nm+1)²` box around its
  nearest grid point.
- `Flag=True` zeroes samples whose UVFITS weight is `≤ 0`.

**Cost** ≈ 2 minutes for 768 channels, 2 polarisations on one MWA snapshot.

### Stage 2 — `myutils.tge.grid.mkbin`

```python
gd.mkbin(GV, dU, Umax, FWHM, f, binUmax, binUmin, Nbin, Mg_min, outf)
```

Run this on the **UAPS** gridded visibilities (single polarisation, e.g. `GV[0]`, with
realisations stacked along the frequency axis). It decides which grid points are usable
and which annulus each belongs to, then writes `{outf}.npz`.

**Grid the UAPS file twice, for two different purposes.** `mkbin` needs the realisations
to average over, so grid channels `0 … Nrea−1` with `nrel = -1` — in a 2-D simulated file
each channel *is* an independent realisation, and `mkbin` averages `V V*` along that axis
to get `M_g`. The normalisation array `ml` used in stage 4 instead needs the same channel
count as the data, so grid the same file with `nrel = n ≥ 0`, which copies realisation `n`
across all channels. Passing the `nrel = 0` array to `mkbin` still runs, but every channel
is then identical and `M_g` is a single realisation with no averaging.

**Contents of `{outf}.npz`**

| Key | Shape | Meaning |
| --- | --- | --- |
| `Nbin`, `dU`, `Umax`, `FWHM`, `f` | scalars | parameters echoed back |
| `NI` | `(Nu, Nu)` | per-grid-point status: `≥0` = bin index (usable), `−1` = outside `[binUmin, binUmax]`, `−2` = `M_g ≤ Mg_min` |
| `ix`, `iy` | `(Ngood,)` | u- and v-index relative to grid centre, usable points only |
| `ni` | `(Ngood,)` | bin index `0 … Nbin−1`, usable points only |
| `lval` | `(Nbin,)` | effective multipole per bin, `ℓ_a = Σ M_g ℓ_g / Σ M_g` with `ℓ_g = 2π\|U\|_g` |

`M_g` is the UAPS self-correlation `⟨V_cg V*_cg⟩` averaged over the realisation
(frequency) axis. Downstream you select points with `mask = NI >= 0` and pass `ni`.

### Stage 3 — `myutils.scf.scf.doscf` (optional)

```python
GV_filtered = sf.doscf(GV, SM, window='hann', method='auto')
```

Removes the spectrally smooth (foreground-dominated) component:

```
V_S = (V_cg * H) / (U * H)          U(nu) = 0 where flagged, 1 otherwise
V_F = V_cg − V_S
```

The division by the convolved flag mask restores the correct normalisation in the presence
of flagged channels. Real and imaginary parts carry independent flag masks (a sample counts
as flagged in a component when that component is exactly zero) and are normalised
separately. Channels flagged in the input have their smooth model forced to zero, so the
filtered output reproduces the input there rather than inventing a value.

Output shape is the input shape with the last axis shortened by `2·NW`,
`NW = int(SM/0.04)` — e.g. `768 → 668` for `SM = 2` MHz.

The suppression is expected in the band `[k_∥]_F/2 ≤ k_∥ ≤ [k_∥]_F` with
`[k_∥]_F = 2π / (r′·N·Δν_c)`.

### Stage 4 — `myutils.clfuncs`

Two entry points, both applying the **cross-TGE** estimator

```
corr(nu_a, nu_b) = Re[ V_cg^XX(nu_a)·V_cg^YY*(nu_b) + V_cg^YY(nu_a)·V_cg^XX*(nu_b) ]
```

Cross-correlating XX with YY removes the noise bias that a self-correlation would carry.
Both require **exactly two polarisations** in axis 0, ordered XX then YY, and grid points
flattened along axis 1:

```python
BIN  = np.load('bin_info_1000007200.npz')
Nbin = int(BIN['Nbin']); ni = BIN['ni']; mask = BIN['NI'] >= 0

corr2d = cfunc.correlate(GV[:, mask], ni, Nbin)   # (Nbin, NC, NC)  — keeps nu_a, nu_b
corr1d = corrf.correlate(GV[:, mask], ni, Nbin)   # (Nbin, NC)      — collapses to dnu
```

Use `clfuncs.correlate` when you want the full `C_l(ν_a, ν_b)` matrix (needed if you care
about the `ν̄` dependence); use `clfuncs.correlate` from the `correlate` module when you
only need `C_l(Δν)` — it accumulates directly into separations and is cheaper in memory
(`Nbin × NC` instead of `Nbin × NC²`).

**Normalisation — do not skip.** `correlate` returns an *unnormalised* sum over grid
points. The estimator is a ratio: run stage 4 on both the data and the UAPS gridded
visibilities and divide.

```python
el = corr_from_data      # numerator
ml = corr_from_uaps      # denominator (the M_g normalisation)
cl = el / ml             # this is C_l
```

This division is **not** provided by the package — it is done by hand in
`Tutorials/clfuncs.ipynb`. The config-driven `pipeline/03_data.py` now performs it. See
[§7](#7-known-issues-and-gotchas).

**Do not apply SCF to the UAPS pass.** The UAPS array is gridded with `nrel = n ≥ 0`, which
copies one realisation across every channel, so `V_cg` is *constant in frequency*. The
smooth component of a constant is that same constant, so the filtered residual is
identically zero and `cl = el/ml` becomes infinite. `M_g` is a per-grid-point geometric
normalisation rather than a spectral one, so it should not be filtered; `ml` is instead
truncated to the number of frequency separations the filtered data has. This is what
`scf.apply_to_uaps: false` (the default) does.

**Coordinate transforms**

```python
maps1 = cfunc.cl_dnu_nubar(cl)      # (Nbin, NC, NC) → (Nbin, NC, 2·NC−1) as (dnu, nubar)
maps2 = cfunc.cl_dnu(maps1)         # average over nubar → (Nbin, NC)
maps2 = cfunc.cl_dnu_nua_nub(cl)    # both steps in one call
```

Unfilled `(Δν, ν̄)` cells are `NaN`, and `cl_dnu` uses `np.nanmean`, so the `ν̄` average
automatically covers only the populated cells.

### Stage 5 — `myutils.psfuncs.psestimation`

**5a. Cosmology and k-vectors**

```python
r, rp, fac, vfac, kper, kpara = pe.build_essential(nuc, dnuc, NE, lval, model='Planck18')
```

| Output | Meaning |
| --- | --- |
| `r` | comoving distance at `nuc`, Mpc |
| `rp` | `dr/dν` at `nuc`, Mpc/MHz — `(c/H(z))·(1+z)²/1420` |
| `fac` | foreground-wedge slope, `r/(r′·nuc)`; wedge boundary is `k_∥ = fac·k_⊥` |
| `vfac` | volume factor `r²·r′·Δν_c·(NE−1)` applied in the cosine transform |
| `kper` | `ℓ/r`, shape `(len(lval),)` |
| `kpara` | `π·m/(r′·Δν_c·(NE−1))`, `m = 0 … NE−1` |

**5b. Cylindrical power spectrum (MLE)**

```python
w    = pe.window(NE)                     # Blackman–Nuttall, normalised to dnu = 0
covi = 1.0 / np.std(cl_noise, axis=0)**2 # inverse noise variance from noise realisations
pk   = pe.func_pk(cl, w, covi, vfac)     # (..., len(lval), NE)
```

For each `ℓ` bin, `P_a = (Aᵀ N⁻¹ A)⁻¹ Aᵀ N⁻¹ · [W·C_ℓa]`, with `A` the cosine-transform
matrix from `calc_A(NE, NE)` and `N` assumed diagonal in `Δν`. The Blackman–Nuttall window
makes `W·C_l(Δν)` periodic so the transform does not ring off the band edges.

**5c. Noise level and the X statistic**

```python
pkn   = pe.func_pk(cl_noise, w, covi, vfac)
dpkn  = np.std(pkn, axis=0)
dpkn *= N_nights**2                       # scale single-realisation noise to the real data
X, mu, sigma = pe.X(pk, dpkn, flag_mask)  # X = P / dP_N over the selected modes
```

`sigma` calibrates the noise: the true error is `dpk = sigma · dpkn`. If the noise model
were perfect and the field signal-free, `X` would have unit variance; `sigma > 1` means the
simulated noise underestimates the real scatter.

**5d. Spherical binning and the final observable**

```python
kk, ppk, dppk   = pe.binned_pk(kper, kpara, pk, sigma*dpkn, NBin, flag_mask)
dk2, dpk2, snr, ul = pe.func_dT(kk, ppk, dppk)
```

`binned_pk` bins `k = √(k_⊥² + k_∥²)` logarithmically over the masked modes using the
inverse-variance (minimum-variance) estimator, and **drops empty bins** — so the returned
length is `nzNBin ≤ NBin`. It prints the per-bin mode count.

`func_dT` returns `Δ²(k) = k³P(k)/2π²`, its `2σ` uncertainty, the SNR `P/δP`, and the
upper limit — `Δ² + 2σ` where `Δ² > 0`, else `2σ` alone (HERA 2020 convention).

**Building `flag_mask`.** Modes inside the foreground wedge and at low `k_∥` must be
excluded by hand:

```python
flag_mask = np.zeros((kper.size, kpara.size), dtype='int')
ks  = np.array([0.135, 0.228, 0.36, 0.5, 0.72, 0.8, 0.92, 1.09, 1.17, 1.39])
ksv = np.array([0, 0, 1, 2, 2, 2])
for ii in range(len(ksv)):
    for jj in range(ksv[ii], len(ksv) - 1):
        m = (kpara >= ks[2*jj]) * (kpara <= ks[2*jj+1])
        flag_mask[ii, m] = 1
np.save('flag_mask.npy', flag_mask)
```

---

## 5. End-to-end worked example

> **In practice, use the config-driven runner instead** — `config.yaml` plus
> `pipeline/01_uaps.py … 05_ps.py`, described in [`RUNNING.md`](RUNNING.md). It wraps
> exactly the calls below while keeping the gridding and SCF parameters identical across
> the three passes, which is the failure mode this example is most likely to introduce if
> you copy it by hand. The example is kept because it shows the raw API with nothing in
> the way.

One MWA snapshot, `1000007200.fits`, 768 channels, XX+YY, 2 MHz SCF.

```python
import numpy as np
import builtins as blt
import myutils.simvis.simvis     as sv
import myutils.tge.grid          as gd
import myutils.scf.scf           as sf
import myutils.clfuncs.correlate as corrf
import myutils.psfuncs.psestimation as pe

DATA  = '/home/MWA_data/1000007200.fits'
INDEX = 7200

# ---- gridding parameters ------------------------------------------------
n1, n2  = 0, 767
Umax    = 250
FWHM    = 23
f       = 0.6
nstokes = [0, 1]          # XX, YY

# ---- Step 0: UAPS reference file ---------------------------------------
blt.nside, blt.Nrea, blt.chunk_size = 512, 100, 400
sv.sim_vis(DATA, f'{INDEX}_uaps.fits', skysimtype='2D',
           seed=1, apsfunc=sv.uaps)

# ---- Step 1: grid the three passes -------------------------------------
GV,  info = gd.grid(DATA, n1, n2, -1, Umax, FWHM, f, True,  nstokes)
GVU       = gd.grid(f'{INDEX}_uaps.fits', n1, n2, 0, Umax, FWHM, f, True, nstokes)[0]
GVN       = gd.grid(DATA, n1, n2, -2, Umax, FWHM, f, True, nstokes, seed=INDEX)[0]
dU, Umax, FWHM, f = info

# ---- Step 2: binning information (UAPS, realisations along freq) -------
# separate grid of the UAPS file: channels 0..Nrea-1 are the Nrea realisations
GVUr = gd.grid(f'{INDEX}_uaps.fits', 0, 99, -1, Umax, FWHM, f, False, [0])[0]
gd.mkbin(GVUr[0], dU, Umax, FWHM, f,
         binUmax=220, binUmin=6.0, Nbin=20, Mg_min=0.01,
         outf=f'bin_info_{INDEX}')
del GVUr

BIN  = np.load(f'bin_info_{INDEX}.npz')
Nbin = int(BIN['Nbin']); ni = BIN['ni']; mask = BIN['NI'] >= 0
lval = BIN['lval'].astype(int)

# ---- Step 3: smooth-component filtering --------------------------------
SM = 2.0
GVf  = sf.doscf(GV,  SM, window='hann', method='fft')   # 768 → 668 channels
GVUf = sf.doscf(GVU, SM, window='hann', method='fft')
GVNf = sf.doscf(GVN, SM, window='hann', method='fft')

# ---- Step 4: correlate and normalise -----------------------------------
el = corrf.correlate(GVf[:,  mask], ni, Nbin)    # data      (Nbin, 668)
ml = corrf.correlate(GVUf[:, mask], ni, Nbin)    # UAPS      (Nbin, 668)
en = corrf.correlate(GVNf[:, mask], ni, Nbin)    # noise     (Nbin, 668)

cl  = el / ml                                    # C_l(dnu)
cln = en / ml                                    # repeat over many noise seeds
                                                 # and stack → (Nreal, Nbin, 668)

# ---- Step 5: power spectrum -------------------------------------------
nuc, dnuc, NE = 154.255, 0.04, 668
r, rp, fac, vfac, kper, kpara = pe.build_essential(nuc, dnuc, NE, lval)

w    = pe.window(NE)
covi = 1.0 / np.std(cln, axis=0)**2
pk   = pe.func_pk(cl,  w, covi, vfac)
pkn  = pe.func_pk(cln, w, covi, vfac)
dpkn = np.std(pkn, axis=0) * 20**2               # 9 nights → factor 20

fm = np.load('flag_mask.npy')
Xs, mu, sigma      = pe.X(pk, dpkn, fm)
kk, ppk, dppk      = pe.binned_pk(kper, kpara, pk, sigma*dpkn, NBin=8, flag_mask=fm)
dk2, dpk2, snr, ul = pe.func_dT(kk, ppk, dppk)
```

Note that `cln` needs **many** noise realisations (the tutorials use 50) to give a stable
`np.std`; loop stage 1 with `nrel=-2` over different `seed` values and stack the results.

---

## 6. Data products and on-disk layout

| Product | Format | Typical size |
| --- | --- | --- |
| `GV_{index}.npy` | complex128 `(2, 457, 457, 768)` | ~5 GB |
| `GV_{index}_Noise_{seed}.npy` | as above, per seed | ~5 GB × Nreal |
| `GV_{index}_UAPS_{nrel}.npy` | as above | ~5 GB |
| `bin_info_{index}.npz` | int/float arrays | KB |
| `cldnu_{index}.npy` | `(Nbin, NC)` float | KB–MB |
| `cldnur_noise_{index}.npy` | `(Nreal, Nbin, NC)` float | MB |
| `./grf/grf_{nside}_{seed}.npy` | float64 `(Nc, 12·nside²)` | GB |

Gridded visibility arrays dominate the footprint. `(2, 457, 457, 768)` complex128 is
5.1 GB per pass; a 50-realisation noise run is ~250 GB if fully materialised. Prefer
correlating each realisation as it is produced and keeping only `(Nbin, NC)` outputs.

Sample outputs for `index = 7200` are checked into `Tutorials/others/`.

---

## 7. Known issues and gotchas

Found by reading the code at `EasyRun_PS_pipeline` HEAD `535fc80` and confirming each one
by running it. Items marked **[fixed]** have since been repaired on this branch and are
kept here so the failure mode is on record — if you have results produced before the fix,
they are affected. Items marked **[open]** are still present.

### 7.1 **[fixed]** `psfuncs.X` crashes on numpy ≥ 1.20

`psestimation.py:645` calls `np.bool(flag_mask)`. `np.bool` was removed in numpy 1.24
(deprecated in 1.20), so this raises `AttributeError` on any supported numpy. The rest of
the module uses the working idiom `flag_mask == 1`. Fixed by switching to
`X[..., flag_mask == 1]`. The same call appeared in the masked-plot cell of
`Tutorials/ps.ipynb` and was changed to `.astype(bool)`. The stored notebook outputs show
this cell having run successfully, so it was last executed on numpy < 1.24.

### 7.2 **[fixed]** SCF was broken outright by its most recent commit

`535fc80` ("Zero flagged modes in GV_SCF", 26 Jul 2026) added

```python
(GV_SCF.real)[NormR[NW:-NW]==0.0] = 0.0
(GV_SCF.imag)[NormI[NW:-NW]==0.0] = 0.0
```

`NormR[NW:-NW]` slices **axis 0** — the polarisation axis — not the frequency axis. With
the usual 1 or 2 polarisations and `NW ≥ 25`, `NW:-NW` is an empty slice, so the boolean
mask has shape `(0, Nu, Nv, nc)` against a `(Npol, Nu, Nv, nc−2·NW)` target:

```
IndexError: boolean index did not match indexed array along dimension 2;
            dimension is 30 but corresponding boolean dimension is 40
```

`doscf` therefore raises for **every** realistic input at `535fc80`. The docstring example
showing `GVr.shape = (2, 457, 457, 668)` predates the commit. Fixed by slicing the last
axis: `NormR[..., NW:-NW]`.

Anything that ran SCF successfully was run before `535fc80`, and so also predates the
imaginary-normalisation fix in §7.2b below.

### 7.2b **[fixed]** SCF normalised the imaginary part with the real part's flag mask

`scf.py:206-207` computes both normalisation convolutions from `NormR`:

```python
NR = convolve(NormR, win_expand, ...)
NI = convolve(NormR, win_expand, ...)   # should be NormI
```

`NormR` and `NormI` differ wherever exactly one of `Re V` / `Im V` is zero, so the
imaginary component of the smooth model was normalised with the wrong weights. Fixed by
convolving `NormI` for the imaginary normalisation.

### 7.3 **[fixed]** `grid` indexed `GV` by polarisation ID rather than by position

`grid.py:406-447` loops `for st in nstokes` and writes `GV[st]`, but `GV` was allocated
with first dimension `len(nstokes)`. This is only correct when `nstokes` starts at 0 and is
contiguous. `nstokes=[1]` (YY only) raised `IndexError`; `nstokes=[0,2]` wrote the wrong
slot. Fixed by enumerating `nstokes` and indexing `GV` by position; the dead
`st = nstokes` line was removed. Verified that `nstokes=[1]` now reproduces slot 1 of a
`nstokes=[0,1]` run exactly.

### 7.4 **[fixed]** Channel count leaked between `correlate` calls via `builtins`

`clfuncs.py` and `correlate.py` pass the channel count to their `@njit` kernels by
assigning `builtins.nc`. numba treats module globals as **compile-time constants**: the
value present at first compilation is frozen into the machine code. Calling `correlate`
again with a different number of channels in the same session silently reuses the old
count — no error, just wrong numbers. Reproduced:

```python
ni = np.zeros(3, dtype=np.int64)
corrf.correlate(np.ones((2, 3, 4), dtype=complex), ni, 1)   # prints "Channels : 4"
c = corrf.correlate(np.ones((2, 3, 6), dtype=complex), ni, 1)
# prints "Channels : 4" again; c[0] == [24. 18. 12.  6.  0.  0.]
# correct answer for 6 channels is  [36. 30. 24. 18. 12.  6.]
```

The returned array has the right *shape*, so nothing downstream notices: separations
beyond the frozen channel count come back as zeros and the rest are computed over too few
channel pairs. This bit whenever SCF smoothing scale, channel range, or dataset changed
within one session — exactly the sweep a parameter study performs. It also polluted the
global namespace for every other module. Fixed in both `clfuncs.py` and `correlate.py` by
passing `nc` as an explicit kernel argument and dropping the `builtins` import.

**If you ran multiple `correlate` calls in one session before this fix, re-run them.**

### 7.5 **[open]** The `C_l` normalisation step is not in the package

`correlate` returns an unnormalised sum; turning it into `C_l` requires dividing by the
UAPS correlation (`cl = el / ml`). That division exists only as a cell in
`Tutorials/clfuncs.ipynb` operating on hand-named `.npy` files. There is no function that
takes data and UAPS and returns `C_l`, which makes the most physically important step of
the pipeline the easiest one to get wrong or forget.

### 7.6 **[fixed]** `numexpr` was an undeclared dependency

`simvis.py:345` imports `numexpr`, which was absent from `pyproject.toml`, so a clean
`pip install` produced a package whose simulation module could not be imported. Added to
`dependencies`.

### 7.7 **[open]** `simvis` parameters travel through `builtins`

`nside`, `Nrea`, `chunk_size`, and internally `r`, `rp`, `nc`, `dnu`, `nu_c`, `b`, `C`,
`KB` are read from / written to `builtins`. Consequences:

- Forgetting to set `blt.nside` gives `NameError` from deep inside the call stack.
- `sky3dgrf` cannot be called standalone despite its docstring showing exactly that — it
  needs `r`, `rp`, `nc`, `dnu` to have been injected by `allskysim`/`sim_vis_hdul` first.
- Nothing is reentrant or thread-safe.

### 7.8 **[open]** `requires-python = ">=3.7"` is inaccurate

The pinned dependency set (numba 0.62, numpy ≥ 1.26, healpy 1.18) requires Python ≥ 3.9,
and the tutorials use the `f"{x = }"` syntax from 3.8.

### 7.9 **[partly fixed]** Build artefacts are shipped in the wheel

Because `[tool.setuptools.packages.find]` defaults to namespace-package discovery, the
built wheel includes `__pycache__/*.cpython-39.pyc` and `.ipynb_checkpoints/` directories
from the source tree — including stale `__init__.cpython-39.pyc` files for `__init__.py`
modules that no longer exist (they were deleted in `1f22adc`; the sub-packages have worked
as PEP 420 namespace packages ever since). A `.gitignore` has been added, but the already
committed `__pycache__` and `.ipynb_checkpoints` files need `git rm --cached`, and the
packages-find table still needs an explicit `exclude`.

### 7.10 Smaller items — all **[open]**

- **Empty-bin division.** `mkbin` computes `lval = wglg/wg` with no guard, so an annulus
  containing no surviving grid point yields `nan` plus a RuntimeWarning. `binned_pk`
  likewise divides by `sum_wk`.
- **`scf.dnuc` is a module-level constant** (`0.04` MHz). Any dataset with a different
  channel width silently gets the wrong smoothing scale. It should be an argument.
- **SCF divide-by-zero warnings.** Where an entire smoothing window is flagged, `NR`/`NI`
  are zero and `GV_SCF.real /= NR` produces `nan` plus a RuntimeWarning. The flagged-mode
  zeroing immediately overwrites those entries with `0.0`, so the result is correct, but
  the warning is noisy and masks genuine ones. Guard the division with `np.divide(...,
  where=NR!=0)` instead.
- **`func_pk` memory.** `vari` is materialised as a dense `(…, NE, NE)` array although it
  is diagonal by construction; `NE=668` with a leading realisation axis of 50 would be
  1.8 GB. Both the diagonal construction and the `(AᵀN⁻¹A)⁻¹AᵀN⁻¹` solve can be done
  without forming the dense matrix.
- **`clfuncs.correlate` mirroring loop.** Filling the symmetric half uses a Python double
  loop over `NC²` = 590k iterations; `corr + corr.transpose(0,2,1) − diag` is equivalent
  and far faster.
- **`keff` shape.** `binned_pk`'s docstring says `keff` is 1-D, but it is computed from
  `sum_wkk/sum_wk` and therefore carries any leading dimensions of `pk`.
- **Docstring typo.** `tge/grid.py` module header says baselines are rejected for
  `|U_i| < |U|_max`; the code rejects `|U_i| > Umax`.
- **Hardcoded paths.** The five near-identical Sphinx driver scripts in
  `Documentation/Run/` hardcode `/home/baijayanta/git_package/ps_pipeline/...`, so they
  only run on their author's machine. `Documentation/Run/docs/build/` (LaTeX intermediates,
  a doctree pickle) is committed.
- **No tests.** There is no test suite, no CI, and no numerical regression fixture, so
  none of the above would have been caught automatically.
