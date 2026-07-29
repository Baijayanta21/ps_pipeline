# Mode selection and published comparisons

Which $(k_\perp, k_\parallel)$ modes enter the spherical average, and what to compare
the answer against.

Code: `src/myutils/masking.py` (selection), `src/myutils/reference.py` (published
numbers), `pipeline/05_ps.py` (applies it), `Tutorials/masking.ipynb` (explore it).

---

## 1. The published strategy

The pipeline's default mask is the mode selection of **TTGE III** — Sarkar, Elahi,
Choudhuri, Bharadwaj, Chatterjee, Bhattacharyya, Sethi & Patwa, *"The Tracking
Tapered Gridded Estimator for the 21-cm power spectrum from the MWA drift scan
observations – III. Improved upper limits at z = 8.2 from multiple pointings"*,
MNRAS 000, 1–17 (2026), [arXiv:2604.24144](https://arxiv.org/abs/2604.24144), in
`Documentation/`.

### It is rescaled to your band automatically

The published numbers are absolute $k$ values, valid only near $\nu_c = 154.2$ MHz.
**They are not used literally.** `myutils.masking.scale_paper` moves every limit to
whatever band the header reports, so the same config works for an LF and an HF
observation and the two remain comparable. Each limit moves by the physics that set
it:

| limit | scales as | why |
|---|---|---|
| `kpara_max`, the tabulated band edges | $1/r'$ | fixed delay scale; the streaks sit at a fixed instrumental delay |
| `kpara_min` | $1/(r'\,\mathrm{SM})$ | it is the SCF floor, so it also moves with `scf.SM_mhz` |
| `kperp_min` | $\nu/r$ | tracks the shortest baseline, with `binUmin` scaled as $\nu$ |
| `kperp_max` | *derived* | chosen so the wedge boundary stays near the floor: $\approx C\,k_{\parallel,\min}/\mathrm{fac}$ |

`kperp_max` is derived rather than scaled because that is the paper's own reasoning —
$C$ comes from its choice and reproduces 0.045 exactly at its band. A shallower wedge
at higher frequency therefore buys you longer baselines.

Worked example, same config at two bands:

| | 154.2 MHz (paper) | 182.4 MHz |
|---|---|---|
| $z$ | 8.21 | 6.78 |
| $r'$ | 16.930 | 15.557 |
| `fac` | 3.519 | 3.078 |
| `kpara_min` | 0.1350 | **0.1469** |
| `kpara_max` | 1.3990 | **1.5227** |
| `kperp_min` | 0.0070 | **0.0087** |
| `kperp_max` | 0.0450 | **0.0560** |

Doubling `scf.SM_mhz` to 4 halves the floor to 0.0675, as it should.

Stage 5 prints a `rescaled` line whenever the run's band differs from the reference,
so the log always records which numbers were actually applied.

The rest of the paper's setup matches this pipeline as-is:

| | paper | this config |
|---|---|---|
| channel width | 40 kHz | 40 kHz |
| SCF smoothing | $N = 50$, 2 MHz | `scf.SM_mhz: 2.0` → `NW = 50` |
| $k_\perp$ grid | 0.007–0.146, 20 bins | `binning.Nbin: 20` |
| declination | −26.7° | −26.7° |

### The three cuts

```
k_perp  <= 0.045 Mpc^-1
k_para  >= 0.135 Mpc^-1
k_para  <= 1.399 Mpc^-1
```

* **$k_\perp \le 0.045$** — beyond that the foreground wedge boundary
  $[k_\parallel]_H = \mathrm{fac}\cdot k_\perp$ rises *above* the SCF smoothing
  scale. At those baselines filtering cannot clean the wedge, because the
  contamination sits at delays SCF does not touch. The paper: *"at large baselines,
  the visibilities … exceeds the smoothing scale of $k_\parallel = 0.135$"*, hence
  *"we therefore restrict the subsequent analysis to $k_\perp \le 0.045$"*.

* **$k_\parallel \ge 0.135$** — below this SCF *itself* removes power, so these
  modes are avoided rather than trusted. This is the one number worth checking if
  you change anything: it is set by the 2 MHz smoothing scale, and it is empirical
  — the paper validates against simulations that signal loss is not significant
  above it, rather than deriving it analytically. **Change `scf.SM_mhz` and this
  floor must be re-derived.**

* **$k_\parallel \le 1.399$** — above that the estimate is noise-dominated and any
  residual contamination is buried anyway.

### The streak mask

Inside that box the paper further masks contaminated modes: a periodic pattern of
horizontal streaks at $\Delta k_\parallel = 0.290$ Mpc⁻¹ (their Figure 5). The
per-$k_\perp$ windows encode it:

```yaml
kpara_ranges:      [0.135, 0.228, 0.36, 0.5, 0.72, 0.8, 0.92, 1.09, 1.17, 1.39]
kpara_start_index: [0, 0, 1, 2, 2, 2]
```

Five $k_\parallel$ bands, and the first band used rises with $k_\perp$. The
six-entry `kpara_start_index` also enforces the $k_\perp \le 0.045$ ceiling on its
own, because the tabulation covers only the first six $k_\perp$ bins.

These bands were already in the repository, carried over from the tutorials without
attribution. They are not ad hoc — they are this published mask.

---

## 2. Using it

### The wedge is not optional

**Every selection excludes the foreground wedge.** There is no switch, and an empty
mask spec still gives a wedge cut. A spherical average that includes the wedge
measures foregrounds, not the 21-cm signal, so it is not something to leave off by
accident.

Only `wedge_buffer` is tunable — extra $k_\parallel$ above the horizon line, useful
because the instrument's frequency response smears the wedge edge.

`use_wedge: false` is a hard error rather than a silent no-op, so old configs that
carry it get told instead of quietly running a different analysis. `use_wedge: true`
is accepted and ignored.

### Everything else is a switch

```yaml
power_spectrum:
  mask:
    preset: paper        # or paper_box, or wedge_only

    wedge_buffer: null   # always-on cut; only the buffer is tunable

    use_limits: null     # false -> ignore the four k limits
    kperp_min: null
    kperp_max: null
    kpara_min: null
    kpara_max: null

    use_tabulated: null  # false -> drop the per-k_perp streak windows
```

**With a preset, `null` means "inherit".** Any real value overrides — including
`false`, which is an override and not an absence. Without a preset, `null` means
unconstrained, and `use_limits` defaults to on whenever any limit is actually set.

| preset | wedge | limits | windows | modes (with_SCF) |
|---|---|---|---|---|
| `paper` | always | ✓ | ✓ | 488 |
| `paper_box` | always | ✓ | — | 1,089 |
| `wedge_only` | always | — | — | 12,570 |

`none` is kept as an alias for `wedge_only`. It used to mean "no selection at all",
which is no longer possible.

Stage 5 prints the resolved preset and the survivor count per constraint, so the
log records what was applied rather than what was typed:

```
mode mask   : built from config (preset: paper)
  wedge (always): excluding k_par <= 3.519 * k_perp
  optional      : k limits, tabulated windows
  modes available              13,360
  outside the wedge [always]   12,570
  inside k limits              1,092
  in tabulated windows         488
  selected (all combined)      488 (3.7%)
  k_perp spanned               0.0078 - 0.0448 Mpc^-1
  k_para spanned               0.1391 - 1.3840 Mpc^-1
```

To depart from it, work in `Tutorials/masking.ipynb`: it recomputes everything
downstream of the mask from the stored `pk` and `dpkn`, so nothing re-runs and the
sweep is instant. Exclusion boxes are notebook-only; save the array if you need them.

### Effect on this run

| variant | modes | $\sigma$ |
|---|---|---|
| `with_SCF` | 488 | **1.35** |
| `without_SCF` | 561 | 8.85 |

$\sigma$ near 1 means the simulated noise describes the observed scatter. Under the
paper mask `with_SCF` reaches 1.35, the closest to 1 any mask has produced here; the
earlier hand-built mask gave 1.75. `without_SCF` at 8.85 is the expected failure —
without SCF the low-$k_\parallel$ foreground residual is still in the data.

`NBin` is 6, not 8: the mask keeps ~500 modes over a narrow $k$ range, and 8 bins
leaves one empty (NaN, silently dropped from the table).

---

## 3. Comparing against published limits

`myutils.reference` holds transcribed upper limits, each carrying its source. Add
them to any spherical figure:

```yaml
plots:
  reference: [ttge3_alpha11, ttge3_case_i]
```

or in a notebook:

```python
mp.spherical_ps(d, reference='ttge3_alpha11')
mp.compare_spherical(runs, reference=['ttge3_alpha11', 'ttge3_case_i'])
print(rf.describe())            # what is available
print(rf.describe('ttge3_case_i'))
```

| name | what |
|---|---|
| `ttge3_alpha11` | TTGE III Table 1, best single pointing centre ($\alpha = 11°$) — **the like-for-like comparison**, since this pipeline processes one field |
| `ttge3_case_i` | TTGE III Table 2, 23 PCs combined incoherently — their headline |
| `ttge3_case_ii` | TTGE III Table 2, 14 PCs, tuned for higher $k$ |
| `ttge3_headline` | just the two abstract numbers, annotated |
| `trott2020_eor0`, `trott2020_eor2` | MWA, as quoted by TTGE III |
| `lofar2025`, `hera2023` | LOFAR and HERA, as quoted by TTGE III |

The headline published numbers:

* best single PC: $\Delta^2_{UL} = (173.13)^2$ mK² at $k = 0.161$ Mpc⁻¹
* 23 PCs combined: $\Delta^2_{UL} = (98.67)^2$ mK² at $k = 0.156$ Mpc⁻¹

Reference curves are drawn in neutral ink with their own markers and dashes, never
in a categorical hue — those identify *our* runs, and overlaying a comparison must
never repaint one. The markers also avoid `D` and `v`, which the spherical figures
already spend on "detection" and "noise dominated".

### Read the field difference before reading the gap

TTGE III finds $\alpha \lesssim 22°$ to be the relatively foreground-free part of
its drift scan, and its headline limits come from there. **This config's field is
RA 39.7°, outside that range**, between the paper's EoR0 (0°) and EoR1 (60°) — and
the paper reports EoR1 as considerably worse than EoR0/EoR2. Our limits sitting
above theirs at low $k$ is expected from the field alone; it is not by itself
evidence of a problem with the run.

The external limits are transcribed from TTGE III's own discussion of its Figure 9,
not from the original papers. Cite the originals before publishing.

---

## 4. Running more than one band from one config

Nothing in the pipeline is pinned to a frequency. Point `paths.input_uvfits` at a
different observation and every frequency-dependent quantity follows:

**Derived from the header, always** — $\nu_c$, $\Delta\nu_c$, `nchan` come from
`CRVAL4/CDELT4/NAXIS4`; then $z = 1420/\nu_c - 1$, $r$, $r'$, the wedge slope `fac`,
the volume factor, and the $k_\perp$/$k_\parallel$ grids. Baselines are converted with
the file's own channel-0 frequency, so $|U|$ in wavelengths is right by construction.

**Rescaled from a reference band** — three settings are properties of the instrument
*at a frequency* rather than of the array, so each carries the frequency it was
calibrated at:

```yaml
gridding:
  FWHM: 23
  FWHM_ref_mhz: 154.255    # beam ~ lambda      -> 19.45 deg at 182.4 MHz
binning:
  binUmin: 6.0
  binUmax: 220
  U_ref_mhz: 154.255       # |U| ~ nu           -> 7.10 - 260.2 lambda at 182.4 MHz
```

`U_ref_mhz` governs `gridding.Umax` too. Scaling $|U|$ keeps the same **antennas** in
play across bands, which is what makes two runs comparable — an unscaled limit would
silently select different baselines. Set either key to `null` to opt out and take the
numbers literally.

`simulation.nside` is raised automatically when the simulated sky can no longer reach
`binUmax` (it must satisfy $|U|_{\max} = (3\,\mathrm{nside}-1)/2\pi \ge$ `binUmax`),
rounded up to the next power of two. Cost goes as $\mathrm{nside}^2$, so the raise is
announced loudly; set `simulation.nside_auto: false` to keep the old value and lose
the outer bins instead.

Every rescaling is echoed at the top of each stage:

```
rescaled band : calibrated at 154.255 MHz -> 182.4150 MHz
                FWHM            23 ->    19.45 deg
                Umax           250 ->    295.6 lambda
                binUmin          6 ->    7.095 lambda
                binUmax        220 ->    260.2 lambda
```

At the reference frequency itself nothing is rescaled and no line is printed, so an
LF run is bit-identical to before this was added.

### What is still *not* band-generic

`plots.reference` is left alone deliberately. The published limits are at $z \approx
8.2$; overlaying them on a $z = 6.8$ spectrum compares different epochs. Drop the key
for a run in another band, or keep it and label the figure yourself.

`power_spectrum.noise_scale` is unrelated to frequency but is still inherited from the
tutorials rather than derived from your integration time — see `docs/PIPELINE.md`.
