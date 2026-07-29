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

It transfers to this pipeline directly, because the setup matches on every number
the selection depends on:

| | paper | this config |
|---|---|---|
| centre frequency | 154.2 MHz | 154.2550 MHz |
| channel width | 40 kHz | 40 kHz |
| SCF smoothing | $N = 50$, 2 MHz | `scf.SM_mhz: 2.0` → `NW = 50` |
| $k_\perp$ grid | 0.007–0.146, 20 bins | 0.0078–0.1466, `binning.Nbin: 20` |
| $k_\parallel$ max | 4.623 Mpc⁻¹ | 4.6390 Mpc⁻¹ |
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

```yaml
power_spectrum:
  mask:
    preset: paper        # or paper_box, or none
    wedge_buffer: null   # null = keep what the preset says
```

**With a preset, `null` means "inherit".** Any real value overrides — including
`false`, which is an override and not an absence. Without a preset, `null` means
unconstrained.

| preset | selection |
|---|---|
| `paper` | the three cuts **and** the streak windows |
| `paper_box` | the three cuts only — more modes, less hand-tuning |
| `none` | everything |

Stage 5 prints the resolved preset and the survivor count per constraint, so the
log records what was applied rather than what was typed:

```
mode mask   : built from config (preset: paper)
  wedge: excluding k_par <= 3.519 * k_perp
  modes available          13,360
  outside the wedge        12,570
  inside k limits           1,092
  in tabulated windows        488
  selected (all combined)     488 (3.7%)
  k_perp spanned           0.0078 - 0.0448 Mpc^-1
  k_para spanned           0.1391 - 1.3840 Mpc^-1
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
