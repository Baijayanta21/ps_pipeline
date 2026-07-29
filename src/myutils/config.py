r"""
Configuration loader for the ps_pipeline
========================================

One YAML file drives every stage. This module finds it, validates it, fills in the
fields that can be derived or read from the input UVFITS header, and exposes the
result with attribute access.

Resolution order for the config path:

#. explicit argument to :func:`load`
#. ``--config PATH`` on the command line
#. ``$PS_CONFIG``
#. ``./config.yaml``
#. ``config.yaml`` beside the installed package's repository root

Typical use in a stage script:

.. code-block:: python

    from myutils.config import load

    cfg = load()
    cfg.echo()                             # print the resolved values

    infits = cfg.paths.input_uvfits        # pathlib.Path
    NE     = cfg.NE                        # derived
    out    = cfg.out('cl.npy')             # path inside output_dir
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

__all__ = ['Config', 'load', 'find_config_path']


# ---------------------------------------------------------------- attribute access

class _Node(dict):
    """A dict whose keys are also attributes, applied recursively."""

    def __init__(self, mapping=None):
        super().__init__()
        for key, value in (mapping or {}).items():
            self[key] = _Node(value) if isinstance(value, dict) else value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"no configuration key '{name}'. Available here: "
                f"{', '.join(sorted(self)) or '(empty)'}"
            ) from None

    def __setattr__(self, name, value):
        self[name] = value


# ---------------------------------------------------------------- locating the file

def find_config_path(explicit=None):
    """Return the config file path, following the documented resolution order."""
    if explicit:
        return Path(explicit).expanduser().resolve()

    # --config on the command line, tolerating unrelated arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config')
    known, _ = parser.parse_known_args()
    if known.config:
        return Path(known.config).expanduser().resolve()

    if os.environ.get('PS_CONFIG'):
        return Path(os.environ['PS_CONFIG']).expanduser().resolve()

    cwd_candidate = Path.cwd() / 'config.yaml'
    if cwd_candidate.is_file():
        return cwd_candidate.resolve()

    # src/myutils/config.py -> repository root
    repo_candidate = Path(__file__).resolve().parents[2] / 'config.yaml'
    if repo_candidate.is_file():
        return repo_candidate

    raise FileNotFoundError(
        "No configuration file found. Pass --config PATH, set $PS_CONFIG, "
        "or run from a directory containing config.yaml."
    )


# ---------------------------------------------------------------- the config object

class Config(_Node):
    """Validated pipeline configuration with derived values."""

    #: sections that must be present
    _REQUIRED_SECTIONS = ('paths', 'observation', 'gridding', 'binning',
                          'simulation', 'scf', 'noise', 'power_spectrum')

    def __init__(self, mapping, source=None):
        super().__init__(mapping)
        self.source = str(source) if source else '<dict>'
        self._validate_sections()
        self._resolve_paths()
        self._read_header()
        self._resolve_derived()

    # -- validation ----------------------------------------------------------

    def _validate_sections(self):
        missing = [s for s in self._REQUIRED_SECTIONS if s not in self]
        if missing:
            raise ValueError(
                f"{self.source}: missing required section(s): {', '.join(missing)}"
            )

        if not self.paths.get('input_uvfits'):
            raise ValueError(f"{self.source}: paths.input_uvfits is required")
        if not self.paths.get('output_dir'):
            raise ValueError(f"{self.source}: paths.output_dir is required")

        nstokes = self.gridding.nstokes
        if not isinstance(nstokes, list) or len(nstokes) != 2 or nstokes[:2] != [0, 1]:
            raise ValueError(
                f"{self.source}: gridding.nstokes must be [0, 1]. The cross-TGE "
                f"estimator hard-codes GV[0] as XX and GV[1] as YY, so a different "
                f"order or a single polarisation cannot be correlated. Got {nstokes!r}."
            )

        if self.scf.window not in ('hann', 'hamming', 'blackman', 'kaiser'):
            raise ValueError(
                f"{self.source}: scf.window must be one of "
                f"hann, hamming, blackman, kaiser. Got {self.scf.window!r}."
            )
        if self.scf.method not in ('fft', 'direct', 'auto'):
            raise ValueError(
                f"{self.source}: scf.method must be fft, direct or auto. "
                f"Got {self.scf.method!r}."
            )
        if self.power_spectrum.cosmology not in ('Planck18', 'FlatLambdaCDM'):
            raise ValueError(
                f"{self.source}: power_spectrum.cosmology must be Planck18 or "
                f"FlatLambdaCDM. Got {self.power_spectrum.cosmology!r}."
            )

    # -- paths ---------------------------------------------------------------

    def _resolve_paths(self):
        p = self.paths

        p.input_uvfits = Path(p.input_uvfits).expanduser()
        if not p.input_uvfits.is_file():
            raise FileNotFoundError(
                f"{self.source}: paths.input_uvfits does not exist: {p.input_uvfits}"
            )
        p.input_uvfits = p.input_uvfits.resolve()

        p.output_dir = Path(p.output_dir).expanduser().resolve()
        p.output_dir.mkdir(parents=True, exist_ok=True)

        if not p.get('index'):
            stem = p.input_uvfits.stem
            if stem.isdigit() and len(stem) >= 4:
                # a bare obsid like 1000007200 -> 7200, matching the tutorials
                p.index = stem[-4:]
            else:
                # anything else: use the whole stem. Scraping digits out of a
                # descriptive name produces nonsense (MWA_LF_RA_39p7..._R5 -> '3975').
                p.index = stem
            self._index_derived = True
        else:
            self._index_derived = False

        p.uaps_uvfits = (Path(p['uaps_uvfits']).expanduser().resolve()
                         if p.get('uaps_uvfits')
                         else p.output_dir / f'{p.index}_uaps.fits')

        p.grf_dir = (Path(p['grf_dir']).expanduser().resolve()
                     if p.get('grf_dir') else p.output_dir / 'grf')
        p.grf_dir.mkdir(parents=True, exist_ok=True)

        # The working copy that 00_flag.py creates and flags. The original input is
        # never opened for writing by any stage.
        sub = (self.get('flagging') or {}).get('work_subdir') or 'UVFITS'
        p.work_dir = p.output_dir / sub
        p.work_uvfits = p.work_dir / p.input_uvfits.name

        ps = self.power_spectrum
        ps.flag_mask = (Path(ps['flag_mask']).expanduser().resolve()
                        if ps.get('flag_mask') else p.output_dir / 'flag_mask.npy')

    # -- header ---------------------------------------------------------------

    def _read_header(self):
        """Fill null observation fields from the input UVFITS header."""
        obs = self.observation
        needed = [k for k in ('nuc_mhz', 'dnuc_mhz', 'nchan') if obs.get(k) is None]
        if needed:
            from astropy.io import fits
            with fits.open(self.paths.input_uvfits, mode='readonly') as hdul:
                hdr = hdul[0].header
                if obs.get('nuc_mhz') is None:
                    obs.nuc_mhz = float(hdr['CRVAL4']) * 1e-6
                if obs.get('dnuc_mhz') is None:
                    obs.dnuc_mhz = abs(float(hdr['CDELT4'])) * 1e-6
                if obs.get('nchan') is None:
                    obs.nchan = int(hdr['NAXIS4'])
            self._header_fields = needed
        else:
            self._header_fields = []

        if not obs.dnuc_mhz:
            raise ValueError(
                f"{self.source}: observation.dnuc_mhz resolved to {obs.dnuc_mhz}. "
                f"The input file may be a 2D simulation, which has CDELT4 = 0 — "
                f"set dnuc_mhz explicitly."
            )

    # -- derived --------------------------------------------------------------

    def _resolve_derived(self):
        obs = self.observation
        if obs.get('n2') is None:
            obs.n2 = obs.nchan - 1
        if not (0 <= obs.n1 <= obs.n2 <= obs.nchan - 1):
            raise ValueError(
                f"{self.source}: need 0 <= n1 <= n2 <= nchan-1, got "
                f"n1={obs.n1}, n2={obs.n2}, nchan={obs.nchan}"
            )

        # The simulated UAPS sky is band-limited at lmax = 3*nside - 1, so it carries
        # no power beyond |U| = lmax/2pi. Grid points past that get M_g ~ 0, are cut by
        # Mg_min, and leave the outer annular bins empty — with no error anywhere.
        lmax = 3 * self.simulation.nside - 1
        self._sim_umax = lmax / (2.0 * np.pi)
        if self._sim_umax < self.binning.binUmax:
            nside_needed = int(np.ceil((2.0 * np.pi * self.binning.binUmax + 1) / 3))
            print(
                f"WARNING: simulation.nside = {self.simulation.nside} gives "
                f"lmax = {lmax}, so the UAPS sky only reaches "
                f"|U| = {self._sim_umax:.1f} lambda, short of binning.binUmax = "
                f"{self.binning.binUmax}. Bins beyond {self._sim_umax:.1f} will be "
                f"starved of modes. Raise nside to >= {nside_needed}, or lower binUmax."
            )

        if self.simulation.Nrea > obs.nchan:
            raise ValueError(
                f"{self.source}: simulation.Nrea ({self.simulation.Nrea}) cannot "
                f"exceed the channel count ({obs.nchan}) — realisations are written "
                f"along the frequency axis."
            )

        ps = self.power_spectrum
        if ps.get('NE') is None:
            ps.NE = self.nchan_out
        elif ps.NE > self.nchan_out:
            raise ValueError(
                f"{self.source}: power_spectrum.NE ({ps.NE}) exceeds the channels "
                f"available after SCF ({self.nchan_out})."
            )

        if ps.get('noise_scale') is None:
            # sigma_N = 60 / sqrt(N_nights); the variance scaling is sigma_N**2.
            # At n_nights = 9 this reproduces the tutorials' hardcoded 20**2 = 400.
            ps.noise_scale = (60.0 / (ps.n_nights ** 0.5)) ** 2

    # -- public helpers -------------------------------------------------------

    # -- flagging -------------------------------------------------------------

    @property
    def flagging_enabled(self):
        return bool((self.get('flagging') or {}).get('enabled', False))

    @property
    def science_uvfits(self):
        """The UVFITS the science stages should read.

        The flagged working copy when flagging is on and it exists, otherwise the
        original. Stages call this rather than ``paths.input_uvfits`` so that turning
        flagging on or off does not require editing any stage.
        """
        if self.flagging_enabled and self.paths.work_uvfits.is_file():
            return self.paths.work_uvfits
        return self.paths.input_uvfits

    def coarse_flag_channels(self):
        """Channel indices flagged by the coarse-band scheme.

        For every coarse band: the first ``coarse_edge_n`` channels, the centre
        channel, and the last ``coarse_edge_n`` channels.
        """
        f = self.get('flagging') or {}
        n_edge = int(f.get('coarse_edge_n', 0) or 0)
        nbands = int(f.get('n_coarse_bands', 24) or 24)
        nchan = int(self.observation.nchan)
        width = f.get('coarse_width') or (nchan // nbands)
        width = int(width)

        if width * nbands != nchan:
            print(f"WARNING: {nbands} coarse bands x {width} channels = "
                  f"{nbands * width}, but the file has {nchan} channels.")
        if n_edge * 2 >= width:
            raise ValueError(
                f"flagging.coarse_edge_n={n_edge} flags {2 * n_edge} of {width} "
                f"channels per band, leaving nothing. Reduce it.")

        idx = []
        for b in range(0, nchan, width):
            idx += list(range(b, min(b + n_edge, nchan)))
            centre = b + width // 2
            if centre < nchan:
                idx.append(centre)
            idx += [i for i in range(b + width - n_edge, b + width) if i < nchan]
        return np.array(sorted(set(idx)), dtype=int)

    @property
    def nchan_used(self):
        """Channels entering stage 1, i.e. n2 - n1 + 1."""
        return self.observation.n2 - self.observation.n1 + 1

    @property
    def NW(self):
        """Channels trimmed from each end by SCF. Zero when SCF is disabled."""
        if not self.scf.enabled:
            return 0
        return int(self.scf.SM_mhz / self.observation.dnuc_mhz)

    @property
    def nchan_out(self):
        """Channels surviving SCF."""
        return self.nchan_used - 2 * self.NW

    @property
    def NE(self):
        """Frequency separations used for the power spectrum."""
        return self.power_spectrum.NE

    @property
    def index(self):
        return self.paths.index

    def out(self, name):
        """Path to *name* inside the output directory."""
        return self.paths.output_dir / name

    def product(self, kind, **kw):
        """Canonical output paths, so every stage agrees on filenames."""
        i = self.index
        table = {
            'bininfo':   f'bin_info_{i}.npz',
            'el':        f'el_{i}.npy',
            'ml':        f'ml_{i}.npy',
            'cl':        f'cl_{i}.npy',
            'el2d':      f'el2d_{i}.npy',
            'cl2d':      f'cl2d_{i}.npy',
            'ml2d':      f'ml2d_{i}.npy',
            'cln':       f'cln_{i}.npy',
            'cln_part':  f'cln_{i}_part{kw.get("task", 0):04d}.npy',
            'ps':        f'ps_{i}.npz',
        }
        if kind not in table:
            raise KeyError(f"unknown product '{kind}'. Known: {', '.join(table)}")
        return self.out(table[kind])

    def apply_simulation_globals(self):
        """Publish nside / Nrea / chunk_size into builtins, as simvis requires."""
        import builtins as blt
        blt.nside = self.simulation.nside
        blt.Nrea = self.simulation.Nrea
        blt.chunk_size = self.simulation.chunk_size

    def echo(self, stage=None):
        """Print the resolved configuration. Called at the top of every stage."""
        obs, g, b, s = self.observation, self.gridding, self.binning, self.scf
        derived = f" (from header: {', '.join(self._header_fields)})" if self._header_fields else ""
        title = f"ps_pipeline — {stage}" if stage else "ps_pipeline"
        line = '=' * 68
        print(line)
        print(title)
        print(line)
        print(f"config        : {self.source}")
        print(f"input         : {self.paths.input_uvfits}")
        print(f"output        : {self.paths.output_dir}")
        origin = ' (auto — set paths.index for a shorter label)' if self._index_derived else ''
        print(f"index         : {self.index}{origin}")
        print(f"observation   : nuc = {obs.nuc_mhz:.4f} MHz, dnuc = {obs.dnuc_mhz:.4f} MHz, "
              f"nchan = {obs.nchan}{derived}")
        print(f"channels      : {obs.n1}-{obs.n2} ({self.nchan_used} used)")
        print(f"gridding      : Umax = {g.Umax}, FWHM = {g.FWHM} deg, f = {g.f}, "
              f"nstokes = {g.nstokes}, flag = {g.flag}")
        print(f"binning       : Nbin = {b.Nbin}, |U| = {b.binUmin}-{b.binUmax}, "
              f"Mg_min = {b.Mg_min}")
        if s.enabled:
            print(f"scf           : SM = {s.SM_mhz} MHz, window = {s.window}, "
                  f"method = {s.method}, NW = {self.NW} -> {self.nchan_out} channels")
        else:
            print(f"scf           : disabled -> {self.nchan_out} channels")
        print(f"simulation    : nside = {self.simulation.nside} "
              f"(lmax = {3 * self.simulation.nside - 1}, |U| <= {self._sim_umax:.0f} "
              f"lambda), Nrea = {self.simulation.Nrea}")
        print(f"noise         : {self.noise.n_realisations} realisations, "
              f"seeds {self.noise.seed_start}-{self.noise.seed_start + self.noise.n_realisations - 1}")
        print(f"power spectrum: NE = {self.NE}, NBin = {self.power_spectrum.NBin}, "
              f"cosmology = {self.power_spectrum.cosmology}, "
              f"noise_scale = {self.power_spectrum.noise_scale:.4f}")
        print(line, flush=True)

    def require(self, *kinds):
        """Fail early with a clear message when an upstream product is missing."""
        stage_of = {
            'bininfo': 'pipeline/02_bininfo.py',
            'el': 'pipeline/03_data.py', 'ml': 'pipeline/03_data.py',
            'cl': 'pipeline/03_data.py', 'cln': 'pipeline/04_noise.py',
        }
        for kind in kinds:
            path = self.product(kind)
            if not path.exists():
                raise FileNotFoundError(
                    f"missing input {path.name} — run {stage_of.get(kind, 'the earlier stage')} first "
                    f"(looked in {self.paths.output_dir})"
                )


# ---------------------------------------------------------------- entry point

def load(path=None, echo=False, stage=None):
    """Load, validate and return the pipeline :class:`Config`."""
    cfg_path = find_config_path(path)
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path}: top level must be a mapping")
    cfg = Config(raw, source=cfg_path)
    if echo:
        cfg.echo(stage)
    return cfg


if __name__ == '__main__':
    # `python -m myutils.config` prints the resolved configuration and exits.
    try:
        load(echo=True, stage='config check')
    except Exception as exc:                                  # noqa: BLE001
        print(f"config error: {exc}", file=sys.stderr)
        sys.exit(1)
