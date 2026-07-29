r"""
Progress reporting for long, silent loops
=========================================

The pipeline spends most of its wall time inside a handful of loops that print
nothing — the baseline loop in :func:`myutils.tge.grid.grid`, the visibility chunk
loop in :func:`myutils.simvis.simvis.visgen_mwa_multi`, the realisation loops in the
sky simulation. On a slow shared filesystem those silences last minutes and are
indistinguishable from a hang, which has repeatedly led to working runs being killed.

:class:`Progress` adapts to where its output is going:

* **Interactive terminal** — a single line redrawn in place with a bar, percentage,
  elapsed time and ETA.
* **Redirected to a log file** — discrete lines at a fixed interval, so a batch log
  stays readable rather than filling with control characters.

Usage::

    p = Progress(len(chunks), 'Visibility')
    for chunk in chunks:
        ...
        p.update()
    p.close()

or as a context manager::

    with Progress(n, 'Gridding') as p:
        for ...:
            p.update()
"""

import shutil
import sys
from datetime import datetime

__all__ = ['Progress', 'fmt_hms']


def fmt_hms(seconds):
    """Format a duration as ``H:MM:SS`` or ``MM:SS`` for short spans."""
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours >= 1:
        return f"{int(hours)}:{int(minutes):02d}:{int(secs):02d}"
    return f"{int(minutes):02d}:{int(secs):02d}"


class Progress:
    """Report progress through a loop of known length.

    Parameters
    ----------
    total : int
        Number of iterations. Values below 1 disable reporting entirely.
    label : str
        Short prefix, e.g. ``'Gridding'``. Padded to a consistent width.
    steps : int, optional
        How many updates to emit when writing to a file. Default 20 (every 5%).
    min_total : int, optional
        Skip reporting for loops shorter than this. Default 1, i.e. always report.
    stream : file, optional
        Where to write. Defaults to ``sys.stdout``.
    enabled : bool, optional
        Set False to silence without restructuring the caller.
    """

    def __init__(self, total, label, steps=20, min_total=1, stream=None,
                 enabled=True):
        self.total = int(total)
        self.label = label
        self.stream = stream if stream is not None else sys.stdout
        self.enabled = bool(enabled) and self.total >= max(1, min_total)
        self.n = 0
        self.start = datetime.now()
        self._last_emit = -1

        try:
            self.tty = self.stream.isatty()
        except Exception:                                          # noqa: BLE001
            self.tty = False

        # in a file, emit at fixed fractions; on a tty, redraw freely but not
        # more than a few times a second
        self.every = max(1, self.total // max(1, steps)) if not self.tty else 1
        self._last_draw = 0.0

    # -- context manager ------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- internals ------------------------------------------------------------

    def _elapsed(self):
        return (datetime.now() - self.start).total_seconds()

    def _render(self, final=False):
        frac = 1.0 if final else min(1.0, self.n / self.total)
        secs = self._elapsed()
        eta = (secs * (1.0 - frac) / frac) if frac > 0 else 0.0

        if self.tty:
            width = max(20, min(shutil.get_terminal_size((100, 24)).columns, 110))
            barw = max(10, width - len(self.label) - 46)
            filled = int(round(barw * frac))
            bar = '█' * filled + '·' * (barw - filled)
            line = (f"\r{self.label:<10s} [{bar}] {100 * frac:5.1f}%  "
                    f"{self.n:,}/{self.total:,}  "
                    f"{fmt_hms(secs)} elapsed  eta {fmt_hms(eta)}")
            self.stream.write(line)
            if final:
                self.stream.write('\n')
        else:
            self.stream.write(
                f"{self.label:<10s} {100 * frac:5.1f}%  "
                f"({self.n:,}/{self.total:,})  "
                f"{fmt_hms(secs)} elapsed  eta {fmt_hms(eta)}\n")
        try:
            self.stream.flush()
        except Exception:                                          # noqa: BLE001
            pass

    # -- public ---------------------------------------------------------------

    def update(self, n=1):
        """Advance by *n* iterations, emitting a line when due."""
        if not self.enabled:
            return
        self.n += n
        if self.tty:
            now = self._elapsed()
            if now - self._last_draw < 0.2 and self.n < self.total:
                return
            self._last_draw = now
            self._render()
        else:
            bucket = self.n // self.every
            if bucket != self._last_emit and self.n < self.total:
                self._last_emit = bucket
                self._render()

    def close(self, note=''):
        """Emit the final line, with total elapsed time."""
        if not self.enabled:
            return
        self.n = self.total
        self._render(final=True)
        secs = self._elapsed()
        suffix = f"  {note}" if note else ''
        self.stream.write(f"{self.label:<10s} done in {fmt_hms(secs)}"
                          f" ({secs:.1f}s){suffix}\n")
        try:
            self.stream.flush()
        except Exception:                                          # noqa: BLE001
            pass
