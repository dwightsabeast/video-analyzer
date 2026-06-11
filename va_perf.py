#!/usr/bin/env python3
"""
va_perf - system capability detection and performance budgets.

One place that answers "how parallel should this machine go?". Everything else
(combined analysis passes, the forensics battery, batch jobs, libvmaf threading,
the preview frame cache) sizes itself from here. No third-party dependencies:
cores via os.cpu_count(), RAM via ctypes/GlobalMemoryStatusEx on Windows or
/proc/meminfo elsewhere, GPU VRAM via nvidia-smi when present.

Modes (VA_PERF env, the GUI Setup tab, or set_mode()):
  max       use every logical core and the GPU; full fan-out (default)
  balanced  leave ~2 cores headroom so the desktop stays responsive
  eco       fewest watt-hours: hardware decode preferred, no oversubscription,
            single batch job. Note "eco" still finishes fast - most energy in
            video work is the decode itself, and the single-decode combined
            passes + fixed-function GPU decode do the saving in every mode.
"""

from __future__ import annotations

import os
import subprocess
import threading
from collections import OrderedDict

CREATIONFLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

MODES = ("max", "balanced", "eco")
_mode_override = None


def cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def mode() -> str:
    """Active performance mode: set_mode() wins, then VA_PERF, then 'max'."""
    if _mode_override in MODES:
        return _mode_override
    env = (os.environ.get("VA_PERF") or "").strip().lower()
    return env if env in MODES else "max"


def set_mode(m) -> str:
    """Set the in-process mode override ('' / None clears it). Returns mode()."""
    global _mode_override
    _mode_override = m if m in MODES else None
    return mode()


# --- Memory -------------------------------------------------------------------

def mem_info() -> dict:
    """{'total_bytes', 'free_bytes'} - free means available-to-allocate.
    Zeros when the platform offers no cheap answer (sizing then falls back
    to conservative fixed caps)."""
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return {"total_bytes": int(st.ullTotalPhys),
                        "free_bytes": int(st.ullAvailPhys)}
        except Exception:   # noqa: BLE001 - detection must never raise
            pass
        return {"total_bytes": 0, "free_bytes": 0}
    total = free = 0
    try:
        with open("/proc/meminfo", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    free = int(line.split()[1]) * 1024
    except OSError:
        pass
    return {"total_bytes": total, "free_bytes": free}


# --- GPU ----------------------------------------------------------------------

_GPU_CACHE = None


def gpus(refresh=False) -> list:
    """[{'name', 'vram_total_mb', 'vram_used_mb'}] via nvidia-smi (cached).
    Empty list when no NVIDIA tooling is present - Intel/AMD decode still
    works through va_hwaccel, there is just no VRAM readout for them."""
    global _GPU_CACHE
    if _GPU_CACHE is not None and not refresh:
        return _GPU_CACHE
    out = []
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=8, creationflags=CREATIONFLAGS)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    try:
                        out.append({"name": parts[0],
                                    "vram_total_mb": int(float(parts[1])),
                                    "vram_used_mb": int(float(parts[2]))})
                    except ValueError:
                        pass
    except (OSError, subprocess.SubprocessError):
        pass
    _GPU_CACHE = out
    return out


# --- Budgets ------------------------------------------------------------------

def _usable_cores() -> int:
    n = cpu_count()
    m = mode()
    if m == "balanced":
        return max(1, n - 2)
    if m == "eco":
        return max(1, n // 2)
    return n


def pool_workers(n_tasks) -> int:
    """Thread-pool size for independent subprocess/numpy stages (forensics
    battery, event passes). Subprocesses do the work, so this is about how
    many children run at once - capped by usable cores."""
    if n_tasks <= 1:
        return 1
    if mode() == "eco":
        return 1
    return max(1, min(int(n_tasks), _usable_cores()))


def batch_jobs(requested=None, n_files=1) -> int:
    """Concurrent files for analyze.py. Serial unless --jobs asks otherwise
    ('auto' sizes from cores; each job already runs a multi-threaded decode,
    so a few jobs saturate). HW-decode session limits on consumer GPUs make
    >4 concurrent decodes counterproductive."""
    if requested in (None, "", 1, "1"):
        return 1
    if str(requested).lower() == "auto":
        if mode() == "eco":
            return 1
        return max(1, min(4, cpu_count() // 4, int(n_files)))
    try:
        return max(1, min(int(requested), 16, int(n_files)))
    except (TypeError, ValueError):
        return 1


def vmaf_threads() -> int:
    """libvmaf n_threads: it is single-threaded unless told otherwise."""
    return _usable_cores()


def ffmpeg_thread_args() -> list:
    """Global ffmpeg thread flags. Empty in max/balanced (ffmpeg's defaults
    already use every core for decode and filter graphs); eco pins decode and
    filter threading to half the cores so clocks stay in the efficient range."""
    if mode() != "eco":
        return []
    n = str(_usable_cores())
    return ["-threads", n, "-filter_threads", n, "-filter_complex_threads", n]


# --- Preview frame cache --------------------------------------------------------

class FrameCache:
    """Thread-safe LRU of decoded preview frames keyed by (generation, index).

    Sized as a fraction of *free* RAM at construction (max 25% / balanced 15% /
    eco 8%), with a floor so it is useful even on a busy machine. Re-showing a
    cached frame costs zero decode work - stepping, scrubbing back and forth,
    and A/B-ing a cut replay from memory instead of re-burning the decoder.
    """

    _FRACTION = {"max": 0.25, "balanced": 0.15, "eco": 0.08}

    def __init__(self, budget_bytes=None):
        if budget_bytes is None:
            free = mem_info().get("free_bytes") or 0
            frac = self._FRACTION.get(mode(), 0.25)
            budget_bytes = int(free * frac) if free else 512 << 20
            budget_bytes = max(128 << 20, budget_bytes)
        self.budget = int(budget_bytes)
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self._d: "OrderedDict" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            f = self._d.get(key)
            if f is None:
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return f

    def put(self, key, frame):
        if frame is None:
            return
        size = int(getattr(frame, "nbytes", 0)) or 1
        if size > self.budget:
            return
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self.bytes -= int(getattr(old, "nbytes", 0)) or 1
            self._d[key] = frame
            self.bytes += size
            while self.bytes > self.budget and self._d:
                _, ev = self._d.popitem(last=False)
                self.bytes -= int(getattr(ev, "nbytes", 0)) or 1

    def clear(self):
        with self._lock:
            self._d.clear()
            self.bytes = 0

    def __len__(self):
        with self._lock:
            return len(self._d)

    def stats(self) -> dict:
        with self._lock:
            return {"frames": len(self._d), "bytes": self.bytes,
                    "budget": self.budget, "hits": self.hits, "misses": self.misses}


# --- Reporting ------------------------------------------------------------------

def _gb(b) -> str:
    return "%.1f GB" % (b / (1 << 30)) if b else "?"


def summary() -> str:
    """Human-readable capability + budget summary for launch.py / the Setup tab."""
    mi = mem_info()
    lines = ["Performance mode: %s" % mode(),
             "CPU: %d logical cores (%d usable in this mode)" % (cpu_count(), _usable_cores()),
             "RAM: %s total, %s free" % (_gb(mi["total_bytes"]), _gb(mi["free_bytes"]))]
    for g in gpus():
        lines.append("GPU: %s - %d MB VRAM (%d MB in use)" % (
            g["name"], g["vram_total_mb"], g["vram_used_mb"]))
    try:
        import va_hwaccel
        seen = getattr(va_hwaccel, "_DECODE_BY_CODEC", {})
        ok = sorted("%s via %s" % (c, m) for c, m in seen.items() if m)
        if ok:
            lines.append("HW decode verified: " + "; ".join(ok))
    except Exception:   # noqa: BLE001
        pass
    lines.append("Budgets: vmaf %d threads, battery pool %d wide, batch auto = %d jobs"
                 % (vmaf_threads(), pool_workers(10), batch_jobs("auto", 99)))
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
