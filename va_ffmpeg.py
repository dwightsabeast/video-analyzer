#!/usr/bin/env python3
"""
va_ffmpeg - ffmpeg/ffprobe discovery and a reliable decode layer.

This module is the engine's I/O foundation. It locates ffmpeg + ffprobe (a copy
bundled next to the scripts is preferred, then the system PATH), probes a file's
video parameters, and decodes frames through an ffmpeg raw-video pipe. ffmpeg
decoding fixes the HEVC / 10-bit / HDR cases where OpenCV's VideoCapture fails,
and lets us tonemap HDR (PQ / HLG / BT.2020) down to a viewable SDR preview.

A cv2.VideoCapture reader is kept as a fallback so the tool still runs when no
ffmpeg binary is available. ``VideoSource`` picks the right one automatically.

No Tkinter here - everything in this module is importable and testable headless.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import json
from functools import lru_cache

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a hard dep for the GUI/renderers
    cv2 = None


CREATIONFLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _safe_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --- Tool discovery ----------------------------------------------------------

def _script_dir() -> str:
    """App root: beside the .exe when frozen, beside the scripts otherwise
    (va_paths.app_dir). tools/ and plugins/ hang off this folder."""
    try:
        import va_paths
        return va_paths.app_dir()
    except ImportError:  # pragma: no cover - va_paths missing (partial checkout)
        return os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=8)
def find_tool(name: str) -> "str | None":
    """Locate an ffmpeg-family tool: a bundled copy beside this script wins, then PATH.

    Platform-aware: a bundled ``name.exe`` is only used on Windows, and a bundled
    POSIX binary must be executable. This stops a Windows ``ffprobe.exe`` shipped
    in the folder from being picked on Linux/macOS, where it cannot run."""
    here = _script_dir()
    candidates = (name + ".exe", name) if os.name == "nt" else (name,)
    for sub in ("tools", "", "mkvtoolnix",
                os.path.join("tools", "mkvtoolnix")):   # tools/ is the canonical home
        for fname in candidates:
            cand = os.path.join(here, sub, fname) if sub else os.path.join(here, fname)
            if os.path.isfile(cand) and (os.name == "nt" or os.access(cand, os.X_OK)):
                return cand
    if os.name == "nt":                       # standard installs (e.g. winget MKVToolNix)
        for env in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
            base = os.environ.get(env)
            if base:
                cand = os.path.join(base, "MKVToolNix", name + ".exe")
                if os.path.isfile(cand):
                    return cand
    return shutil.which(name)


def find_ffmpeg() -> "str | None":
    return find_tool("ffmpeg")


def find_ffprobe() -> "str | None":
    return find_tool("ffprobe")


@lru_cache(maxsize=1)
def _filters() -> frozenset:
    exe = find_ffmpeg()
    if not exe:
        return frozenset()
    try:
        out = subprocess.run([exe, "-hide_banner", "-filters"],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=15,
                             creationflags=CREATIONFLAGS).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            names.add(parts[1])
    return frozenset(names)


def has_filter(name: str) -> bool:
    """True if the bundled/PATH ffmpeg exposes a given filter (e.g. 'libvmaf')."""
    return name in _filters()


def reset_tool_caches():
    """Forget cached tool paths AND capability probes - call after installing
    a tool at runtime so it is discovered without an app restart."""
    find_tool.cache_clear()
    _filters.cache_clear()


# --- Probing -----------------------------------------------------------------

def ffprobe_json(path: str, frames: bool = False, timeout: int = 30) -> "dict | None":
    """Full ffprobe payload (format + streams, optionally first-frame side data)."""
    exe = find_ffprobe()
    if not exe:
        return None
    def _probe(extra):
        args = [exe, "-v", "quiet", "-print_format", "json"] + extra + [path]
        try:
            r = subprocess.run(args, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout,
                               creationflags=CREATIONFLAGS)
            if r.returncode == 0 and r.stdout.strip():
                return json.loads(r.stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError,
                UnicodeDecodeError):
            pass
        return None

    data = _probe(["-show_format", "-show_streams"])
    if data is not None and frames:
        # First frame of the VIDEO stream specifically: a bare %+#1 returns one
        # frame of ANY stream, so audio-first files lost their frame-level HDR
        # side data (MDCV/CLL/HDR10+/DV live there).
        fdata = _probe(["-select_streams", "v:0", "-show_frames",
                        "-read_intervals", "%+#1"])
        data["frames"] = (fdata or {}).get("frames") or []
    return data


def _rate(value: str) -> float:
    if value and "/" in value:
        try:
            n, d = value.split("/")
            d = float(d)
            return float(n) / d if d else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_sar(v) -> float:
    """Stream sample-aspect-ratio -> float (1.0 when unset/invalid/absurd)."""
    s = str(v or "")
    if ":" in s:
        s = s.replace(":", "/")
    sar = _rate(s)
    return sar if 0.1 <= sar <= 10.0 else 1.0


def _parse_rotation(stream) -> int:
    """Display rotation in CLOCKWISE degrees (0/90/180/270), from the legacy
    rotate tag or the Display Matrix side data (whose `rotation` field is
    counter-clockwise, hence the sign flip)."""
    tags = stream.get("tags") or {}
    try:
        return int(round(float(tags.get("rotate")))) % 360
    except (TypeError, ValueError):
        pass
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                return int(round(-float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    return 0


def probe(path: str) -> dict:
    """Normalised video parameters used across the engine and GUI."""
    info = {
        "path": path, "width": 0, "height": 0, "fps": 30.0, "nb_frames": 0,
        "duration": 0.0, "codec": "", "pix_fmt": "", "bit_depth": 8,
        "transfer": "", "primaries": "", "matrix": "",
        "is_hdr": False, "is_hlg": False, "is_pq": False, "is_wide_gamut": False,
        "is_dovi": False, "dovi_ipt": False, "dv_profile": None, "dv_compat": None,
        "sar": 1.0, "rotation": 0, "display_width": 0, "display_height": 0,
        "ok": False,
    }
    data = ffprobe_json(path)
    v = None
    if data:
        v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if v:
        info["ok"] = True
        info["width"] = _safe_int(v.get("width") or 0)
        info["height"] = _safe_int(v.get("height") or 0)
        info["codec"] = v.get("codec_name", "")
        info["pix_fmt"] = v.get("pix_fmt", "")
        info["transfer"] = str(v.get("color_transfer", ""))
        info["primaries"] = str(v.get("color_primaries", ""))
        info["matrix"] = str(v.get("color_space", ""))
        fps = _rate(v.get("avg_frame_rate", "")) or _rate(v.get("r_frame_rate", ""))
        info["fps"] = fps if 0 < fps <= 1000 else 30.0
        nb = v.get("nb_frames")
        dur = v.get("duration") or (data.get("format", {}) or {}).get("duration")
        d = _safe_float(dur)
        info["duration"] = d if 0.0 <= d < 1e8 else 0.0  # guard N/A/inf/junk
        if nb and str(nb).isdigit():
            info["nb_frames"] = int(nb)
        elif info["duration"]:
            info["nb_frames"] = int(round(info["duration"] * info["fps"]))
        pf = info["pix_fmt"]
        if "10" in pf:
            info["bit_depth"] = 10
        elif "12" in pf:
            info["bit_depth"] = 12
        elif "16" in pf:
            info["bit_depth"] = 16
        t, p = info["transfer"], info["primaries"]
        info["is_pq"] = t in ("smpte2084", "16")
        info["is_hlg"] = t in ("arib-std-b67", "18")
        info["is_wide_gamut"] = p in ("bt2020", "bt2020nc", "bt2020c", "9")
        info["is_hdr"] = info["is_pq"] or info["is_hlg"]
        # Dolby Vision: the configuration record rides as stream side data.
        # Profile 5 (and compat-id 0) base layers are IPT-PQ, NOT YCbCr - a
        # standard YCbCr->RGB decode shows the famous magenta/violet wash, so
        # the pipeline must treat them as a distinct kind of HDR ("dovi_ipt").
        tag = str(v.get("codec_tag_string") or "").lower()
        for sd in v.get("side_data_list") or []:
            sdt = str(sd.get("side_data_type", "")).lower()
            if "dovi" in sdt or "dolby vision" in sdt:
                info["is_dovi"] = True
                info["dv_profile"] = _safe_int(sd.get("dv_profile"), None)
                info["dv_compat"] = _safe_int(
                    sd.get("dv_bl_signal_compatibility_id"), None)
                break
        if tag in ("dvh1", "dvhe", "dav1", "dva1"):
            info["is_dovi"] = True      # DV sample entry even if ffprobe is old
        if info["is_dovi"]:
            untagged = not (info["is_pq"] or info["is_hlg"])
            if (info["dv_profile"] == 5
                    or (untagged and info["dv_compat"] == 0)
                    or (untagged and info["dv_profile"] is None
                        and tag in ("dvh1", "dav1") and info["bit_depth"] >= 10)):
                info["dovi_ipt"] = True
                info["is_hdr"] = True           # routes into the HDR pipeline
                info["is_wide_gamut"] = True    # IPT is BT.2020-referred
        # native display geometry: anamorphic pixels (SAR) widen/narrow the
        # picture, rotation metadata turns phone video portrait. The render
        # path uses these so 2.35:1 scope, square, vertical and anamorphic
        # files all show at their true shape.
        info["sar"] = _parse_sar(v.get("sample_aspect_ratio"))
        if info["sar"] == 1.0:
            # some muxers store only DAR; derive the pixel shape from it
            dar = _parse_sar(v.get("display_aspect_ratio"))
            if dar != 1.0 and info["width"] and info["height"]:
                derived = dar * info["height"] / float(info["width"])
                if 0.1 <= derived <= 10.0 and abs(derived - 1.0) > 0.01:
                    info["sar"] = derived
        info["rotation"] = _parse_rotation(v)
        dw = int(round(info["width"] * info["sar"]))
        dh = info["height"]
        if info["rotation"] in (90, 270):
            dw, dh = dh, dw
        info["display_width"], info["display_height"] = dw, dh

    if not info["ok"] and cv2 is not None:  # cv2 fallback probe
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            info["ok"] = True
            info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            f = cap.get(cv2.CAP_PROP_FPS)
            info["fps"] = f if 0 < f <= 1000 else 30.0
            info["nb_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info["duration"] = info["nb_frames"] / info["fps"] if info["fps"] else 0.0
        cap.release()
    if not info["display_width"]:
        info["display_width"], info["display_height"] = info["width"], info["height"]
    return info


def tonemap_vf(operator: str = "hable") -> str:
    """A zscale/tonemap chain mapping PQ/HLG/BT.2020 to BT.709 for preview."""
    return (
        "zscale=transfer=linear:npl=100,"
        "tonemap=tonemap=" + operator + ":desat=0,"
        "zscale=primaries=bt709:transfer=bt709:matrix=bt709:range=tv"
    )


# --- Frame readers -----------------------------------------------------------

class FFmpegReader:
    """Streaming reader: ffmpeg emits raw bgr24 frames on stdout using a chosen
    pipeline (hardware decode / tonemap when available, software otherwise)."""

    def __init__(self, path, pipeline, fps, start_index=0):
        self.path = path
        self.pipeline = pipeline
        self.width = int(pipeline["out_w"])
        self.height = int(pipeline["out_h"])
        self.fps = fps or 30.0
        self.index = int(start_index)
        # Default pipe format is bgr24; pipelines may request a raw 10-bit
        # planar pipe plus a python-side post-process (DV P5 IPT decode).
        self._pipe_fmt = pipeline.get("pipe_fmt", "bgr24")
        if self._pipe_fmt == "yuv420p10le":
            cw, ch = self.width // 2, self.height // 2
            self._frame_bytes = (self.width * self.height + 2 * cw * ch) * 2
        else:
            self._frame_bytes = self.width * self.height * 3
        self._post = None
        if pipeline.get("post") == "ipt":
            import va_ipt
            self._post = va_ipt.IPTDecoder(path)
        self._proc = self._spawn(self.index)

    def _spawn(self, start_index):
        exe = find_ffmpeg()
        if not exe or self._frame_bytes <= 0:
            return None
        ss = max(0, start_index) / self.fps
        args = [exe, "-hide_banner", "-nostdin", "-loglevel", "error"]
        args += list(self.pipeline.get("pre", []))
        if ss > 0:
            args += ["-ss", "%.6f" % ss]
        args += ["-i", self.path, "-vf", self.pipeline["vf"],
                 "-f", "rawvideo", "-pix_fmt", self._pipe_fmt, "-"]
        return subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                bufsize=self._frame_bytes * 4,
                                creationflags=CREATIONFLAGS)

    def read(self):
        if self._proc is None or self._proc.stdout is None:
            return None
        buf = self._proc.stdout.read(self._frame_bytes)
        if not buf or len(buf) < self._frame_bytes:
            return None
        self.index += 1
        if self._post is not None:
            return self._post.decode(buf, self.width, self.height)
        return np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)

    def grab(self, count):
        """Skip frames cheaply (decoded by ffmpeg but not copied to numpy)."""
        if self._proc is None or self._proc.stdout is None:
            return
        for _ in range(max(0, int(count))):
            buf = self._proc.stdout.read(self._frame_bytes)
            if not buf or len(buf) < self._frame_bytes:
                break
            self.index += 1

    def seek(self, index):
        self.close()
        self.index = max(0, int(index))
        self._proc = self._spawn(self.index)

    def close(self):
        if self._proc is not None:
            try:
                if self._proc.stdout:
                    self._proc.stdout.close()
                self._proc.kill()
                self._proc.wait(timeout=1)
            except (OSError, subprocess.SubprocessError):
                pass
            self._proc = None


class CV2Reader:
    """Fallback reader using OpenCV (smooth random access; limited codec support).
    Frames larger than ``max_dim`` are downscaled to match FFmpegReader's cap -
    without ffmpeg there is no tonemap, but at least geometry/memory behave.
    ``sar``/``rotation`` (from probe) are applied manually so anamorphic and
    rotated files keep their native display shape on this path too."""

    _ROT = {90: 0, 180: 1, 270: 2} if cv2 is None else {
        90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE}

    def __init__(self, path: str, fps: float, max_dim: "int | None" = None,
                 sar: float = 1.0, rotation: int = 0):
        self.cap = cv2.VideoCapture(path) if cv2 is not None else None
        self.fps = fps or 30.0
        self.index = 0
        self.max_dim = int(max_dim) if max_dim else 0
        self.sar = float(sar) if sar and 0.1 <= sar <= 10.0 else 1.0
        self.rotation = int(rotation) % 360 if rotation else 0
        # OpenCV >= 4.5 may auto-apply rotation metadata itself; only rotate
        # manually when it does not, or every portrait video flips twice.
        self._apply_rot = self.rotation if self.rotation in (90, 180, 270) else 0
        if self.cap is not None and self._apply_rot:
            try:
                if bool(self.cap.get(cv2.CAP_PROP_ORIENTATION_AUTO)):
                    self._apply_rot = 0
            except (cv2.error, AttributeError):
                pass

    def read(self) -> "np.ndarray | None":
        if self.cap is None or not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame is None:
            return None
        if self._apply_rot:
            frame = cv2.rotate(frame, self._ROT[self._apply_rot])
        if abs(self.sar - 1.0) > 0.01:          # anamorphic: stretch to square px
            h, w = frame.shape[:2]
            sw, sh = (w, int(round(h / self.sar))) if self._apply_rot in (90, 270)                 else (int(round(w * self.sar)), h)
            frame = cv2.resize(frame, (max(2, sw), max(2, sh)),
                               interpolation=cv2.INTER_LINEAR if self.sar > 1
                               else cv2.INTER_AREA)
        if self.max_dim:
            h, w = frame.shape[:2]
            big = max(h, w)
            if big > self.max_dim:
                s = self.max_dim / float(big)
                frame = cv2.resize(frame, (max(2, int(w * s) // 2 * 2),
                                           max(2, int(h * s) // 2 * 2)),
                                   interpolation=cv2.INTER_AREA)
        return frame

    def grab(self, count: int):
        if self.cap is None:
            return
        for _ in range(max(0, int(count))):
            if not self.cap.grab():
                break
            self.index += 1

    def seek(self, index: int):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
            self.index = max(0, int(index))

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# --- Native-signal scope tap --------------------------------------------------

_SWS_MATRIX = {"bt2020nc": "bt2020", "bt2020c": "bt2020", "bt2020": "bt2020",
               "bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601"}


def native_signal_meta(info: dict) -> dict:
    """Transfer/primaries tags describing the file's NATIVE signal, for scopes."""
    if info.get("dovi_ipt") or info.get("is_pq"):
        t = "pq"                        # IPT base layers reconstruct to PQ/2020
    elif info.get("is_hlg"):
        t = "hlg"
    else:
        t = "sdr"
    p = str(info.get("primaries") or "")
    if info.get("is_wide_gamut") or info.get("dovi_ipt"):
        prim = "2020"
    elif p.startswith("smpte43"):       # smpte431 (DCI-P3) / smpte432 (Display P3)
        prim = "P3"
    else:
        prim = "709"
    return {"transfer": t, "primaries": prim,
            "bit_depth": int(info.get("bit_depth") or 8)}


class NativeTap:
    """Second, scope-only decode of the file's NATIVE signal.

    The display path tonemaps HDR to BT.709 SDR for preview, which destroys
    the very information scopes exist to show. This tap decodes WITHOUT
    tonemap, gamut conversion or 8-bit crush: frames arrive as float32 RGB
    (h, w, 3) in 0..1 SIGNAL domain (PQ/HLG/gamma exactly as encoded in the
    file), downscaled for scope analysis. DV profile 5 IPT base layers are
    reconstructed to PQ/BT.2020 via va_ipt (RPU reshape, no tonemap).
    API mirrors FFmpegReader: read / grab / seek / close."""

    MAX_DIM = 480

    @staticmethod
    def needed(info: dict) -> bool:
        """True when the display path distorts colour (tonemap/gamut/bit-crush)."""
        if not (info.get("ok") and find_ffmpeg()):
            return False
        if info.get("is_hdr") or info.get("is_wide_gamut") or info.get("dovi_ipt"):
            return True
        return str(info.get("primaries") or "").startswith("smpte43")

    def __init__(self, path, info, max_dim=None):
        self.path = path
        self.info = info
        self.meta = native_signal_meta(info)
        self.fps = float(info.get("fps") or 30.0)
        self.index = 0
        md = int(max_dim or self.MAX_DIM)
        w = int(info.get("width") or 0)
        h = int(info.get("height") or 0)
        s = min(1.0, md / float(max(w, h, 1)))
        self.width = max(2, (int(w * s) // 2) * 2)
        self.height = max(2, (int(h * s) // 2) * 2)
        self._ipt = None
        if info.get("dovi_ipt"):
            import va_ipt
            self._ipt = va_ipt.IPTDecoder(path)
            self._pix = "yuv420p10le"
            cw, ch = self.width // 2, self.height // 2
            self._frame_bytes = (self.width * self.height + 2 * cw * ch) * 2
            self._vf = "scale=%d:%d:flags=area,format=yuv420p10le" % (
                self.width, self.height)
        else:
            self._pix = "rgb48le"
            self._frame_bytes = self.width * self.height * 6
            m = _SWS_MATRIX.get(str(info.get("matrix") or ""))
            flags = "area" + ((":in_color_matrix=" + m) if m else "")
            self._vf = "scale=%d:%d:flags=%s,format=rgb48le" % (
                self.width, self.height, flags)
        self._proc = self._spawn(0)

    def _spawn(self, start_index):
        exe = find_ffmpeg()
        if not exe or self._frame_bytes <= 0:
            return None
        args = [exe, "-hide_banner", "-nostdin", "-loglevel", "error"]
        ss = max(0, start_index) / self.fps
        if ss > 0:
            args += ["-ss", "%.6f" % ss]
        args += ["-i", self.path, "-map", "0:v:0", "-vf", self._vf,
                 "-f", "rawvideo", "-pix_fmt", self._pix, "-"]
        try:
            return subprocess.Popen(args, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL,
                                    bufsize=self._frame_bytes * 2,
                                    creationflags=CREATIONFLAGS)
        except OSError:
            return None

    def read(self) -> "np.ndarray | None":
        if self._proc is None or self._proc.stdout is None:
            return None
        buf = self._proc.stdout.read(self._frame_bytes)
        if not buf or len(buf) < self._frame_bytes:
            return None
        self.index += 1
        if self._ipt is not None:
            return self._ipt.decode_native(buf, self.width, self.height)
        a = np.frombuffer(buf, "<u2").reshape(self.height, self.width, 3)
        return a.astype(np.float32) / 65535.0

    def grab(self, count):
        if self._proc is None or self._proc.stdout is None:
            return
        for _ in range(max(0, int(count))):
            buf = self._proc.stdout.read(self._frame_bytes)
            if not buf or len(buf) < self._frame_bytes:
                break
            self.index += 1

    def seek(self, index):
        self.close()
        self.index = max(0, int(index))
        self._proc = self._spawn(self.index)

    def close(self):
        if self._proc is not None:
            try:
                if self._proc.stdout:
                    self._proc.stdout.close()
                self._proc.kill()
                self._proc.wait(timeout=1)
            except (OSError, subprocess.SubprocessError):
                pass
            self._proc = None


class VideoSource:
    """Unified decode handle. Prefers a hardware-accelerated ffmpeg pipeline
    (GPU decode + HDR tonemap) when the machine supports it, falls back to
    software, and to OpenCV when ffmpeg is missing or a file needs no help."""

    DECODE_MAX = 1280

    def __init__(self, path, decode_max=None, display_ar=True):
        """``display_ar=True`` (default) decodes to the file's NATIVE DISPLAY
        shape: anamorphic SAR is baked in and rotation metadata applied, so
        2.35:1 scope, square and vertical files come out at their true aspect
        ratio. Pass False for pixel-exact analysis on storage geometry (e.g.
        DCT-grid work, where resampling would smear codec blocks)."""
        self.path = path
        self.decode_max = int(decode_max) if decode_max else self.DECODE_MAX
        self.display_ar = bool(display_ar)
        self.info = probe(path)
        self.width = self.info["width"]
        self.height = self.info["height"]
        self.fps = self.info["fps"]
        self.nb_frames = self.info["nb_frames"]
        self.tonemap = bool(self.info["is_hdr"])
        self.pipeline = None
        self._use_ffmpeg = self._decide()
        self._reader = None
        if self._use_ffmpeg:
            self._ensure_pipeline()

    def _decide(self):
        if not find_ffmpeg():
            return False
        if self.info["is_hdr"] or self.info["bit_depth"] > 8:
            return True
        if self.display_ar and (self.info.get("rotation") or
                                abs((self.info.get("sar") or 1.0) - 1.0) > 0.01):
            return True                      # geometry fixes ride the ffmpeg path
        if self.info["codec"] in ("hevc", "h265", "vp9", "av1"):
            return True
        if cv2 is not None:
            cap = cv2.VideoCapture(self.path)
            ok = cap.isOpened()
            if ok:
                ok2, _ = cap.read()
                ok = ok2
            cap.release()
            if ok:
                return False
        return True

    def _ensure_pipeline(self):
        if self.pipeline is None:
            import va_hwaccel
            self.pipeline = va_hwaccel.choose_video_pipeline(
                self.path, self.info, self.decode_max,
                native_ar=self.display_ar)

    @property
    def backend(self):
        if self._use_ffmpeg:
            return "ffmpeg:" + (self.pipeline["label"] if self.pipeline else "?")
        return "opencv"

    def _make_reader(self, start_index):
        if self._use_ffmpeg:
            self._ensure_pipeline()
            return FFmpegReader(self.path, self.pipeline, self.fps, start_index=start_index)
        r = CV2Reader(self.path, self.fps, max_dim=self.decode_max,
                      sar=(self.info.get("sar") or 1.0) if self.display_ar else 1.0,
                      rotation=(self.info.get("rotation") or 0) if self.display_ar else 0)
        if start_index:
            r.seek(start_index)
        return r

    def start(self, index=0):
        self.close()
        self._reader = self._make_reader(index)

    def read(self):
        if self._reader is None:
            self.start(0)
        return self._reader.read()

    def grab(self, count):
        if self._reader is not None:
            self._reader.grab(count)

    def seek(self, index):
        if self._reader is None:
            self.start(index)
        else:
            self._reader.seek(index)

    def frame_at(self, index):
        self.seek(index)
        return self.read()

    @property
    def index(self):
        return self._reader.index if self._reader else 0

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
