#!/usr/bin/env python3
"""
va_tools - one-click download of optional helper binaries into the tool folder.

Supported tools and sources:
  * dovi_tool / hdr10plus_tool  - quietvoid's GitHub releases
  * ffmpeg (+ffprobe, ffplay)   - BtbN FFmpeg-Builds full GPL build (GitHub,
                                  fixed 'latest' tag; includes libvmaf+libplacebo)
  * mediainfo                   - MediaArea CLI build (mediaarea.net)
  * mkvextract                  - MKVToolNix portable (mkvtoolnix.download,
                                  Windows only; whole folder, needs Qt DLLs)
  * mp4dump (+mp4info)          - Bento4 SDK (bok.net)

Picks the right asset for the current OS/arch, downloads with progress,
extracts the executable(s), and installs them beside the scripts (or in a
vendor subfolder) so the app's find_tool() discovery picks them up.
Standard library only (urllib/zipfile/tarfile/lzma).

    python va_tools.py             # install everything
    python va_tools.py dovi_tool   # install one
"""

from __future__ import annotations

import io
import os
import re
import sys
import json
import shutil
import zipfile
import tarfile
import platform
import tempfile
import subprocess
import urllib.error
import urllib.parse
import urllib.request

# Hide child consoles on Windows (7z/tar spawned from the windowed .exe would
# otherwise flash a cmd window during tool downloads).
CREATIONFLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

# Tool registry. "repo" implies a quietvoid-style GitHub release; "source"
# selects one of the other resolvers below. "exes" lists every executable to
# extract (first one is the primary used for install/discovery checks).
TOOLS = {
    "dovi_tool":      {"repo": "quietvoid/dovi_tool"},
    "hdr10plus_tool": {"repo": "quietvoid/hdr10plus_tool"},
    "ffmpeg":         {"source": "btbn", "exes": ("ffmpeg", "ffprobe", "ffplay")},
    "mediainfo":      {"source": "mediaarea"},
    "mkvextract":     {"source": "mkvtoolnix", "subdir": "mkvtoolnix",
                       "exes": ("mkvextract", "mkvinfo", "mkvmerge")},
    "c2patool":       {"repo": "contentauth/c2pa-rs", "tag_prefix": "c2patool-v"},
    "mp4dump":        {"source": "bento4", "exes": ("mp4dump", "mp4info")},
}
_CHUNK = 256 * 1024


def tool_exes(name):
    return TOOLS[name].get("exes", (name,))


def _platform_profile(system, machine):
    """Per-platform matching rules: (arch marker, OS name markers, archive
    extensions, fallback asset-name suffixes - current naming first)."""
    arch = "aarch64" if str(machine).lower() in ("arm64", "aarch64") else "x86_64"
    s = (system or "").lower()
    if s.startswith("win"):
        return arch, ("windows",), (".zip",), [
            "%s-pc-windows-msvc.zip" % arch, "x86_64-pc-windows-msvc.zip"]
    if s == "darwin" or "mac" in s:
        return arch, ("macos", "apple-darwin"), (".zip", ".tar.gz", ".tgz"), [
            "universal-macOS.zip", "%s-apple-darwin.tar.gz" % arch]
    return arch, ("linux",), (".tar.gz", ".tgz"), [
        "%s-unknown-linux-musl.tar.gz" % arch, "%s-unknown-linux-gnu.tar.gz" % arch]


def select_asset(assets, system=None, machine=None):
    """Choose the best release asset for the platform. Returns (name, url) or (None, None).

    Tries an exact-arch asset first, then a 'universal' build (current macOS
    naming), then any asset that at least matches the OS."""
    system = system or platform.system()
    machine = machine or platform.machine()
    arch, oses, exts, _ = _platform_profile(system, machine)
    named = [((a.get("name") or "").lower(), a) for a in assets]
    named = [(n, a) for n, a in named
             if n.endswith(exts) and any(o in n for o in oses)]
    for want in (arch, "universal", None):
        if want is None and arch == "aarch64" and "linux" in oses:
            break    # an x86_64 binary cannot run on linux/aarch64 (no x64
                     # emulation, unlike Windows-on-ARM / Rosetta): no asset
        for n, a in named:
            if want is None or want in n:
                return a.get("name"), a.get("browser_download_url")
    return None, None


def _pick_member(names, exe_basename):
    base = exe_basename.lower()
    stem = base[:-4] if base.endswith(".exe") else base
    for n in names:
        leaf = n.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if leaf in (base, stem, stem + ".exe"):
            return n
    for n in names:
        leaf = n.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if stem in leaf and (leaf.endswith(".exe") or "." not in leaf):
            return n
    return None


def _archive_names(data, asset_name):
    if asset_name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return z.namelist()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
        return t.getnames()


def _archive_read(data, asset_name, member):
    if asset_name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return z.read(member)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
        f = t.extractfile(member)
        if f is None:
            raise RuntimeError("could not read member from archive")
        return f.read()


def extract_exe(data, asset_name, exe_basename):
    """Pull one executable out of a downloaded .zip / .tar.gz / .tar.xz (bytes)."""
    m = _pick_member(_archive_names(data, asset_name), exe_basename)
    if not m:
        raise RuntimeError("executable not found in archive")
    return _archive_read(data, asset_name, m)


def extract_members(data, asset_name, exe_basenames):
    """Extract several executables. Returns {basename: bytes} for those found."""
    names = _archive_names(data, asset_name)
    out = {}
    for base in exe_basenames:
        m = _pick_member(names, base)
        if m:
            out[base] = _archive_read(data, asset_name, m)
    return out


# --- HTTP ---------------------------------------------------------------------

def _http_get(url, timeout=180, on_progress=None, label="downloading"):
    """GET url into memory; report throttled percent progress when the size is known."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "video-analyzer", "Accept": "application/octet-stream, application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        try:
            total = int(r.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            total = 0
        buf, got, last = [], 0, -1
        while True:
            chunk = r.read(_CHUNK)
            if not chunk:
                break
            buf.append(chunk)
            got += len(chunk)
            if on_progress and total:
                pct = got * 100 // total
                if pct >= last + 5 or got == total:   # throttle UI updates
                    last = pct
                    on_progress("%s %d%%  (%.1f / %.1f MB)" % (
                        label, pct, got / 1048576.0, total / 1048576.0))
        return b"".join(buf)


def _manual_url(name):
    """Where to download a tool by hand when automatic install cannot."""
    spec = TOOLS.get(name, {})
    if "repo" in spec:
        return "https://github.com/%s/releases" % spec["repo"]
    return {"btbn": "https://github.com/BtbN/FFmpeg-Builds/releases",
            "mediaarea": "https://mediaarea.net/en/MediaInfo/Download",
            "mkvtoolnix": "https://mkvtoolnix.download",
            "bento4": "https://www.bok.net/Bento4/binaries/",
            }.get(spec.get("source"), "the project's website")


def _fetch_page(url, tool, manual_url, timeout=30):
    """Fetch a version-listing page; map network failure to a friendly error."""
    try:
        return _http_get(url, timeout=timeout).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError("could not reach %s (%s); check the network, or download "
                           "%s manually from %s and place it beside the scripts"
                           % (url.split("/", 3)[2], e, tool, manual_url))


def _latest_tag(repo, timeout=30, tag_prefix=None):
    """Latest release tag without the GitHub API. Plain repos: follow the
    /releases/latest redirect. Multi-stream repos (tag_prefix set, e.g.
    c2pa-rs publishing both c2pa-v* and c2patool-v*): scan the releases atom
    feed for the newest matching tag, since /latest may point at the wrong
    stream. Immune to API rate limits."""
    if tag_prefix:
        page = _http_get("https://github.com/%s/releases.atom" % repo,
                         timeout=timeout).decode("utf-8", "replace")
        m = re.search(r"/releases/tag/(%s[^\"'<]+)" % re.escape(tag_prefix), page)
        return urllib.parse.unquote(m.group(1)) if m else None
    req = urllib.request.Request("https://github.com/%s/releases/latest" % repo,
                                 headers={"User-Agent": "video-analyzer"}, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        url = r.geturl()
    tag = url.rstrip("/").rsplit("/", 1)[-1]
    return tag if tag and tag != "latest" else None


def _candidate_urls(name, repo, tag, system, machine):
    """Direct download URLs built from the projects' documented asset naming,
    for when the API listing is unavailable."""
    _, _, _, suffixes = _platform_profile(system, machine)
    out, seen = [], set()
    for suf in suffixes:
        # c2pa-rs-style tags already start with the tool name ("c2patool-v0.x"),
        # and assets are "<tag>-<platform>"; classic repos use "<name>-<tag>-...".
        aname = ("%s-%s" % (tag, suf)) if str(tag).startswith(name) \
            else "%s-%s-%s" % (name, tag, suf)
        if aname not in seen:
            seen.add(aname)
            out.append((aname, "https://github.com/%s/releases/download/%s/%s"
                        % (repo, tag, aname)))
    return out


# --- Per-source resolvers ------------------------------------------------------
# Each returns (version_label, [(asset_name, url), ...]) or raises RuntimeError
# with a human-readable reason (e.g. no build for this OS).

def _resolve_quietvoid(name, repo, system, machine, say, tag_prefix=None):
    tag = aname = url = None
    try:
        if tag_prefix:
            api = "https://api.github.com/repos/%s/releases?per_page=30" % repo
            rels = json.loads(_http_get(api, timeout=30).decode("utf-8", "replace"))
            rel = next((r for r in rels if isinstance(r, dict)
                        and str(r.get("tag_name", "")).startswith(tag_prefix)
                        and not r.get("draft") and not r.get("prerelease")), {})
        else:
            api = "https://api.github.com/repos/%s/releases/latest" % repo
            rel = json.loads(_http_get(api, timeout=30).decode("utf-8", "replace"))
        tag = rel.get("tag_name")
        aname, url = select_asset(rel.get("assets", []), system, machine)
    except (urllib.error.URLError, OSError, ValueError):
        say("GitHub API unavailable - trying release page...")
        try:
            # only pass the kwarg when needed (keeps simple mocks/wrappers working)
            tag = _latest_tag(repo, tag_prefix=tag_prefix) if tag_prefix else _latest_tag(repo)
        except (urllib.error.URLError, OSError):
            tag = None
    if not tag and not url:
        raise RuntimeError("could not reach GitHub (API and release page both failed); "
                           "check the network, or download manually from "
                           "https://github.com/%s/releases" % repo)
    cands = [(aname, url)] if url else []
    if not cands and tag:
        cands = _candidate_urls(name, repo, tag, system, machine)
    if not cands:
        raise RuntimeError("no prebuilt %s binary for this platform in release %s"
                           % (name, tag or "?"))
    return tag or "?", cands


def _resolve_btbn(name, repo, system, machine, say):
    """BtbN FFmpeg-Builds publishes a rolling release with the fixed tag 'latest',
    so the URLs are fully constructible without any API."""
    s = str(system).lower()
    arm = str(machine).lower() in ("arm64", "aarch64")
    if s.startswith("win"):
        names = ["ffmpeg-master-latest-%s-gpl.zip" % ("winarm64" if arm else "win64")]
        if arm:
            names.append("ffmpeg-master-latest-win64-gpl.zip")   # x64 emulation fallback
    elif s == "darwin" or "mac" in s:
        raise RuntimeError("BtbN has no macOS ffmpeg builds - install via Homebrew "
                           "(brew install ffmpeg) or evermeet.cx, then place beside the app")
    else:
        names = ["ffmpeg-master-latest-%s-gpl.tar.xz" % ("linuxarm64" if arm else "linux64")]
    base = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/%s"
    return "latest", [(n, base % n) for n in names]


def parse_mediaarea_version(html):
    """Newest CLI version from MediaArea's Windows download page HTML."""
    versions = re.findall(r"MediaInfo_CLI_(\d+(?:\.\d+)+)_Windows", html)
    if not versions:
        return None
    return max(versions, key=lambda v: tuple(int(x) for x in v.split(".")))


def _resolve_mediaarea(name, repo, system, machine, say):
    s = str(system).lower()
    if s == "darwin" or "mac" in s:
        raise RuntimeError("MediaArea ships macOS CLI only as a .dmg - install via "
                           "Homebrew (brew install mediainfo) instead")
    say("querying mediaarea.net for the latest version...")
    html = _fetch_page("https://mediaarea.net/en/MediaInfo/Download/Windows",
                       name, "https://mediaarea.net/en/MediaInfo/Download")
    ver = parse_mediaarea_version(html)
    if not ver:
        raise RuntimeError("could not determine the latest MediaInfo version from "
                           "mediaarea.net - download manually from mediaarea.net")
    arm = str(machine).lower() in ("arm64", "aarch64")
    if s.startswith("win"):
        names = ["MediaInfo_CLI_%s_Windows_%s.zip" % (ver, "ARM64" if arm else "x64")]
    else:
        # the AWS-Lambda build is a static generic-Linux binary
        names = ["MediaInfo_CLI_%s_Lambda_%s.zip" % (ver, "arm64" if arm else "x86_64")]
    base = "https://mediaarea.net/download/binary/mediainfo/%s/%%s" % ver
    return ver, [(n, base % n) for n in names]


def parse_mkvtoolnix_version(html):
    versions = re.findall(r"mkvtoolnix-64-bit-(\d+(?:\.\d+)+)\.7z", html)
    if not versions:
        return None
    return max(versions, key=lambda v: tuple(int(x) for x in v.split(".")))


def _resolve_mkvtoolnix(name, repo, system, machine, say):
    s = str(system).lower()
    if not s.startswith("win"):
        raise RuntimeError("portable MKVToolNix is Windows-only - install via your "
                           "package manager (apt/dnf/brew install mkvtoolnix) instead")
    say("querying mkvtoolnix.download for the latest version...")
    html = _fetch_page("https://mkvtoolnix.download/downloads.html",
                       name, "https://mkvtoolnix.download")
    ver = parse_mkvtoolnix_version(html)
    if not ver:
        raise RuntimeError("could not determine the latest MKVToolNix version - "
                           "download manually from mkvtoolnix.download")
    aname = "mkvtoolnix-64-bit-%s.7z" % ver
    return ver, [(aname, "https://mkvtoolnix.download/windows/releases/%s/%s" % (ver, aname))]


def parse_bento4_listing(html, system, machine):
    """Newest Bento4 SDK zip for the platform from the bok.net directory listing.
    Returns (version_label, asset_name) or (None, None)."""
    s = str(system).lower()
    if s.startswith("win"):
        plats = ["x86_64-microsoft-win32"]
    elif s == "darwin" or "mac" in s:
        plats = ["universal-apple-macosx"]
    elif str(machine).lower() in ("arm64", "aarch64"):
        return None, None                      # no Linux ARM builds published
    else:
        plats = ["x86_64-unknown-linux"]
    best = (None, None)
    for ver, plat in re.findall(r"Bento4-SDK-(\d+-\d+-\d+-\d+)\.([A-Za-z0-9_.-]+)\.zip", html):
        if plat not in plats:
            continue
        key = tuple(int(x) for x in ver.split("-"))
        if best[0] is None or key > best[0]:
            best = (key, "Bento4-SDK-%s.%s.zip" % (ver, plat))
    if best[1] is None:
        return None, None
    return "-".join(str(x) for x in best[0]), best[1]


def _resolve_bento4(name, repo, system, machine, say):
    say("querying bok.net for the latest Bento4 SDK...")
    html = _fetch_page("https://www.bok.net/Bento4/binaries/",
                       name, "https://www.bok.net/Bento4/binaries/")
    ver, aname = parse_bento4_listing(html, system, machine)
    if not aname:
        raise RuntimeError("no prebuilt Bento4 SDK for this platform - build from "
                           "source (github.com/axiomatic-systems/Bento4) or use a package manager")
    return ver, [(aname, "https://www.bok.net/Bento4/binaries/" + aname)]


_RESOLVERS = {"btbn": _resolve_btbn, "mediaarea": _resolve_mediaarea,
              "mkvtoolnix": _resolve_mkvtoolnix, "bento4": _resolve_bento4}


# --- Install ------------------------------------------------------------------

def _write_exe(blob, dest, is_win):
    """Atomic write: .part then os.replace, so a failed download never leaves a
    broken binary behind. Raises a clear error if the target is running."""
    part = dest + ".part"
    with open(part, "wb") as fh:
        fh.write(blob)
    if not is_win:
        try:
            os.chmod(part, 0o755)
        except OSError:
            pass
    try:
        os.replace(part, dest)
    except OSError as e:
        try:
            os.remove(part)
        except OSError:
            pass
        raise RuntimeError("could not replace %s - close any process using it (%s)"
                           % (os.path.basename(dest), e))


def _find_7zip():
    """A real 7-Zip executable: on PATH, or in the standard install folders."""
    for n in ("7z", "7za", "7zr"):
        p = shutil.which(n)
        if p:
            return p
    for env in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        base = os.environ.get(env)
        if base:
            p = os.path.join(base, "7-Zip", "7z.exe")
            if os.path.isfile(p):
                return p
    return None


def _extract_7z(arc, outdir, say):
    """Try 7-Zip, then the py7zr module, then bsdtar. Returns (ok, error_text).
    Windows' bundled tar opens 7z containers but lacks the LZMA codec, so it is
    the last resort, not the first."""
    errs = []
    sz = _find_7zip()
    if sz:
        say("extracting with 7-Zip...")
        r = subprocess.run([sz, "x", "-y", "-o%s" % outdir, arc],
                           capture_output=True, timeout=600,
                           creationflags=CREATIONFLAGS)
        if r.returncode == 0:
            return True, None
        errs.append("7-Zip: " + (r.stderr or r.stdout or b"").decode("utf-8", "replace"))
    try:
        import py7zr
        say("extracting with py7zr...")
        with py7zr.SevenZipFile(arc) as z:
            z.extractall(outdir)
        return True, None
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        errs.append("py7zr: %s" % e)
    tar = shutil.which("tar")
    if tar:
        say("extracting with tar...")
        r = subprocess.run([tar, "-xf", arc, "-C", outdir], capture_output=True,
                           timeout=600, creationflags=CREATIONFLAGS)
        if r.returncode == 0:
            return True, None
        errs.append("tar: " + (r.stderr or b"").decode("utf-8", "replace"))
    return False, " ".join("; ".join(errs or ["no 7-Zip / py7zr / tar available"]).split())


def _install_7z_dir(data, dest_dir, subdir, primary_exe, say):
    """Extract a .7z archive into dest_dir/<subdir>/."""
    tmp = tempfile.mkdtemp(prefix="va_7z_", dir=dest_dir)
    arc = os.path.join(tmp, "pkg.7z")
    with open(arc, "wb") as fh:
        fh.write(data)
    ok, err = _extract_7z(arc, tmp, say)
    if not ok:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError("7z extraction failed (%s). Install 7-Zip (7zip.org) and "
                           "retry - or install MKVToolNix itself (winget install "
                           "MoritzBunkus.MKVToolNix); the app finds it automatically"
                           % err[:140])
    os.remove(arc)
    # find the folder that holds the primary exe (archives have a top-level dir)
    src = None
    for root, _dirs, files in os.walk(tmp):
        if primary_exe in files:
            src = root
            break
    if not src:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError("%s not found inside the archive" % primary_exe)
    final = os.path.join(dest_dir, subdir)
    old = final + ".old"
    try:
        if os.path.isdir(final):
            if os.path.isdir(old):
                shutil.rmtree(old, ignore_errors=True)
            os.replace(final, old)
        os.replace(src, final)
        shutil.rmtree(old, ignore_errors=True)
    except OSError as e:
        raise RuntimeError("could not move into place (%s) - is a tool from "
                           "'%s' still running?" % (e, subdir))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return os.path.join(final, primary_exe)


def tools_dir(base=None) -> str:
    """The canonical home for bundled binaries: <script>/tools (created on
    demand). Keeps ~700 MB of executables out of the code folder root;
    va_ffmpeg.find_tool searches it first among the vendor subfolders."""
    if not base:
        import va_paths
        base = va_paths.app_dir()      # beside the .exe when frozen
    d = os.path.join(base, "tools")
    os.makedirs(d, exist_ok=True)
    return d


_TIDY_NAMES = ("ffmpeg", "ffprobe", "ffplay", "dovi_tool", "hdr10plus_tool",
               "mediainfo", "mp4dump", "mp4info", "mkvmerge", "mkvinfo",
               "c2patool")


def tidy_tool_folder(base=None) -> list:
    """Move known tool executables (and a portable mkvtoolnix/ dir) from the
    script root into tools/. Same-volume os.replace = instant. Returns the
    list of names moved. Safe to call repeatedly."""
    base = base or os.path.dirname(os.path.abspath(__file__))
    dest = tools_dir(base)
    moved = []
    for n in _TIDY_NAMES:
        for fname in (n + ".exe", n):
            p = os.path.join(base, fname)
            if os.path.isfile(p):
                try:
                    os.replace(p, os.path.join(dest, fname))
                    moved.append(fname)
                except OSError:
                    pass
    mkv = os.path.join(base, "mkvtoolnix")
    if os.path.isdir(mkv) and not os.path.isdir(os.path.join(dest, "mkvtoolnix")):
        try:
            os.replace(mkv, os.path.join(dest, "mkvtoolnix"))
            moved.append("mkvtoolnix/")
        except OSError:
            pass
    if moved:
        try:
            import va_ffmpeg
            va_ffmpeg.reset_tool_caches()
        except Exception:   # noqa: BLE001
            pass
    return moved


def install_tool(name, dest_dir, system=None, machine=None, on_progress=None) -> str:
    """Download the latest release of `name` and install its executable(s) into
    dest_dir. Returns the path of the primary executable.

    GitHub-based tools fall back to the /releases/latest redirect plus known
    asset naming when the API is blocked or rate-limited. Executables are
    written atomically (.part + rename)."""
    if name not in TOOLS:
        raise ValueError("unknown tool: %s" % name)
    spec = TOOLS[name]
    system = system or platform.system()
    machine = machine or platform.machine()
    say = on_progress or (lambda m: None)
    is_win = str(system).lower().startswith("win")

    say("querying latest release...")
    if "repo" in spec:
        ver, candidates = _resolve_quietvoid(name, spec["repo"], system, machine, say,
                                             tag_prefix=spec.get("tag_prefix"))
    else:
        ver, candidates = _RESOLVERS[spec["source"]](name, None, system, machine, say)

    data = aname = err = None
    timeout = 1800 if name == "ffmpeg" else 600    # ffmpeg builds are ~150 MB
    for cand_name, cand_url in candidates:
        try:
            data = _http_get(cand_url, timeout=timeout, on_progress=say,
                             label="downloading %s" % cand_name)
            aname = cand_name
            break
        except urllib.error.HTTPError as e:   # e.g. 404 if naming changed; try next
            err = e
        except (urllib.error.URLError, OSError) as e:   # network down/blocked
            err = e
    if data is None:
        raise RuntimeError("download failed: %s - check the network, or download "
                           "%s manually from %s and place it beside the scripts"
                           % (err or "no matching asset", name, _manual_url(name)))

    os.makedirs(dest_dir, exist_ok=True)
    exes = [e + (".exe" if is_win else "") for e in tool_exes(name)]

    if spec.get("subdir"):                     # whole-folder install (.7z portable)
        dest = _install_7z_dir(data, dest_dir, spec["subdir"], exes[0], say)
        say("installed %s (%s)" % (exes[0], ver))
        return dest

    say("extracting...")
    found = extract_members(data, aname, exes)
    if exes[0] not in found:
        raise RuntimeError("%s not found in %s" % (exes[0], aname))
    primary = None
    for exe in exes:                            # primary required, the rest best-effort
        if exe not in found:
            continue
        dest = os.path.join(dest_dir, exe)
        _write_exe(found[exe], dest, is_win)
        if primary is None:
            primary = dest
    say("installed %s (%s, %d KB)" % (", ".join(sorted(found)), ver,
                                      sum(len(b) for b in found.values()) // 1024))
    return primary


def main():
    names = [a for a in sys.argv[1:] if a in TOOLS] or list(TOOLS)
    bad = [a for a in sys.argv[1:] if a not in TOOLS]
    if bad:
        print("unknown tool(s): %s   (available: %s)" % (", ".join(bad), ", ".join(TOOLS)))
        return 2
    rc = 0
    for n in names:
        try:
            print(install_tool(n, tools_dir(),     # app_dir-anchored (frozen-safe)
                               on_progress=lambda m, t=n: print("  [%s] %s" % (t, m))))
        except Exception as exc:  # noqa: BLE001
            print("  [%s] FAILED: %s" % (n, exc))
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
