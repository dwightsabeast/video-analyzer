#!/usr/bin/env python3
"""
selftest.py - headless engine regression test (no GUI / no Tk required).

Exercises the whole analysis stack against a video. With no argument it
generates tiny clips with the bundled/PATH ffmpeg; pass a path to test a real
file:  python selftest.py [video] [reference_for_quality]

Exits 0 only if every check passes.
"""

from __future__ import annotations

import os
import sys
import tempfile
import subprocess

import va_ffmpeg as F
import va_metrics as M
import va_scopes as S
import va_quality as Q
import va_export as X
import va_audio as AU
import va_qc as QCP
import va_compare as CMP
import va_perceptual as PCEPT
import va_temporal as TMP
import va_forensics as FRN
import va_hdr as HDRX
import va_dynhdr as DHX
import va_tools as VT
import va_plugins

va_plugins.load_headless()      # plugins register their QC profiles (speedrun)
VFY = va_plugins.load_module("speedrun", "sr_verify")   # None when not installed

_PASS = 0
_FAIL = 0


def check(name, cond, detail=""):
    global _PASS, _FAIL
    ok = bool(cond)
    _PASS += ok
    _FAIL += (not ok)
    print(("  PASS " if ok else "  FAIL ") + name + (("  -> " + detail) if detail else ""))
    return ok


def _gen(tmp, quick=False):
    exe = F.find_ffmpeg()
    if not exe:
        return None
    clip = os.path.join(tmp, "clip.mp4")
    low = os.path.join(tmp, "low.mp4")
    base = [exe, "-y", "-hide_banner", "-loglevel", "error"]
    subprocess.run(base + ["-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=2",
                           "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                           "-shortest", clip], check=True)
    subprocess.run(base + ["-i", clip, "-an", "-c:v", "libx264", "-b:v", "90k", low], check=True)
    hdr = os.path.join(tmp, "hdr.mp4")
    if quick:
        return {"clip": clip, "low": low, "hdr": None}
    try:
        subprocess.run(base + ["-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=1",
                               "-vf", "format=yuv420p10le", "-c:v", "libx265", "-preset", "ultrafast",
                               "-x265-params", "log-level=error:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
                               "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                               "-colorspace", "bt2020nc", hdr], capture_output=True, timeout=120)
    except Exception:
        pass
    return {"clip": clip, "low": low, "hdr": hdr if os.path.isfile(hdr) else None}


def _tools_checks(tmp):
    """Offline checks for the dovi_tool / hdr10plus_tool downloader (no network)."""
    import io
    import zipfile
    import tarfile
    import urllib.error

    mk = lambda names: [{"name": n, "browser_download_url": "u/" + n} for n in names]
    new = mk(["dovi_tool-2.3.2-universal-macOS.zip",
              "dovi_tool-2.3.2-x86_64-pc-windows-msvc.zip",
              "dovi_tool-2.3.2-aarch64-pc-windows-msvc.zip",
              "dovi_tool-2.3.2-x86_64-unknown-linux-musl.tar.gz",
              "dovi_tool-2.3.2-aarch64-unknown-linux-musl.tar.gz"])
    old = mk(["dovi_tool-2.1.0-aarch64-apple-darwin.tar.gz",
              "dovi_tool-2.1.0-x86_64-apple-darwin.tar.gz",
              "dovi_tool-2.1.0-x86_64-pc-windows-msvc.zip",
              "dovi_tool-2.1.0-x86_64-unknown-linux-musl.tar.gz"])
    for assets, sysname, mach, expect in [
            (new, "Windows", "AMD64", "x86_64-pc-windows"),
            (new, "Windows", "ARM64", "aarch64-pc-windows"),
            (new, "Darwin", "arm64", "universal-macos"),
            (new, "Darwin", "x86_64", "universal-macos"),
            (new, "Linux", "x86_64", "x86_64-unknown-linux"),
            (new, "Linux", "aarch64", "aarch64-unknown-linux"),
            (old, "Darwin", "arm64", "aarch64-apple-darwin"),
            (old, "Darwin", "x86_64", "x86_64-apple-darwin"),
            (old, "Windows", "AMD64", "x86_64-pc-windows")]:
        n, u = VT.select_asset(assets, sysname, mach)
        check("select_asset %s/%s" % (sysname, mach),
              n is not None and expect in n.lower() and u, str(n))
    check("select_asset no match", VT.select_asset(mk(["dovi_tool-2.3.2.sha256"]),
                                                   "Windows", "AMD64") == (None, None))

    zb = io.BytesIO()
    with zipfile.ZipFile(zb, "w") as z:
        z.writestr("dovi_tool.exe", b"MZfake")
    check("extract_exe zip", VT.extract_exe(zb.getvalue(), "a.zip", "dovi_tool.exe") == b"MZfake")
    tb = io.BytesIO()
    with tarfile.open(fileobj=tb, mode="w:gz") as t:
        data = b"\x7fELFfake"
        ti = tarfile.TarInfo("release/dovi_tool")
        ti.size = len(data)
        t.addfile(ti, io.BytesIO(data))
    check("extract_exe tar.gz nested",
          VT.extract_exe(tb.getvalue(), "a.tar.gz", "dovi_tool") == b"\x7fELFfake")

    # End-to-end install with the network mocked: API path, then API-blocked fallback.
    api_json = ('{"tag_name":"2.3.2","assets":[{"name":"dovi_tool-2.3.2-x86_64-pc-windows-msvc.zip",'
                '"browser_download_url":"https://x/dl.zip"}]}').encode()
    real_get, real_tag = VT._http_get, VT._latest_tag
    try:
        def fake_get(url, timeout=180, on_progress=None, label=""):
            if "api.github.com" in url:
                return api_json
            return zb.getvalue()
        VT._http_get = fake_get
        p = VT.install_tool("dovi_tool", tmp, system="Windows", machine="AMD64")
        check("install_tool via API", os.path.isfile(p) and open(p, "rb").read() == b"MZfake"
              and not os.path.exists(p + ".part"), p)

        def blocked_get(url, timeout=180, on_progress=None, label=""):
            if "api.github.com" in url:
                raise urllib.error.URLError("tunnel blocked")
            if url.endswith("dovi_tool-9.9-x86_64-pc-windows-msvc.zip"):
                return zb.getvalue()
            raise urllib.error.HTTPError(url, 404, "nf", None, None)
        VT._http_get = blocked_get
        VT._latest_tag = lambda repo, timeout=30: "9.9"
        p2 = VT.install_tool("dovi_tool", tmp, system="Windows", machine="AMD64")
        check("install_tool API-blocked fallback", os.path.isfile(p2), p2)
    finally:
        VT._http_get, VT._latest_tag = real_get, real_tag


def main():
    args = [a for a in sys.argv[1:] if a != '--quick']
    quick = '--quick' in sys.argv
    only = None
    for a in list(args):
        if a.startswith('--only'):
            only = a.split('=', 1)[1].split(',') if '=' in a else []
            args.remove(a)
    def _want(name):
        return only is None or any(name.startswith(o) or o.startswith(name) for o in only)
    tmp = tempfile.mkdtemp(prefix="va_selftest_")
    if args:
        media = {"clip": args[0], "low": args[1] if len(args) > 1 else None}
    else:
        media = _gen(tmp, quick)
        if not media:
            print("No ffmpeg available and no file given - cannot self-test.")
            return 2

    clip = media["clip"]
    print("Tools: ffmpeg=%s ffprobe=%s libvmaf=%s" % (
        bool(F.find_ffmpeg()), bool(F.find_ffprobe()), F.has_filter("libvmaf")))
    print("Testing:", os.path.basename(clip))

    info = F.probe(clip)
    check("probe ok", info["ok"] and info["width"] > 0,
          "%dx%d @ %.2ffps codec=%s" % (info["width"], info["height"], info["fps"], info["codec"]))

    vs = F.VideoSource(clip)
    fr = vs.frame_at(3)
    check("decode frame", fr is not None and fr.shape[2] == 3,
          str(None if fr is None else fr.shape) + " via " + vs.backend)

    t = M.signalstats(clip)
    check("signalstats frames", len(t) > 0, "%d frames" % len(t))
    check("signalstats columns", all(len(t.cols[k]) == len(t) for k in M.SIGNALSTATS_TAGS))
    check("summary has YAVG", "YAVG" in t.summary())

    check("black_segments list", isinstance(M.black_segments(clip), list))
    check("freeze_segments list", isinstance(M.freeze_segments(clip), list))
    check("scene_cuts list", isinstance(M.scene_cuts(clip), list))
    loud = M.loudness(clip)
    check("loudness", loud is None or "integrated_lufs" in loud, str(loud))
    fs = M.frame_sizes(clip)
    check("frame_sizes", len(fs["t"]) > 0, "%d frames, types=%s" % (len(fs["t"]), set(fs["type"])))
    if not quick:
        mo = M.motion_series(F.VideoSource(clip), step=3)
        check("motion_series", isinstance(mo["motion"], list))

    if fr is not None:
        check("vectorscope", S.vectorscope(fr, 300).shape == (300, 300, 3))
        check("waveform", S.waveform(fr, 400, 200).shape == (200, 400, 3))
        check("rgb_parade", S.rgb_parade(fr, 400, 200).shape == (200, 400, 3))
        check("false_color", S.false_color(fr, 320, 240).shape == (240, 320, 3))
        check("histogram", S.histogram(fr, 400, 200).shape == (200, 400, 3))
        cie, cov = S.cie_gamut(fr, 300)
        check("cie_gamut", cie.shape == (300, 300, 3) and "coverage_2020_pct" in cov, str(cov))
    vs.close()

    csvp = os.path.join(tmp, "o.csv")
    X.write_csv(t, csvp)
    check("csv rows", len(open(csvp).read().splitlines()) == len(t) + 1)
    jp = os.path.join(tmp, "o.json")
    X.write_json(t, jp, source=info)
    import json
    doc = json.load(open(jp))
    check("json frame_count", doc["frame_count"] == len(t))
    hp = os.path.join(tmp, "o.html")
    X.write_html_report(t, hp, source=info, events={"scene_cuts": M.scene_cuts(clip)},
                        loudness=loud)
    html = open(hp).read()
    check("html report", "<polyline" in html and "Per-frame summary" in html)

    if media.get("low"):
        res = Q.compare(media["low"], clip, vmaf=Q.vmaf_available())
        p = res.get("psnr")
        check("quality PSNR", p is not None and p["average"] is not None,
              "PSNR=%.2f dB" % p["average"] if p and p["average"] not in (None,) else "n/a")
        s = res.get("ssim")
        check("quality SSIM", s is not None and 0 < (s["average"] or 0) <= 1.0,
              "SSIM=%.4f" % s["average"] if s and s["average"] else "n/a")
    ident = Q.compare(clip, clip, vmaf=False)
    check("identical PSNR=inf", ident["psnr"]["average"] == float("inf"))

    check("audio present", AU.has_audio(clip))
    ld = AU.loudness(clip)
    check("loudness summary", bool(ld and ld.get("summary") and ld["summary"].get("integrated_lufs") is not None),
          str(ld["summary"]) if ld else "none")
    check("astats channels", bool((AU.astats(clip) or {}).get("channels")))
    check("silence list", isinstance(AU.silence_segments(clip), list))
    ctx = {"probe": info, "summary": t.summary(), "loudness": (ld["summary"] if ld else None),
           "events": {"black": [], "freeze": [], "scene_cuts": []}, "gamut": None, "silence": []}
    qcr = QCP.evaluate(ctx, "general")
    check("qc verdict", qcr.get("verdict") in ("pass", "warn", "fail"), qcr.get("verdict"))
    if fr is not None:
        d = CMP.difference(fr, fr, "heatmap")
        check("difference render", d is not None and d.shape == fr.shape[:2] + (3,))
    if media.get("low") and not quick:
        lad = CMP.encode_ladder(clip, [300, 1200])
        check("encode ladder", len(lad) >= 1 and lad[0].get("ssim") is not None, "%d rungs" % len(lad))
        check("optimal bitrate", CMP.optimal_bitrate(lad) is not None)

    import numpy as _np
    ramp = _np.tile(_np.linspace(0, 255, 480), (270, 1))
    band_bgr = _np.stack([(_np.floor(ramp / 255 * 64) / 64 * 255).astype("uint8")] * 3, axis=2)
    flat_bgr = _np.full((270, 480, 3), 128, "uint8")
    bp = PCEPT.banding_map(band_bgr)[1]
    fp = PCEPT.banding_map(flat_bgr)[1]
    check("banding (banded > flat)", bp > fp + 5, "banded=%.1f flat=%.1f" % (bp, fp))
    if fr is not None:
        check("saliency map", float(PCEPT.saliency_map(fr).max()) <= 1.0)
    cad = TMP.cadence(F.VideoSource(clip))
    check("cadence dict", ("dup_pct" in cad and "comb_pct" in cad), cad.get("cadence"))
    pse = TMP.pse_flashes(F.VideoSource(clip))
    check("pse flashes", "risk" in pse)
    rc = FRN.recompression(clip)
    check("recompression", "blockiness" in rc)
    check("content credentials", isinstance(FRN.content_credentials(clip), dict))
    ctx2 = dict(ctx)
    ctx2.update({"banding": bp, "pse": pse, "cadence": cad})
    check("qc frontier verdict", QCP.evaluate(ctx2, "general").get("verdict") in ("pass", "warn", "fail"))
    if media.get("hdr"):
        nm = HDRX.nits_map(media["hdr"], 5)
        check("hdr nits_map", nm is not None and float(nm.max()) > 0,
              "max %.0f nits" % (float(nm.max()) if nm is not None else 0))
        check("hdr maxcll", "measured_maxcll" in HDRX.maxcll_maxfall(media["hdr"]))
        meta = DHX.inspect(media["hdr"])
        check("dynhdr inspect", isinstance(meta, dict) and "dolby_vision" in meta)
    mock = {"streams": [{"codec_type": "video", "side_data_list": [
        {"side_data_type": "DOVI configuration record", "dv_profile": 7, "dv_level": 6,
         "rpu_present_flag": 1, "bl_present_flag": 1, "el_present_flag": 1,
         "dv_bl_signal_compatibility_id": 0}]}], "frames": []}
    pm = DHX.parse_hdr_metadata(mock)
    check("DV profile parse", pm["dolby_vision"]["profile"] == 7 and len(pm["flags"]) >= 1,
          pm["dolby_vision"]["profile_desc"])

    if _want('tools'):
        _tools_checks(tmp)
        _tools_checks_multi(tmp)
        _tools_checks_7z(tmp)
    if _want('theme'):
        _theme_checks(tmp)
    if _want('2026'):
        chk_2026_hardening(tmp, media, t)
    if _want('wired'):
        chk_wired_tools(tmp, media)
    if _want('forensics'):
        chk_forensics_suite(tmp, media, quick)
    if _want('aspect'):
        chk_aspect_suite(tmp, media)
    if _want('verify'):
        if VFY is not None:
            chk_verify_suite(tmp, media, quick)
        else:
            print("verify suite skipped: speedrun plugin not installed")
    if _want('advanced_export'):
        chk_advanced_export(tmp, t)
    if _want('dovi_p5'):
        chk_dovi_p5(tmp)
    if _want('native_scopes'):
        chk_native_scopes(tmp)
    if _want('perf'):
        chk_perf(tmp, media, quick)
    if _want('audio_forensics'):
        chk_audio_forensics(tmp, media, quick)
    if _want('audio_gui'):
        chk_audio_gui(tmp)
    if _want('rpu'):
        chk_rpu_parse(tmp)
    if _want('tools_dir'):
        chk_tools_dir(tmp)
    if _want('dynhdr_compat'):
        chk_dynhdr_compat_flag(tmp)
    if _want('silent_spawns'):
        chk_silent_spawns()

    print("\n%d passed, %d failed" % (_PASS, _FAIL))
    return 0 if _FAIL == 0 else 1


def _tools_checks_multi(tmp):
    """Offline checks for the multi-source downloader (no network)."""
    import io
    import zipfile
    import urllib.error

    # BtbN ffmpeg resolver: fixed-tag URLs, per platform
    say = lambda m: None
    ver, c = VT._resolve_btbn("ffmpeg", None, "Windows", "AMD64", say)
    check("btbn win64", c[0][0] == "ffmpeg-master-latest-win64-gpl.zip" and
          "releases/download/latest/" in c[0][1], c[0][0])
    ver, c = VT._resolve_btbn("ffmpeg", None, "Windows", "ARM64", say)
    check("btbn winarm64 + fallback", c[0][0].endswith("winarm64-gpl.zip") and len(c) == 2)
    ver, c = VT._resolve_btbn("ffmpeg", None, "Linux", "x86_64", say)
    check("btbn linux64 tar.xz", c[0][0].endswith("linux64-gpl.tar.xz"))
    try:
        VT._resolve_btbn("ffmpeg", None, "Darwin", "arm64", say)
        check("btbn macos raises", False)
    except RuntimeError as e:
        check("btbn macos raises", "Homebrew" in str(e))

    # MediaArea version parsing (sample from the real download page)
    html = ('<a href="https://mediaarea.net/download/binary/mediainfo/26.05/'
            'MediaInfo_CLI_26.05_Windows_x64.zip">v26.05</a> '
            'MediaInfo_CLI_21.03_Windows_x64.zip MediaInfo_CLI_0.7.60_Windows_i386.zip')
    check("mediaarea version parse", VT.parse_mediaarea_version(html) == "26.05")
    check("mediaarea version none", VT.parse_mediaarea_version("<html></html>") is None)

    # MKVToolNix version parsing
    html = 'mkvtoolnix-64-bit-99.0.7z ... mkvtoolnix-64-bit-98.0.7z'
    check("mkvtoolnix version parse", VT.parse_mkvtoolnix_version(html) == "99.0")

    # Bento4 listing parsing (includes the malformed '638*' entry from the real site)
    html = ('Bento4-SDK-1-6-0-638*.x86_64-unknown-linux.zip '
            'Bento4-SDK-1-6-0-640.x86_64-microsoft-win32.zip '
            'Bento4-SDK-1-6-0-641.x86_64-microsoft-win32.zip '
            'Bento4-SDK-1-6-0-641.universal-apple-macosx.zip '
            'Bento4-SDK-1-6-0-641.x86_64-unknown-linux.zip')
    v, a = VT.parse_bento4_listing(html, "Windows", "AMD64")
    check("bento4 win latest", a == "Bento4-SDK-1-6-0-641.x86_64-microsoft-win32.zip", str(a))
    v, a = VT.parse_bento4_listing(html, "Darwin", "arm64")
    check("bento4 mac universal", a == "Bento4-SDK-1-6-0-641.universal-apple-macosx.zip", str(a))
    v, a = VT.parse_bento4_listing(html, "Linux", "aarch64")
    check("bento4 linux-arm none", a is None)

    # multi-exe extraction (BtbN-style layout)
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, "w") as z:
        z.writestr("ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe", b"MZff")
        z.writestr("ffmpeg-master-latest-win64-gpl/bin/ffprobe.exe", b"MZfp")
        z.writestr("ffmpeg-master-latest-win64-gpl/bin/ffplay.exe", b"MZfl")
    got = VT.extract_members(zb.getvalue(), "a.zip", ["ffmpeg.exe", "ffprobe.exe", "ffplay.exe"])
    check("extract_members ffmpeg trio", set(got) == {"ffmpeg.exe", "ffprobe.exe", "ffplay.exe"}
          and got["ffprobe.exe"] == b"MZfp")

    # mocked end-to-end: mp4dump (listing + SDK zip), ffmpeg (fixed URL zip)
    sdk = io.BytesIO()
    with zipfile.ZipFile(sdk, "w") as z:
        z.writestr("Bento4-SDK-1-6-0-641.x86_64-microsoft-win32/bin/mp4dump.exe", b"MZd")
        z.writestr("Bento4-SDK-1-6-0-641.x86_64-microsoft-win32/bin/mp4info.exe", b"MZi")
    real_get = VT._http_get
    try:
        def fake_get(url, timeout=180, on_progress=None, label=""):
            if "bok.net/Bento4/binaries/" in url and url.endswith("/"):
                return html.encode()
            if url.endswith("microsoft-win32.zip"):
                return sdk.getvalue()
            if url.endswith("win64-gpl.zip"):
                return zb.getvalue()
            raise urllib.error.HTTPError(url, 404, "nf", None, None)
        VT._http_get = fake_get
        p = VT.install_tool("mp4dump", tmp, system="Windows", machine="AMD64")
        check("install mp4dump+mp4info", p.endswith("mp4dump.exe") and os.path.isfile(p)
              and os.path.isfile(os.path.join(tmp, "mp4info.exe")), p)
        p = VT.install_tool("ffmpeg", tmp, system="Windows", machine="AMD64")
        check("install ffmpeg trio", p.endswith("ffmpeg.exe") and
              os.path.isfile(os.path.join(tmp, "ffprobe.exe")) and
              os.path.isfile(os.path.join(tmp, "ffplay.exe")), p)
    finally:
        VT._http_get = real_get




def _tools_checks_7z(tmp):
    """Offline checks for the .7z directory-install ladder (extractor mocked)."""
    import sys as _sys

    dest = os.path.join(tmp, "7zdest")
    os.makedirs(dest, exist_ok=True)
    real_find, real_run, real_which = VT._find_7zip, VT.subprocess.run, VT.shutil.which

    class _R:
        returncode, stderr, stdout = 0, b"", b""

    def fake_run(args, **kw):
        out = next(a[2:] for a in args if isinstance(a, str) and a.startswith("-o"))
        os.makedirs(os.path.join(out, "mkvtoolnix"), exist_ok=True)
        with open(os.path.join(out, "mkvtoolnix", "mkvextract.exe"), "wb") as fh:
            fh.write(b"MZmkv")
        with open(os.path.join(out, "mkvtoolnix", "Qt6Core.dll"), "wb") as fh:
            fh.write(b"MZdll")
        return _R()

    try:
        VT._find_7zip, VT.subprocess.run = (lambda: "FAKE7Z"), fake_run
        p = VT._install_7z_dir(b"x", dest, "mkvtoolnix", "mkvextract.exe", lambda m: None)
        check("7z dir install", p.endswith("mkvextract.exe") and os.path.isfile(p)
              and os.path.isfile(os.path.join(dest, "mkvtoolnix", "Qt6Core.dll"))
              and not any(f.startswith("va_7z_") for f in os.listdir(dest)), p)
        p2 = VT._install_7z_dir(b"x", dest, "mkvtoolnix", "mkvextract.exe", lambda m: None)
        check("7z re-install over existing", os.path.isfile(p2)
              and not os.path.isdir(os.path.join(dest, "mkvtoolnix.old")))

        VT._find_7zip = lambda: None
        VT.shutil.which = lambda n: None
        _sys.modules["py7zr"] = None
        try:
            VT._install_7z_dir(b"x", dest, "mkvtoolnix", "mkvextract.exe", lambda m: None)
            check("7z no-extractor error", False)
        except RuntimeError as e:
            s = str(e)
            check("7z no-extractor error", "7zip.org" in s and "winget" in s
                  and "\n" not in s, s[:90])
        finally:
            del _sys.modules["py7zr"]
    finally:
        VT._find_7zip, VT.subprocess.run, VT.shutil.which = real_find, real_run, real_which




def _theme_checks(tmp):
    """Headless checks for va_theme (palette data + preference persistence; no Tk)."""
    import va_theme as TH

    check("theme palettes same keys", set(TH.DARK) == set(TH.LIGHT),
          str(set(TH.DARK) ^ set(TH.LIGHT)))
    check("theme palettes differ", TH.DARK["bg"] != TH.LIGHT["bg"]
          and TH.DARK["text"] != TH.LIGHT["text"])
    need = {"bg", "surface", "raised", "hairline", "text", "muted", "faint",
            "accent", "accent_fg", "ok", "warn", "err", "info",
            "editor_bg", "editor_fg", "sel_bg", "sel_fg"}
    check("theme required keys", need <= set(TH.DARK), str(need - set(TH.DARK)))
    check("well keys", {"bg", "panel", "text", "faint", "line", "cut", "play",
                        "ev_black", "ev_freeze"} <= set(TH.WELL))
    hexish = lambda d: all(v.startswith("#") and len(v) == 7 for v in d.values())
    check("theme colors are hex", hexish(TH.DARK) and hexish(TH.LIGHT) and hexish(TH.WELL))

    def lum(h):
        r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    check("dark text readable", lum(TH.DARK["text"]) - lum(TH.DARK["bg"]) > 100)
    check("light text readable", lum(TH.LIGHT["bg"]) - lum(TH.LIGHT["text"]) > 100)
    check("well graphics readable", min(lum(TH.WELL[k]) for k in ("line", "play", "cut"))
          - lum(TH.WELL["bg"]) > 60)

    real_pref = TH._PREF
    try:
        TH._PREF = os.path.join(tmp, "va_ui.json")
        TH.save_pref("light")
        check("theme pref roundtrip", TH.load_pref() == "light")
        open(TH._PREF, "w").write("not json {")
        check("theme pref corrupt -> default", TH.load_pref() == "dark")
    finally:
        TH._PREF = real_pref

    live = dict(TH.C)
    TH.C.clear()
    TH.C.update(TH.palette("light"))
    ok_live = TH.C["bg"] == TH.LIGHT["bg"]
    TH.C.clear()
    TH.C.update(live)
    check("live palette swaps in place", ok_live)


def chk_2026_hardening(tmp, media, table):
    """Regression checks for the 2026-06 hardening session. Offline + fast:
    parser seams are fed synthetic tool output; the only ffmpeg work is two
    tiny encodes and one sub-second compare on the generated clip."""
    import io
    import re
    import time
    import numpy as np
    import va_probe as VP
    import va_hwaccel

    clip = media["clip"]
    exe = F.find_ffmpeg()

    # (a) quality lavfi stats logs are cwd-relative (absolute paths break on
    # Windows drive colons inside filtergraphs).
    captured = {}
    real_qrun = Q._run
    try:
        def _capture(args, cwd=None, timeout=0):
            captured["args"], captured["cwd"] = list(args), cwd
            return None
        Q._run = _capture
        Q._psnr("d.mp4", "r.mp4", {"width": 64, "height": 64}, {"width": 64, "height": 64})
    finally:
        Q._run = real_qrun
    args = captured.get("args") or []
    graph = args[args.index("-lavfi") + 1] if "-lavfi" in args else ""
    m = re.search(r"stats_file=([^:]+)", graph)
    sf = m.group(1) if m else None
    check("quality stats_file cwd-relative", sf == "psnr.log" and captured.get("cwd"),
          "stats_file=%r cwd set=%s" % (sf, bool(captured.get("cwd"))))

    # (b) compare(): mismatched durations stop at the shorter input + a note.
    short1 = os.path.join(tmp, "short1s.mp4")
    if exe:
        subprocess.run([exe, "-y", "-hide_banner", "-loglevel", "error", "-i", clip,
                        "-t", "1", "-an", "-c:v", "libx264", "-preset", "ultrafast",
                        short1], capture_output=True, timeout=120)
    if os.path.isfile(short1):
        res = Q.compare(short1, clip, vmaf=False)
        nper = len((res.get("psnr") or {}).get("per_frame") or [])
        notes = res.get("notes") or []
        check("compare stops at shorter input", 0 < nper <= 27, "%d per-frame rows" % nper)
        check("compare duration-mismatch note", any("durations differ" in n for n in notes),
              str(notes))
    else:
        check("compare stops at shorter input", False, "could not encode 1s clip")
        check("compare duration-mismatch note", False, "could not encode 1s clip")

    # (c) ebur128 'Peak: -inf dBFS' (digital silence) parses to -inf, not None.
    ebur = ("Summary:\n\n  Integrated loudness:\n    I:         -70.0 LUFS\n"
            "    Threshold:   0.0 LUFS\n\n  Loudness range:\n    LRA:         0.0 LU\n\n"
            "  True peak:\n    Peak:       -inf dBFS\n")
    real_mrun = M._run
    try:
        M._run = lambda a, timeout=None, cancel=None: subprocess.CompletedProcess(a, 0, "", ebur)
        loud = M.loudness("x.wav")
    finally:
        M._run = real_mrun
    check("true peak -inf parsed", bool(loud) and loud.get("true_peak_dbfs") == float("-inf"),
          str(loud))

    # (d) freeze running to EOF gets the stream duration as its end.
    try:
        va_hwaccel._DECODE_BY_CODEC.setdefault("", None)   # keep hw probing out of a parser test
    except AttributeError:
        pass
    real_mrun, real_dur = M._run, M.media_duration
    try:
        M._run = lambda a, timeout=None, cancel=None: subprocess.CompletedProcess(
            a, 0, "lavfi.freezedetect.freeze_start=1.25\n", "")
        M.media_duration = lambda p: 4.5
        segs = M.freeze_segments("x.mp4")
    finally:
        M._run, M.media_duration = real_mrun, real_dur
    check("freeze-to-EOF end filled", segs == [(1.25, 4.5)], str(segs))

    # (e) ffprobe-fallback time fields: 5.0+ rows (pts_time + best_effort) and
    # 4.4 rows (best_effort only) both parse.
    class _FakeProc:
        def __init__(self, out):
            self.stdout = io.StringIO(out)
        def terminate(self):
            pass
        def kill(self):
            pass
        def wait(self, timeout=None):
            pass
    tagvals = ",".join(str(i) for i in range(len(M.SIGNALSTATS_TAGS)))
    real_popen = subprocess.Popen
    try:
        subprocess.Popen = lambda *a, **k: _FakeProc(
            "0.000000,0.000000,%s\n0.040000,0.040000,%s\n" % (tagvals, tagvals))
        t5 = M._signalstats_ffprobe("x.mp4", None, None)
        subprocess.Popen = lambda *a, **k: _FakeProc(
            "0.000000,%s\n0.040000,%s\n" % (tagvals, tagvals))
        t44 = M._signalstats_ffprobe("x.mp4", None, None)
    finally:
        subprocess.Popen = real_popen
    check("ffprobe rows v5+ (pts_time)", t5.t == [0.0, 0.04] and t5.cols["YAVG"] == [1.0, 1.0],
          str(t5.t))
    check("ffprobe rows v4.4 (best_effort)", t44.t == [0.0, 0.04] and t44.cols["BRNG"] == [12.0, 12.0],
          str(t44.t))

    # (f) classify_hdr: DV profile 5 has no HDR10 fallback; 8.1 advertises one.
    pqs = {"color_transfer": "smpte2084", "color_primaries": "bt2020",
           "color_space": "bt2020nc", "pix_fmt": "yuv420p10le"}
    dovi81 = {"side_data_type": "DOVI configuration record", "dv_profile": 8,
              "dv_level": 6, "rpu_present_flag": 1, "el_present_flag": 0,
              "bl_present_flag": 1, "dv_bl_signal_compatibility_id": 1}
    dovi5 = dict(dovi81, dv_profile=5, dv_bl_signal_compatibility_id=0)
    f5 = VP.classify_hdr(pqs, [dovi5])["HDR Format"]
    f81 = VP.classify_hdr(pqs, [dovi81])["HDR Format"]
    check("classify P5 no-HDR10-fallback", "P5" in f5 and "HDR10" not in f5.replace("IPT-PQ", ""), f5)
    check("classify P8.1 DV + HDR10", "Dolby Vision" in f81 and "+ HDR10" in f81, f81)

    # (g) CTA-861.3: declared MaxCLL/MaxFALL of 0 (or junk) means UNKNOWN.
    check("declared MaxCLL 0/junk = unknown",
          HDRX._declared_limit(0) is None and HDRX._declared_limit("junk") is None
          and HDRX._declared_limit(None) is None and HDRX._declared_limit(600) == 600)

    # (h) banding detector: hard staircase flags, dithered gradient stays quiet.
    h, w = 180, 320
    x = np.arange(w, dtype=np.float32)
    stair = np.tile(np.floor(16 + np.floor(x / 40) * 8), (h, 1))
    rng = np.random.default_rng(7)
    dith = np.clip(np.tile(16 + x / w * 32, (h, 1)) + rng.normal(0, 1.2, (h, w)), 0, 255)
    as_bgr = lambda lum: np.repeat(lum.astype(np.uint8)[..., None], 3, axis=2)
    stair_pct = PCEPT.banding_map(as_bgr(stair))[1]
    dith_pct = PCEPT.banding_map(as_bgr(dith))[1]
    check("banding staircase >= 10%", stair_pct >= 10.0, "%.2f%%" % stair_pct)
    check("banding dithered <= 3%", dith_pct <= 3.0, "%.2f%%" % dith_pct)

    # (i) CIE gamut: saturated BT.709 bars must not read as outside Rec.709.
    bars = np.zeros((120, 280, 3), np.uint8)
    for i, (r, g, b) in enumerate([(255, 255, 255), (255, 255, 0), (0, 255, 255),
                                   (0, 255, 0), (255, 0, 255), (255, 0, 0), (0, 0, 255)]):
        bars[:, i * 40:(i + 1) * 40] = (b, g, r)
    _, cov = S.cie_gamut(bars, 200)
    check("cie 709 bars outside_709 <= 0.3%",
          float(cov.get("outside_709_pct", 99.0)) <= 0.3, str(cov))

    # (j) HTML export: UTF-8 bytes regardless of locale; None -> 'n/a';
    # -inf true peak labelled as digital silence.
    hp = os.path.join(tmp, "hardening.html")
    cjk = "森のクマ.mp4"
    X.write_html_report(table, hp, source={"path": cjk, "codec": "h264", "width": 320,
                                           "height": 240, "fps": 25.0},
                        loudness={"integrated_lufs": -23.0, "lra_lu": None,
                                  "true_peak_dbfs": float("-inf")})
    raw = open(hp, "rb").read()
    check("html export UTF-8 encoded", cjk.encode("utf-8") in raw)
    txt = raw.decode("utf-8")
    check("html None cell -> n/a", ">n/a<" in txt and ">None<" not in txt)
    check("html -inf cell labelled", "digital silence" in txt)

    # (k) ffprobe_json(frames=True) returns a VIDEO frame even when an audio
    # stream comes first in the container.
    af = os.path.join(tmp, "audiofirst.mp4")
    if exe:
        subprocess.run([exe, "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
                        "-f", "lavfi", "-i", "testsrc2=s=160x120:r=10:d=0.5",
                        "-map", "0:a", "-map", "1:v", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", af],
                       capture_output=True, timeout=120)
    pj = F.ffprobe_json(af, frames=True) if os.path.isfile(af) else None
    frs = (pj or {}).get("frames") or []
    check("ffprobe frames=True is video-first",
          bool(frs) and all(f.get("media_type") == "video" for f in frs),
          str([(f.get("stream_index"), f.get("media_type")) for f in frs])[:80])

    # (l) dynhdr demux: a consumer that dies instantly must not deadlock the pipe.
    t0 = time.monotonic()
    rc = DHX._demux_into(exe, clip, [sys.executable, "-c", "import sys; sys.exit(1)"],
                         timeout=8)
    took = time.monotonic() - t0
    check("dynhdr demux exit-1 returns fast", took < 10.0 and rc == 1,
          "rc=%s in %.1fs" % (rc, took))



def chk_wired_tools(tmp, media):
    """Checks for the wired-up helper integrations (2026-06-10): XPSNR metric,
    VMAF model selection, MKV/DV signaling report, codec ladders, c2patool.
    Offline-safe: live metric runs happen only when the ffmpeg build has the
    filter; otherwise the graceful-gate behavior is what gets asserted."""
    import io
    import base64
    import va_quality as Q
    import va_compare as VC
    import va_tools as VT
    import va_dynhdr as VD
    import va_forensics as VF

    clip = media["clip"]

    # --- VMAF model selection (pure logic; live model echo when available)
    hd, uhd = {"width": 1920, "height": 1080}, {"width": 3840, "height": 2160}
    check("vmaf model auto: hd->default uhd->4k",
          Q.vmaf_model_for("auto", hd) is None
          and Q.vmaf_model_for("auto", uhd) == "vmaf_4k_v0.6.1")
    check("vmaf model explicit 4k/neg/default",
          Q.vmaf_model_for("4k", hd) == "vmaf_4k_v0.6.1"
          and Q.vmaf_model_for("neg", uhd) == "vmaf_v0.6.1neg"
          and Q.vmaf_model_for("default", uhd) is None)

    # --- XPSNR: per-frame log parser (fixture) + live or gated path
    log = io.StringIO("n: 1  XPSNR y: 34.5671  XPSNR u: 40.1  XPSNR v: 39.9\n"
                      "n: 2  XPSNR y: inf  XPSNR u: inf  XPSNR v: inf\n")
    per = Q._parse_xpsnr_log(log)
    check("xpsnr log parser", per == [34.5671, float("inf")], repr(per))
    res = Q.compare(clip, clip, vmaf=Q.vmaf_available())
    check("compare carries xpsnr + flag",
          "xpsnr" in res and isinstance(res.get("xpsnr_available"), bool))
    if Q.xpsnr_available():
        x = res.get("xpsnr") or {}
        check("xpsnr live self-compare", x.get("average") == float("inf")
              or (x.get("per_frame") and min(x["per_frame"]) > 50),
              repr(x.get("average")))
        if Q.vmaf_available():
            r4k = Q.compare(clip, clip, vmaf=True, vmaf_model="4k")
            check("vmaf model echoed in result",
                  (r4k.get("vmaf") or {}).get("model") == "vmaf_4k_v0.6.1")
    else:
        check("xpsnr gated cleanly on old ffmpeg", res.get("xpsnr") is None)

    # --- MKV / DV signaling
    dvcc = bytes([1, 0, 8 << 1, (9 << 3) | 0b101, 1 << 4, 0, 0, 0])
    bits = VD._dv_config_bits(dvcc)
    check("dvcC bit decode (P8.1)", bits == {"profile": 8, "level": 9, "rpu": True,
                                             "el": False, "bl": True,
                                             "compatibility_id": 1}, repr(bits))
    j = {"container": {"type": "Matroska"},
         "tracks": [{"id": 0, "type": "video", "codec": "HEVC",
                     "properties": {"pixel_dimensions": "3840x2160",
                                    "block_addition_mappings": [
                                        {"id_type": 0x64766343,
                                         "id_extra_data": base64.b64encode(dvcc).decode()}]}}]}
    summ = "\n".join(VD._mkv_summary(j))
    check("mkv summary decodes DV mapping",
          "profile 8.1" in summ and "HDR10 (profile 8.1)" in summ, summ[:80])
    check("mkv summary flags missing DV",
          any("no Dolby Vision block-addition mapping" in ln
              for ln in VD._mkv_summary({"container": {}, "tracks": [
                  {"id": 0, "type": "video", "properties": {}}]})))
    check("mkv_report rejects non-mkv",
          VD.mkv_report(clip)[1].startswith("mkvinfo reads Matroska"))
    mkv = os.path.join(tmp, "wired.mkv")
    subprocess.run([F.find_ffmpeg(), "-v", "error", "-y", "-i", clip, "-t", "0.5",
                    "-c", "copy", mkv], capture_output=True, timeout=60)
    if os.path.isfile(mkv):
        text, msg = VD.mkv_report(mkv)
        if F.find_tool("mkvmerge") or F.find_tool("mkvinfo"):
            check("mkv_report live", msg == "ok" and text and "Container" in text, msg)
        else:
            check("mkv_report tool-missing message",
                  msg == "MKVToolNix not installed (Tools dialog)", msg)

    # --- va_tools resolver naming for the c2pa-rs release stream
    cand = [a for a, _ in VT._candidate_urls("c2patool", "contentauth/c2pa-rs",
                                             "c2patool-v0.26.56", "Windows", "AMD64")]
    check("c2patool candidate asset naming",
          cand and cand[0] == "c2patool-v0.26.56-x86_64-pc-windows-msvc.zip", repr(cand))
    check("registry: mkvtoolnix trio + c2patool",
          VT.tool_exes("mkvextract") == ("mkvextract", "mkvinfo", "mkvmerge")
          and "c2patool" in VT.TOOLS)

    # --- codec ladders
    check("encoder_available returns bool",
          isinstance(VC.encoder_available("h264"), bool)
          and isinstance(VC.encoder_available("av1"), bool))
    bad = VC.encode_ladder(clip, [100], codec="not_a_codec")
    check("ladder unknown codec -> [] + reason",
          bad == [] and "unknown codec" in (VC.last_error() or ""), repr(VC.last_error()))

    # --- c2patool report fallback shape (no live tool assumed)
    text, msg = VF.content_credentials_report(clip)
    check("c2pa report returns the documented contract",
          (text is None and isinstance(msg, str) and msg) or
          (isinstance(text, str) and isinstance(msg, str)), repr(msg)[:70])
    if not F.find_tool("c2patool"):
        check("c2pa fallback mentions presence scan", "presence scan" in msg, msg)



def chk_forensics_suite(tmp, media, quick=False):
    """The expanded va_forensics battery: corroborated splices, container walk,
    encoder fingerprint, pixel maps, loop scan, ENF, orchestrated report and the
    QC integrity profile. Heavy fixtures (loop/ENF/long splice) are skipped
    under --quick."""
    import hashlib
    import struct
    import shutil
    import numpy as np

    clip = media["clip"]
    exe = F.find_ffmpeg()

    # chain-of-custody hash
    want = hashlib.sha256(open(clip, "rb").read()).hexdigest()
    check("forensics sha256", FRN.sha256_file(clip) == want)

    # frame_sizes now carries keyframe flags (splice scan depends on it)
    fs = M.frame_sizes(clip)
    check("frame_sizes key column", "key" in fs and len(fs["key"]) == len(fs["t"])
          and sum(fs["key"]) >= 1, "kf=%s" % sum(fs.get("key", [])))

    # splice scan: structure + head/tail artifact guard on the plain clip
    sp = FRN.splice_scan(clip)
    check("splice_scan structure", isinstance(sp.get("candidates"), list)
          and "stats" in sp and "note" in sp)
    check("splice_scan head guard", all(c["t"] >= 0.3 for c in sp["candidates"]),
          str([c["t"] for c in sp["candidates"]][:4]))

    # container: mp4 walk + appended-mdat tell + AAC-priming elst NOT flagged
    co = FRN.container_report(clip)
    check("container mp4 walk", co.get("format") == "mp4" and co.get("top_level"),
          str(co.get("top_level"))[:60])
    check("container elst priming not flagged",
          not any("edit list" in f["text"] for f in co["findings"]))
    edited = os.path.join(tmp, "edited.mp4")
    shutil.copy(clip, edited)
    with open(edited, "ab") as fh:
        fh.write(struct.pack(">I", 16) + b"mdat" + b"\x00" * 8)
        fh.write(struct.pack(">I", 8 + 70000) + b"free" + b"\x00" * 70000)
    co2 = FRN.container_report(edited)
    check("container appended-mdat tell",
          any("mdat" in f["text"] and f["severity"] == "warn" for f in co2["findings"]),
          "; ".join(f["text"][:40] for f in co2["findings"]))
    check("container free-gap tell",
          any("free" in f["text"] for f in co2["findings"]))

    # encoder fingerprint: x264 SEI + Lavf mux traces in the generated clip
    fp = FRN.encoder_fingerprint(clip)
    check("fingerprint x264 marker", "x264" in (fp.get("markers") or []), str(fp.get("markers")))
    check("fingerprint Lavf trace", any("Lavf" in m or "Lavc" in m for m in fp.get("markers") or []))
    check("fingerprint settings string", bool(fp.get("settings")) and "x264" in fp["settings"])
    mc = FRN.metadata_consistency(clip)
    check("metadata consistency dict", isinstance(mc.get("findings"), list) and "details" in mc)

    # pixel maps
    vs = F.VideoSource(clip)
    fr = vs.frame_at(5)
    vs.close()
    if fr is not None:
        heat, score = FRN.ela_map(fr)
        check("ela map", heat is not None and heat.shape == fr.shape[:2]
              and 0.0 <= float(heat.max()) <= 1.0, "score %s" % score)
        nmap, stats = FRN.noise_map(fr)
        check("noise map", nmap is not None and nmap.shape == fr.shape[:2]
              and "sigma_median" in stats, str(stats)[:60])
    else:
        check("ela map", False, "no frame decoded")
        check("noise map", False, "no frame decoded")
    nc = FRN.noise_consistency(clip, samples=8)
    check("noise consistency", isinstance(nc.get("outliers"), list) and "note" in nc)

    # loop scan structure + no false positives on the plain clip
    lp = FRN.frame_loop_scan(clip)
    check("loop scan clean clip", lp.get("loops") == [] and isinstance(lp.get("periodic"), list),
          str(lp.get("loops")))

    # C2PA ingredient-chain rendering (fixture - no c2patool needed)
    fx = {"active_manifest": "m1", "validation_state": "Valid",
          "manifests": {"m1": {"title": "edit.mp4",
                               "ingredients": [{"title": "cam.mp4", "relationship": "parentOf",
                                                "active_manifest": "m0"}]},
                        "m0": {"title": "cam.mp4",
                               "ingredients": [{"title": "capture", "relationship": "parentOf"}]}}}
    chain = FRN._c2pa_ingredients(fx)
    check("c2pa ingredient chain", len(chain) == 2 and chain[1].startswith("    "),
          str(chain))
    check("c2pa report renders chain", "Ingredients (provenance chain):" in FRN._c2pa_report_text(fx))
    c2 = FRN.c2pa_summary(clip)
    check("c2pa summary shape", set(c2) >= {"present", "validation", "source"})

    # orchestrated report (quick battery on the tiny clip) + marks + render
    rep = FRN.forensics_report(clip, deep=not quick)
    check("forensics report keys", set(rep) >= {"sha256", "findings", "summary",
                                                "splices", "container", "fingerprint"})
    check("forensics findings shape",
          all({"severity", "area", "t", "text"} <= set(f) for f in rep["findings"]))
    txt = FRN.render_report(rep)
    check("forensics render", "FORENSICS" in txt and "SPLICE SCAN" in txt and rep["sha256"] in txt)
    marks = FRN.report_marks(rep)
    check("forensics marks", all(isinstance(t, float) and isinstance(s, str) for t, s in marks))

    # QC integrity profile
    check("qc integrity profile listed", "integrity" in QCP.profile_names())
    r = QCP.evaluate({"probe": {}, "events": {}, "forensics": rep}, "integrity")
    ids = [c["id"] for c in r["checks"]]
    check("qc integrity evaluates", r["verdict"] in ("pass", "warn", "fail")
          and {"provenance", "splices", "loops", "enf"} <= set(ids), r["verdict"])
    r2 = QCP.evaluate({"probe": {}, "events": {}}, "integrity")
    check("qc integrity no-forensics safe", r2["verdict"] == "pass",
          str([(c["id"], c["status"]) for c in r2["checks"] if c["status"] != "pass"]))

    if quick or not exe:
        return

    # --- heavy fixtures -------------------------------------------------------
    base = [exe, "-y", "-hide_banner", "-loglevel", "error"]

    # corroborated splice: two stream-copied parts, forced GOP, distinct audio
    A = os.path.join(tmp, "fA.mp4")
    B = os.path.join(tmp, "fB.mp4")
    cc = os.path.join(tmp, "fcc.txt")
    spliced = os.path.join(tmp, "fsplice.mp4")
    try:
        subprocess.run(base + ["-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=9.5",
                               "-f", "lavfi", "-i", "sine=frequency=440:duration=9.5",
                               "-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p",
                               "-c:a", "aac", "-shortest", A], check=True, timeout=180)
        subprocess.run(base + ["-f", "lavfi", "-i", "smptebars=s=320x240:r=25:d=3",
                               "-f", "lavfi", "-i", "sine=frequency=880:duration=3",
                               "-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p",
                               "-c:a", "aac", "-shortest", B], check=True, timeout=180)
        with open(cc, "w") as fh:
            fh.write("file '%s'\nfile '%s'\n" % (A.replace("\\", "/"), B.replace("\\", "/")))
        subprocess.run(base + ["-f", "concat", "-safe", "0", "-i", cc, "-c", "copy",
                               spliced], check=True, timeout=60)
    except Exception:
        pass
    if os.path.isfile(spliced):
        sp2 = FRN.splice_scan(spliced)
        hits = [c for c in sp2["candidates"] if abs(c["t"] - 9.5) < 0.6]
        check("splice boundary corroborated", bool(hits) and len(hits[0]["signals"]) >= 2,
              str(hits[:1]))
    else:
        check("splice boundary corroborated", False, "fixture encode failed")

    # loop fixture: A + B + A re-encoded in one pass
    s1 = os.path.join(tmp, "fseg1.mp4")
    s2 = os.path.join(tmp, "fseg2.mp4")
    loopf = os.path.join(tmp, "floop.mp4")
    try:
        subprocess.run(base + ["-f", "lavfi", "-i", "cellauto=s=320x240:r=25", "-t", "2",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", s1], check=True, timeout=180)
        subprocess.run(base + ["-f", "lavfi", "-i", "smptebars=s=320x240:r=25", "-t", "2",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", s2], check=True, timeout=180)
        subprocess.run(base + ["-i", s1, "-i", s2, "-i", s1, "-filter_complex",
                               "[0:v][1:v][2:v]concat=n=3:v=1[v]", "-map", "[v]",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", loopf],
                       check=True, timeout=180)
    except Exception:
        pass
    if os.path.isfile(loopf):
        lp2 = FRN.frame_loop_scan(loopf)
        ok = any(abs(L["repeat_t"] - 4.0) < 0.4 and abs(L["src_t"]) < 0.4
                 for L in lp2["loops"])
        check("loop fixture detected", ok, str(lp2["loops"]))
    else:
        check("loop fixture detected", False, "fixture encode failed")

    # ENF fixture: 60 Hz hum + pink noise
    enff = os.path.join(tmp, "fenf.mp4")
    try:
        subprocess.run(base + ["-f", "lavfi", "-i", "testsrc2=s=160x120:r=25:d=12",
                               "-f", "lavfi", "-i", "sine=frequency=60:duration=12",
                               "-f", "lavfi", "-i", "anoisesrc=d=12:c=pink:a=0.02",
                               "-filter_complex", "[1:a][2:a]amix=inputs=2[a]",
                               "-map", "0:v", "-map", "[a]", "-c:v", "libx264",
                               "-pix_fmt", "yuv420p", "-c:a", "aac", enff],
                       check=True, timeout=180)
    except Exception:
        pass
    if os.path.isfile(enff):
        enf = FRN.enf_trace(enff)
        check("enf 60Hz detected", enf.get("present") and enf.get("base_hz") == 60,
              "snr %s" % enf.get("median_snr_db"))
        img = FRN.render_enf(enf)
        check("enf render", img is not None and img.shape[2] == 3)
    else:
        check("enf 60Hz detected", False, "fixture encode failed")
        check("enf render", False, "fixture encode failed")
    enf2 = FRN.enf_trace(clip)
    check("enf absent on 440Hz clip", not enf2.get("present"), enf2.get("note", ""))

def chk_aspect_suite(tmp, media):
    """Native display aspect-ratio rendering: anamorphic (SAR), square,
    vertical and rotation-metadata files must decode at their true display
    shape; storage geometry stays available for pixel-exact analysis."""
    import struct
    exe = F.find_ffmpeg()
    if not exe:
        check("aspect suite", False, "no ffmpeg to build fixtures")
        return
    base = [exe, "-y", "-hide_banner", "-loglevel", "error"]
    ana = os.path.join(tmp, "ar_ana.mp4")
    vert = os.path.join(tmp, "ar_vert.mp4")
    rot = os.path.join(tmp, "ar_rot.mp4")
    try:
        subprocess.run(base + ["-f", "lavfi", "-i", "testsrc2=s=640x540:r=25:d=1",
                               "-vf", "setsar=2/1", "-c:v", "libx264",
                               "-pix_fmt", "yuv420p", ana], check=True, timeout=120)
        subprocess.run(base + ["-f", "lavfi", "-i", "testsrc2=s=360x640:r=25:d=1",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", vert],
                       check=True, timeout=120)
        subprocess.run(base + ["-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=1",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", rot],
                       check=True, timeout=120)
    except Exception as exc:
        check("aspect fixtures", False, str(exc)[:60])
        return
    # write a 90-degree-clockwise display matrix into tkhd (ffmpeg < 6 cannot)
    try:
        with open(rot, "rb") as fh:
            fh.seek(0, 2); size = fh.tell(); fh.seek(0)
            boxes = FRN._walk_mp4(fh, 0, size)
        off = next(b[2] for b in boxes if b[1] == b"tkhd")
        with open(rot, "r+b") as fh:
            fh.seek(off + 8)
            ver = fh.read(1)[0]
            mat_off = off + 8 + 4 + ((8 + 8 + 4 + 4 + 8) if ver == 1
                                     else (4 + 4 + 4 + 4 + 4)) + 8 + 2 + 2 + 2 + 2
            fh.seek(mat_off)
            fh.write(struct.pack(">9i", 0, 0x00010000, 0, -0x00010000, 0, 0,
                                 360 << 16, 0, 0x40000000))
        rot_ok = True
    except (StopIteration, OSError, struct.error):
        rot_ok = False

    p = F.probe(ana)
    check("probe SAR + display dims", abs(p["sar"] - 2.0) < 0.01 and
          (p["display_width"], p["display_height"]) == (1280, 540),
          "sar=%.2f display=%dx%d" % (p["sar"], p["display_width"], p["display_height"]))
    if rot_ok:
        pr = F.probe(rot)
        check("probe rotation -> portrait dims", pr["rotation"] == 90 and
              (pr["display_width"], pr["display_height"]) == (360, 640),
              "rot=%s display=%dx%d" % (pr["rotation"], pr["display_width"], pr["display_height"]))
    else:
        check("probe rotation -> portrait dims", False, "tkhd patch failed")

    def frame_ar(path, **kw):
        vs = F.VideoSource(path, **kw)
        fr = vs.frame_at(3)
        vs.close()
        if fr is None:
            return None, None
        h, w = fr.shape[:2]
        return w / float(h), (w, h)

    ar, dims = frame_ar(ana)
    check("anamorphic decodes 2.37:1", ar is not None and abs(ar - 1280 / 540.0) < 0.05,
          "frame %sx%s" % (dims or ("?",) * 2))
    ar, dims = frame_ar(vert)
    check("vertical decodes 9:16", ar is not None and abs(ar - 360 / 640.0) < 0.02,
          "frame %sx%s" % (dims or ("?",) * 2))
    if rot_ok:
        ar, dims = frame_ar(rot)
        check("rotated decodes portrait", ar is not None and abs(ar - 360 / 640.0) < 0.02,
              "frame %sx%s" % (dims or ("?",) * 2))
    ar, dims = frame_ar(ana, display_ar=False)
    check("storage geometry opt-out", dims == (640, 540), "frame %sx%s" % (dims or ("?",) * 2))
    rd = F.CV2Reader(ana, 25.0, max_dim=1280, sar=2.0, rotation=0)
    fr = rd.read()
    rd.close()
    check("cv2 fallback SAR stretch", fr is not None and
          abs(fr.shape[1] / float(fr.shape[0]) - 1280 / 540.0) < 0.05,
          "frame %dx%d" % (fr.shape[1], fr.shape[0]) if fr is not None else "no frame")


def chk_verify_suite(tmp, media, quick=False):
    """va_verify: platform screen, tempo screen, verdict layer, speedrun QC."""
    clip = media["clip"]
    exe = F.find_ffmpeg()

    # platform screen: a plain ffmpeg-written clip is a pipeline, not a platform
    p = VFY.platform_screen(clip)
    check("verify platform pipeline", p.get("platform") in ("ffmpeg-pipeline",
                                                            "stream-capture"),
          str(p.get("platform")))
    check("verify pipeline not weakened", not p.get("weakened"), str(p.get("weakened")))

    yt = os.path.join(tmp, "vfy_yt.mp4")
    ok = False
    if exe:
        ok = subprocess.run([exe, "-v", "error", "-y", "-f", "lavfi",
                             "-i", "testsrc2=size=192x108:rate=15:duration=2",
                             "-c:v", "libx264", "-preset", "ultrafast",
                             "-metadata:s:v",
                             "handler_name=ISO Media file produced by Google Inc.",
                             yt], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0 and os.path.exists(yt)
    if ok:
        p = VFY.platform_screen(yt)
        check("verify platform youtube", p.get("platform") == "youtube" and
              p.get("reencoded") is True,
              "%s reenc=%s" % (p.get("platform"), p.get("reencoded")))
        check("verify weakened lists splices", "splices" in (p.get("weakened") or []),
              str(p.get("weakened")))
    else:
        check("verify platform youtube", True, "skipped (no ffmpeg/libx264)")

    # tempo screen: 15 fps content re-timed into a 60 fps container with the
    # audio brick-walled at 8 kHz (what a resample-based slowdown leaves)
    slow = os.path.join(tmp, "vfy_slow.mp4")
    ok = False
    if exe:
        ok = subprocess.run([exe, "-v", "error", "-y",
                             "-f", "lavfi", "-i", "testsrc2=size=192x108:rate=15:duration=3",
                             "-f", "lavfi", "-i", "anoisesrc=d=3:c=pink:a=0.5",
                             "-vf", "fps=60", "-af", "aresample=8000,aresample=44100",
                             "-c:v", "libx264", "-preset", "ultrafast",
                             "-c:a", "aac", "-b:a", "128k", "-shortest", slow],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0 and os.path.exists(slow)
    if ok:
        t = VFY.tempo_check(slow)
        check("tempo dup flagged", t["score"] >= 0.4 and
              (t["video"].get("regular_cadence") or t["video"].get("dup_pct", 0) >= 40),
              "score %.2f dup %.1f%%" % (t["score"], t["video"].get("dup_pct", 0)))
        check("tempo audio cutoff flagged",
              any("audio band" in f["text"] for f in t["findings"]),
              "cutoff %s" % ((t.get("audio") or {}).get("cutoff_hz")))
        check("tempo content fps", abs((t["video"].get("content_fps") or 0) - 15.0) < 3.0,
              str(t["video"].get("content_fps")))
    else:
        check("tempo dup flagged", True, "skipped (no ffmpeg/libx264)")
    tc = VFY.tempo_check(clip)
    check("tempo clean clip", tc["score"] < 0.3, "score %.2f" % tc["score"])

    # verdict layer is pure data - no media needed
    fake = {"splices": {"candidates": [
                {"t": 12.0, "score": 4, "signals": ["size", "pts", "audio"]}]},
            "loops": {}, "fingerprint": {}, "metadata": {}, "container": {},
            "noise": {}, "enf": {}, "c2pa": {}, "recompression": {}}
    by = {v["id"]: v for v in VFY.build_verdicts(
        fake, {"weakened": [], "reencoded": False}, {"score": 0.0})}
    check("verdict splices review", by["splices"]["verdict"] == "review" and
          "12.0s" in by["splices"]["plain"], by["splices"]["plain"][:60])
    check("verdict splice caveat", bool(by["splices"]["caveat"]))
    check("verdict core four", all(k in by for k in ("source", "splices",
                                                     "loops", "tempo")))
    by2 = {v["id"]: v for v in VFY.build_verdicts(
        fake, {"weakened": ["splices"], "reencoded": True, "platform": "youtube"},
        {"score": 0.0})}
    check("verdict weakened drops confidence", by2["splices"]["confidence"] == "low"
          and "re-encode" in by2["splices"]["plain"], by2["splices"]["confidence"])
    check("verdict reencode source note", by2["source"]["verdict"] == "note")

    # end-to-end: verify_run + render + marks + the speedrun QC profile
    v = VFY.verify_run(clip, deep=not quick)
    check("verify_run overall", v.get("overall") in ("CLEAR", "NOTE", "REVIEW"),
          "%s - %s" % (v.get("overall"), v.get("summary")))
    txt = VFY.render_verify(v)
    check("verify render sections", all(x in txt for x in (
        "RUN VERIFICATION", "OVERALL", "FULL BATTERY OUTPUT",
        "Indicators, not proof")), "%d chars" % len(txt))
    check("verify marks list", isinstance(VFY.verify_marks(v), list))
    qc = QCP.evaluate({"forensics": v["forensics"],
                       "verify": {"platform": v["platform"], "tempo": v["tempo"]}},
                      "speedrun")
    ids = [c["id"] for c in qc["checks"]]
    check("speedrun profile evaluates", qc["profile"] == "speedrun" and
          "platform" in ids and "tempo" in ids, qc["verdict"])
    check("speedrun in profile_names", "speedrun" in QCP.profile_names())


def chk_advanced_export(tmp, table):
    """The advanced-results cache flows into JSON and HTML exports."""
    import json as _json
    png = None
    try:
        import numpy as np
        import cv2
        okf, buf = cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))
        png = buf.tobytes() if okf else None
    except Exception:
        png = None
    adv = {"banding_map": {"title": "Banding map (frame 3) - 3.1% banding",
                           "generated": "2026-06-10T00:00:00", "text": None,
                           "data": {"frame": 3, "banding_pct": 3.1}, "png": png},
           "mediainfo": {"title": "MediaInfo - clip", "generated": "2026-06-10T00:00:00",
                         "text": "General\nFormat : <MPEG-4>", "data": None, "png": None}}
    jp = os.path.join(tmp, "adv_export.json")
    hp = os.path.join(tmp, "adv_export.html")
    X.write_json(table, jp, advanced=adv)
    with open(jp, encoding="utf-8") as fh:
        doc = _json.load(fh)
    check("adv json data present", doc.get("advanced", {}).get("banding_map", {})
          .get("data", {}).get("banding_pct") == 3.1)
    check("adv json drops image bytes",
          "png" not in _json.dumps(doc.get("advanced", {})) and
          doc["advanced"]["banding_map"]["image_in_html_report"] is (png is not None))
    check("adv json keeps text", "MPEG-4" in (doc["advanced"]["mediainfo"]["text"] or ""))
    X.write_html_report(table, hp, advanced=adv)
    with open(hp, encoding="utf-8") as fh:
        h = fh.read()
    check("adv html section + titles", "Advanced analyses" in h and "MediaInfo - clip" in h)
    check("adv html escapes text", "&lt;MPEG-4&gt;" in h)
    if png:
        check("adv html embeds png", "data:image/png;base64," in h)
    hp2 = X.write_html_report(table, os.path.join(tmp, "adv_none.html"))
    with open(hp2, encoding="utf-8") as fh:
        check("adv html omitted when empty", "Advanced analyses" not in fh.read())

    frep = {"findings": [
        {"severity": "warn", "area": "splice", "t": 1.0, "text": "cut?"},
        {"severity": "info", "area": "noise", "t": 1.5, "text": "sigma"},
        {"severity": "warn", "area": "fingerprint", "t": None, "text": "file-level"}],
        "summary": "test", "splices": {}, "container": {}, "fingerprint": {}}
    hp3 = X.write_html_report(table, os.path.join(tmp, "adv_marks.html"),
                              source={"fps": 25.0}, events={"scene_cuts": [1.2]},
                              forensics=frep)
    with open(hp3, encoding="utf-8") as fh:
        h3 = fh.read()
    check("forensic marks drawn dashed", "stroke-dasharray" in h3 and "#b07fd8" in h3,
          "%d dashes" % h3.count("stroke-dasharray"))
    check("timeline legend entries", "scene cut" in h3 and "splice finding" in h3
          and "noise finding" in h3 and "fingerprint finding" not in h3)
    check("mark hover tooltip", "splice (warn) @ 1.0s" in h3)


def chk_dovi_p5(tmp):
    """Dolby Vision profile 5: detection, DV-aware pipeline routing and the
    software IPTPQc2->BGR decode. Fixture = solid 'fireplace orange' encoded
    into IPT code values, x265-lossless 10-bit, sample entry retagged dvh1
    (the profile-5 signature: DV tag + no colour VUI)."""
    import subprocess as _sp
    import numpy as np
    import va_ipt
    import va_hwaccel as H

    x = np.linspace(0.0, 1.0, 33)
    check("dovi pq roundtrip",
          float(np.abs(va_ipt.pq_oetf(va_ipt.pq_eotf(x)) - x).max()) < 1e-5)
    ident = va_ipt.DOVI_LMS2RGB @ np.linalg.inv(va_ipt.DOVI_LMS2RGB)
    check("dovi lms2rgb invertible", float(np.abs(ident - np.eye(3)).max()) < 1e-9)
    # neutral IPT (P=T=0.5) must decode to neutral grey - the YCbCr-misread
    # bug could never produce this
    gcodes = va_ipt.encode_iptpqc2(np.full((1, 1, 3), 0.01))[0, 0]
    check("dovi neutral ipt is chroma-centred",
          abs(int(gcodes[1]) - 512) < 8 and abs(int(gcodes[2]) - 512) < 8,
          str(gcodes.tolist()))

    exe = F.find_ffmpeg()
    if not exe:
        return
    w, h = 192, 108
    orange = np.linalg.inv(va_ipt.BT2020_TO_BT709) @ \
        (np.array([1.0, 0.45, 0.05]) * 80 / 10000.0)
    codes = va_ipt.encode_iptpqc2(orange[None, None, :])[0, 0]
    y = np.full((h, w), codes[0], "<u2")
    u = np.full((h // 2, w // 2), codes[1], "<u2")
    v = np.full((h // 2, w // 2), codes[2], "<u2")
    raw = os.path.join(tmp, "ipt.raw")
    mp4 = os.path.join(tmp, "p5.mp4")
    with open(raw, "wb") as fh:
        fh.write((y.tobytes() + u.tobytes() + v.tobytes()) * 8)
    r = _sp.run([exe, "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                 "-pix_fmt", "yuv420p10le", "-s", "%dx%d" % (w, h), "-r", "24",
                 "-i", raw, "-c:v", "libx265",
                 "-x265-params", "lossless=1:log-level=none",
                 "-pix_fmt", "yuv420p10le", "-y", mp4],
                capture_output=True, timeout=120)
    if r.returncode != 0 or not os.path.exists(mp4):
        check("dovi p5 fixture (libx265 10-bit unavailable - skipped)", True)
        return
    blob = bytearray(open(mp4, "rb").read())
    i = blob.find(b"stsd")
    j = blob.find(b"hev1", i) if i >= 0 else -1
    if j < 0:
        check("dovi p5 fixture (no hev1 sample entry - skipped)", True)
        return
    blob[j:j + 4] = b"dvh1"
    with open(mp4, "wb") as fh:
        fh.write(bytes(blob))

    info = F.probe(mp4)
    check("dovi p5 detected", bool(info["is_dovi"] and info["dovi_ipt"]
                                   and info["is_hdr"]),
          "profile=%s compat=%s" % (info["dv_profile"], info["dv_compat"]))
    os.environ["VA_NO_HWACCEL"] = "1"
    try:
        H.reset_cache()
        pipe = H.choose_video_pipeline(mp4, info)
        check("dovi p5 pipeline is DV-aware",
              "libplacebo-dovi" in pipe["label"] or "ipt-np" in pipe["label"],
              pipe["label"])
        src = F.VideoSource(mp4)
        fr = src.frame_at(2)
        src.close()
        ok = fr is not None and fr.shape[2] == 3
        check("dovi p5 decode", ok, str(None if fr is None else fr.shape)
              + " via " + src.backend)
        if ok and "ipt-np" in pipe["label"]:
            px = fr[fr.shape[0] // 2, fr.shape[1] // 2].astype(int)
            check("dovi p5 colour sane (orange, not magenta)",
                  bool(px[2] > px[1] > px[0]), "BGR=%s" % px.tolist())
    finally:
        os.environ.pop("VA_NO_HWACCEL", None)
        H.reset_cache()


def chk_native_scopes(tmp):
    """Scopes must read the file's NATIVE colour, not the tonemapped preview."""
    import numpy as np
    import va_ipt

    # --- math layer: same wide-gamut content, native vs display-referred ----
    f = np.zeros((60, 60, 3), np.float32)
    f[:20, :, 0] = 0.75
    f[20:40, :, 1] = 0.75
    f[40:, :, 2] = 0.75                       # pure BT.2020 R/G/B rows
    img, cov = S.cie_gamut_native(f, 320, "pq", "2020")
    check("cie native: 2020 primaries land OUTSIDE 709", cov["outside_709_pct"] > 90,
          "out709=%.1f%%" % cov["outside_709_pct"])
    check("cie native: tagged", cov.get("source") == "native" and cov.get("primaries") == "2020")
    disp = (np.clip(f, 0, 1) * 255).astype(np.uint8)[..., ::-1]
    _, cov2 = S.cie_gamut(disp, 320)
    check("cie display: same values read INSIDE 709 (the old failure mode)",
          cov2["outside_709_pct"] < 5 and cov2.get("source") == "display",
          "out709=%.1f%%" % cov2["outside_709_pct"])

    pq1k = S._nits_to_pq(1000.0)
    g = np.full((40, 64, 3), pq1k, np.float32)
    wf = S.waveform_native(g, 256, 200, "pq", "2020")
    col = wf[:, 128].astype(int).sum(axis=1)
    row, expect = int(np.argmax(col)), int(round(199 - pq1k * 199))
    check("waveform native: 1000-nit trace at PQ height", abs(row - expect) <= 3,
          "row=%d expect=%d" % (row, expect))
    check("waveform native: deterministic",
          np.array_equal(wf, S.waveform_native(g, 256, 200, "pq", "2020")))
    pr = S.rgb_parade_native(f, 300, 160, "pq", "2020")
    check("parade native renders", pr.shape == (160, 300, 3) and pr.dtype == np.uint8)
    hi = S.histogram_native(g, 300, 200, "pq", "2020")
    check("histogram native renders", hi.shape == (200, 300, 3))
    v1 = S.vectorscope_native(f, 240, "pq", "2020")
    check("vectorscope native: deterministic + plots",
          np.array_equal(v1, S.vectorscope_native(f, 240, "pq", "2020"))
          and int((np.abs(v1.astype(int) - 24).sum(axis=2) > 30).sum()) > 200)
    fc = S.false_color_native(g, 300, 200, "pq", "2020")
    check("false colour native: 1000 nit -> orange band", tuple(fc[80, 150]) == (235, 150, 40),
          str(fc[80, 150].tolist()))
    for t, p in (("hlg", "2020"), ("sdr", "P3"), ("sdr", "709")):
        S.cie_gamut_native(f, 200, t, p)
        S.waveform_native(f, 200, 100, t, p)
        S.false_color_native(f, 200, 100, t, p)
        S.histogram_native(f, 200, 150, t, p)
        S.vectorscope_native(f, 160, t, p)
    check("native scopes: hlg/sdr/P3 paths run", True)
    m = S.mark_display_referred(np.zeros((100, 200, 3), np.uint8))
    check("display-referred stamp draws", int(m.sum()) > 0)

    # --- IPT native: reshape WITHOUT tonemap, PQ/2020 out -------------------
    lin = np.zeros((32, 64, 3), np.float32)
    lin[:] = (0.05, 0.02, 0.001)              # 500/200/10 nits (1.0 = 10k)
    codes = va_ipt.encode_iptpqc2(lin)
    buf = (codes[..., 0].astype("<u2").tobytes()
           + codes[::2, ::2, 1].astype("<u2").tobytes()
           + codes[::2, ::2, 2].astype("<u2").tobytes())
    dec = va_ipt.IPTDecoder(os.path.join(tmp, "does-not-exist.mp4"))
    out = dec.decode_native(buf, 64, 32)
    err = float(np.abs(out - va_ipt.pq_oetf(lin)).max())
    check("ipt decode_native: PQ/2020 round-trip", err < 0.01, "maxerr=%.4f" % err)
    bgr = dec.decode(buf, 64, 32)
    check("ipt display decode unchanged", bgr.dtype == np.uint8 and bgr.shape == (32, 64, 3))

    # --- NativeTap against real encodes --------------------------------------
    exe = F.find_ffmpeg()
    if not exe:
        check("native tap (no ffmpeg - skipped)", True)
        return
    base = [exe, "-y", "-hide_banner", "-loglevel", "error"]
    sdr = os.path.join(tmp, "tap_sdr.mp4")
    subprocess.run(base + ["-f", "lavfi", "-i", "color=c=red:s=192x108:d=0.5:r=30",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", sdr], check=True)
    check("tap not needed for 8-bit SDR", not F.NativeTap.needed(F.probe(sdr)))
    pqf = os.path.join(tmp, "tap_pq.mp4")
    r = subprocess.run(base + ["-f", "lavfi", "-i", "color=c=0x8040C0:s=192x108:d=0.5:r=30",
                               "-vf", "scale=out_color_matrix=bt2020:out_range=tv,"
                               "format=yuv420p10le", "-c:v", "libx265",
                               "-x265-params", "lossless=1:log-level=none",
                               "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                               "-colorspace", "bt2020nc", pqf], capture_output=True)
    if r.returncode != 0 or not os.path.exists(pqf):
        check("native tap fixture (libx265 10-bit unavailable - skipped)", True)
        return
    info = F.probe(pqf)
    check("tap needed for PQ/2020", F.NativeTap.needed(info))
    meta = F.native_signal_meta(info)
    check("tap meta pq/2020", meta["transfer"] == "pq" and meta["primaries"] == "2020", str(meta))
    tap = F.NativeTap(pqf, info)
    fr = tap.read()
    ok = fr is not None and fr.dtype == np.float32 and fr.shape == (108, 192, 3)
    check("tap reads float32 native frames", ok)
    if ok:
        exp = np.array([0x80, 0x40, 0xC0], np.float32) / 255.0
        err = float(np.abs(fr[50, 90] - exp).max())
        check("tap values = encoded signal (no tonemap)", err < 0.01, "maxerr=%.4f" % err)
        tap.grab(2)
        i0 = tap.index
        f2 = tap.read()
        check("tap grab/read advances", f2 is not None and tap.index == i0 + 1)
        tap.seek(5)
        check("tap seek + read", tap.read() is not None and tap.index == 6)
    tap.close()
    check("tap close idempotent", tap.read() is None)

    # --- GUI wiring (source-level: Tk won't run headless) --------------------
    gui = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "video-analyzer.py"), encoding="utf-8").read()
    for needle in ("NativeTap.needed", "_tap_loop", "cie_gamut_native",
                   "vectorscope_native", "mark_display_referred", "_stop_tap"):
        check("gui wired: %s" % needle, needle in gui)



def chk_perf(tmp, media, quick=False):
    """Performance layer: va_perf budgets/modes, combined single-decode pass
    parity, chained quality compare, parallel battery parity, frame cache,
    CLI + GUI wiring."""
    import numpy as np
    import va_perf as P

    check("perf cpu detect", P.cpu_count() >= 1)
    mi = P.mem_info()
    check("perf mem detect", mi["total_bytes"] >= 0 and mi["free_bytes"] >= 0)
    check("perf gpus list", isinstance(P.gpus(), list))
    prev = P.mode()
    try:
        P.set_mode("eco")
        check("perf eco single-lane", P.pool_workers(8) == 1
              and P.batch_jobs("auto", 9) == 1)
        check("perf eco thread flags", P.ffmpeg_thread_args()[:1] == ["-threads"])
        P.set_mode("max")
        check("perf max ffmpeg defaults", P.ffmpeg_thread_args() == [])
        check("perf max pool sane", 1 <= P.pool_workers(8) <= max(8, P.cpu_count()))
        check("perf jobs clamp", P.batch_jobs(99, 3) == 3 and P.batch_jobs(None) == 1)
        check("perf vmaf threads", P.vmaf_threads() >= 1)
        check("perf summary text", "CPU:" in P.summary())
    finally:
        P.set_mode(prev if prev in P.MODES else None)

    c = P.FrameCache(budget_bytes=300)
    a = np.zeros(100, np.uint8)
    for i in range(5):
        c.put((1, i), a)
    check("cache LRU eviction", len(c) == 3 and c.get((1, 0)) is None
          and c.get((1, 4)) is not None)
    c.put((1, 9), np.zeros(10 ** 4, np.uint8))
    check("cache rejects oversize", c.get((1, 9)) is None)
    hits = c.stats()["hits"]
    check("cache hit accounting", hits >= 1 and c.stats()["budget"] == 300)
    c.clear()
    check("cache clear", len(c) == 0 and c.stats()["bytes"] == 0)

    clip = media["clip"]
    r = M.analyze_pass(clip)
    check("combined pass runs", r is not None and r["passes_merged"] >= 4,
          "merged=%s" % (None if r is None else r["passes_merged"]))
    if r is not None:
        t1 = M.signalstats(clip)
        check("combined rows match", len(r["table"]) == len(t1),
              "%d vs %d" % (len(r["table"]), len(t1)))
        a1, a2 = r["table"].arrays(), t1.arrays()
        same = len(r["table"]) == len(t1) and all(
            np.allclose(a1[k][~np.isnan(a1[k])], a2[k][~np.isnan(a2[k])], atol=1e-6)
            for k in a1 if a1[k].size)
        check("combined signalstats parity", same)

        def segs_close(x, y, tol=0.1):
            return len(x) == len(y) and all(
                abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol
                for p, q in zip(x, y))
        ev = {"black": M.black_segments(clip), "freeze": M.freeze_segments(clip),
              "scene_cuts": M.scene_cuts(clip)}
        check("combined black parity", segs_close(r["events"]["black"], ev["black"]),
              "%s vs %s" % (r["events"]["black"], ev["black"]))
        check("combined freeze parity", segs_close(r["events"]["freeze"], ev["freeze"]),
              "%s vs %s" % (r["events"]["freeze"], ev["freeze"]))
        check("combined cuts parity",
              len(r["events"]["scene_cuts"]) == len(ev["scene_cuts"])
              and all(abs(x - y) <= 0.1 for x, y in
                      zip(r["events"]["scene_cuts"], ev["scene_cuts"])))
        if AU.has_audio(clip):
            ld = (AU.loudness(clip) or {}).get("summary") or {}
            cs = (r.get("audio") or {}).get("summary") or {}
            li, ci = ld.get("integrated_lufs"), cs.get("integrated_lufs")
            check("combined loudness parity",
                  li is not None and ci is not None and abs(li - ci) <= 0.15,
                  "%s vs %s" % (ci, li))
        rl = M.analyze_pass(media["low"])    # low.mp4 is muxed without audio
        check("combined no-audio path", rl is not None and rl.get("audio") is None
              and rl["silence"] == [] and rl["passes_merged"] == 4)
        rc = M.analyze_pass(clip, cancel=lambda: True)
        check("combined cancel honored", rc is None or rc.get("cancelled"))

    cq = Q.compare(media["low"], clip, vmaf=not quick)
    check("quality single-pass engine",
          str(cq.get("engine", "")).startswith("single-pass"), str(cq.get("engine")))
    di, ri = F.probe(media["low"]), F.probe(clip)
    ps = Q._psnr(media["low"], clip, di, ri) or {}
    ca, sa = (cq.get("psnr") or {}).get("average"), ps.get("average")
    check("quality psnr parity", ca is not None and sa is not None
          and (ca == sa or abs(ca - sa) < 1e-6), "%s vs %s" % (ca, sa))
    check("quality per-frame logs", len((cq.get("ssim") or {}).get("per_frame") or [])
          == len((cq.get("psnr") or {}).get("per_frame") or []) > 0)

    import va_perf as P2
    rp = FRN.forensics_report(clip, deep=False)
    P2.set_mode("eco")                      # eco pool width 1 = serial path
    rs = FRN.forensics_report(clip, deep=False)
    P2.set_mode(prev if prev in P2.MODES else None)
    check("battery parallel==serial", rp.get("sha256") == rs.get("sha256")
          and [(f["area"], f["severity"], f["text"]) for f in rp["findings"]]
          == [(f["area"], f["severity"], f["text"]) for f in rs["findings"]],
          "%d vs %d findings" % (len(rp["findings"]), len(rs["findings"])))
    check("battery key order stable", list(rp.keys()) == list(rs.keys()))

    here = os.path.dirname(os.path.abspath(__file__))
    asrc = open(os.path.join(here, "analyze.py"), encoding="utf-8").read()
    check("cli --jobs wired", '"--jobs"' in asrc and "batch_jobs" in asrc
          and "ThreadPoolExecutor" in asrc)
    check("cli --perf wired", '"--perf"' in asrc and "analyze_pass" in asrc)
    gsrc = open(os.path.join(here, "video-analyzer.py"), encoding="utf-8").read()
    check("gui combined pass wired", "metrics.analyze_pass(" in gsrc)
    check("gui frame cache wired", "va_perf.FrameCache()" in gsrc
          and "self.frame_cache.get(idx)" in gsrc
          and "self.frame_cache.clear()" in gsrc)
    check("gui perf panel wired", "_refresh_perf_panel" in gsrc
          and "_set_perf_mode" in gsrc and 'load_ui_key("perf"' in gsrc)
    check("launch perf summary wired",
          "va_perf" in open(os.path.join(here, "launch.py"), encoding="utf-8").read())


def chk_audio_forensics(tmp, media, quick=False):
    """Audio forensic battery: array-level detectors, the battery contract,
    forensics_report integration, QC + export plumbing, and the -vn guard
    (audio passes must never decode a UHD video stream along the way)."""
    here = os.path.dirname(os.path.abspath(__file__))
    import json
    import numpy as np
    import va_audio as VA
    import va_forensics as VF
    import va_qc as VQ
    import va_export as VE

    sr = 16000
    rng = np.random.default_rng(7)

    # -- detectors on synthetic defects --------------------------------------
    t = np.arange(sr * 6) / sr
    x = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    x[sr * 2:sr * 3] = np.clip(x[sr * 2:sr * 3] * 8, -1, 1)
    cl = VA.clipping_scan(x, sr)
    check("clipping detected at 2s", cl["count"] >= 1
          and abs(cl["events"][0]["t"] - 2.0) < 0.1, str(cl["events"][:1]))
    check("clipping pct sane", 5.0 < cl["clipped_pct"] < 25.0, str(cl["clipped_pct"]))

    x2 = (0.1 * rng.standard_normal(sr * 6)).astype(np.float32)
    x2[sr * 3:sr * 3 + int(0.06 * sr)] = 0.0
    dr = VA.dropout_scan(x2, sr)
    check("dropout 60ms at 3s", len(dr["dropouts"]) == 1
          and abs(dr["dropouts"][0]["t"] - 3.0) < 0.05
          and abs(dr["dropouts"][0]["dur"] - 0.06) < 0.01, str(dr["dropouts"]))

    x3 = (0.2 * np.sin(2 * np.pi * 220 * np.arange(sr * 10) / sr)
          + 0.01 * rng.standard_normal(sr * 10)).astype(np.float32)
    for tt in (2.5, 5.0, 7.5):
        x3[int(tt * sr)] += 0.8
    clicks = VA.dropout_scan(x3, sr)["clicks"]
    check("3 clicks found", len(clicks) == 3
          and all(abs(c["t"] - e) < 0.02 for c, e in zip(clicks, (2.5, 5.0, 7.5))),
          str([c["t"] for c in clicks]))

    a = (0.003 * rng.standard_normal(sr * 4)).astype(np.float32)
    b = (0.04 * rng.standard_normal(sr * 4)).astype(np.float32)
    B = np.fft.rfft(b)
    B[np.fft.rfftfreq(b.size, 1.0 / sr) > 3000] = 0
    b = np.fft.irfft(B).astype(np.float32)
    spl = VA.splice_scan_audio(np.r_[a, b], sr)
    check("splice candidate near 4s", any(abs(c["t"] - 4.0) < 0.5 and c["score"] >= 2
                                          for c in spl["candidates"]),
          str(spl["candidates"][:2]))
    quiet = np.r_[np.zeros(sr * 2, np.float32),
                  (0.05 * rng.standard_normal(sr * 4)).astype(np.float32)]
    spl2 = VA.splice_scan_audio(quiet, sr)
    check("content start out of silence NOT a splice",
          not any(abs(c["t"] - 2.0) < 0.4 for c in spl2["candidates"]),
          str(spl2["candidates"][:2]))

    ser = {"t": [i * 0.1 for i in range(600)], "M": [-30.0] * 300 + [-18.0] * 300}
    steps = VA.loudness_steps(ser)
    check("loudness step found", len(steps) == 1 and 25 < steps[0]["t"] < 32
          and abs(steps[0]["delta_lu"] - 12.0) < 1.5, str(steps))

    # -- file-level: decode, overview, battery --------------------------------
    clip = media["clip"]
    pcm = VA.decode_pcm(clip, sr=8000, mono=True)
    check("decode_pcm mono", pcm is not None and pcm.ndim == 1 and pcm.size > 8000,
          None if pcm is None else str(pcm.shape))
    ov = VA.waveform_overview(clip)
    check("waveform_overview", bool(ov) and ov["buckets"] == len(ov["vmin"])
          == len(ov["vmax"]) == len(ov["rms"]) and ov["duration_s"] > 0,
          None if not ov else "%d buckets %.1fs" % (ov["buckets"], ov["duration_s"]))
    rep = VA.forensic_battery(clip)
    check("battery present + keys", rep["present"] and all(
        k in rep for k in ("clipping", "dropouts", "splices", "channels",
                           "bandwidth", "loudness_steps", "findings", "summary")))
    check("battery json-safe", bool(json.dumps(rep)))
    check("battery render", "AUDIO FORENSICS" in VA.render_battery(rep))

    # -- forensics_report integration -----------------------------------------
    frep = VF.forensics_report(clip, deep=False)
    check("audio stage in battery", (frep.get("audio") or {}).get("present") is not None)
    frep["findings"].append({"severity": "warn", "area": "audio", "t": 1.0,
                             "text": "audio splice signature @ 1.00s (test)"})
    marks = VF.report_marks(frep)
    check("audio finding -> timeline mark", any("audio" in m[1] for m in marks))
    check("render has AUDIO section",
          "AUDIO FORENSICS" in VF.render_report(frep)
          or not (frep.get("audio") or {}).get("present"))

    # scene-cut demotion: fabricate stage outputs then re-merge via report
    cuts = {"splices": {"candidates": [], "stats": {"cut_times": [5.0]}},
            "audio": {"present": True, "findings": [
                {"severity": "warn", "area": "audio", "t": 5.2,
                 "text": "audio splice signature @ 5.20s"},
                {"severity": "warn", "area": "audio", "t": 9.0,
                 "text": "audio splice signature @ 9.00s"}]}}
    demoted = []
    cutv = cuts["splices"]["stats"]["cut_times"]
    for f in cuts["audio"]["findings"]:
        sev = f["severity"]
        if f["t"] is not None and sev == "warn" and "splice" in f["text"] \
                and any(abs(f["t"] - c) <= 0.5 for c in cutv):
            sev = "info"
        demoted.append(sev)
    check("picture-cut demotion logic", demoted == ["info", "warn"], str(demoted))

    # -- QC + export plumbing --------------------------------------------------
    fake = {"audio": {"present": True, "findings": [
        {"severity": "warn", "area": "audio", "t": 5.0, "text": "audio splice @ 5s"}]},
        "findings": [{"severity": "warn", "area": "audio", "t": 5.0,
                      "text": "audio splice @ 5s"}]}
    res = VQ.evaluate({"forensics": fake}, profile="integrity")
    row = next((c for c in res["checks"] if c["id"] == "audio_integrity"), None)
    check("QC audio_integrity warns", row is not None and row["status"] == "warn",
          str(row))
    res2 = VQ.evaluate({"forensics": {"audio": {"present": False}}}, profile="integrity")
    row2 = next((c for c in res2["checks"] if c["id"] == "audio_integrity"), None)
    check("QC audio_integrity no-audio pass", row2 is not None
          and row2["status"] == "pass", str(row2))
    check("export audio mark colour", "audio" in VE._MARK_COLORS)
    check("export adv order", "audio_forensics" in VE._ADV_ORDER)

    # -- the -vn regression guard ----------------------------------------------
    asrc = open(os.path.join(here, "va_audio.py"), encoding="utf-8").read()
    for fn in ("def loudness", "def astats", "def silence_segments", "def correlation"):
        seg = asrc[asrc.index(fn):asrc.index("def ", asrc.index(fn) + 5)]
        check("-vn guard in %s" % fn.split()[1], '"-vn"' in seg or '"0:a:0"' in seg)
    msrc = open(os.path.join(here, "va_metrics.py"), encoding="utf-8").read()
    seg = msrc[msrc.index("def loudness"):]
    seg = seg[:seg.index("def ", 5)]
    check("-vn guard in va_metrics.loudness", '"-vn"' in seg)


def chk_audio_gui(tmp):
    """Audio scrub strip + playback follower wiring (headless: stubbed Tk
    import, geometry math on a bare object, source-level wiring greps)."""
    here = os.path.dirname(os.path.abspath(__file__))
    import sys
    import types
    import importlib.util
    import va_audio as VA
    import va_ffmpeg as VFF

    gsrc = open(os.path.join(here, "video-analyzer.py"), encoding="utf-8").read()
    for needle, why in (
            ("class AudioStrip", "waveform strip class"),
            ("self.audio_strip = AudioStrip(", "strip constructed"),
            ("def _strip_seek", "strip seek handler"),
            ("def _toggle_sound", "speaker toggle"),
            ("def _audition_now", "paused-scrub audition"),
            ("def _wave_worker", "waveform loader"),
            ("def _audio_forensics_done", "battery handler"),
            ("self.follower.start(", "follower started"),
            ("fol.start(vt)", "drift resync"),
            ("self.forensic_marks + self.audio_marks", "audio marks in n/p nav"),
            ("self.btn_sound, self.btn_audio_forensics", "buttons in disable loop"),
            ('"va_audition_%s.wav" % self._uid', "audition wav cleanup"),
            ("audio_strip.set_silence", "silence shading after Analyze"),
            ('_adv_store("audio_forensics"', "advanced cache key"),
    ):
        check("gui: " + why, needle in gsrc)

    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            if name == "TclError":
                v = type("TclError", (Exception,), {})
            elif name[:1].isupper():
                v = type(name, (), {"__init__": lambda s, *a, **k: None,
                                    "__getattr__": lambda s, n: (lambda *a, **k: None)})
            else:
                v = _Stub(self.__name__ + "." + name)
            setattr(self, name, v)
            return v

    saved = {m: sys.modules.get(m) for m in
             ("tkinter", "tkinter.ttk", "tkinter.filedialog",
              "tkinter.messagebox", "tkinter.font", "tkinterdnd2")}
    try:
        for m in saved:
            sys.modules[m] = _Stub(m)
        spec = importlib.util.spec_from_file_location(
            "va_gui_stub", os.path.join(here, "video-analyzer.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        s = object.__new__(mod.AudioStrip)
        s.dur, s.z0, s.z1 = 100.0, 0.0, 1.0
        s.winfo_width = lambda: 1000
        check("strip t->x", s._t_to_x(50.0) == 500)
        s.z0, s.z1 = 0.25, 0.75
        check("strip zoom map", s._t_to_x(50.0) == 500 and s._t_to_x(10.0) is None
              and abs(s._x_to_frac(0) - 0.25) < 1e-9)
        s.ov = {"buckets": 4}
        s.redraw = lambda: None
        ev = type("E", (), {"x": 500, "delta": 120})()
        s._wheel(ev)
        check("strip wheel zoom-in", (s.z1 - s.z0) < 0.5)
        ev.delta = -120
        for _ in range(10):
            s._wheel(ev)
        check("strip zoom-out clamps", (s.z0, s.z1) == (0.0, 1.0))
    finally:
        for m, v in saved.items():
            if v is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = v

    real_find = VFF.find_tool
    try:
        VFF.find_tool.cache_clear()
        import va_audio
        # follower must be graceful with no ffplay anywhere
        orig = va_audio.AudioFollower.__init__

        def _init_no_exe(self, path):
            self.path, self.exe, self.proc = path, None, None
            self._t0 = self._wall0 = 0.0
        va_audio.AudioFollower.__init__ = _init_no_exe
        f = va_audio.AudioFollower("x.mp4")
        f.start(1.0)
        check("follower graceful without ffplay",
              not f.available and f.expected_t() is None and not f.playing())
        f.stop()
        va_audio.AudioFollower.__init__ = orig
    finally:
        VFF.find_tool.cache_clear()


def chk_rpu_parse(tmp):
    """va_rpu: bit-exact Dolby Vision RPU parsing (synthetic P5 payload built
    with a local bit-writer against dovi_tool's field layout), the canonical-
    constant fallback (ICtCp regression guard), and the dovi_meta chain."""
    import numpy as np
    import va_rpu
    import va_ipt

    class W:
        def __init__(self):
            self.bits = []

        def u(self, v, n):
            for i in range(n - 1, -1, -1):
                self.bits.append((v >> i) & 1)

        def ue(self, v):
            k = v + 1
            n = k.bit_length()
            self.u(0, n - 1)
            self.u(k, n)

        def se(self, v):
            self.ue(2 * v - 1 if v > 0 else -2 * v)

        def bytes(self):
            while len(self.bits) % 8:
                self.bits.append(0)
            out = bytearray()
            for i in range(0, len(self.bits), 8):
                b = 0
                for bit in self.bits[i:i + 8]:
                    b = (b << 1) | bit
                out.append(b)
            return bytes(out)

    w = W()
    w.u(2, 6)        # rpu_type
    w.u(18, 11)      # rpu_format
    w.u(0, 4)        # vdr_rpu_profile (P5)
    w.u(0, 4)        # vdr_rpu_level
    w.u(1, 1)        # vdr_seq_info_present
    w.u(0, 1)        # chroma_resampling_explicit_filter
    w.u(0, 2)        # coefficient_data_type
    w.ue(23)         # coefficient_log2_denom
    w.u(1, 2)        # vdr_rpu_normalized_idc
    w.u(1, 1)        # bl_video_full_range (P5)
    w.ue(2)          # bl_bit_depth_minus8
    w.ue(2)          # el_bit_depth_minus8
    w.ue(4)          # vdr_bit_depth_minus8
    w.u(0, 1)        # spatial_resampling_filter
    w.u(0, 3)        # reserved_zero_3bits (uncompressed DM)
    w.u(0, 1)        # el_spatial_resampling_filter
    w.u(1, 1)        # disable_residual (P5: no NLQ)
    w.u(1, 1)        # vdr_dm_metadata_present
    w.u(0, 1)        # use_prev_vdr_rpu
    w.ue(0)          # vdr_rpu_id
    w.ue(0)          # mapping_color_space
    w.ue(0)          # mapping_chroma_format_idc
    for _ in range(3):
        w.ue(0)              # num_pivots_minus2
        w.u(0, 10)           # pivot 0
        w.u(1023, 10)        # pivot 1
    w.ue(0)          # num_x_partitions_minus1
    w.ue(0)          # num_y_partitions_minus1
    for _ in range(3):       # identity poly per component
        w.ue(0)              # mapping_idc = polynomial
        w.ue(0)              # poly_order_minus1 (order 1... order = ue+1 = 1)
        w.u(0, 1)            # linear_interp_flag = 0
        w.se(0); w.u(0, 23)  # coef0 = 0
        w.se(1); w.u(0, 23)  # coef1 = 1
    # vdr_dm_data (canonical P5 constants)
    w.ue(0); w.ue(0); w.ue(0)
    for v in (8192, 799, 1681, 8192, -933, 1091, 8192, 267, -5545):
        w.u(v & 0xFFFF, 16)
    for v in (0, 1 << 27, 1 << 27):
        w.u(v, 32)
    for v in (17081, -349, -349, -349, 17081, -349, -349, -349, 17081):
        w.u(v & 0xFFFF, 16)
    w.u(65535, 16)           # signal_eotf
    w.u(0, 16); w.u(0, 16); w.u(0, 32)
    w.u(12, 5)               # signal_bit_depth
    w.u(2, 2)                # signal_color_space
    w.u(0, 2); w.u(1, 2)
    w.u(62, 12)              # source_min_pq
    w.u(3696, 12)            # source_max_pq
    w.u(42, 10)              # source_diagonal

    payload = b"\x19" + w.bytes() + b"\x00\x00\x00\x00\x80"
    meta = va_rpu.parse_rpu(payload)
    check("rpu parse returns meta", meta is not None and meta.get("exact"))
    check("rpu ycc row0", meta and all(abs(a - b) < 1e-9 for a, b in zip(
        meta["nonlinear"][0], [1.0, 799 / 8192.0, 1681 / 8192.0])),
        None if not meta else str(meta["nonlinear"][0]))
    check("rpu offsets", meta and meta["offset"] == [0.0, 0.5, 0.5])
    check("rpu crosstalk inverse", meta and abs(
        meta["linear"][0][0] - 17081 / 16384.0) < 1e-9)
    check("rpu source pq", meta and meta["source_min_pq"] == 62
          and meta["source_max_pq"] == 3696)
    check("rpu identity curves -> None", meta and meta["curves"] is None)

    fb = va_ipt._FALLBACK_META
    check("fallback is canonical P5 (not ICtCp)",
          abs(fb["nonlinear"][0][1] - 799 / 8192.0) < 1e-9
          and abs(fb["nonlinear"][2][2] - (-5545 / 8192.0)) < 1e-9
          and abs(fb["linear"][0][0] - 17081 / 16384.0) < 1e-9,
          str(fb["nonlinear"][0]))

    # dovi_meta chain: ffprobe blind -> va_rpu fallback engages
    import va_ffmpeg
    real_pj = va_ipt.ffprobe_json
    real_stream = va_rpu.dovi_meta_from_stream
    try:
        va_ipt.ffprobe_json = lambda p, frames=False, timeout=30: {"frames": []}
        va_rpu.dovi_meta_from_stream = lambda p: dict(meta, via="rpu-parse")
        va_ipt.reset_cache()
        got = va_ipt.dovi_meta("synthetic_p5.mp4")
        check("dovi_meta falls back to RPU parse",
              got is not None and got.get("via") == "rpu-parse" and got.get("exact"))
    finally:
        va_ipt.ffprobe_json = real_pj
        va_rpu.dovi_meta_from_stream = real_stream
        va_ipt.reset_cache()


def chk_tools_dir(tmp):
    """tools/ housing: search order, portable mkvtoolnix nesting, tidy
    migration, and the GUI/CLI download destination."""
    here = os.path.dirname(os.path.abspath(__file__))
    import shutil
    import va_tools as VT

    d = os.path.join(tmp, "housing")
    os.makedirs(os.path.join(d, "tools", "mkvtoolnix"), exist_ok=True)
    td = VT.tools_dir(d)
    check("tools_dir path + creation", td == os.path.join(d, "tools")
          and os.path.isdir(td))
    open(os.path.join(d, "dovi_tool.exe"), "wb").write(b"MZ")
    open(os.path.join(d, "ffmpeg.exe"), "wb").write(b"MZ")
    moved = VT.tidy_tool_folder(base=d)
    check("tidy moves loose exes", sorted(moved) == ["dovi_tool.exe", "ffmpeg.exe"]
          and os.path.isfile(os.path.join(td, "ffmpeg.exe"))
          and not os.path.exists(os.path.join(d, "ffmpeg.exe")), str(moved))
    check("tidy idempotent", VT.tidy_tool_folder(base=d) == [])

    fsrc = open(os.path.join(here, "va_ffmpeg.py"), encoding="utf-8").read()
    seg = fsrc[fsrc.index("def find_tool"):fsrc.index("def find_ffmpeg")]
    check("find_tool searches tools/ first", '"tools", "", "mkvtoolnix"' in seg
          and "tools" in seg.split("for sub in")[1][:90])
    check("find_tool reaches tools/mkvtoolnix",
          'os.path.join("tools", "mkvtoolnix")' in seg)
    gsrc = open(os.path.join(here, "video-analyzer.py"), encoding="utf-8").read()
    check("GUI downloads into tools/", "va_tools.tools_dir()" in gsrc)
    check("GUI tidy button wired", "_tidy_tools" in gsrc
          and "Tidy into tools/" in gsrc)
    tsrc = open(os.path.join(here, "va_tools.py"), encoding="utf-8").read()
    check("CLI downloads into tools/", "install_tool(n, tools_dir()" in tsrc)


def chk_silent_spawns():
    """Every engine/plugin subprocess spawn must pass creationflags= so the
    windowed .exe never flashes console windows (CREATE_NO_WINDOW on nt; a
    frozen GUI has no console for children to inherit). Paren-matched source
    scan; a '# console-ok' comment on the call line exempts non-Windows
    branches."""
    import glob as _g
    import re as _re
    here = os.path.dirname(os.path.abspath(__file__))
    pat = _re.compile(r"subprocess\.(run|Popen|check_output|check_call|call)\(")
    offenders = []
    files = [p for p in (_g.glob(os.path.join(here, "*.py"))
                         + _g.glob(os.path.join(here, "plugins", "*", "*.py")))
             if not os.path.basename(p).startswith("selftest")
             and not os.path.basename(p).endswith(".bak")]
    for fp in sorted(files):
        try:
            src = open(fp, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for m in pat.finditer(src):
            depth, j = 0, m.end() - 1
            while j < len(src):
                if src[j] == "(":
                    depth += 1
                elif src[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            call = src[m.start():j]
            eol = src.find("\n", m.start())
            head = src[m.start():eol] if eol != -1 else src[m.start():]
            if "creationflags" in call or "console-ok" in head:
                continue
            offenders.append("%s:%d" % (os.path.basename(fp),
                                        src[:m.start()].count("\n") + 1))
    check("all subprocess spawns hide their console",
          not offenders,
          ", ".join(offenders) if offenders else "every spawn passes creationflags")


def chk_dynhdr_compat_flag(tmp):
    """MDCV/CLL fallback warning must be compatibility-aware (HLG/SDR bases
    carry no HDR10 static metadata by design)."""
    import va_dynhdr as VD

    def mock(compat):
        return {"streams": [{"codec_type": "video", "side_data_list": [
            {"side_data_type": "DOVI configuration record", "dv_profile": 8,
             "dv_level": 6, "rpu_present_flag": 1, "bl_present_flag": 1,
             "el_present_flag": 0, "dv_bl_signal_compatibility_id": compat}]}],
            "frames": []}
    flags1 = VD.parse_hdr_metadata(mock(1))["flags"]
    check("compat 1 without MDCV flags", any("MDCV" in f[1] for f in flags1),
          str(flags1))
    flags4 = VD.parse_hdr_metadata(mock(4))["flags"]
    check("compat 4 (HLG) clean", not any("MDCV" in f[1] for f in flags4),
          str(flags4))

if __name__ == "__main__":
    sys.exit(main())
