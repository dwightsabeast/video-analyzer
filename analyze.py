#!/usr/bin/env python3
"""
analyze.py - headless batch analyzer + QC for video files (no GUI).

    python analyze.py INPUT [-o OUTDIR] [--profile P] [--formats json,html,csv]
                      [--recurse] [--quiet]

INPUT is a single video file or a folder. For each video it runs the full
engine (signalstats, events, loudness, gamut), evaluates a QC profile, and
writes reports. For a folder it also writes a batch dashboard (index.html) with
per-file QC verdicts. Profiles: see --list-profiles.
"""

from __future__ import annotations

import os
import re
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import va_ffmpeg as F
import va_metrics as M
import va_audio as A
import va_scopes as S
import va_qc as QC
import va_export as X
import va_perceptual as P
import va_temporal as T
import va_hdr as HD
import va_dynhdr as DH
import va_forensics as FR
import va_perf as PERF
import va_hwaccel as HW
import va_plugins

PLUGIN_PASSES: dict = {}   # profile -> deep-pass spec from plugins (filled in main)

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".mpg",
             ".mpeg", ".wmv", ".flv", ".3gp", ".mxf", ".m2ts", ".hevc", ".265"}


def _collect(path, recurse):
    if os.path.isfile(path):
        return [path]
    out = []
    for root, _dirs, files in os.walk(path):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in VIDEO_EXT:
                out.append(os.path.join(root, fn))
        if not recurse:
            break
    return out


def _slug(text):
    """Filesystem/URL-safe output stem derived from a relative path."""
    out = re.sub(r"[\\/]+", "_", str(text))
    out = re.sub(r"[^\w.\- ]+", "_", out)
    return out.strip(" ._") or "file"


def _assign_output_bases(files, input_path, recurse):
    """Unique flat output stem per input file. Returns ({path: stem}, [notes]).

    Without this, a/clip.mp4 and b/clip.mp4 silently overwrite each other's
    reports. Collisions are uniquified deterministically (case-insensitive for
    Windows filesystems): relative-path slugs under --recurse, else -2/-3
    suffixes; the layout stays flat."""
    groups = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        groups.setdefault(stem.lower(), []).append(f)
    root = input_path if os.path.isdir(input_path) else (os.path.dirname(input_path) or ".")
    bases, notes, taken = {}, [], set()
    for key, group in groups.items():          # unique names keep their stem
        if len(group) == 1:
            bases[group[0]] = os.path.splitext(os.path.basename(group[0]))[0]
            taken.add(key)
    for key in sorted(groups):                 # then resolve collision groups
        group = sorted(groups[key])
        if len(group) == 1:
            continue
        for f in group:
            stem = os.path.splitext(os.path.basename(f))[0]
            cand = _slug(os.path.splitext(os.path.relpath(f, root))[0]) if recurse else stem
            uniq, i = cand, 1
            while uniq.lower() in taken:
                i += 1
                uniq = "%s-%d" % (cand, i)
            taken.add(uniq.lower())
            bases[f] = uniq
            if uniq != stem:
                notes.append("NOTE: duplicate output name %r - writing %s as %s.*" % (
                    stem, os.path.relpath(f, root), uniq))
    return bases, notes


def analyze_file(path, outdir, formats, profile, quiet=False, out_base=None,
                 forensics=False, echo=print):
    info = F.probe(path)
    if not info.get("ok"):
        why = "cannot open"
        try:
            if A.has_audio(path):
                why = "no video stream (audio-only?)"
        except Exception:
            pass
        echo("  SKIP (%s): %s" % (why, os.path.basename(path)))
        return None
    cp = M.analyze_pass(path)              # 1 decode for metrics+events+audio
    if cp is not None:
        table, events, silence = cp["table"], cp["events"], cp["silence"]
        loud = (cp.get("audio") or {}).get("summary")
        if not quiet:
            echo("  combined pass: %d analyses, 1 decode (%s), %.1fs" % (
                cp["passes_merged"], cp["decode"], cp["elapsed"]))
    else:                                   # per-pass fallback (old behavior)
        table = M.signalstats(path)
        events = {"black": M.black_segments(path), "freeze": M.freeze_segments(path),
                  "scene_cuts": M.scene_cuts(path)}
        aud = A.loudness(path) if A.has_audio(path) else None
        loud = aud["summary"] if aud and aud.get("summary") else None
        silence = A.silence_segments(path) if A.has_audio(path) else []
    gamut = None
    fr = None
    try:
        vs = F.VideoSource(path)
        fr = vs.frame_at(max(1, (info["nb_frames"] or 2) // 2))
        vs.close()
        if fr is not None:
            _, gamut = S.cie_gamut(fr, 320)
    except Exception:
        gamut = None
    banding = None
    if fr is not None:
        try:
            _, banding = P.banding_map(fr)
        except Exception:
            banding = None
    try:
        cad = T.cadence(F.VideoSource(path))
    except Exception:
        cad = None
    try:
        pse = T.pse_flashes(F.VideoSource(path))
    except Exception:
        pse = None
    hdr_cll = HD.maxcll_maxfall(path) if info.get("is_hdr") else None
    hdr_meta = DH.inspect(path) if info.get("is_hdr") else None
    plugin_passes = [(prof, p) for prof, p in sorted(PLUGIN_PASSES.items())
                     if p.get("always") or prof == profile]
    frep = None
    if forensics or profile == "integrity" or any(
            p["needs_forensics"] for _pr, p in plugin_passes):
        if not quiet:
            echo("  forensics battery (hash/splice/container/noise/loops/ENF)...")
        frep = FR.forensics_report(path)
    plugin_reps = []
    for _prof, p in plugin_passes:
        if not quiet:
            echo("  %s..." % p["echo"])
        plugin_reps.append((p, p["run"](path, forensics=frep, cadence=cad)))

    ctx = {"probe": info, "summary": table.summary(), "loudness": loud,
           "events": events, "gamut": gamut, "silence": silence,
           "banding": banding, "cadence": cad, "pse": pse, "hdr_cll": hdr_cll,
           "hdr_meta": hdr_meta, "forensics": frep}
    for p, rep in plugin_reps:
        if rep is not None and p["ctx"]:
            ctx[p.get("ctx_key", "verify")] = p["ctx"](rep)
    ctx.setdefault("verify", None)
    pass_json = {p.get("ctx_key", "verify"):
                 (p["json"](rep) if (rep is not None and p["json"]) else None)
                 for p, rep in plugin_reps}
    qc = QC.evaluate(ctx, profile)

    base = out_base or os.path.splitext(os.path.basename(path))[0]
    report = None
    if "csv" in formats:
        X.write_csv(table, os.path.join(outdir, base + ".csv"))
    if "json" in formats:
        X.write_json(table, os.path.join(outdir, base + ".json"), source=info,
                     events=events, loudness=loud, extra={"qc": qc, "gamut": gamut, "banding": banding, "cadence": cad,
                            "pse": pse, "hdr_cll": hdr_cll, "hdr_meta": hdr_meta,
                            "forensics": frep,
                            "verify": pass_json.get("verify"),
                            "plugin_passes": ({k: v for k, v in pass_json.items()
                                               if k != "verify"} or None)})
    if "html" in formats:
        report = os.path.join(outdir, base + ".report.html")
        X.write_html_report(table, report, source=info, events=events,
                             loudness=loud, gamut=gamut, qc=qc, forensics=frep)
    if frep is not None:
        with open(os.path.join(outdir, base + ".forensics.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(FR.render_report(frep))
    for p, rep in plugin_reps:
        if rep is not None and p["render"]:
            with open(os.path.join(outdir, base + p["suffix"]), "w",
                      encoding="utf-8") as fh:
                fh.write(p["render"](rep))

    if not quiet:
        echo("  %-40s %5dx%-4d %-5s %-4s  QC[%s]: %s" % (
            os.path.basename(path)[:40], info["width"], info["height"], info["codec"],
            "HDR" if info["is_hdr"] else "SDR", profile, qc["verdict"].upper()))
        for c in qc["checks"]:
            if c["status"] != "pass":
                echo("        %-6s %-22s %s" % (c["status"].upper(), c["label"], c["detail"]))
    return {"name": os.path.basename(path), "codec": info["codec"],
            "resolution": "%dx%d" % (info["width"], info["height"]), "hdr": info["is_hdr"],
            "loudness": (loud.get("integrated_lufs") if loud else "-"),
            "verdict": qc["verdict"], "report": report}


def main():
    ap = argparse.ArgumentParser(description="Headless video analysis + QC.")
    ap.add_argument("input", nargs="?", help="video file or folder")
    ap.add_argument("-o", "--out", default="va_reports", help="output folder (default: va_reports)")
    ap.add_argument("--profile", default="general", help="QC profile (--list-profiles)")
    ap.add_argument("--formats", default="json,html", help="comma list of json,html,csv")
    ap.add_argument("--recurse", action="store_true", help="recurse into subfolders")
    ap.add_argument("--forensics", action="store_true",
                    help="run the forensics battery (SHA-256, splice scan, container "
                         "walk, encoder fingerprint, noise/loop/ENF) and write "
                         "<name>.forensics.txt (implied by --profile integrity and "
                         "by plugin profiles such as speedrun, which also write "
                         "their own report, e.g. <name>.verify.txt)")
    ap.add_argument("--jobs", default="1", metavar="N",
                    help="files to analyze concurrently (default 1 = serial; "
                         "'auto' sizes from your cores; each job already runs "
                         "a multi-threaded single-decode pass)")
    ap.add_argument("--perf", choices=("max", "balanced", "eco"),
                    help="performance mode (default max; eco = lowest power: "
                         "hw decode, no oversubscription). Also via VA_PERF env.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--list-profiles", action="store_true")
    args = ap.parse_args()

    hooks = va_plugins.load_headless()
    PLUGIN_PASSES.update(hooks.passes)
    for e in hooks.errors:
        print("plugin error: %s" % e, file=sys.stderr)

    if args.list_profiles:
        print("QC profiles:", ", ".join(QC.profile_names()))
        return 0
    if args.profile not in QC.profile_names():
        print("unknown profile %r - run --list-profiles. Domain profiles "
              "(e.g. 'speedrun') come from plugins/; is the plugin "
              "installed and enabled?" % args.profile, file=sys.stderr)
        return 2
    if not args.input:
        ap.error("an input file or folder is required")
    if not F.find_ffmpeg() and not F.find_ffprobe():
        print("No ffmpeg/ffprobe found - put ffmpeg(.exe)/ffprobe(.exe) beside the scripts or on PATH.")
        return 2
    if args.profile not in QC.profile_names():
        print("Unknown profile %r. Available: %s" % (args.profile, ", ".join(QC.profile_names())))
        return 2
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    bad = [f for f in formats if f not in ("json", "html", "csv")]
    if bad or not formats:
        print("Unknown format(s) %r. Available: json, html, csv" % ", ".join(bad or [args.formats]))
        return 2
    if not os.path.exists(args.input):
        print("input not found:", args.input)
        return 2

    files = _collect(args.input, args.recurse)
    if not files:
        print("No video files found in:", args.input)
        return 1
    bases, notes = _assign_output_bases(files, args.input, args.recurse)
    os.makedirs(args.out, exist_ok=True)
    print("Analyzing %d file(s) -> %s   (profile: %s)" % (len(files), args.out, args.profile))
    for n in notes:
        print(n)
    if args.perf:
        PERF.set_mode(args.perf)
    jobs = PERF.batch_jobs(args.jobs, len(files))
    if jobs > 1:
        print("Parallel batch: %d jobs (mode: %s) - output is buffered per file" % (
            jobs, PERF.mode()))
        try:
            HW.decode_method(files[0])      # warm the hw-decode cache once,
        except Exception:                   # not once per worker thread
            pass
    entries = []
    write_errors = 0

    def _post(e, f):
        if e and os.path.isdir(args.input) and \
                bases.get(f, "") != os.path.splitext(os.path.basename(f))[0]:
            e["name"] = os.path.relpath(f, args.input).replace(os.sep, "/")
        return e

    if jobs <= 1:
        for f in files:
            try:
                e = analyze_file(f, args.out, formats, args.profile, args.quiet,
                                 out_base=bases.get(f), forensics=args.forensics)
            except OSError as exc:
                # an unwritable/unreachable output path must not kill the batch
                print("  ERROR %-38s could not write report: %s" % (
                    os.path.basename(f)[:38], exc))
                write_errors += 1
                continue
            if _post(e, f):
                entries.append(e)
    else:
        import threading
        from concurrent.futures import ThreadPoolExecutor

        plock = threading.Lock()
        results = {}

        def _work(f):
            buf = []
            echo = lambda *a: buf.append(" ".join(str(x) for x in a))
            try:
                e = analyze_file(f, args.out, formats, args.profile, args.quiet,
                                 out_base=bases.get(f), forensics=args.forensics,
                                 echo=echo)
            except OSError as exc:
                echo("  ERROR %-38s could not write report: %s" % (
                    os.path.basename(f)[:38], exc))
                e = OSError
            with plock:
                for ln in buf:
                    print(ln)
            return e

        with ThreadPoolExecutor(max_workers=jobs) as ex:
            for f, e in zip(files, ex.map(_work, files)):
                results[f] = e
        for f in files:                     # dashboard rows in input order
            e = results.get(f)
            if e is OSError:
                write_errors += 1
            elif _post(e, f):
                entries.append(e)
    if entries and (len(entries) > 1 or os.path.isdir(args.input)):
        dash = os.path.join(args.out, "index.html")
        try:
            X.write_batch_dashboard(entries, dash, args.profile)
            print("Batch dashboard:", os.path.abspath(dash))
        except OSError as exc:
            print("ERROR could not write the batch dashboard: %s" % exc)
            write_errors += 1
    fails = sum(1 for e in entries if e["verdict"] == "fail")
    if write_errors:
        print("%d file(s) could not be written - check the output folder permissions." % write_errors)
    print("Done. %d analysed, %d FAIL, %d WARN." % (
        len(entries), fails, sum(1 for e in entries if e["verdict"] == "warn")))
    print("Reports -> %s" % os.path.abspath(args.out))
    if not entries or write_errors:
        return 1                               # every file skipped / unwritable
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
