#!/usr/bin/env python3
"""
va_export - turn an analysis into portable data + a standalone report.

Writes the per-frame metrics table to CSV (universal) and a structured JSON
"analysis document" (source params + aggregate summary + events + loudness +
per-frame rows), and renders a self-contained HTML report. Pure compute.
"""

from __future__ import annotations

import csv
import base64
import json
import os
import time

import numpy as np

from va_metrics import SIGNALSTATS_TAGS

VERSION = "2.0"


def _clean(v):
    """NaN -> None so it survives CSV/JSON cleanly."""
    try:
        if v != v:  # NaN
            return None
    except TypeError:
        return v
    return v


def _jsafe(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def table_rows(table):
    a = table.arrays()
    n = len(table)
    for i in range(n):
        row = {"frame": i, "t": round(float(a["t"][i]), 4)}
        for tag in SIGNALSTATS_TAGS:
            row[tag] = _clean(float(a[tag][i]))
        yield row


def write_csv(table, path, extra=None) -> str:
    """Per-frame CSV. ``extra`` = {col_name: array} for aligned derived series."""
    extra = extra or {}
    cols = ["frame", "t"] + SIGNALSTATS_TAGS + list(extra.keys())
    a = table.arrays()
    n = len(table)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(n):
            row = [i, round(float(a["t"][i]), 4)]
            for tag in SIGNALSTATS_TAGS:
                v = a[tag][i]
                row.append("" if v != v else v)
            for k, arr in extra.items():
                row.append(arr[i] if i < len(arr) else "")
            w.writerow(row)
    return path


def build_document(table, source=None, events=None, loudness=None,
                   extra=None) -> dict:
    src = {}
    if source:
        for k in ("path", "width", "height", "fps", "nb_frames", "duration",
                  "codec", "pix_fmt", "bit_depth", "transfer", "primaries",
                  "is_hdr", "is_wide_gamut"):
            if k in source:
                src[k] = os.path.basename(source[k]) if k == "path" else source[k]
    return {
        "tool": "video-analyzer",
        "version": VERSION,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": src,
        "frame_count": len(table),
        "summary": table.summary(),
        "events": events or {},
        "loudness": loudness,
        "extra": extra or {},
        "frames": list(table_rows(table)),
    }


def write_json(table, path, source=None, events=None, loudness=None,
               extra=None, advanced=None) -> str:
    doc = build_document(table, source, events, loudness, extra)
    if advanced:
        doc["advanced"] = _advanced_json(advanced)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, default=_jsafe)
    return path


# --- Advanced-tool results (GUI session cache: text / data / PNG image) ------

_ADV_ORDER = ["qc", "verify", "forensics", "audio_forensics", "content_credentials", "banding_map",
              "ela_map", "noise_map", "enf", "hdr_metadata", "dynamic_vs_content",
              "dynamic_vs_static", "hdr_multi_display", "dovi_l1_plot",
              "hdr10plus_plot", "mediainfo", "mp4_boxes", "mkv_structure"]


def _adv_keys(adv):
    return [k for k in _ADV_ORDER if k in adv] +            [k for k in sorted(adv) if k not in _ADV_ORDER]


def _advanced_json(adv) -> dict:
    """JSON-facing view of the advanced-results cache: keeps title/data/text,
    drops image bytes (the HTML report embeds those)."""
    out = {}
    for k in _adv_keys(adv):
        e = adv.get(k) or {}
        out[k] = {"title": e.get("title"), "generated": e.get("generated"),
                  "data": e.get("data"), "text": e.get("text"),
                  "image_in_html_report": bool(e.get("png"))}
    return out


def _advanced_section(adv) -> str:
    """HTML section for the advanced-results cache; images embedded base64."""
    parts = ["<h2>Advanced analyses</h2>",
             "<p>Results of the Advanced-menu tools run during this session. "
             "Tools that were not run are not listed.</p>"]
    for k in _adv_keys(adv):
        e = adv.get(k) or {}
        parts.append("<h3>%s</h3>" % _esc(e.get("title") or k))
        if e.get("generated"):
            parts.append("<p class='mm'>generated %s</p>" % _esc(e["generated"]))
        data = e.get("data")
        if isinstance(data, dict) and data and                 all(not isinstance(v, (dict, list, tuple)) for v in data.values()):
            parts.append(_kv_table(sorted(data.items())))
        elif data is not None:
            parts.append("<pre>%s</pre>" % _esc(
                json.dumps(data, indent=1, default=_jsafe)[:20000]))
        if e.get("text"):
            parts.append("<pre>%s</pre>" % _esc(str(e["text"])[:40000]))
        if e.get("png"):
            parts.append('<img style="max-width:100%%;border:1px solid #333" '
                         'src="data:image/png;base64,%s"/>' %
                         base64.b64encode(e["png"]).decode("ascii"))
    return "".join(parts)


# --- Self-contained HTML report ----------------------------------------------

def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _polyline(values, w, h, color, vmin=None, vmax=None) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return ""
    lo = min(vals) if vmin is None else vmin
    hi = max(vals) if vmax is None else vmax
    rng = (hi - lo) or 1.0
    n = len(values)
    stride = max(1, n // 800)
    pts = []
    for i in range(0, n, stride):
        v = values[i]
        if v is None:
            continue
        x = (i / (n - 1) * w) if n > 1 else 0
        y = h - ((v - lo) / rng) * h
        pts.append("%.1f,%.1f" % (x, y))
    return '<polyline fill="none" stroke="%s" stroke-width="1" points="%s"/>' % (
        color, " ".join(pts))


_MARK_COLORS = {"splice": "#b07fd8", "loops": "#5fb4ff", "noise": "#ff9f40",
                "audio": "#4fd1c5",
                "enf": "#6dd06d"}
_MARK_OTHER = "#d870a0"


def _forensic_marks(forensics):
    """[(t, area, severity)] for every timestamped forensic finding."""
    out = []
    for f in (forensics or {}).get("findings", []) or []:
        if f.get("t") is not None:
            out.append((float(f["t"]), f.get("area", "forensic"),
                        f.get("severity", "info")))
    return out


def _timeline_legend(cuts, marks) -> str:
    """Legend for the chart overlays - only entries actually drawn."""
    items = []
    if cuts:
        items.append(("scene cut", "#ff6060", "solid"))
    seen = []
    for _t, area, _s in (marks or []):
        if area not in seen:
            seen.append(area)
    items += [("%s finding" % a, _MARK_COLORS.get(a, _MARK_OTHER), "dashed")
              for a in seen]
    if not items:
        return ""
    spans = "".join(
        '<span style="margin-right:18px;white-space:nowrap">'
        '<span style="display:inline-block;width:20px;border-top:3px %s %s;'
        'vertical-align:middle;margin-right:6px"></span>%s</span>'
        % (sty, col, _esc(lab)) for lab, col, sty in items)
    note = ("&nbsp;&middot;&nbsp; bright dash = corroborated (warn), dim = "
            "informational &middot; hover a dashed line for details" if seen else "")
    return '<p class="mm" style="margin:4px 0 0 0">%s%s</p>' % (spans, note)


def _chart(title, values, color, fps=25.0, cuts=None, vmin=None, vmax=None,
           marks=None) -> str:
    w, h = 900, 120
    vals = [v for v in values if v is not None]
    lo = ("%.2f" % min(vals)) if vals else "-"
    hi = ("%.2f" % max(vals)) if vals else "-"
    ticks = ""
    n = len(values)
    if cuts:
        for c in cuts:
            fi = c * fps
            if 0 < fi < n:
                x = fi / (n - 1) * w
                ticks += '<line x1="%.1f" y1="0" x2="%.1f" y2="%d" stroke="#ff6060" stroke-width="1" opacity="0.5"/>' % (x, x, h)
    for t, area, sev in (marks or []):
        fi = t * fps
        if 0 < fi < n:
            x = fi / (n - 1) * w
            ticks += ('<line x1="%.1f" y1="0" x2="%.1f" y2="%d" stroke="%s" '
                      'stroke-width="1.5" stroke-dasharray="6,4" opacity="%s">'
                      '<title>%s</title></line>'
                      % (x, x, h, _MARK_COLORS.get(area, _MARK_OTHER),
                         "0.9" if sev == "warn" else "0.45",
                         _esc("%s (%s) @ %.1fs" % (area, sev, t))))
    return (
        '<div class="chart"><div class="ct">%s <span class="mm">min %s / max %s</span></div>'
        '<svg viewBox="0 0 %d %d" preserveAspectRatio="none" class="cv">'
        '<rect width="%d" height="%d" fill="#181818"/>%s%s</svg></div>'
        % (_esc(title), lo, hi, w, h, w, h, ticks, _polyline(values, w, h, color, vmin, vmax))
    )


def _fmt_cell(v):
    """Readable cell text for missing/degenerate values (None, -inf)."""
    if v is None:
        return "n/a"
    if isinstance(v, float) and v == float("-inf"):
        return "-inf (digital silence)"
    return v


def _kv_table(rows) -> str:
    body = "".join("<tr><td>%s</td><td>%s</td></tr>" % (_esc(k), _esc(_fmt_cell(v)))
                   for k, v in rows)
    return "<table class='kv'>%s</table>" % body


def write_html_report(table, path, source=None, events=None, loudness=None,
                      quality=None, gamut=None, qc=None, forensics=None,
                      advanced=None) -> str:
    a = table.arrays()
    fps = (source or {}).get("fps", 25.0) or 25.0
    cuts = (events or {}).get("scene_cuts") or []
    summ = table.summary()

    src_rows = []
    if source:
        for k in ("path", "codec", "pix_fmt", "bit_depth"):
            if source.get(k):
                src_rows.append((k, os.path.basename(source[k]) if k == "path" else source[k]))
        if source.get("width"):
            src_rows.append(("resolution", "%sx%s" % (source["width"], source["height"])))
        src_rows.append(("fps", round(fps, 3)))
        if source.get("duration"):
            src_rows.append(("duration", "%.2fs" % source["duration"]))
        src_rows.append(("HDR", "yes" if source.get("is_hdr") else "no"))
        src_rows.append(("wide gamut", "yes" if source.get("is_wide_gamut") else "no"))

    sum_rows = ""
    for k in ("YAVG", "YMIN", "YMAX", "SATAVG", "SATMAX", "HUEAVG", "BRNG", "TOUT"):
        if k in summ:
            d = summ[k]
            sum_rows += ("<tr><td>%s</td><td>%.2f</td><td>%.2f</td><td>%.2f</td>"
                         "<td>%.2f</td><td>%.2f</td></tr>" %
                         (k, d["min"], d["p1"], d["mean"], d["p99"], d["max"]))

    ev_html = ""
    if events:
        for name in ("black", "freeze"):
            segs = events.get(name) or []
            if segs:
                ev_html += "<p><b>%s:</b> %s</p>" % (
                    name, ", ".join("%.2f-%.2fs" % (s, e) for s, e in segs))
        if cuts:
            ev_html += "<p><b>scene cuts:</b> %s</p>" % ", ".join("%.2fs" % c for c in cuts)
    if not ev_html:
        ev_html = "<p>None detected.</p>"

    loud_html = "<p>No audio analysed.</p>"
    if loudness:
        loud_html = _kv_table([
            ("Integrated (LUFS)", loudness.get("integrated_lufs")),
            ("Loudness range (LU)", loudness.get("lra_lu")),
            ("True peak (dBFS)", loudness.get("true_peak_dbfs")),
        ])

    qual_html = ""
    if quality:
        qrows = []
        for m in ("psnr", "ssim", "xpsnr", "vmaf"):
            d = quality.get(m)
            if d and d.get("average") is not None:
                qrows.append((m.upper(), round(d["average"], 3)))
        vm = quality.get("vmaf") or {}
        if vm.get("model"):
            qrows.append(("VMAF model", vm["model"]))
        note = "" if quality.get("vmaf_available") else " (VMAF unavailable in this ffmpeg build)"
        qual_html = ("<h2>Reference quality</h2><p>%s vs %s%s</p>%s" % (
            _esc(quality.get("distorted", "")), _esc(quality.get("reference", "")),
            note, _kv_table(qrows)))

    gam_html = ""
    if gamut:
        gam_html = ("<h2>Gamut</h2>%s" % _kv_table([
            ("Rec.2020 coverage", "%s%%" % gamut.get("coverage_2020_pct", "-")),
            ("Outside Rec.709", "%s%%" % gamut.get("outside_709_pct", "-")),
        ]))

    yavg = list(a.get("YAVG", []))
    # signalstats reports native code values (0..255 for 8-bit, 0..1023 for
    # 10-bit PQ sources, ...): scale the chart axis by source bit depth, and
    # autoscale with 5% headroom when the depth is unknown.
    y_title = "Luma average (YAVG)"
    try:
        bd = int((source or {}).get("bit_depth"))
        y_top = float(2 ** bd - 1) if bd >= 8 else None
    except (TypeError, ValueError):
        y_top = None
    if y_top is None:
        finite = [v for v in yavg if v is not None and v == v]
        y_top = max(finite) * 1.05 if finite else 255.0
    elif y_top != 255.0:
        y_title = "Luma average (YAVG, %d-bit scale)" % bd
    marks = _forensic_marks(forensics)
    charts = (
        _timeline_legend(cuts, marks)
        + _chart(y_title, yavg, "#4ec9b0", fps, cuts, 0, y_top, marks=marks)
        + _chart("Saturation average (SATAVG)", list(a.get("SATAVG", [])), "#f0c040", fps, cuts, marks=marks)
        + _chart("Broadcast-range pixels (BRNG)", list(a.get("BRNG", [])), "#ff6060", fps, cuts, marks=marks)
    )

    css = ("body{background:#1e1e1e;color:#d4d4d4;font-family:Segoe UI,Arial,sans-serif;margin:24px;}"
           "h1{color:#f0c040;font-weight:500;}h2{color:#9cdcfe;font-weight:500;border-bottom:1px solid #333;padding-bottom:4px;}"
           "table{border-collapse:collapse;margin:8px 0;}td,th{border:1px solid #333;padding:4px 10px;font-size:13px;}"
           ".kv td:first-child{color:#9cdcfe;}.chart{margin:10px 0;}.ct{font-size:13px;color:#aaa;}.mm{color:#666;}"
           ".cv{width:100%;height:120px;border:1px solid #333;}th{color:#f0c040;}"
           "pre{background:#181818;border:1px solid #333;padding:8px;font-size:12px;overflow-x:auto;white-space:pre-wrap;}")

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Video analysis report</title>"
        "<style>%s</style></head><body>"
        "<h1>Video analysis report</h1>"
        "<p>Generated %s by video-analyzer v%s · %d frames analysed</p>"
        "<h2>Source</h2>%s"
        "<h2>Per-frame summary</h2>"
        "<table><tr><th>metric</th><th>min</th><th>p1</th><th>mean</th><th>p99</th><th>max</th></tr>%s</table>"
        "<h2>Timeline</h2>%s"
        "<h2>Events</h2>%s"
        "<h2>Loudness</h2>%s"
        "%s%s"
        "</body></html>"
    ) % (css, time.strftime("%Y-%m-%d %H:%M"), VERSION, len(table),
         _kv_table(src_rows), sum_rows, charts, ev_html, loud_html, qual_html, gam_html)

    if qc:
        html = html.replace("</body></html>", _qc_section(qc) + "</body></html>")
    if forensics:
        html = html.replace("</body></html>", _forensics_section(forensics) + "</body></html>")
    if advanced:
        html = html.replace("</body></html>", _advanced_section(advanced) + "</body></html>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


def _forensics_section(rep) -> str:
    """Findings + splice candidates + chain-of-custody hash from a
    va_forensics.forensics_report() dict."""
    colors = {"warn": "#f0c040", "info": "#9cdcfe"}
    rows = ""
    for f in rep.get("findings", []):
        t = ("%.2fs" % f["t"]) if f.get("t") is not None else "-"
        rows += ("<tr><td style='color:%s'>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                 % (colors.get(f.get("severity"), "#aaa"), _esc(f.get("severity", "").upper()),
                    _esc(f.get("area", "")), t, _esc(f.get("text", ""))))
    if not rows:
        rows = "<tr><td colspan='4'>No findings - nothing surfaced by the battery.</td></tr>"
    sp_rows = ""
    for c in (rep.get("splices") or {}).get("candidates", [])[:12]:
        sp_rows += ("<tr><td>%.2fs</td><td>%d</td><td>%s</td><td>%s</td></tr>" % (
            c["t"], c["score"], _esc("+".join(c["signals"])),
            "yes" if c.get("at_scene_cut") else "no"))
    sp_html = ("<h3>Splice candidates</h3><table><tr><th>time</th><th>score</th>"
               "<th>signals</th><th>at scene cut</th></tr>%s</table>" % sp_rows) if sp_rows else ""
    kv = [("SHA-256", rep.get("sha256")),
          ("Summary", rep.get("summary"))]
    rc = rep.get("recompression") or {}
    if rc.get("dct_double_quant") is not None:
        kv.append(("Double-quantization", rc.get("dct_double_quant")))
    c2 = rep.get("c2pa") or {}
    kv.append(("Content Credentials", ("present (%s)" % (c2.get("validation") or "unvalidated"))
               if c2.get("present") else "none"))
    enf = rep.get("enf") or {}
    if enf:
        kv.append(("ENF (mains hum)", enf.get("note", "")))
    lp = rep.get("loops") or {}
    if lp:
        kv.append(("Repeated sequences", "%d loop(s), %d periodic pattern(s)" % (
            len(lp.get("loops") or []), len(lp.get("periodic") or []))))
    return ("<h2>Forensics / integrity</h2>%s"
            "<h3>Findings</h3><table><tr><th>severity</th><th>area</th><th>time</th>"
            "<th>finding</th></tr>%s</table>%s"
            "<p style='color:#888'>Indicators, not proof - corroborate before "
            "drawing conclusions.</p>") % (_kv_table(kv), rows, sp_html)


def _qc_section(qc) -> str:
    colors = {"pass": "#4ec9b0", "warn": "#f0c040", "fail": "#ff6060"}
    rows = "".join(
        "<tr><td>%s</td><td style='color:%s'>%s</td><td>%s</td></tr>" % (
            _esc(c["label"]), colors.get(c["status"], "#aaa"),
            c["status"].upper(), _esc(c["detail"])) for c in qc.get("checks", []))
    v = qc.get("verdict", "?")
    return ("<h2>QC report - profile '%s' - <span style='color:%s'>%s</span></h2>"
            "<table><tr><th>check</th><th>status</th><th>detail</th></tr>%s</table>" % (
                _esc(qc.get("profile", "")), colors.get(v, "#aaa"), v.upper(), rows))


def write_batch_dashboard(entries, path, profile="") -> str:
    """entries: list of {name, codec, resolution, hdr, loudness, verdict, report}."""
    colors = {"pass": "#4ec9b0", "warn": "#f0c040", "fail": "#ff6060"}
    rows = ""
    for e in entries:
        v = e.get("verdict", "")
        link = e.get("report")
        name = ("<a href='%s'>%s</a>" % (_esc(os.path.basename(link)), _esc(e["name"]))
                if link else _esc(e["name"]))
        loud = e.get("loudness")
        rows += ("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                 "<td style='color:%s'>%s</td></tr>" % (
                     name, _esc(e.get("codec", "")), _esc(e.get("resolution", "")),
                     "HDR" if e.get("hdr") else "SDR",
                     _esc("-" if loud is None else str(loud)),
                     colors.get(v, "#aaa"), v.upper()))
    css = ("body{background:#1e1e1e;color:#d4d4d4;font-family:Segoe UI,Arial,sans-serif;margin:24px}"
           "h1{color:#f0c040;font-weight:500}table{border-collapse:collapse}"
           "td,th{border:1px solid #333;padding:5px 10px;font-size:13px}th{color:#f0c040}a{color:#9cdcfe}")
    html = ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Batch QC</title>"
            "<style>%s</style></head><body><h1>Batch analysis - %d file(s)%s</h1>"
            "<table><tr><th>file</th><th>codec</th><th>resolution</th><th>range</th>"
            "<th>loudness</th><th>QC</th></tr>%s</table></body></html>") % (
                css, len(entries), (" - profile '%s'" % profile if profile else ""), rows)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path
