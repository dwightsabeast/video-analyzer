"""Timer / timecode OCR plugin. GUI-only (the region must be boxed by hand);
the engine lives in ocr_core, the calibration dialog in ocr_dialogs."""

from __future__ import annotations

import os

import ocr_core

STATE = {"rect": None, "bank": None, "overlay": False}
_DLG = {}


def register(api):
    m = api.add_toolbar_menu("Timer OCR ▾")
    api.add_command(m, "Select region && calibrate...", lambda: _open(api))
    api.add_command(m, "Audit timer across file", lambda: _scan(api))
    api.add_command(m, "Show/hide region overlay", lambda: _toggle_overlay(api))
    api.add_separator(m)
    api.add_command(m, "Clear calibration", lambda: _clear(api))
    # rect + glyph bank survive a file change: same overlay, next video
    api.bind_key("<t>", lambda: _read_at_playhead(api),
                 "read the calibrated timer at the playhead")
    # other plugins can read any boxed region with the calibrated font
    api.provide("ocr.read_region",
                lambda gray: (STATE["bank"].read(gray)
                              if STATE["bank"] else ("", 1.0)))
    api.provide("ocr.parse_timer", ocr_core.parse_timer)


def _draw_region(frame, _idx):
    r = STATE.get("rect")
    if r is None:
        return None
    try:
        import cv2
        x0, y0, x1, y1 = [int(v) for v in r]
        cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 180, 255), 2)
        cv2.putText(frame, "OCR", (x0, max(12, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 180, 255), 1,
                    cv2.LINE_AA)
    except Exception:  # noqa: BLE001
        return None
    return frame


def _toggle_overlay(api):
    STATE["overlay"] = not STATE["overlay"]
    if STATE["overlay"]:
        if STATE.get("rect") is None:
            api.status("Timer OCR: no region boxed yet "
                       "(Select region & calibrate...)", "warn")
            STATE["overlay"] = False
            return
        api.add_overlay("ocr_region", _draw_region)
        api.status("Timer OCR: region overlay ON", "ok")
    else:
        api.remove_overlay("ocr_region")
        api.status("Timer OCR: region overlay off", "info")


def _read_at_playhead(api):
    if STATE["bank"] is None or STATE["rect"] is None or not api.path:
        return
    _idx, bgr = api.current_frame()
    if bgr is None:
        return
    import numpy as np
    x0, y0, x1, y1 = [int(v) for v in STATE["rect"]]
    g = np.asarray(bgr).mean(axis=2).astype("float32")[y0:y1, x0:x1]
    s, _d = STATE["bank"].read(g)
    p = ocr_core.parse_timer(s, api.fps)
    api.status("Timer OCR @ playhead: %r -> %s"
               % (s, "%.3fs" % p[1] if p else "unparsed"),
               "ok" if p else "warn")


def _open(api):
    if not api.path:
        return
    import ocr_dialogs
    dlg = _DLG.get("w")
    if dlg is not None and dlg.win.winfo_exists():
        dlg.win.lift()
        return
    _DLG["w"] = ocr_dialogs.RegionDialog(api, STATE)


def _clear(api):
    STATE["rect"] = None
    STATE["bank"] = None
    api.status("Timer OCR calibration cleared", "info")


def _scan(api):
    if not api.path:
        return
    if STATE["rect"] is None or STATE["bank"] is None:
        api.status("Timer OCR: select the region and calibrate first "
                   "(Timer OCR menu)", "err")
        return
    digits = set("0123456789") & set(STATE["bank"].coverage())
    if len(digits) < 10:
        api.status("Timer OCR: only %d/10 digits trained (%s) - calibrate on "
                   "more frames" % (len(digits),
                                    "".join(sorted(digits)) or "none"), "warn")
    api.status("Timer OCR: reading the overlay on every frame...", "warn")
    path, rect, bank = api.path, STATE["rect"], STATE["bank"]

    def worker():
        res = ocr_core.scan(
            path, rect, bank,
            on_progress=lambda m: api.post(
                lambda m=m: api.status("Timer OCR: %s..." % m, "warn")),
            cancel=api.closing)
        api.post(_done, api, path, res)

    api.run_bg(worker)


def _done(api, path, res):
    if path != api.path or api.closing():
        return
    if not res.get("ok"):
        api.status("Timer OCR: %s" % res.get("error", "failed"), "err")
        return
    api.add_marks("ocr", ocr_core.marks(res))
    ser = res.get("series") or {}
    if ser.get("v"):
        api.add_series("Timer drift (s)", ser["t"], ser["v"])
    warns = sum(1 for e in res["events"] if e["severity"] == "warn")
    slope = res.get("overall_slope")
    title = ("Timer audit - %.0f%% readable, %s, %d warn"
             % (100 * res["read_rate"],
                "slope %.2f%%" % (100 * slope) if slope is not None
                else "no slope", warns))
    api.adv_store("timer_ocr", title, text=ocr_core.render_report(res),
                  data={k: res[k] for k in ("read_rate", "events", "segments",
                                            "overall_slope", "frames", "fps")})
    api.status("Timer OCR: %s" % title, "warn" if warns else "ok")
    api.text_popup("Timer audit - %s" % os.path.basename(path),
                   ocr_core.render_report(res))
