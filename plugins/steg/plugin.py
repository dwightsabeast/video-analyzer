"""Hidden-data / steganography plugin.

Adds 'Hidden data v' to the toolbar (full scan + LSB-plane overlay), the
'steg' QC profile, and the analyze.py pass writing <name>.steg.txt. The
detectors live in steg_container / steg_bitplane / steg_codec, orchestrated
by steg_scan."""

from __future__ import annotations

import os

import steg_scan

STATE = {"last": None, "overlay": False}


def register(api):
    import va_qc
    va_qc.register_profile("steg", steg_scan.qc_checks())
    m = api.add_toolbar_menu("Hidden data ▾")
    api.add_command(m, "Scan for hidden data", lambda: _scan(api, force=False))
    api.add_command(m, "Scan (force LSB even if lossy)",
                    lambda: _scan(api, force=True))
    api.add_command(m, "QC check (steg profile)...",
                    lambda: api.run_qc("steg"))
    api.add_command(m, "Show/hide LSB-plane overlay",
                    lambda: _toggle_overlay(api))
    api.on_open(lambda _p: STATE.update(last=None))


def _scan(api, force=False):
    if not api.path:
        return
    api.status("Hidden-data scan: container / LSB / codec...", "warn")
    path = api.path

    def worker():
        rep = steg_scan.scan(
            path, force_lsb=force,
            on_progress=lambda msg: api.post(
                lambda msg=msg: api.status("Hidden-data: %s..." % msg, "warn")),
            cancel=api.closing)
        api.post(_done, api, path, rep)

    api.run_bg(worker)


def _done(api, path, rep):
    if path != api.path or api.closing():
        return
    if not rep.get("ok"):
        api.status("Hidden-data scan: %s" % rep.get("error", "failed"), "err")
        return
    STATE["last"] = rep
    api.add_marks("steg", steg_scan.marks(rep))
    res = rep.get("residual") or {}
    if res.get("v"):
        api.add_series("Hidden: coded-size residual (MAD)", res["t"], res["v"])
    warns = rep.get("warn", 0)
    title = "Hidden-data: %d finding(s), %d warn" % (len(rep["findings"]), warns)
    api.adv_store("steg", title, text=steg_scan.render_report(rep),
                  data={"stats": rep["stats"], "findings": rep["findings"],
                        "lsb": rep.get("lsb"), "gated": rep.get("gated"),
                        "codec": {k: v for k, v in (rep.get("codec") or {}).items()
                                  if k != "sizes"}})
    api.status(title, "warn" if warns else "ok")
    api.text_popup("Hidden-data scan - %s" % os.path.basename(path),
                   steg_scan.render_report(rep))


def _toggle_overlay(api):
    STATE["overlay"] = not STATE["overlay"]
    if STATE["overlay"]:
        api.add_overlay("steg_lsb", _draw_lsb)
        api.status("Hidden data: LSB-plane overlay ON (low bit of each pixel)",
                   "ok")
    else:
        api.remove_overlay("steg_lsb")
        api.status("Hidden data: LSB-plane overlay off", "info")


def _draw_lsb(frame, _idx):
    """Replace the preview with the amplified LSB plane: structured patterns
    here are the tell-tale of pixel embedding; clean content looks like noise."""
    try:
        import numpy as np
        g = np.asarray(frame)
        luma = (g.mean(axis=2) if g.ndim == 3 else g).astype(np.uint8)
        plane = ((luma & 1) * 255).astype(np.uint8)
        return np.stack([plane, plane, plane], axis=2) if g.ndim == 3 else plane
    except Exception:  # noqa: BLE001
        return None


def register_headless(api):
    api.register_qc_profile("steg", steg_scan.qc_checks())
    api.register_profile_pass(
        "steg",
        run=lambda path, forensics=None, cadence=None: steg_scan.scan(path),
        echo="hidden-data scan (container / LSB / codec)",
        ctx=lambda rep: rep,
        json_extract=lambda rep: {"stats": rep.get("stats"),
                                  "findings": rep.get("findings"),
                                  "lsb": rep.get("lsb"),
                                  "gated": rep.get("gated")},
        render=steg_scan.render_report,
        suffix=".steg.txt",
        needs_forensics=False,
        ctx_key="steg")
