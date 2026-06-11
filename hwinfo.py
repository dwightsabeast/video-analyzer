#!/usr/bin/env python3
"""
hwinfo.py - report which hardware decode / HDR-tonemap paths actually work here.

    python hwinfo.py "C:\\path\\to\\some_hdr_clip.mkv"

Pass an HDR file to test decode of THAT codec plus the tonemap paths. Paste the
output back if a GPU path you expect to work is reported FAIL - the ffmpeg error
line says why.
"""

import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import va_ffmpeg as F
import va_hwaccel as H


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    ff = F.find_ffmpeg()
    print("ffmpeg :", ff or "NOT FOUND")
    print("ffprobe:", F.find_ffprobe() or "NOT FOUND")
    if ff:
        try:
            v = subprocess.run([ff, "-hide_banner", "-version"],
                               capture_output=True, text=True, timeout=15,
                               creationflags=F.CREATIONFLAGS).stdout.splitlines()
            print("version:", v[0] if v else "?")
            if "full" not in (v[0] if v else "") and "essentials" in (v[0] if v else ""):
                print("  NOTE: this is the 'essentials' build - it has NO libplacebo / OpenCL")
                print("        tonemap. For GPU HDR tonemap, install the gyan.dev 'full' build.")
        except Exception:
            pass
    print("hwaccels:", ", ".join(sorted(H.hwaccels())) or "none")
    print("filters :", ", ".join(
        "%s=%s" % (f, "yes" if F.has_filter(f) else "no")
        for f in ("libplacebo", "tonemap_opencl", "tonemap_vaapi", "zscale", "scale_cuda")))

    if not path:
        print("\nDECODE - GPU device creation (pass a file to also test codec decode):")
        for m in H._decode_methods():
            ok, err = H._probe_device(m)
            print("  %-9s device %s%s" % (m, "OK" if ok else "FAIL",
                                          "" if ok else "   -> " + err[:150]))
        print('\nPass an HDR clip to test fully:  python hwinfo.py "C:\\path\\to\\hdr.mkv"')
        return
    if not os.path.isfile(path):
        print("\nFile not found:", path)
        return

    rep = H.diagnostics(path)
    print("\nfile codec:", rep["codec"])
    print("\nDECODE - device present AND can hardware-decode '%s'?" % rep["codec"])
    for d in rep["decode"]:
        dev = "device OK" if d["device_ok"] else "device FAIL"
        dec = "decodes %s OK" % rep["codec"] if d["decode_ok"] else "cannot decode %s" % rep["codec"]
        tail = "" if d["device_ok"] else "   -> " + d["error"][:150]
        print("  %-9s %-12s %s%s" % (d["method"], dev, dec, tail))

    print("\nHDR TONEMAP - tested on %s:" % os.path.basename(path))
    for t in rep["tonemap"]:
        print("  %-16s %s%s" % (t["method"], "OK" if t["ok"] else "FAIL",
                                "" if t["ok"] else "   -> " + t["error"][:150]))

    info = F.probe(path)
    H.reset_cache()
    pl = H.choose_video_pipeline(path, info)
    print("\nis_hdr=%s  %dx%d" % (info["is_hdr"], info["width"], info["height"]))
    print("CHOSEN PIPELINE: %s   (output %dx%d)" % (pl["label"], pl["out_w"], pl["out_h"]))
    print("  ffmpeg flags:", " ".join(pl["pre"]) or "(none)")
    print("  filtergraph :", pl["vf"])


if __name__ == "__main__":
    main()
