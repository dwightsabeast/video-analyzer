#!/usr/bin/env python3
"""pack_plugins - publish the plugins/ folders so other installs can pull them.

Zips every plugins/<name>/ into dist/<name>-<version>.zip (reproducibly, so
unchanged plugins keep the same sha256) and writes dist/registry.json in the
format the in-app Plugins manager consumes (plugins/registry.example.json).

    python pack_plugins.py --base-url https://github.com/ORG/REPO/releases/download/TAG
    python pack_plugins.py --only steg,ocr --out dist

Publish flow (GitHub):
    1. run this script with the release tag you are about to create
    2. create that release and upload dist/*.zip as assets
    3. commit dist/registry.json to the repo root as registry.json; users
       point the Plugins manager at its stable raw URL:
       https://raw.githubusercontent.com/ORG/REPO/main/registry.json

Stdlib only; no third-party imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile

EXCLUDE_DIRS = {"__pycache__", ".git"}
EXCLUDE_EXT = {".pyc", ".pyo", ".bak"}
EXCLUDE_FILES = {".DS_Store", "Thumbs.db"}
ZIP_DATE = (2020, 1, 1, 0, 0, 0)   # fixed timestamp => deterministic zips


def _manifest(plugdir: str) -> dict:
    with open(os.path.join(plugdir, "plugin.json"), "r", encoding="utf-8") as fh:
        m = json.load(fh)
    if not isinstance(m, dict) or not m.get("name"):
        raise ValueError("plugin.json must be an object with a 'name'")
    m.setdefault("title", m["name"])
    m.setdefault("version", "0")
    m.setdefault("description", "")
    return m


def _files(plugdir: str):
    """Sorted relative paths to include (sorted => deterministic zips)."""
    out = []
    for root, dirs, files in os.walk(plugdir):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS)
        for fn in sorted(files):
            if fn in EXCLUDE_FILES or os.path.splitext(fn)[1] in EXCLUDE_EXT:
                continue
            full = os.path.join(root, fn)
            out.append(os.path.relpath(full, plugdir))
    return out


def pack(plugdir: str, name: str, version: str, out_dir: str) -> str:
    """Zip plugdir as <name>/<file...> into out_dir; returns the zip path."""
    zpath = os.path.join(out_dir, "%s-%s.zip" % (name, version))
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in _files(plugdir):
            arc = name + "/" + rel.replace(os.sep, "/")
            zi = zipfile.ZipInfo(arc, date_time=ZIP_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            with open(os.path.join(plugdir, rel), "rb") as fh:
                zf.writestr(zi, fh.read())
    return zpath


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="",
                    help="URL prefix for zip_url entries, e.g. "
                         "https://github.com/ORG/REPO/releases/download/TAG "
                         "(blank leaves a FILL-ME placeholder)")
    ap.add_argument("--only", default="",
                    help="comma-separated plugin names (default: all)")
    ap.add_argument("--out", default=os.path.join(here, "dist"),
                    help="output directory (default: dist/)")
    args = ap.parse_args(argv)

    plugroot = os.path.join(here, "plugins")
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    base = args.base_url.rstrip("/")
    os.makedirs(args.out, exist_ok=True)

    entries, bad = [], []
    for fn in sorted(os.listdir(plugroot)):
        d = os.path.join(plugroot, fn)
        if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, "plugin.json")):
            continue
        if only and fn not in only:
            continue
        try:
            m = _manifest(d)
        except Exception as exc:  # noqa: BLE001 - report and continue
            bad.append("%s: %s" % (fn, exc))
            continue
        zpath = pack(d, m["name"], m["version"], args.out)
        zname = os.path.basename(zpath)
        entries.append({
            "name": m["name"],
            "title": m["title"],
            "version": m["version"],
            "description": m["description"],
            "zip_url": (base + "/" + zname) if base
                       else "https://FILL-ME/" + zname,
            "sha256": sha256(zpath),
        })
        print("  %-12s v%-8s %s  (%.1f KB)" % (
            m["name"], m["version"], zname, os.path.getsize(zpath) / 1024.0))

    reg = os.path.join(args.out, "registry.json")
    with open(reg, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)
        fh.write("\n")
    print("wrote %s  (%d plugin(s))" % (reg, len(entries)))
    if not base:
        print("note: no --base-url given; edit the FILL-ME zip_url values "
              "before publishing")
    for b in bad:
        print("skipped %s" % b, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
