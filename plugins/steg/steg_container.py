"""Container-level hidden-data forensics.

A video file can carry a passenger while still playing normally. The reliable
tells, all structural:

  * appended data after the last legitimate box/element (players stop at the
    end of the media and ignore the rest);
  * polyglots - an embedded ZIP/RAR/7z/PDF/PNG/... whose magic signature sits
    at an offset it has no business being at (ZIP is read from the end, so a
    video + appended archive is a single valid file of both types);
  * content-bearing padding atoms (free/skip/wide in MP4, Void in MKV) that
    hold something other than zeros;
  * mdat bytes no sample table references (a gap between the frames the index
    actually points at).

Plus a whole-file entropy curve for context: a high-entropy *trailing* or
*padding* region looks like a compressed/encrypted payload, whereas the main
encoded video is expected to be high-entropy and is not itself suspicious.

Stdlib + numpy only. Pure helpers (entropy_series, classify_region) are unit-
tested directly; scan() walks a real file."""

from __future__ import annotations

import os
import struct

import numpy as np

# magic -> (label, min trailing/padding severity bump). Signatures chosen to be
# specific enough that a chance hit in compressed video is unlikely.
_MAGIC = [
    (b"PK\x03\x04", "ZIP archive"),
    (b"PK\x05\x06", "ZIP end-of-central-directory"),
    (b"Rar!\x1a\x07", "RAR archive"),
    (b"7z\xbc\xaf\x27\x1c", "7-Zip archive"),
    (b"%PDF-", "PDF document"),
    (b"\x89PNG\r\n\x1a\n", "PNG image"),
    (b"GIF89a", "GIF image"),
    (b"GIF87a", "GIF image"),
    (b"\x1f\x8b\x08", "gzip stream"),
    (b"BZh", "bzip2 stream"),
    (b"\xfd7zXZ\x00", "xz stream"),
    (b"\x7fELF", "ELF executable"),
    (b"SQLite format 3\x00", "SQLite database"),
    (b"-----BEGIN ", "PEM key/cert block"),
    (b"\x50\x4b\x07\x08", "ZIP data descriptor"),
]
_MAX_HITS = 40
_MP4_BRANDS = (b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"mfra",
               b"moof", b"meta", b"uuid", b"pdin", b"sidx", b"styp")


def shannon(buf: bytes) -> float:
    """Shannon entropy (bits/byte, 0..8) of a byte string."""
    if not buf:
        return 0.0
    counts = np.bincount(np.frombuffer(buf, dtype=np.uint8), minlength=256).astype(np.float64)
    p = counts[counts > 0] / len(buf)
    return float(-(p * np.log2(p)).sum())


def entropy_series(path, window=65536, step=None):
    """(offsets[], entropy[]) sliding-window entropy across the whole file."""
    size = os.path.getsize(path)
    step = step or window
    offs, ent = [], []
    with open(path, "rb") as fh:
        pos = 0
        while pos < size:
            fh.seek(pos)
            buf = fh.read(window)
            if not buf:
                break
            offs.append(pos)
            ent.append(shannon(buf))
            pos += step
    return offs, ent


def _read_chunks(fh, total, chunk=8 << 20, overlap=64):
    pos = 0
    tail = b""
    while pos < total:
        fh.seek(pos)
        buf = fh.read(chunk)
        if not buf:
            break
        yield pos - len(tail), tail + buf
        tail = buf[-overlap:]
        pos += len(buf)


def magic_scan(path, skip_ranges=None) -> list:
    """Embedded-format signatures at any offset. skip_ranges = [(lo,hi)] of
    offsets to ignore (e.g. the file's own ftyp at 0). Returns
    [{offset, label, signature}]."""
    skip_ranges = skip_ranges or []
    hits = []
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        for base, buf in _read_chunks(fh, size):
            for sig, label in _MAGIC:
                start = 0
                while True:
                    i = buf.find(sig, start)
                    if i < 0:
                        break
                    off = base + i
                    start = i + 1
                    if off < 0:
                        continue
                    if any(lo <= off < hi for lo, hi in skip_ranges):
                        continue
                    hits.append({"offset": int(off), "label": label,
                                 "signature": sig.hex()})
                    if len(hits) >= _MAX_HITS:
                        return hits
    return hits


# --- MP4 -----------------------------------------------------------------------

def _top_boxes(path):
    """[(type, offset, size)] top-level MP4 boxes via va_forensics._walk_mp4."""
    import va_forensics
    with open(path, "rb") as fh:
        boxes = va_forensics._walk_mp4(fh, 0, os.path.getsize(path))
    return [(bt, off, sz) for depth, bt, off, sz, _pl in boxes if depth == 0]


def _mp4_sample_extent(path):
    """(referenced_bytes, mdat_range) from stsz+stco/co64. Best-effort; returns
    (None, None) if the tables can't be read."""
    import va_forensics
    try:
        with open(path, "rb") as fh:
            boxes = va_forensics._walk_mp4(fh, 0, os.path.getsize(path))
            mdat = next(((off, sz) for d, bt, off, sz, _p in boxes if bt == b"mdat"), None)
            if not mdat:
                return None, None
            sizes_total = 0
            for d, bt, off, sz, _p in boxes:
                if bt == b"stsz":
                    fh.seek(off + 12)
                    ver_sample, count = struct.unpack(">II", fh.read(8))
                    if ver_sample:
                        sizes_total += ver_sample * count   # uniform sample size
                    else:
                        data = fh.read(min(count, 2_000_000) * 4)
                        arr = np.frombuffer(data[: (len(data) // 4) * 4], dtype=">u4")
                        sizes_total += int(arr.sum())
            return sizes_total, mdat
    except Exception:  # noqa: BLE001 - any parse surprise => "unknown"
        return None, None


def _mp4_report(path, findings, stats):
    boxes = _top_boxes(path)
    if not boxes:
        return None       # not a box file we understand
    size = os.path.getsize(path)
    last_end = max((off + sz for _bt, off, sz in boxes), default=0)
    stats["last_box_end"] = last_end
    skip = []
    ftyp = next(((off, sz) for bt, off, sz in boxes if bt == b"ftyp"), None)
    if ftyp:
        skip.append((ftyp[0], ftyp[0] + ftyp[1]))
    if last_end < size:
        trailing = size - last_end
        with open(path, "rb") as fh:
            fh.seek(last_end)
            head = fh.read(min(trailing, 65536))
        ent = shannon(head)
        findings.append({"kind": "appended_data", "severity": "warn",
                         "offset": last_end, "size": trailing,
                         "text": "%d bytes after the last MP4 box (entropy %.2f "
                                 "bits/byte - %s)" % (
                                     trailing, ent,
                                     "looks compressed/encrypted" if ent > 7.0
                                     else "low-entropy, may be padding/text")})
    for bt, off, sz in boxes:
        if bt in (b"free", b"skip", b"wide") and sz > 16:
            with open(path, "rb") as fh:
                fh.seek(off + 8)
                body = fh.read(min(sz - 8, 65536))
            if body and body.count(0) < len(body) * 0.98:
                findings.append({"kind": "padding_atom", "severity": "warn",
                                 "offset": off, "size": sz,
                                 "text": "'%s' atom carries %d non-zero bytes "
                                         "(entropy %.2f) - padding atoms are "
                                         "normally zeros"
                                         % (bt.decode("latin1"), sz - 8,
                                            shannon(body))})
    ref, mdat = _mp4_sample_extent(path)
    if ref is not None and mdat is not None:
        _off, msz = mdat
        gap = (msz - 8) - ref
        stats["mdat_size"] = msz
        stats["mdat_referenced"] = ref
        if gap > max(4096, 0.02 * msz):
            findings.append({"kind": "mdat_gap", "severity": "warn",
                             "offset": mdat[0], "size": gap,
                             "text": "%d bytes inside mdat are not referenced by "
                                     "the sample table (%.1f%% of the media data)"
                                     % (gap, 100.0 * gap / max(1, msz))})
    return skip


# --- MKV / WebM ----------------------------------------------------------------

def _read_vint(fh, keep_marker):
    b0 = fh.read(1)
    if not b0:
        return None, 0
    first = b0[0]
    if first == 0:
        return None, 1
    length = 8 - first.bit_length() + 1
    rest = fh.read(length - 1)
    if len(rest) < length - 1:
        return None, length
    raw = bytes([first]) + rest
    if keep_marker:
        return int.from_bytes(raw, "big"), length
    val = first & ((1 << (8 - length)) - 1)
    for c in rest:
        val = (val << 8) | c
    # all-ones => unknown size
    if val == (1 << (7 * length)) - 1:
        return None, length
    return val, length


def _mkv_report(path, findings, stats):
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != b"\x1a\x45\xdf\xa3":
            return None
        fh.seek(0)
        last_end = 0
        skip = []
        # top level: EBML header, then Segment
        while fh.tell() < size:
            eid, _l1 = _read_vint(fh, keep_marker=True)
            esz, _l2 = _read_vint(fh, keep_marker=False)
            if eid is None:
                break
            data_start = fh.tell()
            if esz is None:                      # unknown-size (streamed) element
                stats["mkv_unknown_size"] = True
                last_end = size                  # can't bound structurally
                break
            end = data_start + esz
            last_end = end
            if eid == 0x18538067:                # Segment: scan children for Void
                child = data_start
                while child < end and child < size:
                    fh.seek(child)
                    cid, _a = _read_vint(fh, keep_marker=True)
                    csz, _b = _read_vint(fh, keep_marker=False)
                    if cid is None or csz is None:
                        break
                    cdata = fh.tell()
                    if cid == 0xEC and csz > 16:   # Void
                        body = fh.read(min(csz, 65536))
                        if body and body.count(0) < len(body) * 0.98:
                            findings.append({"kind": "padding_atom",
                                             "severity": "warn", "offset": child,
                                             "size": csz,
                                             "text": "Void element carries %d "
                                                     "non-zero bytes (entropy "
                                                     "%.2f)" % (csz, shannon(body))})
                    child = cdata + csz
            fh.seek(end)
        stats["last_box_end"] = last_end
        if last_end < size:
            trailing = size - last_end
            fh.seek(last_end)
            head = fh.read(min(trailing, 65536))
            ent = shannon(head)
            findings.append({"kind": "appended_data", "severity": "warn",
                             "offset": last_end, "size": trailing,
                             "text": "%d bytes after the last EBML element "
                                     "(entropy %.2f bits/byte - %s)"
                                     % (trailing, ent,
                                        "looks compressed/encrypted" if ent > 7.0
                                        else "low-entropy")})
        return skip


def scan(path, on_progress=None, cancel=None) -> dict:
    """Container hidden-data forensics for one file."""
    say = on_progress or (lambda m: None)
    size = os.path.getsize(path)
    findings, stats = [], {"file_size": size}
    say("walking container")
    skip = None
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
        if head[4:8] == b"ftyp" or head[4:8] in _MP4_BRANDS:
            stats["container"] = "mp4"
            skip = _mp4_report(path, findings, stats)
        elif head[:4] == b"\x1a\x45\xdf\xa3":
            stats["container"] = "mkv"
            skip = _mkv_report(path, findings, stats)
        else:
            stats["container"] = "unknown"
    except Exception as exc:  # noqa: BLE001
        stats["walk_error"] = repr(exc)
    say("scanning for embedded files")
    for h in magic_scan(path, skip_ranges=skip or []):
        last_end = stats.get("last_box_end", 0)
        where = ("trailing" if last_end and h["offset"] >= last_end
                 else "within media/header")
        sev = "warn" if where == "trailing" or h["label"].startswith(
            ("ZIP", "RAR", "7-Zip", "PDF", "SQLite")) else "info"
        findings.append({"kind": "embedded_file", "severity": sev,
                         "offset": h["offset"], "size": 0,
                         "text": "%s signature at offset %d (%s)"
                                 % (h["label"], h["offset"], where)})
    say("entropy sweep")
    offs, ent = entropy_series(path)
    stats["entropy_mean"] = round(float(np.mean(ent)), 3) if ent else 0.0
    stats["entropy_max"] = round(float(np.max(ent)), 3) if ent else 0.0
    warns = sum(1 for f in findings if f["severity"] == "warn")
    return {"ok": True, "findings": findings, "stats": stats,
            "entropy": {"offset": offs, "v": ent}, "warn": warns}
