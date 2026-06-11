# Overnight Stress Test & Build Report — 2026-06-10 (second overnight run)

Scope requested: stress every component against the three new HDR videos, fix
autonomously, verify every button, improve how the tool is stored/housed, and
expand the audio side (forensic tools, scrubbing like video, findings in
exports).

Test material: Dolby "Patterns of Nature" UHD/24 in three editions —
DV Profile 5 (IPT, iOS), HDR10 + DV 8.1, HLG + DV 8.4 — each with
DD+JOC (E-AC-3 Atmos) 768 kbps 5.1 audio. All three are the same master,
which made cross-codec ground-truthing possible.

---

## 1. Critical fix: DV Profile 5 rendered real content with wrong hues

**Symptom found tonight:** the P5 file's green river valleys and moss rendered
magenta/red through the software IPT path (the exact class of bug the
fireplace fix targeted — it survived on real content).

**Root cause:** the fallback colour matrices were BT.2100 **ICtCp**; real
Dolby P5 streams carry Dolby's own IPT constants (chroma columns differ by up
to ~20x: e.g. Ct coefficient 0.0086 vs Dolby's 0.0975). Last night's selftest
fixture couldn't catch it: it encoded with the same wrong matrices it decoded
with, so the error cancelled.

**Fix:**
- New **`va_rpu.py`** — a pure-Python Dolby Vision RPU bitstream parser
  (NAL 62 → header → mapping → DM payload, following dovi_tool's field
  layout). Recovers the file's true `ycc_to_rgb` / `rgb_to_lms` matrices,
  offsets, `source_min/max_pq`, and per-component reshaping curves
  (polynomial + MMR) on **any** ffmpeg build — previously exact colour needed
  ffprobe ≥ 5.1.
- `va_ipt.dovi_meta()` chain is now: ffprobe side data → direct RPU parse →
  canonical P5 constants (the new fallback; the ICtCp matrix is gone).
- **Validated against ground truth:** decoded the same frame from the P5 and
  the HDR10 editions of the same master; hue-histogram cosine similarity went
  from **0.01 → 0.73** and visual inspection shows green forests/blue water.
  (Residual difference is the labelled-approximate tonemap, not hue.)
- Your file's RPU parsed bit-exact to the canonical Dolby values, including
  the 4000-nit mastering levels (`source_min/max_pq` 62/3696).

## 2. Major audio bug: every audio pass decoded the UHD video too

`loudness/astats/silence_segments/correlation` (va_audio) and
`va_metrics.loudness` ran ffmpeg without `-vn` — on these files each "audio"
pass software-decoded 3840x2160 HEVC at ~2 fps. That is why audio analysis
could "time out" on real UHD material. All audio passes now decode only the
audio stream; a selftest guard greps for the flags so it cannot regress.
Measured: full E-AC-3 track loudness pass **0.3 s** (was: >43 s, timed out).

## 3. New: audio forensic battery (`va_audio.forensic_battery`)

One 16 kHz mono decode feeds sample-level scans; short extra decodes cover
channels/bandwidth:

| Scan | What it flags |
|---|---|
| clipping_scan | runs of full-scale samples (hard clipping), % of samples at ceiling |
| dropout_scan | ≥20 ms runs of exact zeros mid-stream; click/pop transients (robust MAD outliers) |
| splice_scan_audio | background **noise-floor steps** + **spectral-rolloff jumps** that persist away from silence — classic edit points; candidates at picture cuts are demoted to "normal sound editing" |
| channel_check | silent channels, fake stereo (mono fold), phase-inverted pairs |
| bandwidth_analysis | spectrum cutoff vs Nyquist — reveals earlier lossy generations / resampled speed changes |
| loudness_steps | sustained momentary-loudness jumps (level edits) |

Wired everywhere the video battery already was:
- **`va_forensics.forensics_report`** runs it as a stage; audio findings merge
  into the unified findings list, the dashed timeline marks (teal), `n`/`p`
  issue navigation, `analyze.py --forensics` text/JSON/HTML reports.
- **QC**: `integrity` profile gained an `audio_integrity` check.
- **Exports**: HTML timeline gets audio marks + legend; JSON carries the full
  battery; the GUI Advanced-results cache includes `audio_forensics`.

On your pristine Dolby demo files the battery correctly reports *no concerns*
(after tuning: sound-design transitions at picture cuts are not "splices");
on a deliberately degraded proxy it correctly fingerprints the low-bitrate
AAC ancestor (band ends at 5.3 kHz = 24% of Nyquist) and the silent LFE.

## 4. New: audio scrubbing + audible playback in the GUI

- **Waveform strip** (Audio tab): min/max envelope + RMS body of the whole
  track. Drag to scrub the video (frame-accurate, same path as the timeline),
  mouse-wheel zoom centred on the cursor down to 0.5% of the file,
  double-click to fit. Playhead stays in sync both directions. Silence
  segments shade after Analyze; audio forensic findings draw as ticks.
- **🔊 Audio follow** (transport bar): real audio during playback via the
  bundled `ffplay` (no new dependencies). The video keeps the clock; audio is
  restarted on seeks and drift-corrected every chart tick (>0.4 s → resync).
  Preference persists across sessions.
- **Paused-scrub audition**: each settled scrub plays ~1/3 s of audio at the
  playhead (Windows winsound; extraction off-thread).

## 5. Storage / housing

- All helper binaries (~700 MB: ffmpeg/ffprobe/ffplay, dovi_tool,
  hdr10plus_tool, mediainfo, mp4dump/mp4info, c2patool, mkvtoolnix/) moved
  into **`tools/`**. Discovery order: `tools/` → script folder → `mkvtoolnix/`
  → `tools/mkvtoolnix/` → standard installs → PATH. Downloads (GUI + CLI)
  now land in `tools/`. A **"Tidy into tools/"** button migrates strays.
- The stuck `.git` remnant is still locked from this side (Windows holds the
  permissions) — please delete it manually when convenient.

## 6. Button audit (all 36 handlers + 16 bindings)

Static + behavioural audit: every `command=`/`bind` target exists; no
duplicate menu labels; every button that is ever disabled is re-enabled by
some path; export menu gating, Setup-tab download/refresh cache-reset paths,
and the Advanced-cache keys all check out. New controls (🔊, Audio
forensics..., strip interactions, Tidy) follow the same enable/disable
discipline and are covered by new selftest greps. AudioStrip geometry/zoom
math is unit-tested headless.

## 7. Other fixes along the way

- `va_dynhdr`: the "no HDR10 static fallback (MDCV/CLL)" warning now only
  fires for DV streams that actually declare HDR10 compatibility (8.1) — HLG
  (8.4) and SDR (8.2) bases don't carry MDCV by design. Your P8.4 file no
  longer warns; P8.1 (which has proper MDCV) stays clean.
- `va_audio.loudness_steps` reports the settled step size rather than the
  first-crossing midpoint.
- `_open_done` now clears stale per-file audio state (`self.audio`,
  waveform, marks) so a new file can't inherit the previous file's silence
  list in exports.
- numpy-2-safe: scan dicts use native floats (JSON-export safe on your
  Windows numpy).

## 8. Verification

- `selftest.py` gained `--only=<suite,...>` (faster targeted runs for you
  too: e.g. `python selftest.py --quick --only=audio_forensics`).
- New suites: `chk_audio_forensics` (detectors on synthetic defects, battery
  contract, report/QC/export integration, `-vn` guard), `chk_audio_gui`
  (stubbed-Tk import, strip math, follower grace without ffplay, wiring
  greps), `chk_rpu_parse` (bit-exact synthetic RPU built with a local
  bit-writer; canonical-constant regression guard; dovi_meta fallback chain),
  `chk_tools_dir`, `chk_dynhdr_compat_flag`.
- Full sandbox regression: **all suites green** (520 line-checks across six
  chunked runs incl. ~60 new ones; the sandbox 45 s per-call cap forces the
  chunking — Windows runs it in one go).
- Real-file battery on your three videos: probe/classify, DV config parse,
  native scopes (PQ + HLG + IPT), nits map (3461-nit peak measured on P8.1),
  MaxCLL measured-vs-declared, banding, analyze_pass single-decode merge,
  forensics + audio battery, verify, CLI `analyze.py --forensics`, HTML/JSON
  exports — all exercised directly on the uploads.

## Needs your eyes on Windows (can't be clicked from the sandbox)

1. **DV P5 colour**: open the P5 file — vegetation should be green both in the
   preview (libplacebo path) and if you force the software path; the status
   bar should say *rendered via DV RPU* or *software IPT decode*.
2. **Audio follow**: 🔊 on → Play → sound in sync; seek while playing → audio
   follows; pause → scrub the waveform strip → audition snippets.
3. **Audio forensics...** button on a file you've edited/clipped — expect
   splice/clipping marks on the strip and in `n`/`p`.
4. **Tools dialog**: statuses should all read *installed* (now from `tools/`);
   try one Re-download to confirm it lands in `tools/`.
5. Delete the stuck `.git` folder manually.
