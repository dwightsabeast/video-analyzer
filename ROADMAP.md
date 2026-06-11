# Video Analyzer — Roadmap to a one-stop video data-science workbench

> Status: phases 1-4 IMPLEMENTED (2026-06-09). Engine modules va_ffmpeg / va_probe /
> va_metrics / va_scopes / va_quality / va_export plus the video-analyzer.py GUI now
> cover decode, per-frame metrics, scopes, events, loudness, quality, gamut and export.
> Run `python selftest.py` to verify the engine headlessly (23 checks). Sections below
> remain the design reference; shell commands are templates to confirm against ffmpeg.
>
> 2026-06-10 — FORENSICS EXPANSION shipped: va_forensics grew from 3 signals to a
> full battery — corroborated splice scan (size/PTS/IDR-cadence/audio signals scored
> per timestamp), MP4/MKV container edit-history walk, encoder fingerprinting
> (x264/x265 SEI + Lavf/camera byte-scan vs tags), metadata coherence audit, ELA +
> noise-consistency maps, repeated-sequence/loop + periodicity detection, ENF
> (mains-hum) continuity, SHA-256, C2PA ingredient chains, and a unified
> forensics_report() with merged findings. Wired everywhere: GUI Advanced ▾ (report
> with dashed timeline markers + n/p nav, ELA/noise maps, ENF plot), CLI
> `--forensics` (+ `<name>.forensics.txt`), QC profile `integrity`, HTML/JSON
> export sections. selftest: 148 checks (forensics suite included).

## 1. Vision

Turn the current real-time monitor into a workbench that treats a video file as a
**dataset**: every signal metric as a time-series across all frames, aggregate
distributions, automatic anomaly/event detection, reference-based quality scoring,
audio loudness, and exportable data + reports — all from one window.

Three standing project goals drive every decision:

1. **Usability** — one window, sensible defaults, nothing hidden behind a manual.
2. **How information is conveyed** — charts and a scrubbable timeline over raw
   numbers; flag problems, don't just display values.
3. **Dependency reliance** — keep Python deps at `opencv-python` + `numpy`; push
   heavy lifting onto the bundled `ffmpeg`/`ffprobe` binary.

## 2. Where it stands today

- Single-file Tkinter GUI (`video-analyzer.py`, ~1190 lines).
- Playback with seek / play / pause and frame nudging.
- HDR & stream metadata via bundled `ffprobe` (HDR10, HDR10+, HLG, Dolby Vision;
  primaries / transfer / matrix; MDCV / CLL side data).
- Live hue x saturation vectorscope (samples 6000 pixels) + Rec.709/2020-weighted
  luma histogram with mean / 99th-percentile markers.
- Charts rendered straight into RGB buffers with cv2/numpy (no matplotlib).
- Known limitation: OpenCV decode fails on some HEVC / 10-bit / HDR files; charts
  reflect an 8-bit decode, not the full HDR signal.

## 3. Decisions locked in

- **ffmpeg is the analysis engine.** `ffprobe` is part of ffmpeg; its filters
  (`signalstats`, `cropdetect`, `psnr`, `ssim`, `libvmaf`, `ebur128`,
  `blackdetect`, `freezedetect`, `select=scene`) already produce nearly every
  metric we want. We parse their output rather than reimplementing them.
- **Bundle full ffmpeg + ffprobe** (static build) beside the script, replacing the
  standalone 213 MB `ffprobe.exe`. A combined static build is typically
  ~80-120 MB, so we gain reliable decoding *and* shrink the footprint. The
  existing auto-detect (bundled first, then PATH) stays.
- **Python deps stay at `opencv-python` + `numpy`.** No matplotlib, no Pillow, no
  pandas for core features. cv2/numpy keep doing resize, color conversion, and
  chart drawing.

## 4. Architecture direction

- **Decode through ffmpeg, not `cv2.VideoCapture`.** Spawn
  `ffmpeg -i FILE -f rawvideo -pix_fmt bgr24 -` and read frames off stdout into
  numpy. This fixes HEVC / 10-bit / HDR decode and gives one reliable path. For
  HDR previews, tonemap in the same pass (see Phase 1).
- **Analysis runs as a background pass.** On load, kick off an ffprobe
  `signalstats` job that streams per-frame metrics into in-memory arrays (one row
  per frame), feeding the timeline. Long files analyze progressively.
- **Keep Tk on the main thread.** The existing rule stays: worker threads do pure
  compute and marshal widget updates back via `root.after`. Widget sizes stay
  cached (`_update_sizes`) so workers never call Tk.
- **One data model.** All metrics land in a single per-frame table (frame index,
  pts, Y/U/V min-avg-max, sat, hue, diffs, broadcast-range %, loudness, scene
  score, ...) that drives charts, the timeline, export, and reports.

## 5. Competitive landscape

| Camp | Tools | What they nail | What they miss |
| --- | --- | --- | --- |
| Colorist scopes | Nobe OmniScope, DaVinci Resolve / Premiere Lumetri | Real-time waveform, parade, vectorscope, CIE, false color | Whole-file analysis, time-series, export |
| Preservation / QC | QCTools (open), Telestream & Interra Baton (paid) | Per-frame metric graphs, anomaly flags, export | HDR / wide-gamut depth, modern scopes |
| Reference quality | FFMetrics, MSU VQMT, Netflix VMAF | PSNR / SSIM / VMAF vs a master | Almost everything else |
| Codec / raw internals | YUView, Elecard StreamEye | YUV planes, motion vectors, macroblocks | Friendly UX, color / QC analytics |

**Closest target: QCTools.** It is the open-source incarnation of "data science of
a video," built on the same ffmpeg filters we're adopting. We aim to match its
analytical depth while beating it on HDR/color, modern scopes, and UX.

### Capability matrix

| Capability | This tool now | QCTools | Colorist scopes | VMAF tools | Target |
| --- | :-: | :-: | :-: | :-: | :-: |
| Reliable decode (HEVC / 10-bit / HDR) | ~ | Y | Y | Y | Y |
| Live scopes (waveform / parade / CIE) | ~ | ~ | Y | - | Y |
| HDR & color metadata | Y | ~ | Y | - | Y |
| Per-frame metric time-series | - | Y | - | ~ | Y |
| Anomaly & event detection | - | Y | - | - | Y |
| Reference quality (PSNR / SSIM / VMAF) | - | ~ | - | Y | Y |
| Audio loudness (EBU R128) | - | Y | ~ | - | Y |
| Gamut volume (CIE / % Rec.2020) | - | - | Y | - | Y |
| Data export (CSV / JSON) | - | Y | - | ~ | Y |
| Reports (PDF / HTML) | - | ~ | - | ~ | Y |

Legend: `Y` = full, `~` = partial, `-` = absent.

## 6. Phased build plan

Each phase ends with something usable. Commands are reference templates — confirm
flags against the bundled ffmpeg build. On Windows, `movie=`/`amovie=` source
paths may need their `\` and `:` escaped.

### Phase 1 — Foundation (the leap to "data science")

Goal: reliable decode + a per-frame metrics engine + a scrubbable timeline + export.

- **ffmpeg decode pipeline.** Replace `cv2.VideoCapture` with an ffmpeg stdout
  pipe. Single frame on seek:
  `ffmpeg -ss {t} -i {file} -f rawvideo -pix_fmt bgr24 -vframes 1 -`;
  a long-lived pipe for playback. HDR->SDR preview tonemap (reference):
  `-vf zscale=t=linear:npl=100,tonemap=hable,zscale=p=bt709:t=bt709:m=bt709:r=tv,format=bgr24`.
- **Metrics engine.** Background ffprobe pass:
  `ffprobe -f lavfi -i "movie={file},signalstats" -show_entries frame=pkt_pts_time:frame_tags=lavfi.signalstats.YMIN,lavfi.signalstats.YAVG,lavfi.signalstats.YMAX,lavfi.signalstats.YDIF,lavfi.signalstats.SATAVG,lavfi.signalstats.SATMAX,lavfi.signalstats.HUEAVG,lavfi.signalstats.TOUT,lavfi.signalstats.VREP,lavfi.signalstats.BRNG -of csv`
  Parse rows into numpy arrays in the shared data model.
- **Timeline graph.** A full-width strip under the video plotting selectable
  metrics across the whole file, with a playhead synced to the seek bar. Click to
  seek. This is the centerpiece of goal #2.
- **Export.** Write the per-frame table to CSV and JSON. (Parquet optional later;
  avoid adding pandas just for this.)

Acceptance: opens a 10-bit HEVC HDR file, plays it, shows a YAVG/SAT timeline that
scrubs with playback, and exports a CSV whose row count matches the frame count.

### Phase 2 — Scopes, audio, anomalies

Goal: match colorist scopes and start flagging problems, not just showing numbers.

- **Scopes (numpy-rendered, same approach as today):** luma waveform, RGB parade,
  false color, plus the existing vectorscope. Reference ffmpeg equivalents for
  validation: `waveform`, `vectorscope`, `histogram`.
- **Audio loudness (EBU R128).** Summary:
  `ffmpeg -i {file} -af ebur128=peak=true -f null -` (parse Integrated I, LRA, true
  peak). Time-series:
  `ffprobe -f lavfi -i "amovie={file},ebur128=metadata=1" -show_entries frame_tags=lavfi.r128.M,lavfi.r128.S -of csv`.
- **Event detection.**
  - Black frames: `ffmpeg -i {file} -vf blackdetect=d=0.1:pic_th=0.98 -f null -`.
  - Freeze: `freezedetect=n=-60dB:d=2`.
  - Scene cuts: `ffprobe -f lavfi -i "movie={file},select=gt(scene\,0.4)" -show_entries frame=pkt_pts_time -of csv`.
  - Broadcast-range / temporal outliers: from signalstats `BRNG` / `TOUT` (Phase 1).
  - PSE flash risk: derive from frame-to-frame luma deltas (`YDIF`) and the area of change.
  Surface events as markers on the timeline + a jump-to-issue list.

Acceptance: timeline shows clickable markers for black / freeze / scene-cut, and an
audio panel reports integrated LUFS and true peak.

### Phase 3 — Reference quality & gamut

Goal: encoding / QC workflows and color-volume analysis.

- **Compare two files (master vs encode).** Run together:
  `ffmpeg -i {distorted} -i {reference} -lavfi "[0:v][1:v]libvmaf=log_path=vmaf.json:log_fmt=json" -f null -`
  for VMAF; the `psnr` and `ssim` filters for those metrics. (Confirm input order:
  libvmaf treats the first labeled input as the distorted one — validate per build.)
  Plot per-frame scores on the timeline; show mean / harmonic-mean VMAF.
- **Gamut volume.** Convert sampled pixels to CIE xy, draw the 1931 chromaticity
  diagram with Rec.709 / P3 / 2020 triangles, and report the percent of Rec.2020
  area the content actually occupies. Pure numpy math; no new deps.
- **Reports.** Export a self-contained HTML report (charts as inline PNGs + summary
  tables). PDF only later if needed — avoid a heavy PDF dependency.

Acceptance: pick two files, get a VMAF / PSNR / SSIM timeline + summary, and export
an HTML report that opens standalone.

### Phase 4 — Advanced (stretch)

- Motion: optical-flow magnitude over time (cv2); camera-vs-subject motion.
- Codec internals: per-frame size and I/P/B type from `ffprobe -show_frames`
  (`pict_type`, `pkt_size`) -> bitrate-over-time graph.
- Shot detection: integrate PySceneDetect if its detectors beat ffmpeg's
  `select=scene` for your content (optional dep, feature-gated).
- Optional ML: shot/scene tagging, face / object counts — opt-in only, must not
  bloat the base install.

## 7. Data model & export

One per-frame table is the single source of truth:

`frame, pts_time, YMIN, YAVG, YMAX, UAVG, VAVG, SATAVG, SATMAX, HUEAVG, YDIF, TOUT, VREP, BRNG, scene_score, loudness_M, loudness_S, [vmaf, psnr, ssim]`

- CSV (universal) and JSON (structured) in Phase 1.
- HTML report in Phase 3.
- Keep a stable column schema so exports stay diff-able across versions.

## 8. Dependency & packaging plan

- Replace `ffprobe.exe` (213 MB) with a combined static `ffmpeg` + `ffprobe`
  (~80-120 MB) beside the script. Keep auto-detect: bundled first, then PATH.
- Confirm the static build includes `libvmaf` (Phase 3) and the `zscale` /
  `tonemap` filters (HDR preview). Record the exact build used.
- pip deps remain: `opencv-python`, `numpy`. Anything else (PySceneDetect, etc.)
  must be optional and feature-gated.
- Verify large files write intact (this repo has hit Write-tool NUL-padding
  truncation before): build big files in parts and check with `wc` / `tail`.

## 8.5 Performance & power layer (DONE 2026-06-10)

`va_perf.py` + single-decode combined passes: Analyze runs signalstats/black/
freeze/scene/R128/silence in one ffmpeg graph (parity selftested); quality
compare chains PSNR/SSIM/XPSNR/VMAF in one process with `n_threads`; the
forensics battery runs on a core-sized pool; `analyze.py --jobs N|auto` batches
files in parallel (opt-in); preview frames replay from a free-RAM-sized LRU
cache; Max/Balanced/Eco modes (GUI Tools panel, `VA_PERF`, `--perf`). Deferred:
NVDEC session-aware job scheduling >4, GPU-resident scopes, VMAF-CUDA (needs a
build that ships `libvmaf_cuda`).

## 9. Open questions / risks

- HDR metric fidelity: signalstats on an 8-bit decode under-represents the true
  signal. Decide where to compute in native bit depth vs. accept a documented
  approximation.
- Analysis cost on long / 4K files: signalstats is a full decode pass. Need
  progressive results + cancelation, and possibly a downscaled analysis pass.
- libvmaf 
## 2026-06-10 — Run-verification layer (speedrun moderation use case)

Shipped `va_verify.py`: `platform_screen()` (YouTube/stream-capture/ffmpeg-pipeline
detection; lists which forensic checks a platform re-encode weakened),
`tempo_check()` (regular dup-frame insertion + frame-rate bookkeeping + audio
spectral-cutoff probe), `verify_run()`/`render_verify()` (plain-language verdicts:
clear/note/review + confidence + caveats; battery render appended). Wired: QC
profile `speedrun` (platform-aware recompress), `analyze.py --profile speedrun`
writes `<name>.verify.txt`, GUI Advanced ▾ → "Verify run (splice / tempo / source)"
reuses forensic timeline marks + n/p nav. Selftest: chk_verify_suite (18 checks,
162 total under --quick in the sandbox).

Known limits (candidate follow-ups): no retimer/load-screen detection yet (RTA/
loadless timing); tempo screen cannot see frame-DROP speedups or <25% changes
(caveats say so); Twitch-specific fingerprints not modelled beyond stream-capture/
Lavf heuristics.

## 2026-06-10 — Overnight 2: audio forensics, audio scrub/playback, DV-P5 colour exactness, tools/ housing

Shipped:
- **DV Profile 5 colour fixed for real content.** The software IPT fallback used
  BT.2100 ICtCp matrices; actual Dolby P5 streams use Dolby's own IPT constants
  (~20x different chroma scaling) — greens rendered magenta. New `va_rpu.py`
  parses the RPU NAL directly (header → mapping → DM payload, dovi_tool field
  layout) so exact per-file matrices + reshaping curves now load on ANY ffmpeg;
  canonical P5 constants are the new last-resort default. Verified frame-aligned
  against the HDR10 (P8.1) edition of the same master.
- **Audio forensic battery** (`va_audio.forensic_battery`): hard clipping,
  digital dropouts, click/pop transients, splice signatures (background noise
  floor + spectral-rolloff steps away from silence; demoted to info at picture
  cuts), silent/fake-stereo/phase-inverted channels, bandwidth history (earlier
  lossy generations), sustained loudness steps. Runs as a stage of
  `va_forensics.forensics_report` → findings/marks/QC/exports all inherit it.
  QC `integrity` profile gained `audio_integrity`.
- **Audio scrubbing + playback in the GUI**: waveform strip in the Audio tab
  (drag-scrub, wheel zoom, playhead sync both directions, silence shading,
  forensic ticks), 🔊 audio-follow via bundled ffplay (drift-corrected against
  the video clock, restart-on-seek), paused-scrub audition snippets (winsound).
- **Every audio pass now decodes ONLY the audio stream** (`-vn` / `-map 0:a:0`)
  — loudness/astats/silence/correlation used to drag a full UHD video decode
  along (minutes instead of seconds on 4K files; GUI audio "timeouts").
- **tools/ housing**: binaries live in `tools/` (downloads land there, search
  order prefers it, portable `mkvtoolnix/` nests inside, "Tidy into tools/"
  button migrates loose executables). Repo root is code + docs again.
- MDCV/CLL fallback warning is now DV-compatibility-aware (HLG/SDR bases carry
  no HDR10 static metadata by design).
- `selftest.py --only=<suite,...>` filter; new suites: chk_audio_forensics (61),
  chk_audio_gui, chk_rpu_parse, chk_tools_dir, chk_dynhdr_compat_flag.

Known limits / follow-ups: audio splice scan is mono-folded (per-channel scan
would catch single-channel edits); ffplay follow is wall-clock synced (no PTS
slaving — fine for review, not for lip-sync QC); audition snippets are
Windows-only (winsound); ENF remains in va_forensics (mains hum), not fused
with the audio battery's splice scoring yet.

## 2026-06-11 — Speedrun workbench (toolbar rework + four moderator tools)

Shipped:
- **Toolbar/menus reorganised**: issue-jump buttons now bracket the transport
  cluster; Advanced ▾ grew per-family submenus (QC & picture / HDR & dynamic
  metadata / Container & structure / Forensics & integrity); new top-level
  **Speedrun ▾** menu owns Verify run + a now-exposed QC `speedrun` profile
  (the GUI QC worker also feeds cached verify/forensics results into
  `va_qc.evaluate`, so platform/tempo checks report instead of "not measured").
- **Retimer** (`va_runtools.RetimeDialog`): mark start/end on the playhead,
  fps-aware RTA + optional game-fps frame count, paste-ready mod note,
  LRT line when a load scan exists. Offline equivalent of yt-frame-timer.
- **Load remover** (`va_loads`): whole-file reduced-res sweep flags black,
  frozen, and reference-matching frames (AutoSplit-style captures from the
  playhead, strict/normal/loose presets); segments → timeline marks, n/p
  navigation, exports, and the retimer's LRT.
- **Music continuity scan** (`va_music`): chunked STFT spectral-flux novelty
  normalised by local activity; flags one-hop music-bed jumps (the splice
  tell from the Groobo Diablo 3:12 investigation), checks spectrum
  before/after, and matches re-used/looped segments via per-second
  fingerprints with a continuation-margin test. Plot popup + marks + exports.
- **Luck calculator** (`va_luck`): exact log-space binomial upper tails
  (Dream-report style) with per-event selection ("pick-of") correction,
  verdict bands, and honest caveats. Stdlib-only maths.
- `selftest_runtools.py`: 24 headless checks (exact-tail cross-validation
  against rational arithmetic, synthetic-video load detection incl. ref
  match, synthetic-music control/splice/verbatim-repeat cases).

Known limits / follow-ups: load "static" kind also fires on pause menus and
held cutscene frames (label says so — moderator judges); music scan flags
loud SFX and legit track changes (step through marks before concluding);
luck events assume independence; HUD-region continuity tracking, on-screen
timer OCR, and version fingerprinting (reference stills) are the next tier.

## 2026-06-11 — Plugin architecture (core = video forensics; speedrun = plugin)

Shipped:
- **`va_plugins.py`**: plugin framework. Plugins are `plugins/<name>/`
  folders (`plugin.json` manifest + `plugin.py` entry; private modules import
  by plain name — the folder joins `sys.path`). GUI side: `register(api)`
  against a stable **AppApi v1** facade (toolbar menus anchored left of
  Export ▾ with file-state management, run_bg/post/status, timeline marks +
  n/p issues via prefix-owned `add_marks`, adv_store/export integration,
  popups, frame access, `run_qc`, on-file-open hooks); widgets are tracked
  per-plugin so disable/reload tears them down cleanly. Headless side:
  `register_headless(api)` adds QC profiles (`va_qc.register_profile`) and
  per-profile deep passes for analyze.py. Broken plugins are isolated
  everywhere. Zip install (path-traversal-safe, replace-in-place), JSON
  registry fetch/download with optional sha256 (`registry.example.json`),
  trust-confirmation before any install.
- **`va_plugins_ui.py`** + toolbar **Plugins** button: manager dialog
  (installed list, enable/disable persisted as a disabled-set so new plugins
  auto-enable, reload without restart, install-from-zip, registry browse +
  download, open folder).
- **Speedrun pack extracted to `plugins/speedrun/`** (`sr_verify`, `sr_loads`,
  `sr_music`, `sr_luck`, `sr_dialogs`, `plugin.py`): identical functionality,
  now removable. The generic audio probes (`_audio_spectrum`/`audio_cutoff`)
  moved INTO core `va_audio` (the audio battery's bandwidth check no longer
  silently depends on the speedrun module); `va_qc` lost its speedrun
  checks/profile and gained `register_profile()`.
- **analyze.py is plugin-aware**: deep passes generalised (`PLUGIN_PASSES`),
  `--profile speedrun` works exactly as before when the plugin is present,
  `--list-profiles` includes plugin profiles, and unknown profiles exit 2
  with a hint instead of silently falling back to "general".
- selftest.py loads plugins for the verify suite (skips cleanly when the
  plugin is absent); new `selftest_plugins.py` (16 checks: discovery,
  enable persistence, crash isolation, zip install + hostile-zip rejection,
  registry validation, CLI guard); `selftest_runtools.py` now tests the
  plugin's modules. Verified: full verify suite 60/60, CLI end-to-end writes
  verify.txt through the plugin pass.

Known limits / follow-ups: plugin reload re-imports entry modules but cannot
unload already-imported private modules (a true code update mid-session may
need a restart); plugin menus always insert left of Export ▾ (no ordering
control yet); the registry has no signing beyond per-zip sha256 — host it
somewhere trusted; AppApi is v1 — additions are fine, breaking changes need
a version bump and manifest `api` gating.

## 2026-06-11 — Three domain packs: analog artifacts, captions QC, timer OCR

Shipped (zero core-file edits — first real exercise of the plugin API):
- **`plugins/analog/`** — dropout streaks (row-residual MAD outliers spanning
  most of the width, bottom band excluded), head-switching noise (bottom-band
  HF energy ratio), time-base line wobble (per-row gradient-correlation
  shifts with parabolic sub-pixel refinement, global pan subtracted), tonal
  luma flicker (in-band spectral peak vs median AC power — a flat noise floor
  scores ~10x, AGC pumping scores hundreds; a naive band-power fraction
  false-flags white noise at ~25%). Timeline marks + `analog` QC profile +
  `<name>.analog.txt` CLI pass.
- **`plugins/captions/`** — track discovery (text/bitmap/608 flag),
  ffmpeg-SRT extraction or sidecar, speech map from silencedetect, QC:
  coverage %, systematic sync offset (median cue-start vs nearest speech
  onset, spread-gated), CPS, overlaps, short cues, line layout. `captions`
  profile + `<name>.captions.txt` pass. Bitmap/CC = present-only.
- **`plugins/ocr/`** — calibration-based glyph OCR (box region → segment
  glyphs by column projection → user types the reading → templates;
  incremental until all 10 digits trained), timer parsing (H:MM:SS.mmm /
  M:SS.cc / SMPTE :FF / bare counters), full-file audit: backward jumps,
  forward skips, freezes, and per-segment clock-drift slope with recursive
  changepoint splitting (caught a 0.93x section to the frame in the
  fixture). Region/calibration dialog; GUI-only.
- `selftest_packs.py` (31 checks, all synthetic fixtures: 'VHS' video with
  injected artifacts, mkv with offset SRT + burst audio, rendered timer
  video with splice + drift). README Plugins section + quick start updated.

2026-06-11 follow-up polish: (1) ttk **Treeview was unstyled** (the plugin
manager is the first Treeview in the app — clam's stock white field made rows
unreadable); va_theme.apply() now styles Treeview + Treeview.Heading in both
palettes, so the theme toggle covers it. (2) **Plugin menus consolidated**:
instead of one toolbar menubutton per plugin, the toolbar has a single
**Plugins ▾** menu — *Manage plugins...* on top, then one cascade per enabled
plugin (Advanced ▾ style). `AppApi.add_toolbar_menu` keeps its signature but
now creates cascades (trailing "▾" stripped from labels); a `_CascadeRef`
adapter exposes `.config(state=)` so the app's existing enable/disable button
loops flip cascade entries untouched; teardown deletes cascades by label, so
disable/reload works the same. Plugins needed zero changes.

Known limits / follow-ups: analog dot-crawl/rainbowing detector not yet
implemented (needs chroma planes; scan is luma-only today); captions sync
uses speech onsets only (no per-cue ASR alignment); OCR assumes a fixed-
position overlay and one font (proportional-font timestamps with kerning
changes may segment differently per frame — calibrate on more frames).

## 2026-06-11 — Plugin API v2 (the malleability round)

Shipped:
- **Timeline series**: `api.add_series(label, t, v, vmin=, vmax=)` puts any
  per-time curve in the metrics timeline + selector (explicit timestamps map
  to container-frame x; draws even before Analyze; per-file lifetime; the
  selector refreshes on add/remove/teardown). Analog publishes "Analog: line
  jitter (px)", OCR publishes "Timer drift (s)" — a splice is a step, a
  slowdown is a ramp, in one glance.
- **Preview overlays**: `api.add_overlay(name, fn)`; fn(frame_bgr, idx) runs
  in `_render_video` (the one render choke point) on a copy, exception-
  isolated. OCR gained "Show/hide region overlay".
- **Inter-plugin services**: module-level registry; `api.provide`/`require`
  on both GUI and headless APIs. OCR provides `ocr.read_region` +
  `ocr.parse_timer` (any pack can read pixels with the calibrated font).
- **Hotkeys**: `api.bind_key("<t>", fn)` — core-reserved set refused,
  per-seq dispatcher dict so teardown is a safe no-op (avoids Tk's unbind
  bugs), respects the text-entry guard. OCR binds `t` = read timer at
  playhead.
- **on_analyze hook** (fires in `_analysis_done` before the timeline redraw)
  + `api.events` property.
- **QC profile EXTENSION**: `va_qc.extend_profile(profile, checks, owner)` /
  `retract_extensions(owner)`; `api.extend_qc_profile` auto-owners by plugin
  name; evaluate() composes base + extensions; reload-safe.
- **CLI multi-pass**: pass specs gained `ctx_key` (default "verify") and
  `always`; analyze.py runs every matching pass, stores each under its own
  ctx key, writes each report file, and JSON gains `plugin_passes{key:...}`
  (speedrun stays on the legacy "verify" key untouched). Latent GUI gap fixed
  the same way: `_qc_worker` now bridges ALL cached tool results into
  `ctx["adv"]`, so "QC check (analog/captions profile)" sees its own scan
  (it previously read speedrun's verify slot and said "not measured").
  Plugin check helpers read own-key → adv → legacy verify.
- API_VERSION=2; analog/captions/ocr manifests now require api 2 (speedrun
  stays 1). selftest_plugins +11 v2 checks (fake-tk surface lifecycle);
  selftest_packs +6 (series separation, ctx-key/adv/legacy QC paths, drift
  series shows the jump + slowdown).

Known limits / follow-ups: overlays run on the playback render path — keep
them cheap (no per-frame decode work); plugin series are single-line plots
(no multi-line/bands yet); scope-tab registration, plugin-registered tool
downloads, and the Python console plugin are the next API round.

## 2026-06-11 — Hidden-data / steganography plugin (plugins/steg)

Shipped, three layers (full scope incl. codec-domain, per request):
- **steg_container** (reliable): appended data after the last MP4 box / EBML
  element (entropy of the tail labels it encrypted vs padding); polyglot scan
  for embedded-format magic at unexpected offsets (ZIP/RAR/7z/PDF/PNG/ELF/…),
  classified trailing vs in-media; content-bearing free/skip/Void atoms;
  mdat reference-gap (stsz/mdat extent); whole-file sliding-window entropy.
  Reused va_forensics._walk_mp4; added a minimal EBML vint walker for MKV.
- **steg_bitplane**: RS analysis (Fridrich) as the PRIMARY detector + chi-
  square (Westfeld) as CONTEXT only. Hard-won lessons: (1) chi-square over-
  flags smooth histograms — useless as a standalone clean/dirty test, so it
  never triggers a finding alone (this is why RS/SPA superseded it); (2) the
  RS quadratic term-mapping is easy to get backwards — verify clean→~0,
  full-embed→~1; (3) synthetic fixtures are treacherous — noise-added images
  false-positive RS (LSBs already random), noiseless gradients false-positive
  chi (smooth pairs); the right cover model is 1/f noise + a JPEG roundtrip
  (combs the histogram for chi, keeps spatial smoothness for RS). Own incomplete-
  gamma (_gammq) for the chi survival fn (no scipy). Implemented a custom
  incomplete-gamma rather than depending on scipy.
- **steg_codec** (best-effort, honestly hedged): SEI user_data_unregistered
  NAL scan (parse_sei walks Annex-B NALs after ffmpeg mp4toannexb demux; H.264
  type 6, HEVC 39/40; emulation-prevention stripped) — every libx264 file
  carries one ~590-byte version SEI, so the detector keys on >2KB; plus a
  frame-size-vs-complexity robust-regression residual (coefficient/MV
  embedding inflates coded size). True QDCT/MV bitstream steganalysis is NOT
  done and the report says so.
- **steg_scan** orchestrator: container always; LSB gated to lossless 8-bit
  (LOSSLESS codec set; force flag); codec for H.264/HEVC + the residual.
  steg QC profile (6 checks) + analyze.py steg pass (.steg.txt, ctx_key steg)
  + LSB-plane overlay + size-residual timeline series.
- selftest_packs t_steg (12 checks incl. real polyglot, encrypted append,
  lossless FFV1 clean-vs-LSB-embedded end to end). All 5 plugins load clean
  (steg api 2); full selftest sweep green.

DROPPED (deliberately): audio LSB steganalysis. Spatial RS saturates on a
high-amplitude 1-D audio signal (neighbour diffs dwarf the ±1 flip), and clean
PCM LSBs are frequently already random (dither/noise floor) so naive tests
over-flag — needs a dedicated method, deferred. Other future steg work:
dot-crawl/chroma analysis, true coefficient-domain steganalysis, StegExpose-
style multi-estimator fusion (add Sample Pairs alongside RS).
