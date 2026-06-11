# Video Analyzer

A one-stop **video data-science & QC workbench**. It plays a video and treats it
as a dataset: HDR-aware decode, a full set of live scopes, per-frame metric
timelines, automatic event/QC detection, deep audio analysis, reference-quality
comparison, and a set of frontier analyses (perceptual banding/visibility,
temporal integrity, forensics, HDR light-levels, and Dolby Vision / HDR10+
dynamic metadata) — all on a lean stack (`opencv-python` + `numpy`), with
**ffmpeg** doing the heavy lifting and the GPU used when available.

It runs as a desktop GUI **and** as a headless CLI for batch QC over folders.

---

## Highlights

- **Reliable, hardware-accelerated decode** — HEVC / AV1 / 10-bit / HDR via ffmpeg
  (NVDEC / Quick Sync / D3D11VA), with automatic HDR→SDR tonemapping; OpenCV fallback.
- **Native aspect-ratio rendering** — every file shows at its true display shape:
  anamorphic SAR is baked in (2.35:1 scope stored as 16:9 renders 2.35:1), rotation
  metadata makes phone video portrait, square/vertical/odd ratios letterbox cleanly.
  The info panel shows a "Display AR" line whenever storage and display geometry
  differ; analysis that needs untouched codec pixels opts out internally.
- **Scopes** — vectorscope, luma waveform, RGB parade, false colour, histogram, CIE gamut.
  On HDR / wide-gamut files every scope reads the **native signal** (a second,
  scope-only decode: no tonemap, no gamut conversion, no 8-bit crush), so the CIE
  plot really shows points outside BT.709, the waveform is graticuled in **nits**
  (PQ), false colour uses nits bands, and the vectorscope is true Y'CbCr BT.2020
  with computed 75% targets. Scopes are stamped `NATIVE PQ/BT.2020` (gold) - or
  `DISPLAY-REFERRED` (amber) if the native tap is unavailable (no ffmpeg). During
  playback the tap chases the playhead (latest frame wins); paused or stepping it
  converges on the exact frame. DV profile 5 IPT is reshaped to PQ/BT.2020 via the
  RPU without tonemapping.
- **Metrics timeline** — per-frame signalstats (luma/chroma/saturation/hue, frame
  diffs, broadcast-range, temporal outliers) with a scrubbable, click-to-seek timeline.
- **Events** — black / freeze / scene-cut detection with jump-to-issue navigation.
- **Audio** — EBU R128 loudness (+ over time), per-channel peak/RMS/DC/dynamic range,
  silence, stereo correlation, spectrogram & waveform.
- **Quality** — PSNR / SSIM / VMAF compare-two-files, an A/B viewer (diff / heatmap /
  split), and an encode-ladder that finds the optimal per-title bitrate.
- **QC profiles** — `broadcast-r128`, `web-streaming`, `general` produce pass/fail
  verdicts over loudness, levels, events, banding, flash safety, cadence, and HDR metadata.
- **Frontier analyses** — banding/contouring, artifact-visibility (JND) maps,
  saliency-weighted "visually lossless?" verdicts; cadence/telecine + PSE flash safety;
  per-pixel nits maps, MaxCLL/MaxFALL verification, multi-display previews; Dolby
  Vision & HDR10+ inspection and verification.
- **Forensics / integrity battery** — corroborated splice detection (coding-cost +
  PTS + off-cadence-IDR + audio discontinuities scored per timestamp), container
  edit-history tells (MP4 box walk: multiple `mdat`, free-space gaps, edit lists;
  MKV writing apps), encoder fingerprinting (x264/x265 SEI settings, Lavf/Lavc and
  camera-vendor byte-scan vs claimed tags), ELA + sensor-noise consistency maps,
  repeated-sequence/loop detection, ENF (mains-hum) continuity, SHA-256
  chain-of-custody, C2PA validation with ingredient chains — merged into one
  findings report with timeline markers. Indicators, not proof.
- **Run verification (submission moderation)** — one-click "is this one continuous,
  unmanipulated capture?" for speedrun/contest/evidence review: the forensics battery
  plus a platform re-encode screen (YouTube / stream-capture / ffmpeg-pipeline
  detection that says which forensic signals it weakened, instead of letting them
  mislead), a tempo screen (regular duplicate-frame insertion, frame-rate bookkeeping,
  audio spectral-cutoff probe for resample slowdowns), and plain-language verdicts
  with confidence ratings and "what this does NOT prove" caveats.
- **Headless CLI** — analyze a file or a whole folder, emit JSON / HTML / CSV reports
  and a batch dashboard with QC verdicts.

---

## Install

**No-Python option (single exe):** run `build.bat` once on any machine with
Python — it produces a standalone **`video-analyzer.exe`** in this folder
(PyInstaller; ~150–200 MB, first launch unpacks for ~10 s). Target machines
need nothing installed: share the folder (exe + `tools\` + `plugins\`) as a
zip. The exe finds `tools\`, `plugins\`, and writes `va_ui.json` **beside
itself**, and doubles as the CLI:

```
video-analyzer.exe                     GUI (same as python launch.py)
video-analyzer.exe analyze CLIP ...    headless batch QC (= analyze.py)
video-analyzer.exe hwinfo CLIP        hardware decode/tonemap probe
video-analyzer.exe tools               download ffmpeg + helper binaries
video-analyzer.exe pack-plugins        zip plugins for sharing
```

`tools\` stays external on purpose — bundling 844 MB of ffmpeg/mkvtoolnix into
the exe would mean a ~1 GB file unpacking to %TEMP% on every launch. A fresh
exe with no `tools\` still works: it offers the in-app download on first run.
(Windows Defender sometimes flags freshly built PyInstaller exes — if so, add
an exclusion or submit the file as a false positive.)

**Running from source instead:**

1. **Python 3.9+**, then:
   ```
   pip install -r requirements.txt        # opencv-python, numpy
   pip install tkinterdnd2                 # optional: drag-and-drop file open
   ```
2. **ffmpeg + ffprobe** (required for full functionality; `ffplay` from the same
   build enables audio playback). Use the **Tools** button in the app to download
   the BtbN full GPL build automatically (includes `libplacebo`, hardware
   decoders, and `libvmaf`), or put the executables in the **`tools/`** subfolder,
   beside the scripts, or on `PATH`. Without ffmpeg the tool falls back to
   OpenCV with reduced capability.
3. **Optional helper binaries** — all installable from the **Tools** dialog in the app
   (or `python va_tools.py [name]`), or place them beside the scripts / on PATH:
   - [`dovi_tool`](https://github.com/quietvoid/dovi_tool) — Dolby Vision RPU extract / L1 plot.
   - [`hdr10plus_tool`](https://github.com/quietvoid/hdr10plus_tool) — HDR10+ metadata extract / plot.
   - [`mediainfo`](https://mediaarea.net/en/MediaInfo) — deep container/stream metadata report.
   - [`mkvextract`](https://mkvtoolnix.download) — lossless MKV track extraction (Windows portable).
   - [`mp4dump`](https://www.bento4.com) — MP4 box inspector for DV `dvcC`/`dvvC` signaling.
   - [`c2patool`](https://github.com/contentauth/c2pa-rs) — Content Credentials (C2PA)
     manifest validation + ingredient chains (a presence byte-scan works without it).

   The plots and reports live under **Advanced ▾** once a file is loaded.

   **Where binaries live:** downloads land in the **`tools/`** subfolder, which is
   searched first (then the script folder, then a portable `mkvtoolnix/` dir in
   either, then standard installs and `PATH`). A **Tidy into tools/** button in
   the Tools dialog moves loose executables out of the code folder for you.

## Quick start

```
python launch.py                 # preflight-checks ffmpeg, prints capabilities, opens the GUI
python launch.py movie.mkv       # open a file directly
run.bat                          # Windows convenience launcher

python analyze.py FILE_OR_FOLDER --profile broadcast-r128   # headless analysis + QC
python hwinfo.py movie.mkv       # report which hardware decode / tonemap paths work
python selftest.py               # headless engine self-test
python selftest_plugins.py       # plugin-framework self-test (discovery/install/registry)
python selftest_runtools.py      # speedrun-plugin self-test (loads/music/luck)
python selftest_packs.py         # domain-pack self-test (analog/captions/ocr/steg)
```

---

## The GUI

Toolbar: **Open**, then the transport cluster bracketed by **⏮ Issue / Issue ⏭**
(jump between detected events — forensic findings appear here too, and as dashed
markers on the timeline after a forensics scan) around the frame-step and
Play/Stop buttons; then **Analyze** (runs the metrics pass and fills the
timeline, Audio tab, and events), **Compare ▾** (quality metrics / A-B viewer /
encode ladder), **Advanced ▾** with submenus — *QC & picture* (QC check, banding
map), *HDR & dynamic metadata* (multi-display preview, metadata report,
dynamic-metadata vs content, dynamic-vs-static A-B, DV/HDR10+ plots),
*Container & structure* (MediaInfo / mp4dump / mkvinfo), *Forensics & integrity*
(**forensics report**, C2PA, ELA map, noise map, ENF trace), any menus added
by installed **plugins**, **Export ▾** (CSV / JSON / HTML report), and
**Plugins ▾** — *Manage plugins...* at the top, then one submenu per enabled
plugin (Speedrun, Analog, Captions, Timer OCR), laid out like the
Advanced ▾ submenus. The HDR badge and the live decode backend
(e.g. `ffmpeg:cuda+libplacebo`) show in the status bar.

## Plugins

The core app is a general video-forensics workbench; domain workbenches are
**plugins** — folders under `plugins/<name>/` with a `plugin.json` manifest
and a `plugin.py` entry (`register(api)` for the GUI, `register_headless(api)`
for `analyze.py`/selftests). Each enabled plugin appears as a submenu of the
toolbar's **Plugins ▾**; **Plugins ▾ → Manage plugins...** opens the manager:
enable/disable installed packs, **Install from zip**, browse/download from a
JSON **registry** (URL persisted in settings; format in
`plugins/registry.example.json`, sha256-checked downloads), open the plugins
folder, and reload. Plugins are ordinary Python running with your full
permissions — install only code you trust. A broken plugin is isolated at
load and reported in the status bar; everything else keeps working.

<<<<<<< Updated upstream
**Publishing plugins** — `pack_plugins.py` builds everything the registry
needs (stdlib only, deterministic zips so unchanged plugins keep their
sha256):

```bash
python pack_plugins.py --base-url https://github.com/ORG/REPO/releases/download/plugins-v1
```

That writes `dist/<name>-<version>.zip` per plugin plus `dist/registry.json`
with matching `zip_url`/`sha256` entries. Publish on GitHub: create release
`plugins-v1` and upload the zips as assets, then commit `registry.json` to
the repo root so its raw URL stays stable across releases. Other users paste
that URL — `https://raw.githubusercontent.com/ORG/REPO/main/registry.json` —
into **Manage plugins... → registry → Fetch** and click Install. To ship
copies pre-pointed at your registry, set `DEFAULT_REGISTRY_URL` in
`va_plugins.py`. The repo must be public (the downloader sends no auth);
keep private packs on an internal HTTP(S) server instead — same script,
different `--base-url`. To update a plugin, bump `version` in its
`plugin.json`, re-run the script with a new tag, and replace the committed
`registry.json`.
=======
**Publishing plugins** — each plugin ships as its own zip listed in
`dist/registry.json`, so users install packs individually from
**Manage plugins...** (the app comes pre-pointed at this repo's registry via
`DEFAULT_REGISTRY_URL` in `va_plugins.py`). `pack_plugins.py` builds it all
(stdlib only, deterministic zips so unchanged plugins keep their sha256):

```bash
python pack_plugins.py --base-url https://raw.githubusercontent.com/dwightsabeast/video-analyzer/main/dist
git add dist && git commit -m "publish plugins" && git push
```

To update one plugin: bump `version` in its `plugin.json`, re-run the
script, push. Users see the new version on the next **Fetch** and reinstall
just that pack. The repo must stay public (the downloader sends no auth);
for private packs host `dist/` on an internal HTTP(S) server instead — same
script, different `--base-url`.
>>>>>>> Stashed changes

**Plugin API v2** lets packs reach deep into the app: **timeline series**
(per-time curves join the metric selector — timer drift, line jitter),
**preview overlays** (draw on the video frame — the OCR region box),
**inter-plugin services** (`api.provide`/`api.require` — e.g. other packs can
borrow the OCR pack's calibrated reader), **post-Analyze hooks**, **hotkey
registration** (core keys are reserved), **extending existing QC profiles**
(owner-tracked, so disable/reload retracts cleanly), and **multiple CLI deep
passes per run** (each pass owns a ctx key; `always=True` passes run on every
`analyze.py` invocation). Everything a plugin adds is torn down when it is
disabled or reloaded.

**Speedrun forensics plugin** (ships in `plugins/speedrun/`) — the
run-verification workbench: **Verify run** (forensics battery +
platform/tempo screens with a moderator verdict), **QC check (speedrun
profile)**, a frame-accurate **Retimer** (mark start/end on the playhead →
RTA + paste-ready mod note, like yt-frame-timer but offline), **Load remover
/ LRT** (finds black/frozen stretches plus frames matching captured reference
load screens, AutoSplit-style; marks the timeline and the retimer then also
quotes load-removed time), **Music continuity scan** (spectral-flux
discontinuities in the music bed — the classic splice tell — plus
re-used-segment matching; results plot + timeline marks), and a **Luck
calculator** (exact binomial odds for visible RNG events with a
selection-bias correction, Dream-report style). Scan marks join the `n`/`p`
issue navigation and travel with JSON/HTML exports. Headless, the plugin
registers the `speedrun` QC profile and the `analyze.py` deep pass that
writes `<name>.verify.txt`.

**Analog artifact detector** (`plugins/analog/`) — tape-era artifact scan for
digitised analog video: dropout streaks (bright line flashes), VHS
head-switching noise in the bottom band, time-base line wobble (no/weak TBC),
and tonal luma flicker (AGC pumping / film transfer). Timeline marks, an
`analog` QC profile, and an `analyze.py` pass writing `<name>.analog.txt`.

**Captions QC** (`plugins/captions/`) — subtitle compliance: track discovery
(text / bitmap / EIA-608 flag), extraction of text tracks or a `.srt`
sidecar, speech coverage from an audio silence map, systematic sync offset
vs speech onsets, reading speed (CPS), overlaps, minimum durations and line
layout. `captions` QC profile + `analyze.py` pass writing
`<name>.captions.txt`. Bitmap (PGS/DVB) and CC tracks report present-only.

**Hidden-data / steganography scan** (`plugins/steg/`) — find data hidden in a
video file, three layers. *Container forensics* (always, reliable): data
appended after the last box/element, polyglot archives/documents (ZIP, RAR,
7z, PDF… by magic signature at an unexpected offset), content-bearing padding
atoms (`free`/`skip`/Void), bytes in `mdat` no sample table references, and a
whole-file entropy map (a high-entropy *trailing* region looks
encrypted/compressed). *Pixel-LSB steganalysis* (RS analysis primary, chi-square
as corroborating context) — gated to lossless 8-bit sources, because pixel-LSB
cannot survive lossy coding; force-scan available. *Compressed-domain
indicators* — SEI `user_data_unregistered` NAL scan (a documented payload
carrier in H.264/HEVC) and a frame-size-vs-complexity residual (coefficient/MV
embedding inflates coded frames). `steg` QC profile + `analyze.py` pass writing
`<name>.steg.txt`; an LSB-plane preview overlay; the size residual plots as a
timeline series. Audio LSB is intentionally not attempted (see ROADMAP — clean
audio LSBs are often already random, and spatial RS doesn't transfer to a 1-D
signal). Indicators, not proof.

**Timer / timecode OCR** (`plugins/ocr/`) — burned-in timer audit with no OCR
engine: box the overlay once, type what it reads, and every glyph becomes a
template (calibrate on a couple of frames until all ten digits are trained).
Then every frame is read and audited: backward jumps and forward skips
(splice-class evidence), freezes (loads/pauses), and per-segment clock-drift
slope with changepoint splitting — a timer running at 93% of video speed is
the cleanest slowed-footage tell. Works on LiveSplit overlays, in-game
timers, SMPTE burn-ins and CCTV timestamps. GUI-only (the region is boxed by
hand).

Panels: the video; a **scope notebook** (Vectorscope, Waveform, RGB parade, False
colour, CIE gamut, Histogram) plus an **Audio** tab; the **Stream & HDR Information**
text; and the full-width **Metrics Timeline** with a metric selector, event markers,
and a playhead (click to seek).

**Audio**: the 🔊 toolbar button makes the sound follow playback (bundled
`ffplay`, restarted on every seek and drift-corrected against the video clock),
and while paused each scrub plays a ~1/3 s audition snippet (Windows). The
Audio tab shows a **waveform strip** — drag to scrub exactly like the video
timeline, mouse-wheel to zoom (double-click resets), playhead synced both ways,
silence shaded after Analyze. **Audio forensics...** runs the battery
(hard clipping, digital dropouts, clicks/pops, splice signatures in the
background noise floor + spectral rolloff, silent/duplicated/phase-inverted
channels, bandwidth history, sustained loudness steps); findings mark the
waveform strip, join the `n`/`p` issue navigation, and travel with exports.

Shortcuts: `Space` play/pause · `,`/`.` step one frame back/forward (also the `◀|` / `|▶` buttons; stepping pauses playback) · `←/→` seek 1 s · `Home` stop · `n`/`p` next/previous
issue · `o` open. Drag-and-drop a video onto the window if `tkinterdnd2` is installed.
On the audio strip: drag = scrub, wheel = zoom, double-click = fit.

Every Advanced- or Speedrun-menu result you generate (QC, verify, forensics, C2PA, HDR
reports, MediaInfo/mp4dump/mkvinfo, banding/ELA/noise maps, ENF, plots...) is
cached for the session and included in **Export ▾ → Analysis JSON** (data + text;
cleared when a new file is opened) and **HTML report** (which also embeds the
map/plot images, and overlays timestamped forensic findings on the timeline
charts as dashed colour-coded lines - legend included, hover for details).
Run a tool first and its results travel with the export.

## The CLI (`analyze.py`)

```
python analyze.py INPUT [-o OUTDIR] [--profile P] [--formats json,html,csv]
                  [--forensics] [--recurse] [--jobs N|auto] [--perf MODE]
                  [--quiet] [--list-profiles]
```

`INPUT` is a file or a folder. For each video it runs the full engine, evaluates a
QC profile, and writes per-file reports; for a folder it also writes `index.html` — a
batch dashboard with per-file QC verdicts. Exit code is non-zero if any file FAILs QC.

QC profiles:
- **broadcast-r128** — EBU R128 loudness (-23 LUFS), true peak ≤ -1 dBTP, no black/freeze,
  broadcast-range levels, audio present, plus PSE flash safety, cadence, and HDR metadata.
- **web-streaming** — ~-14 LUFS target, true peak, banding, PSE, HDR metadata.
- **general** — sanity checks: black/freeze/banding/PSE/cadence/silence/audio/HDR.
- **integrity** — the forensics battery as pass/fail checks: provenance (C2PA),
  container integrity, corroborated splices, loops/periodicity, noise-floor
  consistency, ENF continuity, recompression. Implies `--forensics`, which also
  writes a `<name>.forensics.txt` report (and embeds findings in JSON/HTML).
- **speedrun** *(provided by the speedrun plugin)* — run/submission
  verification: source-pipeline screen (platform re-encodes weaken
  splice/encoder/noise evidence and are called out as such), tempo/speed
  integrity, splices, loops, noise floor, ENF, container, provenance.
  Implies the forensics battery and also writes `<name>.verify.txt` — a
  moderator-readable verdict report (GUI: Plugins ▾ → Speedrun → Verify run). Plugin
  profiles appear in `--list-profiles` whenever the plugin is installed
  and enabled.

`--jobs N` (or `--jobs auto`) analyses several files at once — output is buffered
per file so logs stay readable; the default stays serial. `--perf max|balanced|eco`
picks the performance mode for this run (see *Performance & power* below).

Example:
```
python analyze.py "D:\deliverables" --profile broadcast-r128 --formats json,html --jobs auto
```

## Hardware acceleration

Decode (and HDR tonemapping) run on the GPU when possible. The tool **probes** what
actually works on your machine and picks the best pipeline, falling back to software:

- Decode: `cuda` (NVDEC) / `qsv` (Quick Sync) / `d3d11va` / `dxva2` — verified per
  codec, so e.g. AV1 only uses HW where the GPU truly supports it.
- HDR tonemap: `libplacebo` (best, needs the full build) → GPU decode + `scale_cuda`
  downscale + tonemap → `tonemap_opencl` → software. Output is capped to 1280px on
  the long side so the preview stays fast even at 4K.
- **Dolby Vision Profile 5** (`dvh1`, no HDR10 fallback) stores pixels in Dolby's
  IPT colour space — decoded as ordinary YCbCr it comes out magenta/violet, so it
  gets its own pipeline: `libplacebo` with `apply_dolbyvision` (exact, applies the
  per-frame RPU) → built-in software IPT decode (`va_ipt.py`, numpy; reads the
  file's own DM colour matrices + reshaping curves via ffprobe, or by parsing
  the RPU NAL directly (`va_rpu.py`) on older ffmpeg builds — the canonical
  Dolby P5 constants are the last-resort default. Labelled *approximate* in the status
  bar because per-scene reshaping is applied statically). DV profiles with an
  HDR10/SDR/HLG base layer (8.1/8.2/8.4, 7) ride the normal pipelines.

Run `python hwinfo.py yourfile.mkv` to see a per-path PASS/FAIL report with the actual
ffmpeg error when something can't be used. Env vars: `VA_NO_HWACCEL=1` forces software;
`VA_HWACCEL_DEBUG=1` logs the chosen pipeline.

## Performance & power

The engine sizes itself from your machine (`va_perf.py`: cores, free RAM, GPU VRAM
via `nvidia-smi`) and is built around one principle: **the decode is where the
watts go**, so never decode the same file twice for answers you can compute in
one pass.

- **Analyze = one decode.** Per-frame signalstats, black/freeze detection, scene
  cuts, R128 loudness and silence all run as one ffmpeg filter graph
  (`va_metrics.analyze_pass`) instead of six separate full decodes — same numbers
  (the selftest asserts parity), ~4–6× less decode work. The status line reports
  it: `6 analyses · 1 decode (cuda) · 12s`.
- **Compare = two decodes.** PSNR → SSIM → XPSNR → VMAF run chained in a single
  process (distorted and reference each decoded once, not once per metric), and
  libvmaf gets `n_threads` = your usable cores — it is single-threaded by default.
- **Forensics battery fans out.** The ten independent stages (hash, splice scan,
  container walk, noise, loops, ENF, …) run on a thread pool sized to your cores.
- **Frame cache.** Decoded preview frames are kept in an LRU cache sized from
  free RAM (~25% in `max` mode), so stepping and scrubbing back over recent
  frames costs zero decode work.
- **Hardware decode everywhere.** All analysis passes reuse the verified
  per-codec HW decode (NVDEC/Quick Sync) — fixed-function silicon at a fraction
  of the CPU's power draw, leaving the cores for the math.

Modes (Tools window → PERFORMANCE, `--perf`, or the `VA_PERF` env var):
**max** (default) uses every core and full fan-out; **balanced** leaves two cores
of headroom for the rest of the desktop; **eco** spends the fewest watt-hours —
HW decode, no oversubscription, single batch lane. Every mode keeps the
single-decode combined passes: doing the work once is the biggest saving of all.

---

## Module map

| File | Role |
| --- | --- |
| `video-analyzer.py` | Tkinter GUI (entry point) |
| `launch.py` | preflight launcher + capability report |
| `analyze.py` | headless CLI / batch QC |
| `hwinfo.py` | hardware-accel diagnostic |
| `selftest.py` | headless engine regression test |
| `va_ffmpeg.py` | tool discovery + `VideoSource` decode |
| `va_hwaccel.py` | codec-aware HW decode + HDR tonemap pipeline selection |
| `va_probe.py` | HDR10/HDR10+/HLG/Dolby Vision classification + ffprobe formatting |
| `va_ipt.py` | Dolby Vision P5 (IPTPQc2) software decode fallback |
| `va_rpu.py` | direct DV RPU bitstream parser (matrices + reshaping curves on ANY ffmpeg) |
| `va_metrics.py` | signalstats + black/freeze/scene events + ebur128 + frame sizes + motion |
| `va_perf.py` | system detection (cores/RAM/VRAM), performance modes + budgets, preview frame cache |
| `va_scopes.py` | scope renderers (vectorscope / waveform / parade / false colour / histogram / CIE), display + native-signal (PQ/HLG/BT.2020) variants |
| `va_quality.py` | PSNR / SSIM / VMAF |
| `va_audio.py` | loudness + per-channel stats + silence + correlation + spectrogram/waveform + PCM decode, waveform overview, playback follower, audio forensic battery |
| `va_qc.py` | configurable pass/fail QC profiles |
| `va_compare.py` | A/B difference/heatmap + encode-ladder |
| `va_export.py` | CSV / JSON / HTML report + batch dashboard |
| `va_perceptual.py` | banding, JND visibility, saliency, visually-lossless |
| `va_temporal.py` | duplicate/cadence/telecine + PSE flash safety |
| `va_plugins.py` | plugin framework: discovery/manifests, GUI + headless loading, zip install, registry |
| `va_plugins_ui.py` | Plugins manager dialog |
| `plugins/speedrun/` | speedrun plugin: run verification (platform/tempo screens + verdicts), retimer, load remover, music continuity, luck calculator |
<<<<<<< Updated upstream
| `va_forensics.py` | corroborated splice scan, container/encoder/metadata forensics, ELA + no
=======
| `va_forensics.py` | corroborated splice scan, container/encoder/metadata forensics, ELA + noise maps, loop detection, ENF trace, SHA-256, C2PA validation |
| `va_hdr.py` | PQ nits analysis, MaxCLL/MaxFALL verification, multi-display preview |
| `va_dynhdr.py` | Dolby Vision / HDR10+ dynamic-metadata inspect, verify, and plots |
| `va_theme.py` | dark theme + persisted UI preferences (`va_ui.json`) |
| `va_tools.py` | helper-binary downloads (ffmpeg, dovi_tool, ...) into `tools/` |
| `va_paths.py` | app/bundle dir resolution + console attach for the frozen exe |
| `pack_plugins.py` | builds `dist/` plugin zips + `registry.json` for publishing |
| `plugins/analog/` | analog artifact detector pack (dropouts, head-switch, TBC wobble, flicker) |
| `plugins/captions/` | captions QC pack (sync, reading speed, coverage, layout) |
| `plugins/ocr/` | timer/timecode OCR pack (calibrated glyph templates, drift audit) |
| `plugins/steg/` | hidden-data / steganography pack (container, LSB, codec layers) |
>>>>>>> Stashed changes
