#!/usr/bin/env python3
"""
va_plugins - the analyzer's plugin framework.

The core app is a general video-forensics workbench; domain packs (speedrun
verification, etc.) live in plugins. A plugin is a folder:

    plugins/<name>/plugin.json     manifest (see below)
    plugins/<name>/plugin.py       entry module
    plugins/<name>/*.py            private modules (the folder joins sys.path)

Manifest keys: name (folder-safe id), title, version, description,
entry (default "plugin.py"), api (default 1).

GUI side: the app builds an AppApi facade and load_gui() calls each enabled
plugin's register(api). Headless side (analyze.py, selftests): load_headless()
calls register_headless(api) so plugins can add QC profiles and per-profile
analysis passes without any Tk.

Plugins are ordinary Python executed in-process: install only code you trust.
This module stays import-safe headless (tkinter is only touched inside AppApi
methods, which the GUI alone calls).

Plugin authors: every subprocess spawn MUST pass
``creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0``
- the app ships as a windowed .exe, so an unflagged child pops up a console
window on the user's screen (selftest chk_silent_spawns enforces this for
bundled plugins)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import zipfile

import va_theme

API_VERSION = 2
DEFAULT_REGISTRY_URL = ("https://raw.githubusercontent.com/dwightsabeast/"
                        "video-analyzer/main/dist/registry.json")
                                 # overridable in the Plugins manager (ui key
                                 # "plugin_registry_url"); format in
                                 # plugins/registry.example.json; published by
                                 # pack_plugins.py


SERVICES: dict = {}      # inter-plugin services: name -> object/callable


def provide_service(name, obj):
    SERVICES[str(name)] = obj


def require_service(name, default=None):
    return SERVICES.get(str(name), default)


def plugins_dir(base=None) -> str:
    """The canonical plugin home: <script>/plugins (created on demand)."""
    if not base:
        import va_paths
        base = va_paths.app_dir()      # beside the .exe when frozen
    d = os.path.join(base, "plugins")
    os.makedirs(d, exist_ok=True)
    return d


def _read_manifest(d: str) -> dict:
    mf = os.path.join(d, "plugin.json")
    with open(mf, "r", encoding="utf-8") as fh:
        m = json.load(fh)
    if not isinstance(m, dict) or not m.get("name"):
        raise ValueError("plugin.json must be an object with a 'name'")
    m.setdefault("title", m["name"])
    m.setdefault("version", "0")
    m.setdefault("description", "")
    m.setdefault("entry", "plugin.py")
    m.setdefault("api", 1)
    return m


def _disabled() -> set:
    raw = va_theme.load_ui_key("plugins_disabled", "") or ""
    return {s for s in raw.split(",") if s}


def set_enabled(name: str, enabled: bool):
    d = _disabled()
    (d.discard if enabled else d.add)(name)
    va_theme.save_ui_key("plugins_disabled", ",".join(sorted(d)))


def registry_url() -> str:
    return va_theme.load_ui_key("plugin_registry_url",
                                DEFAULT_REGISTRY_URL) or ""


def set_registry_url(url: str):
    va_theme.save_ui_key("plugin_registry_url", (url or "").strip())


def discover(base=None) -> list:
    """[{name,title,version,description,dir,entry,api,enabled,error}] for every
    plugin folder, including broken ones (error set, never raises)."""
    root = plugins_dir(base)
    out = []
    dis = _disabled()
    for fn in sorted(os.listdir(root)):
        d = os.path.join(root, fn)
        if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, "plugin.json")):
            continue
        info = {"name": fn, "title": fn, "version": "?", "description": "",
                "dir": d, "entry": "plugin.py", "api": 1,
                "enabled": fn not in dis, "error": None}
        try:
            m = _read_manifest(d)
            info.update({k: m[k] for k in ("name", "title", "version",
                                           "description", "entry", "api")})
            if int(m.get("api", 1)) > API_VERSION:
                info["error"] = ("needs plugin API v%s; this app provides v%d"
                                 % (m.get("api"), API_VERSION))
        except Exception as exc:  # noqa: BLE001 - a broken manifest must not abort discovery
            info["error"] = "bad plugin.json: %s" % exc
        out.append(info)
    return out


def _import_entry(info: dict):
    """Import a plugin's entry module (plugin dir joins sys.path so the
    plugin's private modules import by plain name)."""
    if info["dir"] not in sys.path:
        sys.path.insert(0, info["dir"])
    modname = "va_plugin_" + info["name"]
    path = os.path.join(info["dir"], info["entry"])
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def load_module(plugin_name: str, module_name: str, base=None):
    """Import one private module from an installed plugin (for selftests and
    other out-of-band consumers). Returns the module or None."""
    for info in discover(base):
        if info["name"] == plugin_name and not info["error"]:
            if info["dir"] not in sys.path:
                sys.path.insert(0, info["dir"])
            try:
                return importlib.import_module(module_name)
            except Exception:  # noqa: BLE001
                return None
    return None


# --- GUI loading --------------------------------------------------------------

def load_gui(api, base=None) -> list:
    """Import every enabled plugin and call register(api). Returns
    [{name, ok, error}] - a broken plugin never takes the app down."""
    report = []
    for info in discover(base):
        if not info["enabled"]:
            report.append({"name": info["name"], "ok": None, "error": "disabled"})
            continue
        if info["error"]:
            report.append({"name": info["name"], "ok": False, "error": info["error"]})
            continue
        try:
            mod = _import_entry(info)
            api._begin(info["name"])
            if hasattr(mod, "register"):
                mod.register(api)
            report.append({"name": info["name"], "ok": True, "error": None})
        except Exception as exc:  # noqa: BLE001
            api._abort(info["name"])
            report.append({"name": info["name"], "ok": False,
                           "error": "%s: %s" % (type(exc).__name__, exc)})
    return report


def unload_gui(api):
    """Destroy every widget plugins created (for reload / disable-apply)."""
    api._unload_all()


# --- Headless loading (analyze.py, selftests) ---------------------------------

class HeadlessApi:
    """What plugins may extend without a GUI: QC profiles and per-profile
    analysis passes (extra deep pass + ctx contribution + report writer)."""

    api_version = API_VERSION

    def __init__(self):
        self.passes = {}
        self.errors = []

    def register_qc_profile(self, name, checks):
        import va_qc
        va_qc.register_profile(name, checks)

    def extend_qc_profile(self, profile, checks, owner="plugin"):
        """Append checks to an EXISTING profile (core or another plugin's).
        Re-registration under the same owner replaces, so reloads are safe."""
        import va_qc
        va_qc.extend_profile(profile, checks, owner=owner)

    def provide(self, name, obj):
        provide_service(name, obj)

    def require(self, name, default=None):
        return require_service(name, default)

    def register_profile_pass(self, profile, run, echo="plugin pass",
                              ctx=None, json_extract=None, render=None,
                              suffix=".txt", needs_forensics=True,
                              ctx_key="verify", always=False):
        """run(path, forensics=, cadence=) -> rep; ctx(rep) -> the value
        stored at ctx[ctx_key] for QC checks; json_extract(rep) -> JSON-safe
        dict; render(rep) -> text written as <name><suffix>. always=True runs
        the pass on every analyze.py invocation regardless of the chosen
        profile (several plugin passes can coexist when their ctx_keys
        differ)."""
        self.passes[str(profile)] = {
            "run": run, "echo": str(echo), "ctx": ctx,
            "json": json_extract, "render": render, "suffix": str(suffix),
            "needs_forensics": bool(needs_forensics),
            "ctx_key": str(ctx_key), "always": bool(always)}


def load_headless(base=None) -> HeadlessApi:
    """Import enabled plugins, call register_headless(api), return the hooks.
    Never raises; failures land in api.errors."""
    api = HeadlessApi()
    for info in discover(base):
        if not info["enabled"] or info["error"]:
            if info["error"]:
                api.errors.append("%s: %s" % (info["name"], info["error"]))
            continue
        try:
            mod = _import_entry(info)
            if hasattr(mod, "register_headless"):
                mod.register_headless(api)
        except Exception as exc:  # noqa: BLE001
            api.errors.append("%s: %s: %s" % (info["name"],
                                              type(exc).__name__, exc))
    return api


# --- Install / registry --------------------------------------------------------

def _safe_extract(zf: zipfile.ZipFile, dest: str):
    for m in zf.namelist():
        p = os.path.normpath(m)
        if p.startswith("..") or os.path.isabs(p) or ":" in p.split(os.sep)[0][1:2]:
            raise ValueError("unsafe path in zip: %r" % m)
    zf.extractall(dest)


def install_zip(zip_path: str, base=None) -> dict:
    """Install a plugin zip into plugins/<name>. The zip may contain the
    plugin folder itself or its contents (plugin.json at root). Replaces any
    existing install of the same name. Returns the manifest."""
    tmp = tempfile.mkdtemp(prefix="va_plug_")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            _safe_extract(zf, tmp)
        src = tmp
        if not os.path.isfile(os.path.join(src, "plugin.json")):
            subs = [d for d in os.listdir(tmp)
                    if os.path.isdir(os.path.join(tmp, d))]
            if len(subs) == 1 and os.path.isfile(
                    os.path.join(tmp, subs[0], "plugin.json")):
                src = os.path.join(tmp, subs[0])
            else:
                raise ValueError("no plugin.json found in zip")
        m = _read_manifest(src)
        if int(m.get("api", 1)) > API_VERSION:
            raise ValueError("plugin needs API v%s; app provides v%d"
                             % (m.get("api"), API_VERSION))
        dest = os.path.join(plugins_dir(base), m["name"])
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        shutil.move(src, dest)
        return m
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fetch_registry(url=None, timeout=30) -> list:
    """Download and validate the plugin registry JSON:
    [{name,title,version,description,zip_url,sha256?}, ...]"""
    import va_tools
    url = url or registry_url()
    if not url:
        raise ValueError("no registry URL configured")
    data = va_tools._http_get(url, timeout=timeout)
    reg = json.loads(data.decode("utf-8", "replace"))
    if not isinstance(reg, list):
        raise ValueError("registry must be a JSON list")
    out = []
    for e in reg:
        if isinstance(e, dict) and e.get("name") and e.get("zip_url"):
            e.setdefault("title", e["name"])
            e.setdefault("version", "?")
            e.setdefault("description", "")
            out.append(e)
    return out


def download_plugin(entry: dict, base=None, on_progress=None) -> dict:
    """Download a registry entry's zip (sha256-checked when provided) and
    install it. Returns the manifest."""
    import va_tools
    say = on_progress or (lambda m: None)
    say("downloading %s..." % entry.get("title", entry["name"]))
    data = va_tools._http_get(entry["zip_url"], timeout=600,
                              on_progress=on_progress,
                              label="downloading %s" % entry["name"])
    want = (entry.get("sha256") or "").lower().strip()
    if want:
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise ValueError("sha256 mismatch: expected %s got %s"
                             % (want[:16], got[:16]))
    fd, tmp = tempfile.mkstemp(suffix=".zip", prefix="va_plug_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        say("installing...")
        return install_zip(tmp, base)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# --- The GUI-side facade --------------------------------------------------------

class _CascadeRef:
    """Adapter that lets the app's enable/disable button loops treat a
    Plugins-menu cascade entry like a widget: .config(state=...) flips the
    entry's state (entries are addressed by their unique label)."""

    def __init__(self, menu, label):
        self._menu = menu
        self._label = label

    def config(self, state=None, **_kw):
        if state is not None:
            try:
                self._menu.entryconfig(self._label, state=state)
            except Exception:  # noqa: BLE001 - entry may already be torn down
                pass


class AppApi:
    """Stable surface plugins program against (v1). Thin delegates onto the
    app; every widget a plugin creates is tracked by owner so disable/reload
    can cleanly tear it down."""

    api_version = API_VERSION

    def __init__(self, app):
        self._app = app
        self._owner = None
        self._owned = {}          # name -> {"buttons": [..], "hooks": [..]}

    # lifecycle (framework use)
    def _begin(self, name):
        self._owner = name
        self._owned.setdefault(name, {"buttons": [], "hooks": [],
                                      "services": [], "series": [],
                                      "overlays": [], "keys": [],
                                      "analyze_hooks": []})

    def _abort(self, name):
        self._teardown(name)

    def _teardown(self, name):
        own = self._owned.pop(name, None)
        if not own:
            return
        app = self._app
        for ref, label, menu in own["buttons"]:
            try:
                if menu in app._menus:
                    app._menus.remove(menu)
                if ref in app.plugin_buttons:
                    app.plugin_buttons.remove(ref)
                app.plugins_menu.delete(label)
            except Exception:  # noqa: BLE001
                pass
        for h in own["hooks"]:
            try:
                app.plugin_open_hooks.remove(h)
            except ValueError:
                pass
        for h in own.get("analyze_hooks", ()):
            try:
                app.plugin_analyze_hooks.remove(h)
            except ValueError:
                pass
        for n in own.get("services", ()):
            SERVICES.pop(n, None)
        for n in own.get("overlays", ()):
            app.plugin_overlays.pop(n, None)
        for n in own.get("keys", ()):
            app.plugin_keys.pop(n, None)
        if own.get("series"):
            for n in own["series"]:
                app.plugin_series.pop(n, None)
            try:
                app._refresh_metric_choices()
            except Exception:  # noqa: BLE001
                pass
        try:
            import va_qc
            va_qc.retract_extensions(name)
        except Exception:  # noqa: BLE001
            pass

    def _unload_all(self):
        for name in list(self._owned):
            self._teardown(name)

    # --- properties -----------------------------------------------------
    @property
    def root(self):
        return self._app.root

    @property
    def path(self):
        return self._app.path

    @property
    def fps(self):
        return self._app.fps

    @property
    def total_frames(self):
        return self._app.total_frames

    @property
    def cur_index(self):
        return self._app.cur_index

    @property
    def mono_font(self):
        return self._app.mono_font

    def closing(self) -> bool:
        return bool(self._app._closing)

    # --- plugin menus -------------------------------------------------------
    def add_toolbar_menu(self, label, needs_file=True):
        """A submenu (cascade) under the toolbar's Plugins ▾ menu, laid out
        like the Advanced ▾ submenus. Returns the menu; any trailing "▾" in
        the label is dropped (cascades draw their own arrow)."""
        import tkinter as tk
        app = self._app
        label = str(label).replace("▾", "").strip()
        menu = tk.Menu(app.plugins_menu, tearoff=0)
        va_theme.style_menu(menu)
        app.plugins_menu.add_cascade(label=label, menu=menu)
        ref = _CascadeRef(app.plugins_menu, label)
        if needs_file:
            ref.config(state=tk.NORMAL if app.path else tk.DISABLED)
            app.plugin_buttons.append(ref)
        app._menus.append(menu)
        self._owned[self._owner]["buttons"].append((ref, label, menu))
        return menu

    def add_command(self, menu, label, command):
        menu.add_command(label=label, command=command)

    def add_separator(self, menu):
        menu.add_separator()

    # --- app services -----------------------------------------------------
    def run_bg(self, fn, *args):
        self._app._run_bg(fn, *args)

    def post(self, fn, *args):
        self._app._post(fn, *args)

    def status(self, text, kind="info"):
        from va_theme import C
        col = {"info": C["muted"], "ok": C["ok"], "warn": C["warn"],
               "err": C["err"]}.get(kind, C["muted"])
        self._app.lbl_status.config(text=text, foreground=col)

    def current_frame(self):
        """(index, BGR frame copy | None) at the playhead."""
        app = self._app
        with app.frame_lock:
            fr = None if app.current_bgr is None else app.current_bgr.copy()
        return app.cur_index, fr

    def goto_frame(self, idx):
        self._app._goto_frame(idx)

    def add_marks(self, prefix, marks):
        """Replace this prefix's timeline/issue marks: [(t_seconds, label)].
        Labels get the prefix prepended so each tool owns its own marks."""
        app = self._app
        pre = str(prefix)
        app.scan_marks = ([m for m in app.scan_marks
                           if not m[1].startswith(pre)]
                          + [(float(t), pre + ": " + str(lab))
                             for t, lab in marks])
        app._refresh_scan_issues()

    def adv_store(self, key, title, text=None, data=None, img=None):
        self._app._adv_store(key, title, text=text, data=data, img=img)

    def text_popup(self, title, text):
        self._app._text_popup(title, text)

    def image_popup(self, title, rgb):
        self._app._image_popup(title, rgb)

    def run_qc(self, profile="general"):
        self._app._run_qc(profile)

    def on_open(self, callback):
        """callback(path) runs on the Tk thread when a new file opens."""
        self._app.plugin_open_hooks.append(callback)
        self._owned[self._owner]["hooks"].append(callback)

    def on_analyze(self, callback):
        """callback(path) runs on the Tk thread when the Analyze pass lands
        (events/table are populated; add series here and they draw at once)."""
        self._app.plugin_analyze_hooks.append(callback)
        self._owned[self._owner]["analyze_hooks"].append(callback)

    @property
    def events(self):
        """The Analyze pass event dict ({} before Analyze)."""
        return self._app.events or {}

    # --- inter-plugin services --------------------------------------------
    def provide(self, name, obj):
        """Publish a callable/object other plugins can require()."""
        provide_service(name, obj)
        self._owned[self._owner]["services"].append(str(name))

    def require(self, name, default=None):
        return require_service(name, default)

    # --- timeline series ----------------------------------------------------
    def add_series(self, label, t, v, vmin=None, vmax=None):
        """Put a per-time curve on the metrics timeline: it joins the metric
        selector under `label`. t = seconds, v = values (None gaps allowed).
        Re-adding a label replaces it; series clear when a new file opens."""
        app = self._app
        label = str(label)
        app.plugin_series[label] = {"t": [float(x) for x in t],
                                    "v": list(v), "vmin": vmin, "vmax": vmax}
        if label not in self._owned[self._owner]["series"]:
            self._owned[self._owner]["series"].append(label)
        app._refresh_metric_choices()
        app._draw_timeline()

    def remove_series(self, label):
        self._app.plugin_series.pop(str(label), None)
        self._app._refresh_metric_choices()
        self._app._draw_timeline()

    # --- preview overlays ----------------------------------------------------
    def add_overlay(self, name, fn):
        """fn(frame_bgr, frame_index) draws on (or returns a replacement for)
        a COPY of the display frame. Runs on the render path - keep it cheap
        and thread-tolerant (pure numpy/cv2). Replace by re-adding the name."""
        app = self._app
        app.plugin_overlays[str(name)] = fn
        if str(name) not in self._owned[self._owner]["overlays"]:
            self._owned[self._owner]["overlays"].append(str(name))
        app._refresh_preview()

    def remove_overlay(self, name):
        self._app.plugin_overlays.pop(str(name), None)
        self._app._refresh_preview()

    # --- shortcuts ------------------------------------------------------------
    def bind_key(self, sequence, fn, description=""):
        """Register a hotkey (Tk sequence like "<t>"). Returns False if the
        key is reserved by the core app or another plugin. Bindings respect
        the app's text-entry guard and die with the plugin."""
        app = self._app
        seq = str(sequence)
        if seq in getattr(app, "RESERVED_KEYS", ()) or seq in app.plugin_keys:
            return False
        app.plugin_keys[seq] = fn
        app.root.bind(seq, app._hk(
            lambda s=seq: app.plugin_keys.get(s, lambda: None)()))
        self._owned[self._owner]["keys"].append(seq)
        return True

    # --- QC --------------------------------------------------------------------
    def extend_qc_profile(self, profile, checks):
        """Append checks to an existing QC profile; owner-tracked, so plugin
        disable/reload retracts them and re-registration never duplicates."""
        import va_qc
        va_qc.extend_profile(profile, checks, owner=self._owner)
