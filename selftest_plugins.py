"""Headless self-test for the plugin framework (va_plugins): discovery,
manifests, enable/disable persistence, zip install (incl. hostile zips),
headless loading, registry validation, and the analyze.py unknown-profile
guard. UI prefs are monkeypatched in-memory so running this never touches
your real settings. Run: python selftest_plugins.py"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile

import va_theme
import va_plugins

FAILS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, cond):
    print("  [%s] %s" % ("ok" if cond else "FAIL", name))
    if not cond:
        FAILS.append(name)


def _mem_prefs():
    store = {}
    va_theme.load_ui_key = lambda k, d=None: store.get(k, d)
    va_theme.save_ui_key = lambda k, v: store.__setitem__(k, v)
    return store


def _mk_plugin(base, name, body):
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=True)
    json.dump({"name": name, "title": name, "version": "0.1",
               "description": "test", "entry": "plugin.py", "api": 1},
              open(os.path.join(d, "plugin.json"), "w"))
    open(os.path.join(d, "plugin.py"), "w").write(body)
    return d


def t_discover_and_toggle(tmp):
    print("discovery / enable persistence:")
    base = os.path.join(tmp, "app1")
    os.makedirs(base)
    _mk_plugin(os.path.join(base, "plugins"), "alpha", "def register(api):\n    pass\n")
    bad = os.path.join(base, "plugins", "broken")
    os.makedirs(bad)
    open(os.path.join(bad, "plugin.json"), "w").write("{not json")
    infos = {p["name"]: p for p in va_plugins.discover(base)}
    check("finds plugin", "alpha" in infos and infos["alpha"]["enabled"])
    check("broken manifest isolated",
          "broken" in infos and infos["broken"]["error"] is not None)
    va_plugins.set_enabled("alpha", False)
    infos = {p["name"]: p for p in va_plugins.discover(base)}
    check("disable persists", not infos["alpha"]["enabled"])
    va_plugins.set_enabled("alpha", True)
    infos = {p["name"]: p for p in va_plugins.discover(base)}
    check("re-enable persists", infos["alpha"]["enabled"])


def t_headless_load(tmp):
    print("headless loading:")
    base = os.path.join(tmp, "app2")
    _mk_plugin(os.path.join(base, "plugins"), "prof", (
        "def register_headless(api):\n"
        "    api.register_qc_profile('zz_test', [])\n"
        "    api.register_profile_pass('zz_test', run=lambda p, forensics=None,"
        " cadence=None: {'x': 1}, echo='zz pass', suffix='.zz.txt')\n"))
    _mk_plugin(os.path.join(base, "plugins"), "crasher",
               "raise RuntimeError('boom')\n")
    hooks = va_plugins.load_headless(base)
    import va_qc
    check("profile registered", "zz_test" in va_qc.profile_names())
    check("pass registered", "zz_test" in hooks.passes
          and hooks.passes["zz_test"]["echo"] == "zz pass")
    check("crashing plugin isolated",
          any("crasher" in e and "boom" in e for e in hooks.errors))
    va_qc.PROFILES.pop("zz_test", None)


def t_zip_install(tmp):
    print("zip install:")
    base = os.path.join(tmp, "app3")
    os.makedirs(os.path.join(base, "plugins"))
    src = _mk_plugin(tmp, "zippy", "def register(api):\n    pass\n")
    zp = os.path.join(tmp, "zippy.zip")
    with zipfile.ZipFile(zp, "w") as zf:
        for fn in os.listdir(src):
            zf.write(os.path.join(src, fn), "zippy/" + fn)
    m = va_plugins.install_zip(zp, base)
    check("zip installs", m["name"] == "zippy" and os.path.isfile(
        os.path.join(base, "plugins", "zippy", "plugin.json")))
    m2 = va_plugins.install_zip(zp, base)          # replace existing
    check("zip reinstall replaces", m2["name"] == "zippy")
    evil = os.path.join(tmp, "evil.zip")
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../evil.py", "print('escaped')")
    try:
        va_plugins.install_zip(evil, base)
        check("hostile zip rejected", False)
    except (ValueError, Exception) as exc:  # noqa: BLE001
        check("hostile zip rejected", "unsafe" in str(exc) or "plugin.json" in str(exc))


def t_registry(tmp):
    print("registry:")
    import va_tools
    real = va_tools._http_get
    payload = json.dumps([
        {"name": "good", "title": "Good", "version": "1", "zip_url": "http://x/g.zip"},
        {"bogus": True},
        {"name": "noزip"},
    ]).encode()
    va_tools._http_get = lambda url, **kw: payload
    try:
        reg = va_plugins.fetch_registry("http://example/registry.json")
        check("registry validated", [e["name"] for e in reg] == ["good"])
    finally:
        va_tools._http_get = real
    real_ru = va_plugins.registry_url   # neutralise the shipped default +
    va_plugins.registry_url = lambda: ""  # any saved URL: "" must still raise
    try:
        va_plugins.fetch_registry("")
        check("empty url rejected", False)
    except ValueError:
        check("empty url rejected", True)
    finally:
        va_plugins.registry_url = real_ru


def t_cascade_api():
    """AppApi cascade lifecycle against a fake tkinter (no display needed)."""
    print("AppApi cascade lifecycle:")
    import types

    class FakeMenu:
        def __init__(self, *a, **k):
            self.entries = []
            self.states = {}

        def add_cascade(self, label=None, menu=None):
            self.entries.append((label, menu))

        def add_command(self, label=None, command=None):
            self.entries.append((label, None))

        def add_separator(self):
            self.entries.append(("---", None))

        def entryconfig(self, label, state=None):
            self.states[label] = state

        def delete(self, label):
            self.entries = [(l, m) for l, m in self.entries if l != label]

        def configure(self, **kw):
            pass

    real_tk = sys.modules.get("tkinter")
    fake_tk = types.ModuleType("tkinter")
    fake_tk.Menu = FakeMenu
    fake_tk.NORMAL = "normal"
    fake_tk.DISABLED = "disabled"
    sys.modules["tkinter"] = fake_tk
    try:
        class FakeApp:
            def __init__(self):
                self.plugins_menu = FakeMenu()
                self.plugins_menu.add_command(label="Manage plugins...")
                self.plugins_menu.add_separator()
                self._menus = [self.plugins_menu]
                self.plugin_buttons = []
                self.plugin_open_hooks = []
                self.path = None

        app = FakeApp()
        api = va_plugins.AppApi(app)
        api._begin("p1")
        m = api.add_toolbar_menu("Pack One ▾")
        api.add_command(m, "Do thing...", lambda: None)
        api._begin("p2")
        api.add_toolbar_menu("Pack Two ▾")
        labels = [l for l, _m in app.plugins_menu.entries]
        check("cascades appended (arrow stripped)",
              labels == ["Manage plugins...", "---", "Pack One", "Pack Two"])
        check("needs_file starts disabled",
              app.plugins_menu.states.get("Pack One") == "disabled")
        for b in app.plugin_buttons:
            b.config(state="normal")
        check("state loop enables cascades",
              app.plugins_menu.states.get("Pack Two") == "normal")
        api._teardown("p1")
        labels = [l for l, _m in app.plugins_menu.entries]
        check("teardown removes one cascade",
              labels == ["Manage plugins...", "---", "Pack Two"]
              and len(app.plugin_buttons) == 1)
        api._unload_all()
        check("unload_all leaves core entries",
              [l for l, _m in app.plugins_menu.entries]
              == ["Manage plugins...", "---"] and not app.plugin_buttons)
    finally:
        if real_tk is not None:
            sys.modules["tkinter"] = real_tk
        else:
            sys.modules.pop("tkinter", None)


def t_v2_api():
    """API v2 surfaces against a fake tkinter: series, overlays, keys,
    services, analyze hooks, QC extension - incl. teardown."""
    print("AppApi v2 surfaces:")
    import types

    class FakeRoot:
        def __init__(self):
            self.bound = {}

        def bind(self, seq, fn):
            self.bound[seq] = fn

    class FakeApp:
        RESERVED_KEYS = frozenset(("<space>", "<n>"))

        def __init__(self):
            self.plugins_menu = types.SimpleNamespace(
                add_cascade=lambda **k: None, entryconfig=lambda *a, **k: None,
                delete=lambda *a: None)
            self._menus = []
            self.plugin_buttons = []
            self.plugin_open_hooks = []
            self.plugin_analyze_hooks = []
            self.plugin_series = {}
            self.plugin_overlays = {}
            self.plugin_keys = {}
            self.path = "x.mp4"
            self.events = {"black": []}
            self.root = FakeRoot()
            self.refreshes = 0

        def _refresh_metric_choices(self):
            self.refreshes += 1

        def _draw_timeline(self):
            pass

        def _refresh_preview(self):
            pass

        def _hk(self, fn):
            return fn

    real_tk = sys.modules.get("tkinter")
    fake_tk = types.ModuleType("tkinter")
    fake_tk.Menu = lambda *a, **k: types.SimpleNamespace(
        add_cascade=lambda **kw: None, configure=lambda **kw: None)
    fake_tk.NORMAL, fake_tk.DISABLED = "normal", "disabled"
    sys.modules["tkinter"] = fake_tk
    try:
        app = FakeApp()
        api = va_plugins.AppApi(app)
        api._begin("vt")
        api.add_series("Curve A", [0, 1, 2], [5, 6, 7], vmin=0)
        check("series stored + selector refreshed",
              "Curve A" in app.plugin_series and app.refreshes == 1)
        api.add_overlay("box", lambda f, i: f)
        check("overlay stored", "box" in app.plugin_overlays)
        check("reserved key refused",
              api.bind_key("<space>", lambda: None) is False)
        hits = []
        check("key bound", api.bind_key("<t>", lambda: hits.append(1)) is True)
        app.root.bound["<t>"]()
        check("key dispatches", hits == [1])
        check("duplicate key refused",
              api.bind_key("<t>", lambda: None) is False)
        api.provide("vt.fn", lambda x: x + 1)
        check("service provide/require",
              api.require("vt.fn")(1) == 2 and api.require("nope", 9) == 9)
        seen = []
        api.on_analyze(lambda p: seen.append(p))
        app.plugin_analyze_hooks[0]("f.mp4")
        check("analyze hook registered", seen == ["f.mp4"])
        import va_qc
        api.extend_qc_profile("general",
                              [("vtx", "VT", lambda c: ("warn", "x"))])
        ids = [c["id"] for c in va_qc.evaluate({}, "general")["checks"]]
        check("profile extended", "vtx" in ids)
        api._teardown("vt")
        ids = [c["id"] for c in va_qc.evaluate({}, "general")["checks"]]
        check("teardown clears everything",
              not app.plugin_series and not app.plugin_overlays
              and not app.plugin_keys and not app.plugin_analyze_hooks
              and va_plugins.require_service("vt.fn") is None
              and "vtx" not in ids)
        app.root.bound["<t>"]()          # stale binding is a safe no-op
        check("stale key no-op", hits == [1])
    finally:
        if real_tk is not None:
            sys.modules["tkinter"] = real_tk
        else:
            sys.modules.pop("tkinter", None)


def t_cli_guard():
    print("analyze.py profile guard:")
    r = subprocess.run([sys.executable, os.path.join(HERE, "analyze.py"),
                        "nonexistent.mp4", "--profile", "no_such_profile"],
                       capture_output=True, text=True, timeout=120)
    check("unknown profile exits 2", r.returncode == 2)
    check("guard mentions plugins", "plugin" in (r.stderr + r.stdout).lower())
    r2 = subprocess.run([sys.executable, os.path.join(HERE, "analyze.py"),
                         "--list-profiles"],
                        capture_output=True, text=True, timeout=120)
    check("list-profiles includes plugin profile",
          "speedrun" in r2.stdout)


if __name__ == "__main__":
    _mem_prefs()
    with tempfile.TemporaryDirectory(prefix="va_st_plug_") as tmp:
        t_discover_and_toggle(tmp)
        t_headless_load(tmp)
        t_zip_install(tmp)
        t_registry(tmp)
    t_cascade_api()
    t_v2_api()
    t_cli_guard()
    print()
    if FAILS:
        print("FAILED: %d check(s): %s" % (len(FAILS), ", ".join(FAILS)))
        sys.exit(1)
    print("all plugin-framework checks passed")
