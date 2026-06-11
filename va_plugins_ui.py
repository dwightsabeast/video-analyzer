"""Plugins manager dialog - the GUI side of va_plugins.

Lists installed plugins (enable/disable/reload), installs from a local zip,
and browses/downloads from a JSON registry (URL persisted as the
"plugin_registry_url" UI preference; format in plugins/registry.example.json).
Plugins execute in-process with the app's full permissions, so installation
always confirms trust with the user first."""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import va_plugins
import va_theme


def clr(key):
    return va_theme.C[key]


TRUST = ("Plugins are ordinary Python code that runs inside the app with "
         "your full user permissions.\n\nInstall only plugins you trust. "
         "Continue?")


class PluginManagerDialog:
    def __init__(self, app):
        self.app = app
        self.win = tk.Toplevel(app.root)
        self.win.title("Plugins")
        self.win.geometry("860x560")
        self.win.configure(bg=clr("bg"))

        ttk.Label(self.win, text="INSTALLED",
                  foreground=clr("muted")).pack(anchor="w", padx=10,
                                                pady=(10, 2))
        self.tree = ttk.Treeview(self.win, height=6, show="headings",
                                 columns=("plugin", "version", "status", "desc"))
        for col, w, txt in (("plugin", 150, "plugin"), ("version", 70, "version"),
                            ("status", 90, "status"), ("desc", 480, "description")):
            self.tree.heading(col, text=txt)
            self.tree.column(col, width=w, anchor="w")
        self.tree.pack(fill=tk.X, padx=10)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        row = ttk.Frame(self.win)
        row.pack(fill=tk.X, padx=10, pady=6)
        ttk.Button(row, text="Enable / disable",
                   command=self._toggle).pack(side=tk.LEFT)
        ttk.Button(row, text="Reload plugins",
                   command=self._reload).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Install from zip...",
                   command=self._install_zip).pack(side=tk.LEFT, padx=(18, 0))
        ttk.Button(row, text="Open plugins folder",
                   command=self._open_folder).pack(side=tk.LEFT, padx=6)

        reg = ttk.Frame(self.win)
        reg.pack(fill=tk.X, padx=10, pady=(10, 2))
        ttk.Label(reg, text="REGISTRY",
                  foreground=clr("muted")).pack(side=tk.LEFT)
        self.url_var = tk.StringVar(value=va_plugins.registry_url())
        ttk.Entry(reg, textvariable=self.url_var).pack(side=tk.LEFT, padx=8,
                                                       fill=tk.X, expand=True)
        ttk.Button(reg, text="Fetch list",
                   command=self._fetch).pack(side=tk.LEFT)
        self.rtree = ttk.Treeview(self.win, height=5, show="headings",
                                  columns=("plugin", "version", "desc"))
        for col, w, txt in (("plugin", 150, "plugin"), ("version", 70, "version"),
                            ("desc", 570, "description")):
            self.rtree.heading(col, text=txt)
            self.rtree.column(col, width=w, anchor="w")
        self.rtree.pack(fill=tk.X, padx=10, pady=(4, 0))
        self.rtree.bind("<<TreeviewSelect>>", self._on_rtree_select)
        row2 = ttk.Frame(self.win)
        row2.pack(fill=tk.X, padx=10, pady=6)
        ttk.Button(row2, text="Download selected",
                   command=self._download).pack(side=tk.LEFT)
        ttk.Label(row2, text="registry format: plugins/registry.example.json",
                  foreground=clr("muted")).pack(side=tk.RIGHT)

        self.out = scrolledtext.ScrolledText(self.win, height=7,
                                             font=app.mono_font, wrap=tk.WORD)
        self.out.configure(bg=clr("editor_bg"), fg=clr("editor_fg"),
                           insertbackground=clr("editor_fg"), relief="flat")
        self.out.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2, 10))
        self._registry = []
        self._installed = {}
        self._say("Plugins extend the core forensics tool with domain "
                  "workbenches (the speedrun pack lives here). They are "
                  "Python code running with your full permissions - install "
                  "only what you trust.")
        self._say("Select any plugin to read its full description here.")
        self._refresh()

    # --- helpers ---------------------------------------------------------
    def _say(self, msg):
        self.out.insert(tk.END, msg.rstrip() + "\n")
        self.out.see(tk.END)

    def _refresh(self):
        self.tree.delete(*self.tree.get_children())
        self._installed = {}
        for p in va_plugins.discover():
            self._installed[p["name"]] = p
            status = ("error" if p["error"] else
                      "enabled" if p["enabled"] else "disabled")
            self.tree.insert("", tk.END, iid=p["name"],
                             values=(p["title"], p["version"], status,
                                     p["error"] or p["description"]))

    def _detail(self, head, body):
        """Show a full, wrapped detail block in the bottom box (the tree
        columns truncate; this is where the whole description lives)."""
        self.out.insert(tk.END, "\n--- %s ---\n%s\n" % (head, body))
        self.out.see(tk.END)

    def _on_tree_select(self, _event=None):
        name = self._selected()
        p = self._installed.get(name) if name else None
        if not p:
            return
        status = ("ERROR" if p["error"] else
                  "enabled" if p["enabled"] else "disabled")
        head = "%s  v%s  [%s]" % (p["title"], p["version"], status)
        body = p["error"] or p["description"] or "(no description)"
        self._detail(head, body)

    def _on_rtree_select(self, _event=None):
        sel = self.rtree.selection()
        if not sel:
            return
        e = next((x for x in self._registry if x["name"] == sel[0]), None)
        if not e:
            return
        self._detail("%s  v%s  (registry)" % (e["title"], e["version"]),
                     e.get("description") or "(no description)")

    def _selected(self):
        sel = self.tree.selection()
        return sel[0] if sel else None

    # --- actions ---------------------------------------------------------
    def _toggle(self):
        name = self._selected()
        if not name:
            self._say("select an installed plugin first")
            return
        cur = {p["name"]: p["enabled"] for p in va_plugins.discover()}
        va_plugins.set_enabled(name, not cur.get(name, True))
        self._refresh()
        self._say("%s %s - Reload plugins to apply"
                  % (name, "disabled" if cur.get(name, True) else "enabled"))

    def _reload(self):
        app = self.app
        va_plugins.unload_gui(app.plugin_api)
        report = va_plugins.load_gui(app.plugin_api)
        app.plugin_report = report
        for r in report:
            self._say("reload: %-14s %s" % (r["name"],
                      "ok" if r["ok"] else (r["error"] or "?")))
        self._refresh()

    def _install_zip(self):
        if not messagebox.askokcancel("Install plugin", TRUST,
                                      parent=self.win):
            return
        fn = filedialog.askopenfilename(parent=self.win,
                                        title="Plugin zip",
                                        filetypes=[("zip", "*.zip")])
        if not fn:
            return
        try:
            m = va_plugins.install_zip(fn)
        except Exception as exc:  # noqa: BLE001
            self._say("install failed: %s" % exc)
            return
        self._say("installed %s %s" % (m["name"], m["version"]))
        self._reload()

    def _open_folder(self):
        d = va_plugins.plugins_dir()
        try:
            if os.name == "nt":
                os.startfile(d)                          # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", d])            # console-ok (not Windows)
            else:
                subprocess.Popen(["xdg-open", d])        # console-ok (not Windows)
        except OSError as exc:
            self._say("could not open %s: %s" % (d, exc))

    def _fetch(self):
        url = self.url_var.get().strip()
        va_plugins.set_registry_url(url)
        if not url:
            self._say("no registry URL set - paste one and Fetch again "
                      "(see plugins/registry.example.json for the format)")
            return
        self._say("fetching %s ..." % url)
        app = self.app

        def worker():
            try:
                reg = va_plugins.fetch_registry(url)
                app._post(self._fetched, reg, None)
            except Exception as exc:  # noqa: BLE001
                app._post(self._fetched, [], str(exc))

        app._run_bg(worker)

    def _fetched(self, reg, err):
        if not self.win.winfo_exists():
            return
        if err:
            self._say("registry fetch failed: %s" % err)
            return
        self._registry = reg
        self.rtree.delete(*self.rtree.get_children())
        for e in reg:
            self.rtree.insert("", tk.END, iid=e["name"],
                              values=(e["title"], e["version"],
                                      e["description"]))
        self._say("%d plugin(s) in registry" % len(reg))

    def _download(self):
        sel = self.rtree.selection()
        if not sel:
            self._say("select a registry plugin first")
            return
        entry = next((e for e in self._registry if e["name"] == sel[0]), None)
        if entry is None:
            return
        if not messagebox.askokcancel("Install plugin", TRUST,
                                      parent=self.win):
            return
        app = self.app

        def worker():
            try:
                m = va_plugins.download_plugin(
                    entry, on_progress=lambda msg: app._post(
                        lambda msg=msg: self.win.winfo_exists()
                        and self._say(msg)))
                app._post(self._downloaded, m, None)
            except Exception as exc:  # noqa: BLE001
                app._post(self._downloaded, None, str(exc))

        app._run_bg(worker)

    def _downloaded(self, m, err):
        if not self.win.winfo_exists():
            return
        if err:
            self._say("download failed: %s" % err)
            return
        self._say("installed %s %s" % (m["name"], m["version"]))
        self._reload()
