#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""THE CODE-CLOSURE WRAPPER -- binds a harness run to the exact code and data it ran on.

    python stack/scripts/closure_run.py --out <closure.json> --result <harness PASS json> \
        [--data <file> ...] [--data-root <dir> ...] [--binding] [--commit <sha>] [--tree <root>] \
        -- <script.py> <args ...>

SPEC_REFCV7 A11 (section 16, 2026-09-27): the launch gate's overfit records (G-MAP-OVERFIT,
G-BOX-OVERFIT) bind to a CODE CLOSURE that must be EQUAL at the launch commit, instead of to the
launch commit's sha -- so the binding GPU runs can start as soon as the MODEL code has landed. The
wrapped script runs UNCHANGED (`runpy.run_path(script, run_name="__main__")`, `sys.argv` =
[script, *args]); when it ends -- normally, through `SystemExit`, or by an uncaught exception --
this writes `--out`:

* `modules`: (tree-relative path, git blob) of EVERY module in `sys.modules` whose file lies inside
  the run's tree -- anywhere in it, so a helper loaded BY PATH from a Research Lab package is bound
  too -- plus the wrapped script and every tree `.py` file the process OPENED (source read through
  `exec`, `runpy`, ...). READ FROM THE PROCESS, never from a hand list;
* `probed`: (path, blob) of tree modules the import system was ASKED for that never ended up in
  `sys.modules` (an import that raised, a `find_spec` probe) -- their content can still decide the
  run;
* `absent`: every module NAME the import system was asked for that was never imported and that did
  not exist in the tree at the run (a guarded `try: import ...` that fell through). The gate
  refuses a launch tree on which any of them NOW resolves: the launch code would take a path the
  PASS never ran;
* `outside_tree`: modules imported from files that are in NEITHER the tree NOR the interpreter's
  own installation (a shadowing editable install, a side folder) -- the gate refuses any;
* `children`: processes the run spawned -- their imports are NOT in this record;
* `data`: (path, sha256) of every `--data` file AND every file OPENED under a `--data-root`
  (a `sys.addaudithook` on the `open` event): the frames, GT, prereg and payloads the run read;
* `closure_sha256` over (modules, probed, absent, data); `argv` + `argv_sha256`; the wrapped
  script's exit status (a crash still writes its record, with a non-zero status); the harness PASS
  JSON by path + sha256 (`--result`); `binding` (ONLY with `--binding` -- an early record is
  unusable by construction); torch / timm (distribution versions) + CUDA / cuDNN; the tree's
  archived commit (informative only).

stdlib only: this file imports nothing from the tree, so it never enters its own closure.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import platform
import runpy
import site
import sys
import sysconfig
import time
from pathlib import Path

SCHEMA = "tanitad.launch_gate.closure/1"
OWN_TOPS = ("tanitad", "taniteval")
#: where the tree's own modules resolve from, relative to the tree root (+ the script's own dir)
IMPORT_ROOTS = ("stack", "taniteval", "stack/scripts", "tools")
_CHILD_EVENTS = ("subprocess.Popen", "os.fork", "os.forkpty", "os.posix_spawn", "os.spawn",
                 "os.exec", "os.system", "os.startfile")


def git_blob(data: bytes) -> str:
    """`git hash-object` of the bytes (no filters): sha1("blob <n>\\0" + bytes)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256_file(p: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def closure_digest(modules: list, probed: list, absent: list, data: list) -> str:
    """ONE definition, mirrored by `launch_gate._closure_digest` (a test pins them together)."""
    return hashlib.sha256(canonical({"modules": modules, "probed": probed, "absent": absent,
                                     "data": data})).hexdigest()


def resolve_in_tree(tree: Path, name: str, extra_roots: tuple = ()) -> Path | None:
    """The tree file a module NAME resolves to through the tree's import roots, else None."""
    parts = [x for x in name.split(".") if x]
    if not parts:
        return None
    for base in [*(tree / r for r in IMPORT_ROOTS), *extra_roots]:
        q = base.joinpath(*parts)
        for cand in (q.parent / (q.name + ".py"), q / "__init__.py"):
            if cand.is_file():
                return cand
    return None


def dist_versions() -> dict:
    """torch / timm by DISTRIBUTION metadata (identical whether or not the run imported them),
    CUDA / cuDNN from torch. `launch_gate._host_env` uses the same definition."""
    out = {"python": platform.python_version(), "platform": sys.platform}
    try:
        from importlib import metadata
    except ImportError:                                   # pragma: no cover
        metadata = None
    for d in ("torch", "timm"):
        try:
            out[d] = metadata.version(d) if metadata else None
        except Exception:                                 # noqa: BLE001 -- absent => None
            out[d] = None
    # CUDA / cuDNN only from a torch the RUN imported (never imported here: a run without torch
    # records None, and the gate then reads a mismatch on any CUDA host -- loud, not silent)
    t = sys.modules.get("torch")
    out["cuda"] = getattr(getattr(t, "version", None), "cuda", None) if t is not None else None
    try:
        out["cudnn"] = t.backends.cudnn.version() if t is not None else None
    except Exception:                                     # noqa: BLE001
        out["cudnn"] = None
    return out


def _env_roots() -> list[Path]:
    """The interpreter's own installation: stdlib, prefixes, site-packages."""
    c = {sys.prefix, sys.base_prefix, sys.exec_prefix, getattr(sys, "base_exec_prefix", "")}
    try:
        c.update(site.getsitepackages())
    except Exception:                                     # noqa: BLE001
        pass
    try:
        c.add(site.getusersitepackages())
    except Exception:                                     # noqa: BLE001
        pass
    for k in ("stdlib", "platstdlib", "purelib", "platlib"):
        try:
            c.add(sysconfig.get_path(k))
        except Exception:                                 # noqa: BLE001
            pass
    out = []
    for x in c:
        if x:
            try:
                out += [Path(os.path.abspath(x)), Path(x).resolve()]
            except OSError:
                pass
    return list(dict.fromkeys(out))


def _under(p: Path, root: Path) -> bool:
    return p == root or root in p.parents


class _Lookups:
    """A meta-path OBSERVER: records every module NAME the import system is asked for, and finds
    nothing itself (it returns None, so the real finders answer)."""

    def __init__(self):
        self.names: set[str] = set()
        self.on = True

    def find_spec(self, fullname, path=None, target=None):  # noqa: D401 -- importlib protocol
        if self.on:
            self.names.add(fullname)
        return None


class Recorder:
    def __init__(self, a, script: Path, argv: list[str]):
        self.tree = Path(a.tree).resolve()
        self.out = Path(a.out)
        self.data = [Path(x) for x in a.data]
        self.roots = [Path(x).resolve() for x in a.data_root]
        self.result = Path(a.result) if a.result else None
        self.binding = bool(a.binding)
        self.commit = a.commit
        self.script = script
        self.argv = argv
        self.opened: set[str] = set()
        self.code_opened: set[str] = set()
        self.children: list[str] = []
        self.lookups = _Lookups()
        self.status = None
        self.error = None
        self.written = False
        self.t0 = time.time()

    def audit(self, event, args):
        if self.written:
            return
        if event in _CHILD_EVENTS or event.startswith("os.spawn") or event.startswith("os.exec"):
            if len(self.children) < 200:
                try:
                    self.children.append(f"{event}: {str(args[:2])[:240]}")
                except Exception:                         # noqa: BLE001
                    self.children.append(event)
            return
        if event != "open" or not args:
            return
        p = args[0]
        if isinstance(p, int):
            return
        try:
            rp = Path(os.fsdecode(p)).resolve()
        except (OSError, ValueError, TypeError):
            return
        if rp.suffix == ".py" and _under(rp, self.tree):
            self.code_opened.add(str(rp))
        for root in self.roots:
            if _under(rp, root):
                self.opened.add(str(rp))
                return

    def _rel(self, p: Path) -> str | None:
        try:
            return p.resolve().relative_to(self.tree).as_posix()
        except (ValueError, OSError):
            return None

    def _modules(self) -> tuple[list, list, list, list]:
        env_roots = _env_roots()
        me = Path(__file__).resolve()                     # the wrapper: never its own closure
        mods: dict[str, str] = {}
        outside = []
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None)
            if not f:
                continue
            p = Path(f)
            if p.suffix == ".pyc":                        # a cached module: record its SOURCE
                src = Path(str(p)[:-1])
                p = src if src.is_file() else p
            try:
                raw = Path(os.path.abspath(p))
                p = p.resolve()
            except OSError:
                continue
            if p == me:
                continue
            rel = self._rel(p)
            if rel is not None:
                if p.is_file():
                    mods[rel] = git_blob(p.read_bytes())
                continue
            # the interpreter's own files, judged on the path AS IMPORTED and AS RESOLVED: on
            # Debian /usr/lib/python3.X/sitecustomize.py is a symlink into /etc/python3.X/
            # (MEASURED 2026-09-27 on Thor)
            env = any(_under(q, r) for q in (raw, p) for r in env_roots)
            if name.split(".")[0] in OWN_TOPS or not env:
                outside.append({"module": name, "file": str(f)})
        for f in [self.script, *(Path(x) for x in sorted(self.code_opened))]:
            if f.resolve() == me:
                # the wrapper's own source, read back by linecache for a warning attributed to
                # its frame (MEASURED 2026-09-27 on Thor) -- never part of its own closure
                continue
            rel = self._rel(f)
            if rel is None:
                outside.append({"module": "__main__ (the wrapped script)", "file": str(f)})
            elif f.is_file():
                mods[rel] = git_blob(f.read_bytes())
        # the names the import system was asked for but that never became modules
        extra = (self.script.resolve().parent,)
        probed: dict[str, str] = {}
        absent: set[str] = set()
        for n in sorted(self.lookups.names - set(sys.modules)):
            hit = resolve_in_tree(self.tree, n, extra)
            if hit is None:
                absent.add(n)
            else:
                rel = self._rel(hit)
                if rel is not None and rel not in mods:
                    probed[rel] = git_blob(hit.read_bytes())
        return (sorted([k, v] for k, v in mods.items()), sorted([k, v] for k, v in probed.items()),
                sorted(absent), sorted(outside, key=lambda d: (d["module"], d["file"])))

    def write(self, status, error=None):
        if self.written:
            return
        self.lookups.on = False
        modules, probed, absent, outside = self._modules()
        self.written = True
        data: dict[str, str] = {}
        for p in [*self.data, *(Path(x) for x in sorted(self.opened))]:
            if p.is_file():
                data[str(p.resolve())] = sha256_file(p)
        data_l = sorted([k, v] for k, v in data.items())
        rec = {"schema": SCHEMA, "binding": self.binding, "tree": str(self.tree),
               "commit_informative": self.commit,
               "script": self._rel(self.script) or str(self.script),
               "argv": self.argv, "argv_sha256": hashlib.sha256(canonical(self.argv)).hexdigest(),
               "exit_status": status, "error": error,
               "modules": modules, "n_modules": len(modules), "probed": probed,
               "absent": absent, "outside_tree": outside, "children": self.children,
               "data": data_l, "n_data": len(data_l),
               "closure_sha256": closure_digest(modules, probed, absent, data_l),
               "result": ({"path": str(self.result), "sha256": sha256_file(self.result)}
                          if self.result is not None and self.result.is_file() else
                          {"path": str(self.result) if self.result else None, "sha256": None}),
               "env": dist_versions(), "elapsed_s": round(time.time() - self.t0, 1),
               "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out.with_suffix(self.out.suffix + ".tmp")
        tmp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, self.out)
        print(f"[closure_run] {len(modules)} modules, {len(probed)} probed, {len(absent)} absent "
              f"names, {len(data_l)} data files, exit {status} -> {self.out}", file=sys.stderr,
              flush=True)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("closure_run: usage: [options] -- <script.py> <args ...>")
    i = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--result", default=None, help="the harness's PASS JSON (bound by sha256)")
    ap.add_argument("--data", action="append", default=[])
    ap.add_argument("--data-root", action="append", default=[],
                    help="every file OPENED under this directory is recorded by sha256")
    ap.add_argument("--binding", action="store_true", help="a BINDING run (else: early)")
    ap.add_argument("--commit", default=None, help="the tree's archived commit (informative)")
    ap.add_argument("--tree", default=str(Path(__file__).resolve().parents[2]),
                    help="the run's tree root (default: this file's repo root)")
    a = ap.parse_args(argv[:i])
    rest = argv[i + 1:]
    if not rest:
        raise SystemExit("closure_run: no script after --")
    script = Path(rest[0])
    rec = Recorder(a, script, rest)
    sys.meta_path.insert(0, rec.lookups)
    sys.addaudithook(rec.audit)
    atexit.register(lambda: rec.write(rec.status if rec.status is not None else "unknown",
                                      rec.error or "the process ended outside the wrapper"))
    sys.argv = [str(script), *rest[1:]]
    # exactly `python <script>`: sys.path[0] is the SCRIPT's directory (it was this file's)
    here = Path(__file__).resolve().parent
    if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == here:
        sys.path[0] = str(script.resolve().parent)
    else:
        sys.path.insert(0, str(script.resolve().parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        rec.status = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        rec.write(rec.status, None if rec.status == 0 else f"SystemExit({e.code!r})")
        raise
    except BaseException as e:                            # noqa: BLE001 -- recorded, re-raised
        rec.status, rec.error = 1, f"{type(e).__name__}: {e}"[:600]
        rec.write(1, rec.error)
        raise
    rec.status = 0
    rec.write(0)


if __name__ == "__main__":
    main()
