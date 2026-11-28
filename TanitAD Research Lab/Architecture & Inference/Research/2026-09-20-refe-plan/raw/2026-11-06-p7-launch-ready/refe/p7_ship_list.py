#!/usr/bin/env python3
"""P7-SHIP (BACKLOG PKG-SHIP-COMPLETE): the files the refe_p7 G-SUITE READS, derived from the suite itself, checked against a package directory.

WHY.  A pod package built as a directory subset (code/ refe/ splits/) missed four files its own G-SUITE reads (`eval/rls1{b,d,e}_arms.json`,
`raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.md`).  The launch gate refused, correctly, after about 40 minutes of idle A40 (RESULT.md of 2026-10-22-p6-newpod-rebuild).
The ship step had a hand-kept directory list; nothing tied it to what the suite opens.

  derive  --pkg <FULL package root> --out <list.json> [--profile refe_p7] [--extra-script name:TOKEN ...] [--trace] [--trace-file opened.json]
          1. the suite: `launch_gate.PROFILES[profile]["suite"]` (+ any --extra-script, e.g. test_p7_chain.py:ZZP7CHAIN_PASS), read from the package's own launch_gate.py;
          2. STATIC scan: the AST of every suite script and, transitively, of every refe/ module it imports or names; every string constant and every `a / "b" / "c"` chain that RESOLVES to
             a file under the package root (or, for a bare name, under refe/) is a read candidate (an over-approximation: a shipped file too many is harmless, a missing one is a 40-minute loss);
          3. TRACED run (--trace): each suite entry is run with a `sys.addaudithook("open")` sitecustomize on PYTHONPATH (inherited by child pythons); every file opened for reading under the
             package root is recorded.  --trace-file merges a previous trace instead of re-running (the suite takes ~40 min); it is REFUSED when the closure changed since that trace was taken.
          Output: the files with sha256 + how found (static / trace) + which script reads them, and the sha256 of every script in the closure (the staleness binding).
  check   --pkg <package dir to SHIP> --list <list.json>
          REFUSES (rc 1, `ZZP7SHIP_BAD ...`) on: a listed file that is missing, a listed file whose bytes differ (CRLF-only differences are accepted), a suite script whose bytes differ from the ones the
          list was derived from (STALE: regenerate the list at ship time, as SHIP_MANIFEST.json is), a missing shipped root, a missing SHIP_MANIFEST.json.   rc 0: `ZZP7SHIP_OK files=N`.
The check needs no git and no full tree: it runs on the pod, as stage 0 of p7_pod.sh, in seconds, before any CPU-hour or GPU-minute is spent.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "refe.p7_ship_list/1"
SHIPPED_ROOTS = ("refe", "code", "splits")                 # == gate_token.TREE_ROOTS
REQUIRED_TOP = ("SHIP_MANIFEST.json",)
SKIP_PARTS = ("__pycache__", ".pytest_cache")

TRACE_SITE = '''
import os, sys
_out = os.environ.get("P7_TRACE_OUT"); _root = os.environ.get("P7_TRACE_ROOT", "")
if _out and _root:
    _nroot = os.path.normcase(os.path.abspath(_root))
    def _hook(event, args):
        if event != "open":
            return
        try:
            p, mode, flags = args
            if isinstance(p, (bytes, int)) or p is None:
                return
            m = "r" if mode is None else str(mode)
            if any(c in m for c in "wax+"):
                return
            if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT):
                return
            ap = os.path.abspath(str(p))
            if os.path.normcase(ap).startswith(_nroot + os.sep):
                with open(_out, "a", encoding="utf-8") as f:
                    f.write(ap + "\\n")
        except Exception:
            return
    sys.addaudithook(_hook)
'''


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_pair(p: Path) -> tuple:
    """(raw sha256, LF-normalised sha256): a CRLF checkout of an LF blob is not a different file."""
    b = p.read_bytes()
    return sha256_bytes(b), sha256_bytes(b.replace(b"\r\n", b"\n"))


def rel_of(root: Path, p: Path) -> str:
    return p.resolve().relative_to(root.resolve()).as_posix()


# ------------------------------------------------------------------------------------------------ the suite
def load_suite(pkg: Path, profile: str, extra: list) -> list:
    path = pkg / "refe" / "launch_gate.py"
    if not path.is_file():
        raise SystemExit(f"ZZP7SHIP_BAD {path} is missing")
    sys.path.insert(0, str(pkg / "refe"))
    spec = importlib.util.spec_from_file_location("_p7ship_launch_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prof = mod.PROFILES[profile]
    suite = [dict(e) for e in prof["suite"]]
    for x in extra:
        name, _, token = x.partition(":")
        suite.append({"script": name, "token": token or "ZZTEST_PASS"})
    return suite


# ------------------------------------------------------------------------------------------------ the static scan
def _leaves(node):
    """Left-to-right leaves of a `a / b / c` chain (a non-Div node is its own leaf)."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _leaves(node.left) + _leaves(node.right)
    return [node]


def candidates(src: str) -> set:
    tree = ast.parse(src)
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            s = n.value.strip()
            if s and "\n" not in s and " " not in s and len(s) < 260:
                out.add(s[2:] if s.startswith("./") else s)
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            lv = _leaves(n)
            consts = []
            for x in lv:
                if isinstance(x, ast.Constant) and isinstance(x.value, str):
                    consts.append(x.value.strip("/"))
                else:
                    consts = []                         # restart after any non-constant leaf: the anchor (HERE / PKG / tmp) is not part of the relative path
            if consts:
                out.add("/".join(consts))
    return out


def resolve(pkg: Path, cand: str):
    if not cand or cand.startswith(("/", "\\")) or ".." in cand.split("/") or "\\" in cand or ":" in cand or "*" in cand or "{" in cand:
        return None
    for base, need_slash in ((pkg, True), (pkg / "refe", False)):
        if need_slash and "/" not in cand:
            continue                                    # a bare name is searched only under refe/ (no accidental match of a top-level document)
        p = base / cand
        if p.is_file() and not any(part in SKIP_PARTS for part in p.parts):
            return p
    return None


def closure(pkg: Path, suite: list) -> dict:
    """{rel path: Path} of every suite script and every refe/ module they import or name, transitively."""
    refe = pkg / "refe"
    todo = [refe / e["script"] for e in suite if "script" in e]
    seen: dict = {}
    while todo:
        p = todo.pop()
        if not p.is_file():
            raise SystemExit(f"ZZP7SHIP_BAD suite script {p} is missing")
        rel = rel_of(pkg, p)
        if rel in seen:
            continue
        seen[rel] = p
        if p.suffix != ".py":
            continue
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src)
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                names |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                names.add(n.module.split(".")[0])
        for nm in names:
            if (refe / f"{nm}.py").is_file():
                todo.append(refe / f"{nm}.py")
        for c in candidates(src):
            q = refe / c
            if c.endswith((".py", ".sh")) and "/" not in c and q.is_file():
                todo.append(q)
    return seen


def closure_digest(scripts: dict) -> str:
    return sha256_bytes(json.dumps(scripts, sort_keys=True).encode())


def static_reads(pkg: Path, scripts: dict) -> dict:
    out: dict = {}
    for rel, p in scripts.items():
        if p.suffix != ".py":
            continue
        for c in candidates(p.read_text(encoding="utf-8")):
            q = resolve(pkg, c)
            if q is not None:
                out.setdefault(rel_of(pkg, q), set()).add(rel)
    return out


# ------------------------------------------------------------------------------------------------ the traced run
def traced_reads(pkg: Path, suite: list, timeout: int) -> dict:
    """Run every suite entry under the audit-hook sitecustomize; {rel path: {script, ...}} of files opened for reading under the package root."""
    out: dict = {}
    with tempfile.TemporaryDirectory(prefix="p7trace_") as td:
        Path(td, "sitecustomize.py").write_text(TRACE_SITE, encoding="utf-8")
        for e in suite:
            name = e.get("script", "raw")
            argv = e["raw"] if "raw" in e else [sys.executable, str(pkg / "refe" / e["script"]), *e.get("args", [])]
            tf = Path(td) / f"opened_{name}.txt"
            env = dict(os.environ, PYTHONIOENCODING="utf-8", P7_TRACE_OUT=str(tf), P7_TRACE_ROOT=str(pkg),
                       PYTHONPATH=os.pathsep.join([td] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
            r = subprocess.run(argv, cwd=str(pkg / "refe"), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
            ok = r.returncode == 0 and e["token"] in r.stdout
            tokline = next((ln.strip() for ln in reversed(r.stdout.splitlines()) if e["token"] in ln), "")
            print(f"ZZP7SHIP_TRACE {name} rc={r.returncode} token_seen={e['token'] in r.stdout} result={tokline[:80]!r}", flush=True)
            if not ok:
                raise SystemExit(f"ZZP7SHIP_BAD the traced run of {name} did not pass (rc {r.returncode}): a trace of a failing suite is not a list; tail: {r.stdout[-300:]!r} {r.stderr[-300:]!r}")
            if tf.is_file():
                for ln in tf.read_text(encoding="utf-8").splitlines():
                    q = Path(ln)
                    try:
                        rel = rel_of(pkg, q)
                    except ValueError:
                        continue
                    if q.is_file() and not any(part in SKIP_PARTS for part in q.parts) and not rel.endswith(".pyc"):
                        out.setdefault(rel, set()).add(name)
    return out


# ------------------------------------------------------------------------------------------------ derive / check
def derive(a) -> int:
    pkg = Path(a.pkg).resolve()
    suite = load_suite(pkg, a.profile, a.extra_script or [])
    scripts = closure(pkg, suite)
    cdig = closure_digest({k: sha_pair(p)[1] for k, p in scripts.items()})
    files: dict = {}
    for rel, readers in static_reads(pkg, scripts).items():
        files.setdefault(rel, {"via": set(), "readers": set()})
        files[rel]["via"].add("static")
        files[rel]["readers"] |= readers
    trace_note = "not run"
    traced = None
    if a.trace:
        traced = traced_reads(pkg, suite, a.timeout)
        Path(a.trace_out or (str(a.out) + ".trace.json")).write_text(json.dumps({"closure_digest": cdig, "opened": {k: sorted(v) for k, v in sorted(traced.items())}}, indent=1), encoding="utf-8")
        trace_note = "run now"
    elif a.trace_file:
        t = json.loads(Path(a.trace_file).read_text(encoding="utf-8"))
        if t.get("closure_digest") != cdig and not a.allow_stale_trace:
            print(f"ZZP7SHIP_BAD the trace in {a.trace_file} was taken at closure {str(t.get('closure_digest'))[:12]}, the suite is now {cdig[:12]}: re-run with --trace (or --allow-stale-trace, disclosed)")
            return 1
        traced = {k: set(v) for k, v in t["opened"].items()}
        trace_note = "merged from file" + ("" if t.get("closure_digest") == cdig else " (STALE, allowed)")
    for rel, readers in (traced or {}).items():
        files.setdefault(rel, {"via": set(), "readers": set()})
        files[rel]["via"].add("trace")
        files[rel]["readers"] |= readers
    out_files = {}
    for rel in sorted(files):
        p = pkg / rel
        raw, lf = sha_pair(p)
        out_files[rel] = {"sha256": raw, "sha256_lf": lf, "via": sorted(files[rel]["via"]), "readers": sorted(files[rel]["readers"])[:6],
                          "outside_shipped_roots": rel.split("/")[0] not in SHIPPED_ROOTS}
    doc = {"schema": SCHEMA, "profile": a.profile, "suite": [e.get("script", "raw") for e in suite], "closure_digest": cdig,
           "scripts": {k: sha_pair(p)[1] for k, p in sorted(scripts.items())}, "trace": trace_note, "files": out_files,
           "required_top": list(REQUIRED_TOP), "shipped_roots": list(SHIPPED_ROOTS)}
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    outside = [k for k, v in out_files.items() if v["outside_shipped_roots"]]
    print(f"ZZP7SHIP_DERIVED files={len(out_files)} outside_roots={len(outside)} scripts={len(scripts)} trace={trace_note} closure={cdig[:12]}")
    for k in outside:
        print(f"  outside {k}  via={'+'.join(out_files[k]['via'])}  readers={','.join(out_files[k]['readers'][:3])}")
    return 0


def check(a) -> int:
    pkg = Path(a.pkg)
    try:
        doc = json.loads(Path(a.list).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"ZZP7SHIP_BAD the ship list {a.list} is unreadable: {type(e).__name__}: {e}")
        return 1
    if doc.get("schema") != SCHEMA:
        print(f"ZZP7SHIP_BAD {a.list} is not a {SCHEMA} list")
        return 1
    missing, differs, stale = [], [], []
    for r in doc.get("shipped_roots", SHIPPED_ROOTS):
        if not (pkg / r).is_dir():
            missing.append(r + "/")
    for r in doc.get("required_top", REQUIRED_TOP):
        if not (pkg / r).is_file():
            missing.append(r)
    for rel, lf in sorted(doc.get("scripts", {}).items()):
        p = pkg / rel
        if not p.is_file():
            missing.append(rel)
        elif sha_pair(p)[1] != lf:
            stale.append(rel)
    for rel, meta in sorted(doc.get("files", {}).items()):
        p = pkg / rel
        if not p.is_file():
            missing.append(rel)
        else:
            raw, lf = sha_pair(p)
            if raw != meta["sha256"] and lf != meta.get("sha256_lf"):
                differs.append(rel)
    missing, differs, stale = sorted(set(missing)), sorted(set(differs)), sorted(set(stale))
    if missing or differs or stale:
        first = (missing or differs or stale)[:4]
        print(f"ZZP7SHIP_BAD missing={len(missing)} differs={len(differs)} stale_scripts={len(stale)} first={first} "
              f"(ship the listed files; a STALE script means the list predates a landed change: regenerate it at ship time, like SHIP_MANIFEST.json)")
        return 1
    print(f"ZZP7SHIP_OK files={len(doc.get('files', {}))} scripts={len(doc.get('scripts', {}))} closure={str(doc.get('closure_digest'))[:12]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("derive")
    p.add_argument("--pkg", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--profile", default="refe_p7")
    p.add_argument("--extra-script", action="append", default=[], metavar="NAME:TOKEN")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--trace-out", default=None)
    p.add_argument("--trace-file", default=None)
    p.add_argument("--allow-stale-trace", action="store_true")
    p.add_argument("--timeout", type=int, default=7200)
    p = sub.add_parser("check")
    p.add_argument("--pkg", required=True)
    p.add_argument("--list", required=True)
    a = ap.parse_args(argv)
    return {"derive": derive, "check": check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
