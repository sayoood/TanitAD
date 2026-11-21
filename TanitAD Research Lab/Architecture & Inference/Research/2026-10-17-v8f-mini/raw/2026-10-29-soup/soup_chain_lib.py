#!/usr/bin/env python3
"""soup_chain_lib.py -- the helpers of soup_chain.sh, the dev-box CPU chain of PREREG_V8F_SOUP section 13a.  ZERO training, ZERO GPU, no network, no remote shell, no git.  Every verdict the chain acts on is computed HERE (so it can be tested
and mutation-tested), and the shell only orders the calls.

    python soup_chain_lib.py <command> ...          (run it with -I)
      treedigest --tree T                            the pinned tool-tree digest (stack/tanitad + taniteval *.py + the T1 tool; reproduces LAUNCH-INTENT's 579183a3... on tree_li)
      plan --plan P --tree T --out O [--field k]     validate the rendered plan; print one line per step:  idx kind tag ckpt  (tab separated)
      argv --plan P --idx N                          the N-th step's tokens, one per line
      outputs --plan P --out O --idx N               the step's expected output paths (the files the chain moves aside), one per line
      stepdone / stepcheck / stepwrite               a step's done record: skip test (same ckpt sha, tool digest, command), post-run output check, writing it
      guards --ram-min-mb N --disk D --disk-min-gb G --c-min-mb M     RAM / output drive / C: (rc 1 = LOW: the chain waits, never deletes)
      ready --plan P --spec S                        every checkpoint the plan touches is what the spec / its soup.json says it is (readiness, section 13a)
      pins-write / pins-verify                       PINS.txt: the sha256 of every input the chain reads; ONE difference stops the chain
      manifest --li-runs R --plan P --out O --spec S --write W      the soup reader's manifest (LAUNCH-INTENT runs.json entries + SOUP / SOUPR / SOUP4 + t0_dev)
      parsecheck --tool T -- ARGS...                 parse one rendered argv with the tool's own argparse (nothing runs)
      world                                          rc 0 iff SOUP_WORLD names a valid throw-away world (test only)
      fmt -- ARGS...                                 the shell-quoted spelling logged for a command
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PY_PROD = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
OUT_PROD = "F:/TanitAD-scratch/soup_t1_out"
TREE_PROD = "F:/TanitAD-scratch/launch_intent_t1/tree_li"
TOOL_DIGEST = "579183a362f7f7843d36bd6c2cc339457b41511c3721e476bfc126eee2321224"      # tree_li, PREREG section 13a; soup_read.py's K-TREE requires the same digest in every done record
NDUMP = 141
KINDS = ("causal", "logged", "t0fresh", "t0std", "t0dev")
T1_TOOL, T0_TOOL = "v8f_t1_eval.py", "v8f_mini_eval.py"
WORLD_MARKER, WORLD_TOKEN = "SOUP_TESTWORLD.marker", "SOUP-THROWAWAY-WORLD"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class PlanError(Exception):
    pass


# ------------------------------------------------------------------------------------------------------------------------------ small things
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_sha(argv: list) -> str:
    """sha256 of the NUL-joined tokens (the LAUNCH-INTENT chain's rule)."""
    return hashlib.sha256(b"".join(a.encode() + b"\0" for a in argv)).hexdigest()


def fmt(args: list) -> str:
    return " ".join(f"'{a}'" if re.search(r"[^A-Za-z0-9_./:=,@%+-]", a) else a for a in args)


def norm(p: str) -> str:
    return p.replace("\\", "/")


def world_ok() -> bool:
    """A test world is honoured only when SOUP_WORLD names an existing directory carrying the marker file and living OUTSIDE the production output and the soup directory.  Real mode never sets it."""
    w = os.environ.get("SOUP_WORLD", "")
    if not w or not os.path.isdir(w):
        return False
    mk = os.path.join(w, WORLD_MARKER)
    if not os.path.isfile(mk) or open(mk, encoding="utf-8").read().split("\n")[0].strip() != WORLD_TOKEN:
        return False
    r = os.path.realpath(w).replace("\\", "/").lower().rstrip("/")
    for bad in (OUT_PROD, HERE, TREE_PROD):
        b = os.path.realpath(bad).replace("\\", "/").lower().rstrip("/")
        if r == b or r.startswith(b + "/") or b.startswith(r + "/"):
            return False
    return True


# ------------------------------------------------------------------------------------------------------------------------------ the tool tree
def digest_set(root: str) -> list:
    paths = set()
    for r in ("stack/tanitad", "taniteval"):
        for dp, _dn, fn in os.walk(os.path.join(root, r)):
            for f in fn:
                if f.endswith(".py"):
                    paths.add(os.path.relpath(os.path.join(dp, f), root).replace("\\", "/"))
    paths.add("stack/scripts/v8f_t1_eval.py")
    return sorted(paths, key=lambda s: s.encode())


def tree_digest(root: str) -> str:
    lines = "".join(f"{sha256_file(os.path.join(root, p))} *{p}\n" for p in digest_set(root))
    return hashlib.sha256(lines.encode()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------ the plan
def _flag(tok: list, name: str, line: int):
    idx = [i for i, t in enumerate(tok) if t == name]
    if len(idx) != 1 or idx[0] + 1 >= len(tok):
        raise PlanError(f"plan line {line}: {name} must occur exactly once with a value (found {len(idx)})")
    return tok[idx[0] + 1]


def parse_plan(path: str, tree: str, out: str, py: str = PY_PROD) -> list:
    """The rendered commands (argv/PLAN.txt) -> steps.  Every step is validated; the plan is the ONLY source of what runs."""
    text = open(path, encoding="utf-8", newline="").read()
    if "\r" in text:
        raise PlanError("plan has CR characters (LF only)")
    tree, out = norm(tree).rstrip("/"), norm(out).rstrip("/")
    steps, seen = [], set()
    for ln_no, ln in enumerate(text.split("\n"), 1):
        if not ln.strip() or ln.startswith("#"):
            continue
        tok = shlex.split(ln)
        if len(tok) < 8 or tok[0] != py:
            raise PlanError(f"plan line {ln_no}: first token must be {py}")
        tool = tok[1]
        if not tool.startswith(tree + "/stack/scripts/") or os.path.basename(tool) not in (T1_TOOL, T0_TOOL) or tok[2] != "run":
            raise PlanError(f"plan line {ln_no}: the tool must be {tree}/stack/scripts/{T1_TOOL} or {T0_TOOL}, subcommand run (got {tool!r} {tok[2]!r})")
        ckpt, tag, od = norm(_flag(tok, "--ckpt", ln_no)), _flag(tok, "--tag", ln_no), norm(_flag(tok, "--out-dir", ln_no))
        if not ckpt.endswith("/ckpt_last.pt"):
            raise PlanError(f"plan line {ln_no}: --ckpt must end in /ckpt_last.pt ({ckpt})")
        if tag in seen or not re.match(r"^[A-Za-z0-9_.,-]+$", tag):
            raise PlanError(f"plan line {ln_no}: tag {tag!r} is duplicated or not a plain name")
        seen.add(tag)
        if os.path.basename(tool) == T1_TOOL:
            kind = _flag(tok, "--limit", ln_no)
            if kind not in ("causal", "logged") or od != out or norm(_flag(tok, "--dump-ticks", ln_no)) != out + "/ticks":
                raise PlanError(f"plan line {ln_no}: a T1 step needs --limit causal|logged, --out-dir {out} and --dump-ticks {out}/ticks")
            if "--dump-heads" not in tok:
                raise PlanError(f"plan line {ln_no}: --dump-heads is required (the heads are the decisive rows)")
        else:
            fresh = "--ages" in tok
            if fresh and _flag(tok, "--ages", ln_no) != "fresh":
                raise PlanError(f"plan line {ln_no}: --ages must be 'fresh' when present")
            kind = "t0fresh" if fresh else ("t0dev" if tag.endswith("-dev") else "t0std")
            if not od.startswith(out + "/t0"):
                raise PlanError(f"plan line {ln_no}: a T0 step writes under {out}/t0*, not {od}")
        steps.append({"idx": len(steps), "kind": kind, "tag": tag, "ckpt": ckpt, "out": out, "out_dir": od, "tool": tool, "tokens": tok, "line": ln_no})
    if not steps:
        raise PlanError("the plan has no steps")
    return steps


def expected(step: dict) -> dict:
    """The files and directories a finished step leaves, and its done record."""
    o, t = step["out"], step["tag"]
    if step["kind"] in ("causal", "logged"):
        return {"files": [f"{o}/t1v8f_{t}.json", f"{o}/ticks/ticks_{t}.npz", f"{o}/argv/{t}.argv"], "dumps": f"{o}/dump_{t}", "done": f"{o}/done_{t}.txt", "extra": []}
    d = step["out_dir"]
    return {"files": [f"{d}/eval_{t}.json", f"{d}/eval_{t}.npz"], "dumps": None, "done": f"{o}/done_t0_{t}.txt", "extra": [f"{d}/eval_{t}_proj.npz"]}


def n_dumps(path: str) -> int:
    try:
        return sum(1 for f in os.listdir(path) if re.match(r"^ep.*\.npz$", f))
    except OSError:
        return 0


def step_problems(step: dict) -> list:
    ex, probs = expected(step), []
    for f in ex["files"]:
        if not os.path.isfile(f) or os.path.getsize(f) == 0:
            probs.append(f"missing or empty: {f}")
    if ex["dumps"] and n_dumps(ex["dumps"]) < NDUMP:
        probs.append(f"{ex['dumps']}: {n_dumps(ex['dumps'])} dumps < {NDUMP}")
    return probs


def done_record(step: dict, ckpt_sha: str, tool_digest: str, cmd: str, wall_s: int, before: str = None, after: str = None) -> dict:
    ex = expected(step)
    return {"tag": step["tag"], "kind": step["kind"], "limit": step["kind"] if step["kind"] in ("causal", "logged") else None, "ckpt": step["ckpt"], "ckpt_sha256": ckpt_sha,
            "ckpt_sha256_before": before or ckpt_sha, "ckpt_sha256_after": after or ckpt_sha, "tool_digest": tool_digest, "cmd_sha256": cmd,
            "n_dumps": n_dumps(ex["dumps"]) if ex["dumps"] else None, "wall_s": wall_s, "smoke": False, "max_windows": None}


def step_done(step: dict, ckpt_sha: str, tool_digest: str, cmd: str) -> tuple:
    """-> (done?, why not).  A finished step is skipped only if its record names THIS checkpoint (three times), THIS tool digest and THIS command, and its outputs are all there."""
    ex = expected(step)
    if not os.path.isfile(ex["done"]) or os.path.getsize(ex["done"]) == 0:
        return False, "no done record"
    try:
        d = json.load(open(ex["done"], encoding="utf-8"))
    except (OSError, ValueError) as e:
        return False, f"unreadable done record ({e})"
    for k, want in (("ckpt_sha256", ckpt_sha), ("ckpt_sha256_before", ckpt_sha), ("ckpt_sha256_after", ckpt_sha), ("tool_digest", tool_digest), ("cmd_sha256", cmd), ("tag", step["tag"])):
        if d.get(k) != want:
            return False, f"done record {k} is {str(d.get(k))[:16]!r}, this run has {str(want)[:16]!r}"
    if d.get("smoke") is not False or d.get("max_windows") is not None:
        return False, "done record is a smoke / window-limited run"
    probs = step_problems(step)
    return (False, probs[0]) if probs else (True, "complete")


# ------------------------------------------------------------------------------------------------------------------------------ guards
def ram_avail_mb() -> int:
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("ml", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                    ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    m = MS()
    m.l = ctypes.sizeof(MS)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(m.ap // (1 << 20))


def _hook(name: str):
    """A test world may override a reading with <world>/hook_<name>.txt; real mode never does."""
    if not world_ok():
        return None
    fp = os.path.join(os.environ["SOUP_WORLD"], f"hook_{name}.txt")
    return open(fp, encoding="utf-8").read().strip() if os.path.isfile(fp) else None


def guards(ram_min_mb: int, disk: str, disk_min_gb: float, c_min_mb: int) -> tuple:
    """-> (rc, text).  0 = every guard holds, 1 = LOW (the chain waits, never deletes)."""
    low = []
    ram = ram_avail_mb()
    if _hook("ram_mb") is not None:
        ram = int(_hook("ram_mb"))
    if ram < ram_min_mb:
        low.append(f"RAM available {ram} MB < {ram_min_mb} MB")
    free = shutil.disk_usage(disk).free / (1 << 30)
    if _hook("disk_gb") is not None:
        free = float(_hook("disk_gb"))
    if free < disk_min_gb:
        low.append(f"{disk} free {free:.1f} GB < {disk_min_gb} GB")
    cfree = shutil.disk_usage("C:/").free / (1 << 20)
    if _hook("c_mb") is not None:
        cfree = float(_hook("c_mb"))
    if cfree < c_min_mb:
        low.append(f"C: free {cfree:.0f} MB < {c_min_mb} MB")
    txt = f"ram={ram}MB disk_free={round(free, 1)}GB c_free={round(cfree)}MB"
    return (1, "LOW " + "; ".join(low) + " | " + txt) if low else (0, "OK " + txt)


# ------------------------------------------------------------------------------------------------------------------------------ readiness and pins
def ready(plan_steps: list, spec: dict) -> list:
    """Section 13a: a soup is READY when its soup.json says K-BASIN PASS and records the output file's sha256 (and the file hashes to it); a control checkpoint is the spec's member, byte for byte."""
    probs, by_path = [], {norm(m["path"]).lower(): (n, m["sha256"]) for n, m in spec["members"].items()}
    done = set()
    for st in plan_steps:
        ck = st["ckpt"]
        if ck in done:
            continue
        done.add(ck)
        if not os.path.isfile(ck):
            probs.append(f"{ck}: not a file")
            continue
        got = sha256_file(ck)
        rec = os.path.join(os.path.dirname(ck), "soup.json")
        if os.path.isfile(rec):
            r = json.load(open(rec, encoding="utf-8"))
            if r.get("output_sha256") != got:
                probs.append(f"{ck}: sha256 {got[:16]} != soup.json output_sha256 {str(r.get('output_sha256'))[:16]}")
            if (r.get("K-BASIN") or {}).get("PASS") is not True:
                probs.append(f"{rec}: K-BASIN did not PASS")
        elif norm(ck).lower() in by_path:
            n, want = by_path[norm(ck).lower()]
            if got != want:
                probs.append(f"{ck}: sha256 {got[:16]} != the spec's {n} {want[:16]}")
        else:
            probs.append(f"{ck}: neither a soup (no soup.json) nor a member of the spec")
    return probs


def pin_entries(files: dict, plan_steps: list, tree: str) -> list:
    """The pin lines: a label=path sha256 per input file, one per distinct checkpoint, and the tool tree digest."""
    ents = [f"file={lab}|{norm(p)} {sha256_file(p)}" for lab, p in files.items()]
    for ck in sorted({s['ckpt'] for s in plan_steps}):
        ents.append(f"ckpt={ck} {sha256_file(ck)}")
    t0 = f"{norm(tree).rstrip('/')}/stack/scripts/{T0_TOOL}"      # the T0 tool is not in the tree digest (which must stay the pinned 579183a3...): pinned by itself
    ents.append(f"tool={t0} {sha256_file(t0)}")
    ents.append(f"tree={norm(tree)} {tree_digest(tree)}")
    return ents


def pins_check(pins_path: str, files: dict, plan_steps: list, tree: str, expect_tree_digest: str | None) -> list:
    """Recompute every pin and compare.  Returns the list of differences (empty = verified).  A pin line that no longer names an input, and an input with no pin line, are differences too."""
    if not os.path.isfile(pins_path):
        return [f"no PINS file at {pins_path}"]
    want = {}
    for ln in open(pins_path, encoding="utf-8").read().split("\n"):
        if ln and not ln.startswith("#"):
            k, _, v = ln.rpartition(" ")
            want[k] = v
    have = {}
    for e in pin_entries(files, plan_steps, tree):
        k, _, v = e.rpartition(" ")
        have[k] = v
    diffs = [f"pin {k}: {have.get(k, 'ABSENT')[:16]} != pinned {want.get(k, 'NO PIN')[:16]}" for k in sorted(set(have) | set(want)) if have.get(k) != want.get(k)]
    if expect_tree_digest is not None and have.get(f"tree={norm(tree)}") != expect_tree_digest:
        diffs.append(f"tool tree digest {str(have.get(f'tree={norm(tree)}'))[:16]} != the pinned {expect_tree_digest[:16]}")
    return diffs


# ------------------------------------------------------------------------------------------------------------------------------ the manifest
def build_manifest(li_runs: dict, steps: list, spec: dict, out: str) -> dict:
    """The soup reader's manifest: the LAUNCH-INTENT entries for CTRL / REP / A / Arep (CTRL and A gain t0_dev) plus SOUP / SOUPR / SOUP4, every path taken from the plan's steps.  Refuses if a step's outputs are absent."""
    by_tag = {s["tag"]: s for s in steps}
    for s in steps:
        pr = step_problems(s)
        if pr:
            raise PlanError(f"step {s['tag']}: {pr[0]}")
        if not os.path.isfile(expected(s)["done"]):
            raise PlanError(f"step {s['tag']}: no done record")

    def one(tag):
        if tag not in by_tag:
            raise PlanError(f"the plan has no step with tag {tag}")
        return by_tag[tag]

    def t1(tag):
        s = one(tag)
        o = s["out"]
        return {"tag": tag, "ticks": f"{o}/ticks/ticks_{tag}.npz", "dump_dir": f"{o}/dump_{tag}", "json": f"{o}/t1v8f_{tag}.json", "argv": f"{o}/argv/{tag}.argv", "done": f"{o}/done_{tag}.txt"}

    def t0(tag):
        s = one(tag)
        return {"json": f"{s['out_dir']}/eval_{tag}.json", "npz": f"{s['out_dir']}/eval_{tag}.npz"}
    arms = {r: json.loads(json.dumps(li_runs["arms"][r])) for r in ("CTRL", "REP", "A", "Arep")}
    seeds = [str(x) for x in li_runs.get("seeds", [0, 1])]
    for r, stem in (("CTRL", "dagoacc"), ("A", "a")):
        for s in seeds:
            arms[r][s]["t0_dev"] = t0(f"M2b-{stem}_{s}-dev")
    for name, sp in spec["soups"].items():
        if sp.get("expect"):
            continue
        role, s = sp["role"], str(sp["seed"])
        arms.setdefault(role, {})[s] = {"ckpt_tail": f"{sp['run_dir']}/ckpt_last.pt", "soup_dir": os.path.dirname(one(f"{name}_m10_causal_heads")["ckpt"]),
                                         "t1": t1(f"{name}_m10_causal_heads"), "t1_logged": t1(f"{name}_m10_logged_heads"), "t0": t0(f"M2b-{name.lower()}"), "t0_fresh": t0(f"M2b-{name.lower()}-fresh")}
    return {"seeds": [int(x) for x in seeds], "arms": arms, "build": li_runs.get("build"),
            "provenance": {"prereg": "PREREG_V8F_SOUP.md", "built_by": "soup_chain_lib.py manifest (the dev-box chain, soup_chain.sh)", "n_steps": len(steps), "out": norm(out)}}


# ------------------------------------------------------------------------------------------------------------------------------ parse check
_CAP = ("import argparse, importlib.util, json, sys\nclass Cap(Exception): pass\norig = argparse.ArgumentParser.parse_args\n"
        "def pa(self, args=None, namespace=None):\n    ns = orig(self, args, namespace)\n    print('ZZPARSED ' + json.dumps(vars(ns), default=str))\n    raise Cap()\n"
        "argparse.ArgumentParser.parse_args = pa\nspec = importlib.util.spec_from_file_location('soup_tool', sys.argv[1])\nm = importlib.util.module_from_spec(spec)\nsys.modules['soup_tool'] = m\n"
        "spec.loader.exec_module(m)\ntry:\n    m.main(sys.argv[2:])\nexcept Cap:\n    pass\n")


def parsecheck(tool: str, args: list) -> tuple:
    """Parse ARGS with the tool's own argparse; the run is cut off by a patched parse_args BEFORE anything executes."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    env["TANITEVAL_STACK_OVERRIDE"] = os.path.dirname(os.path.dirname(os.path.abspath(tool)))
    r = subprocess.run([sys.executable, "-I", "-B", "-c", _CAP, tool, *args], capture_output=True, env=env, stdin=subprocess.DEVNULL)
    for ln in r.stdout.decode("utf-8", "replace").splitlines():
        if ln.startswith("ZZPARSED "):
            return True, json.loads(ln[9:])
    return False, (r.stderr.decode("utf-8", "replace") or r.stdout.decode("utf-8", "replace"))[-400:]


# ------------------------------------------------------------------------------------------------------------------------------ CLI
def _files(a) -> dict:
    return {k: v for k, v in (("plan", a.plan), ("spec", a.spec), ("builder", a.builder), ("lib", a.lib), ("chain", a.chain), ("reader", a.reader), ("li_reader", a.li_reader), ("banked_li_read", a.banked),
                              ("li_runs", a.li_runs)) + tuple((f"tmpl:{os.path.basename(t)}", t) for t in sorted(a.tmpl or [])) if v}


def _expect_digest(a):
    """None = no pin (a test world's stub tree); otherwise the pinned production digest unless the caller names another."""
    if a.expect_tree_digest == "none":
        return None
    return a.expect_tree_digest or TOOL_DIGEST


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(newline="\n", encoding="utf-8")
        except Exception:      # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("treedigest", "plan", "argv", "outputs", "stepdone", "stepcheck", "stepwrite", "guards", "ready", "pins-write", "pins-verify", "pinsha", "manifest", "parsecheck", "world", "fmt", "sha", "cmdsha"):
        p = sub.add_parser(nm)
        for f in ("--tree", "--plan", "--out", "--spec", "--tool", "--pins", "--builder", "--lib", "--li-reader", "--banked", "--li-runs", "--write", "--ckpt-sha", "--tool-digest", "--cmd-sha",
                  "--disk", "--expect-tree-digest", "--before", "--after", "--chain", "--reader", "--ckpt"):
            p.add_argument(f)
        for f in ("--idx", "--ram-min-mb", "--c-min-mb", "--wall"):
            p.add_argument(f, type=int)
        p.add_argument("--disk-min-gb", type=float)
        p.add_argument("--tmpl", action="append")
        p.add_argument("--py", default=PY_PROD)
        p.add_argument("rest", nargs="*")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "world":
            return 0 if world_ok() else 1
        if a.cmd == "fmt":
            print(fmt(a.rest))
            return 0
        if a.cmd == "sha":
            print(sha256_file(a.rest[0]))
            return 0
        if a.cmd == "cmdsha":
            print(cmd_sha(a.rest))
            return 0
        if a.cmd == "treedigest":
            print(tree_digest(a.tree))
            return 0
        if a.cmd == "parsecheck":
            ok, ns = parsecheck(a.tool, a.rest)
            print(("PARSE-OK " if ok else "PARSE-FAIL ") + (json.dumps(ns) if ok else str(ns)))
            return 0 if ok else 1
        if a.cmd == "guards":
            rc, txt = guards(a.ram_min_mb, a.disk, a.disk_min_gb, a.c_min_mb)
            print(txt)
            return rc
        steps = parse_plan(a.plan, a.tree, a.out, a.py)
        if a.cmd == "plan":
            for s in steps:
                print(f"{s['idx']}\t{s['kind']}\t{s['tag']}\t{s['ckpt']}")
            return 0
        if a.cmd == "argv":
            print("\n".join(steps[a.idx]["tokens"]))
            return 0
        if a.cmd == "outputs":
            ex = expected(steps[a.idx])
            print("\n".join(ex["files"] + ([ex["dumps"]] if ex["dumps"] else []) + ex["extra"] + [ex["done"]]))
            return 0
        if a.cmd in ("stepdone", "stepcheck", "stepwrite"):
            st = steps[a.idx]
            if a.cmd == "stepdone":
                ok, why = step_done(st, a.ckpt_sha, a.tool_digest, a.cmd_sha)
                print(why)
                return 0 if ok else 1
            if a.cmd == "stepcheck":
                pr = step_problems(st)
                print("; ".join(pr) if pr else "outputs complete")
                return 1 if pr else 0
            rec = done_record(st, a.ckpt_sha, a.tool_digest, a.cmd_sha, a.wall or 0, a.before, a.after)
            rec["command"] = fmt(st["tokens"])
            with open(expected(st)["done"], "w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec) + "\n")
            return 0
        spec = json.load(open(a.spec, encoding="utf-8")) if a.spec else None
        if a.cmd == "ready":
            pr = ready(steps, spec)
            print("; ".join(pr) if pr else f"{len({s['ckpt'] for s in steps})} checkpoints ready")
            return 1 if pr else 0
        if a.cmd == "pins-write":
            ents = pin_entries(_files(a), steps, a.tree)
            if _expect_digest(a) is not None and tree_digest(a.tree) != _expect_digest(a):
                raise PlanError(f"the tool tree {a.tree} does not hash to the pinned {_expect_digest(a)[:16]}: no PINS written")
            with open(a.write, "w", encoding="utf-8", newline="\n") as f:
                f.write("# PINS of soup_chain.sh (PREREG_V8F_SOUP section 13a).  The chain STOPs on ANY difference.  Rewrite only deliberately: bash soup_chain.sh --write-pins\n" + "\n".join(ents) + "\n")
            print(f"wrote {a.write} ({len(ents)} pins)")
            return 0
        if a.cmd == "pins-verify":
            d = pins_check(a.pins, _files(a), steps, a.tree, _expect_digest(a))
            print("\n".join(d) if d else "pins verified")
            return 1 if d else 0
        if a.cmd == "pinsha":
            for ln in open(a.pins, encoding="utf-8").read().splitlines():
                if ln.startswith("ckpt=") and ln.rpartition(" ")[0] == f"ckpt={norm(a.ckpt)}":
                    print(ln.rpartition(" ")[2])
                    return 0
            return 1
        if a.cmd == "manifest":
            m = build_manifest(json.load(open(a.li_runs, encoding="utf-8")), steps, spec, a.out)
            with open(a.write, "w", encoding="utf-8", newline="\n") as f:
                json.dump(m, f, indent=1)
            print(f"manifest {a.write}: {sum(len(v) for r, v in m['arms'].items())} slots")
            return 0
    except (PlanError, OSError, KeyError, ValueError, IndexError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
