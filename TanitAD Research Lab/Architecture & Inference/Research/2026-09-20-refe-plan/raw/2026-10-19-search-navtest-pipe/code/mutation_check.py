#!/usr/bin/env python3
"""Mutation check of SEARCH-NAVTEST-PIPE's guards: every `# @GUARD:<name>` line of the OPERATIVE files is disabled ONCE, in a scratch copy of the package (code/, ref/, raw/), and the suite (test_navpipe.py +
test_navpipe_s0.py + test_navpipe_guards.py + test_navpipe_licence.py) MUST then fail.  A guard whose removal leaves the suite green is untested.  The scratch copy is the only thing ever mutated: the package (and any chain running from it) is never touched.

    CUDA_VISIBLE_DEVICES=-1 python mutation_check.py --out ../raw/mutations.json [--only NAME[,NAME]]
Exit 0 only if every mutation is caught AND the unmutated copy passes first (the control).
The mutation is DERIVED from the marked line, so a new guard cannot be added without being mutated:
  * `if <cond>:` lines  ->  `if False:` (the refusal never fires);  override table OVERRIDE below for the forms where that is not the right disablement;
  * every other marked line needs an OVERRIDE entry (the run refuses to start without one).
KNOWN_UNTESTABLE lists the guards the suite cannot reach without the GPU; they are reported as such, never as caught.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
FILES = ("navpipe_core.py", "navpipe_chain.py", "navpipe_launch.py", "navpipe_analyse.py", "make_navtest_bundle.py", "navpipe_bridge.py", "navpipe_gpu_stage.py")
TESTS = ("test_navpipe_licence.py", "test_navpipe.py", "test_navpipe_s0.py", "test_navpipe_guards.py")
MARK = re.compile(r"@GUARD:([A-Za-z0-9\-]+)\s*$")
# guard -> (kind, payload): "replace": (old-substring-of-the-line, new); "prepend": a statement inserted before the line (same indent); "line": the whole replacement line body
OVERRIDE = {
    "C-CRASH-RETRY": ("replace", ("if rc not in CRASH_CODES or att == CRASH_RETRIES:", "if True:")),                   # no retry at all
    "C-SEAL-FIRST": ("prepend", 'json.load(open(c.token_file, encoding="utf-8"))'),                                    # data is opened BEFORE the seal is checked
    "C-S0-SEAL-FIRST": ("prepend", 'json.load(open(c.token_file, encoding="utf-8"))'),
    "G-IDENT-REF": ("replace", ("bad = [t for i, t in enumerate(tokens) if ref.has(t) and not ref.equal(t, traj_raw[i], logits[i], pick[i], seam)]", "bad = []")),
    "G-T1ID-IDENTITY": ("replace", ("ident_ok = bool(len(tokens) > 0 and np.isfinite(d) and d <= tol)", "ident_ok = True")),
    "G-T1ID-CONTROL": ("replace", ("ctl_ok = bool(changed > 0)", "ctl_ok = True")),
    "C-DELETE-CALLED": ("replace", ('NC.delete_ctx(st, cid, c.cdir(cid, "bundle", "ctx.npy"))', "pass")),
}
KNOWN_UNTESTABLE = {
    "MB-IDENT-REF": "the builder's per-token identity gate runs only inside make_navtest_bundle.py on a GPU forward; the suite cannot execute it. Its comparison is `ref.equal`, the same call G-IDENT-REF mutates, and its bridge judge (BR-*) is mutated.",
}


def find_guards():
    out = []
    for f in FILES:
        for i, line in enumerate(open(os.path.join(HERE, f), encoding="utf-8").read().split("\n")):
            m = MARK.search(line)
            if m and not line.lstrip().startswith(("#", '"', "`")):
                out.append((f, i, m.group(1), line))
    return out


def mutate(text: str, lineno: int, name: str):
    lines = text.split("\n")
    line = lines[lineno]
    indent = line[:len(line) - len(line.lstrip())]
    if name in OVERRIDE:
        kind, pay = OVERRIDE[name]
        if kind == "replace":
            old, new = pay
            if old not in line:
                return None
            lines[lineno] = line.replace(old, new, 1)
        elif kind == "prepend":
            lines[lineno:lineno] = [indent + pay]
        return "\n".join(lines)
    m = re.match(r"^(\s*)(if|elif) (.*):(\s*#.*)?$", line)
    if not m:
        return None
    lines[lineno] = f"{m.group(1)}{m.group(2)} False:  # MUTATED {name}"
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--keep-going", action="store_true")
    a = ap.parse_args(argv)
    guards = find_guards()
    names = [g[2] for g in guards]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        print("ZZMUT_ABORT duplicate guard names", dup)
        return 3
    unhandled = [g[2] for g in guards if g[2] not in OVERRIDE and mutate(open(os.path.join(HERE, g[0]), encoding="utf-8").read(), g[1], g[2]) is None]
    if unhandled:
        print("ZZMUT_ABORT no mutation derivable for", unhandled)
        return 3
    only = set(a.only.split(",")) if a.only else None
    res, caught = [], 0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    runs = [None] + [g for g in guards if (only is None or g[2] in only)]
    for g in runs:
        tmp = tempfile.mkdtemp(prefix="npmut_", dir=os.environ.get("NPMUT_DIR"))
        try:
            for d in ("code", "ref", "raw"):
                if os.path.isdir(os.path.join(PKG, d)):
                    shutil.copytree(os.path.join(PKG, d), os.path.join(tmp, d), ignore=shutil.ignore_patterns("__pycache__"))
            label = "CONTROL (unmutated)"
            if g is not None:
                f, ln, name, line = g
                label = name
                if name in KNOWN_UNTESTABLE:
                    res.append({"guard": name, "file": f, "status": "UNTESTABLE_WITHOUT_GPU", "why": KNOWN_UNTESTABLE[name]})
                    print("ZZMUT UNTESTABLE", f, "|", name, flush=True)
                    continue
                p = os.path.join(tmp, "code", f)
                s = open(p, encoding="utf-8").read()
                new = mutate(s, ln, name)
                if new is None or new == s:
                    res.append({"guard": name, "file": f, "status": "MUTATION_NOT_APPLICABLE"})
                    continue
                open(p, "w", encoding="utf-8", newline="\n").write(new)
            print("ZZMUT_START", label, flush=True)                                                           # so a runaway mutant is identifiable while it runs
            cmd = [sys.executable, "-m", "pytest", *TESTS, "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={os.path.join(tmp, 'pt')}"]
            r = subprocess.run(cmd, cwd=os.path.join(tmp, "code"), capture_output=True, text=True, env=env)
            ok = r.returncode == 0
            if g is None:
                res.append({"guard": label, "status": "CONTROL_GREEN" if ok else "CONTROL_RED", "tail": r.stdout[-300:]})
                if not ok:
                    print("ZZMUT_CONTROL_FAILED", r.stdout[-800:])
                    json.dump(res, open(a.out, "w"), indent=1)
                    return 2
                print("ZZMUT_CONTROL_GREEN", flush=True)
            else:
                failed = [l for l in r.stdout.splitlines() if l.startswith("FAILED")][:2]
                infra = (not ok) and any(m in r.stdout for m in ("No space left", "Errno 28", "ENOSPC", "MemoryError", "Access is denied"))      # a suite that died of the BOX is not a suite that caught the mutation
                status = "SURVIVED" if ok else ("INFRA_ERROR" if infra else "CAUGHT")
                res.append({"guard": label, "file": g[0], "status": status, "first_failing": failed})
                caught += (status == "CAUGHT")
                print("ZZMUT", {"CAUGHT": "CAUGHT  ", "SURVIVED": "SURVIVED", "INFRA_ERROR": "INFRA_ERROR"}[status], g[0], "|", label, "|", (failed[0][:150] if failed else "(no FAILED line)"), flush=True)
                if ok and not a.keep_going:
                    pass
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    judged = [x for x in res if x["status"] in ("CAUGHT", "SURVIVED", "MUTATION_NOT_APPLICABLE")]
    judged += [x for x in res if x["status"] == "INFRA_ERROR"]
    bad = [x for x in judged if x["status"] != "CAUGHT"]
    unt = [x for x in res if x["status"] == "UNTESTABLE_WITHOUT_GPU"]
    json.dump({"caught": caught, "judged": len(judged), "survived_or_inapplicable": bad, "untestable_without_gpu": unt, "results": res}, open(a.out, "w"), indent=1)
    print(f"ZZMUT_SUMMARY caught {caught} of {len(judged)}; survived/inapplicable: {[x['guard'] for x in bad]}; untestable without the GPU: {[x['guard'] for x in unt]}", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
