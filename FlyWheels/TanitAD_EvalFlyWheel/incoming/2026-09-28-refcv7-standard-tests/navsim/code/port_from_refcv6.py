#!/usr/bin/env python3
"""Port the refcv6 NavSim suite's MODEL-AGNOSTIC drivers into this package (any python).

The source is the CLEAN TREE (``git archive 0c444082`` -> ``C:/Users/Admin/ev7nav``), never a
working copy. Each ported file gets a provenance header naming its source path and git blob, and
the EXACT list of textual substitutions applied; everything else is byte-identical (EOL kept).
Re-running is idempotent. A substitution that matches nothing REFUSES (a stale rule is a defect).

    python code/port_from_refcv6.py            # writes code/<name7>, prints the manifest
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.environ.get("TANITAD_REPO", "C:/Users/Admin/ev7nav")
SRC = os.path.join(TREE, "FlyWheels", "TanitAD_EvalFlyWheel", "incoming",
                   "2026-09-23-refcv6-standard-tests", "navsim", "code")
GIT_DIR = "C:/Users/Admin/tanitad-push/.git"
COMMIT = "0c444082a08efdcbeef36585f1b3baabbfafc1fc"
REL = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code/"

#: dst -> (src, [(old, new, min_count)])
PORTS = {
    "score_arm7.py": ("score_arm6.py", [
        ('ap.add_argument("--exp-tag", default="e6")', 'ap.add_argument("--exp-tag", default="e7")', 1),
    ]),
    "score_navtest7.py": ("score_navtest6.py", [
        ('if not a.label.startswith("r6"):', 'if not a.label.startswith("r7"):', 1),
        ("⛔ labels must start with 'r6'", "⛔ labels must start with 'r7'", 1),
    ]),
    "check_harness_repro7.py": ("check_harness_repro.py", []),
    "import_floors7.py": ("import_floors.py", [
        ('es = os.path.join(PKG, "raw", "scores_warmup_s1000")',
         'es = os.path.join(PKG, "raw", "harness_repro")   # refcv7: ECHO is re-scored with CV/STOP', 1),
    ]),
    "score_queue7.sh": ("score_queue6.sh", [
        ('code/score_arm6.py', 'code/score_arm7.py', 2),
    ]),
    # ---- analysis: arm names R6_ -> R7_, the step-aware stamp -> refcv7's, PRIOR as a model arm
    "parse7.py": ("parse6.py", [
        ("import model_stamp6 as M6", "import model_stamp7 as M6", 1),
        ('MODEL_ARMS = ("R6_A1", "R6_A1_s1", "R6_A1NT", "R6_BLIND", "R6_NAVOFF", "R6_VMAXOFF")',
         'MODEL_ARMS = ("R7_A1", "R7_A1_s1", "R7_A1NT", "R7_BLIND", "R7_NAVOFF", "R7_VMAXOFF",\n'
         '              "R7_FILTOFF", "R7_CEILDECL_d", "PRIOR_ha0p")', 1),
        ('         ("R6_A1", "R6_A1NT", "time construction (ST - NT)"),',
         '         ("R6_A1", "R6_A1NT", "time construction (ST - NT)"),\n'
         '         ("R6_A1", "R7_FILTOFF", "SPEC_REFCV7 A2: ceiling filter ON - OFF (real forward; SPEC A1: the emitted plan never sees the filter)"),\n'
         '         ("R6_A1", "R7_CEILDECL_d", "DIAGNOSTIC (SPEC A1): as built - the ceiling as DECLARED on the emitted pick"),\n'
         '         ("R6_A1", "PRIOR_ha0p", "the residual: A1 - its own prior P alone"),\n'
         '         ("PRIOR_ha0p", "STOP_zero", "the prior vs doing nothing"),\n'
         '         ("PRIOR_ha0p", "CV_official", "the prior vs CV"),\n'
         '         ("PRIOR_ha0p", "ECHO_ha0_ext", "the prior (pose-derived) vs the echo (EgoStatus a, ay)"),', 1),
        ("R6_", "R7_", 10),
    ]),
    "parse_navtest7.py": ("parse_navtest6.py", [
        ("import model_stamp6 as M6", "import model_stamp7 as M6", 1),
        ('k.startswith("R6_")', 'k.startswith(("R7_", "PRIOR_"))', 3),
        ('k.split("__minus__")[1].startswith("R6_")',
         'k.split("__minus__")[1].startswith(("R7_", "PRIOR_"))', 1),
        ("R6_A1", "R7_A1", 8),
    ]),
    "decompose7.py": ("decompose6.py", [
        ('ap.add_argument("--arm", default="R6_A1")', 'ap.add_argument("--arm", default="R7_A1")', 1),
        ('n.startswith("R6_")', 'n.startswith(("R7_", "PRIOR_"))', 1),
    ]),
    "families7.py": ("families6.py", [
        ('preds = {"refcv6": np.asarray', 'preds = {str(z["arm"]): np.asarray', 1),
    ]),
    "plan_deltas7.py": ("plan_deltas.py", [
        ('X[t]["source"] == "refcv6"', 'X[t]["source"] == "refcv7"', 1),
        ('r["source"] == "refcv6"', 'r["source"] == "refcv7"', 1),
        ("R6_", "R7_", 20),
    ]),
    "step_compare7.py": ("step_compare.py", [
        ("import model_stamp6 as M6", "import model_stamp7 as M6", 1),
        ('lab = f"r6s{step}_{arm}"', 'lab = f"r7s{step}_{arm}"', 1),
        ("R6_A1", "R7_A1", 10),
    ]),
}


def blob(path_rel: str) -> str:
    r = subprocess.run(["git", f"--git-dir={GIT_DIR}", "rev-parse", f"{COMMIT}:{path_rel}"],
                       capture_output=True, text=True)
    b = r.stdout.strip()
    if len(b) != 40:
        raise SystemExit(f"⛔ blob of {path_rel} at {COMMIT[:8]} unreadable: {r.stderr.strip()!r}")
    return b


def main() -> int:
    man = {"tree": TREE, "commit": COMMIT, "files": {}}
    for dst, (src, subs) in PORTS.items():
        raw = open(os.path.join(SRC, src), "rb").read()
        crlf = b"\r\n" in raw
        text = raw.decode("utf-8").replace("\r\n", "\n")
        applied = []
        for old, new, n_min in subs:
            n = text.count(old)
            if n < n_min:
                raise SystemExit(f"⛔ {src}: substitution {old!r} matched {n} < {n_min}")
            text = text.replace(old, new)
            applied.append({"old": old, "new": new, "count": n})
        b = blob(REL + src)
        cm = "#"
        hdr = (f"{cm} PORTED from the refcv6 NavSim suite: {REL}{src} (git blob {b} at "
               f"{COMMIT[:12]}) by code/port_from_refcv6.py; substitutions: "
               f"{json.dumps([(x['old'], x['new'], x['count']) for x in applied], ensure_ascii=False)}\n")
        lines = text.split("\n")
        if lines and lines[0].startswith("#!"):
            lines.insert(1, hdr.rstrip("\n"))
        else:
            lines.insert(0, hdr.rstrip("\n"))
        out = "\n".join(lines)
        if crlf:
            out = out.replace("\n", "\r\n")
        p = os.path.join(HERE, dst)
        with open(p, "wb") as fh:
            fh.write(out.encode("utf-8"))
        man["files"][dst] = {"src": REL + src, "src_blob": b, "substitutions": applied,
                             "sha256": hashlib.sha256(out.encode("utf-8")).hexdigest()}
    json.dump(man, open(os.path.join(os.path.dirname(HERE), "raw", "PORT_MANIFEST.json"), "w",
                        encoding="utf-8"), indent=1, ensure_ascii=False)
    for k, v in man["files"].items():
        print(k, "<-", v["src_blob"][:10], len(v["substitutions"]), "subs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
