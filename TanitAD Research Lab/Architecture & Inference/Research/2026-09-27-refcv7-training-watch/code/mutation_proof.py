"""Mutation proof for the refcv7 Training Watch: every alarm's historical defect, reintroduced into a COPY of
the builder, must turn `test_build_watch_refcv7.py` RED. A guard whose test stays green under its own defect
guards nothing (CLAUDE.md: "a mutation that reintroduces the real defect and must go RED").

    python mutation_proof.py <tree> <out.json>

<tree> holds taniteval/tools/training_watch/{build_watch_refcv7.py, watch.css},
taniteval/tests/test_build_watch_refcv7.py and taniteval/conftest.py. A scratch copy is made under
C:/Users/Admin/w7mut_<pid>; the source tree is never edited. Each anchor must occur EXACTLY once, or the
mutation is recorded as NOT-APPLIED (a mutation that cannot land proves nothing and is never a pass).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time

FILES = ("taniteval/tools/training_watch/build_watch_refcv7.py", "taniteval/tools/training_watch/watch.css",
         "taniteval/tests/test_build_watch_refcv7.py", "taniteval/conftest.py")
B = "taniteval/tools/training_watch/build_watch_refcv7.py"

# (id, what it reintroduces, anchor, replacement)
MUTATIONS = [
    ("M1", "a key ABSENT from the row is drawn as 0",
     'if row is None or k not in row:\n        return "missing", None',
     'if row is None or k not in row:\n        return "value", 0.0'),
    ("M2", "an UNDEFINED (null) IoU is drawn as 0",
     'if v is None:\n        return "undef", None',
     'if v is None:\n        return "value", 0.0'),
    ("M3", "a 20-60 m reading at or below 0.05 is RED (the informative amber promoted past the registered rule)",
     'out["amber"].append((r["step"], c, b, v, THIN_FLOOR))',
     'out["breaches"].append((r["step"], c, b, v, THIN_FLOOR))'),
    ("M4", "the registered start step dropped (a step-4,500 dip alarms)",
     "THIN_FROM_STEP = 5000", "THIN_FROM_STEP = 0"),
    ("M5", "only the LATEST eval is barred (the LOGGING_SPEC 5.3 latch lost)",
     'if r["step"] < THIN_FROM_STEP:\n            continue',
     'if r["step"] < THIN_FROM_STEP or r is not ev[-1]:\n            continue'),
    ("M6", "a missing key prints its literal name (the gate's text check is fooled)",
     'tds.append(f\'<td class="hm unav" data-cell="{c} {b}" title=',
     'tds.append(f\'<td class="hm unav" data-cell="{c} {b}" data-key="{map_key(stat, c, b)}" title='),
    ("M7", "the A10 band widened to admit refcv6's 3.40",
     "BOX_RATIO_BAND = (0.5, 1.5)", "BOX_RATIO_BAND = (0.5, 4.0)"),
    ("M8", "the ratio cross-check against the pooled counts removed",
     "if abs(want - d[\"ratio\"]) > 1e-4 * max(1.0, abs(want)):",
     "if False:"),
    ("M9", "a declared ga_* key ABSENT from the row is not an alarm",
     'if v == "ABSENT" or _num(v) is None:\n                out["absent"].append(k)',
     'if False:\n                out["absent"].append(k)'),
    ("M10", "a declared ga_* key at EXACTLY 0 is not an alarm",
     'elif _num(v) == 0:\n                out["zero"].append(k)',
     'elif False:\n                out["zero"].append(k)'),
    ("M12", "the IoU-from-counts cross-check reads agreement when it checked nothing",
     "        if not nchk:\n", "        if False:\n"),
    ("M13", "a class with labelled cells and ZERO pull is not an alarm (LOGGING_SPEC_MAP10 5.2)",
     "if gv is not None and gv == 0 and nc:", "if False:"),
    ("M14", "the box ratio alarm armed from the first eval (the warm-up ruling dropped)",
     "BOX_ARM_STEP = 5000", "BOX_ARM_STEP = 0"),
    ("M15", "beyond 60 m coloured like 0-60 m (the values-only ruling dropped)",
     "            elif b in PLAIN_BANDS:\n", "            elif False:\n"),
    ("M16", "a ground-truth condition added to the registered 0-20 m rule (the literal rule narrowed)",
     'kind, v = _cell(r, map_key("iou", c, RED_BAND))\n            if kind == "value" and v <= THIN_FLOOR:',
     'kind, v = _cell(r, map_key("iou", c, RED_BAND))\n            if kind == "value" and v <= THIN_FLOOR and '
     '(_num(r.get(map_key("n", c, RED_BAND))) or 0) > 0:'),
    ("M17", "the trainer's ratio flag held against the ARMED state (a false disagreement while warming up)",
     "bool(fl) != outside:", 'bool(fl) != d["alarm"]:'),
    ("M11", "the NavSim count guard disabled (a partial split is shown)",
     '    navtest token subset (it names its token file and carries its own n). Otherwise the refusal text."""\n',
     '    navtest token subset (it names its token file and carries its own n). Otherwise the refusal text."""\n'
     '    return None\n'),
]


def _py() -> str:
    return sys.executable


def run_suite(tree: str) -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               PYTHONPATH=os.pathsep.join([os.path.join(tree, "taniteval"), tree]))
    t0 = time.time()
    r = subprocess.run([_py(), "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rf",
                        "taniteval/tests/test_build_watch_refcv7.py"], cwd=tree, env=env,
                       capture_output=True, timeout=900)
    out = r.stdout.decode("utf-8", "replace")
    failed = sorted(set(re.findall(r"^FAILED [^:]+::(\S+)", out, flags=re.M)))
    m = re.search(r"(\d+) passed", out)
    f = re.search(r"(\d+) failed", out)
    return {"rc": r.returncode, "passed": int(m.group(1)) if m else 0, "failed": int(f.group(1)) if f else 0,
            "failed_tests": failed, "s": round(time.time() - t0, 1), "tail": out.strip().splitlines()[-1:] }


def main(argv) -> int:
    src, out_json = argv[0], argv[1]
    scratch = os.path.join(r"C:\Users\Admin", f"w7mut_{os.getpid()}")
    if os.path.exists(scratch):
        raise SystemExit(f"scratch {scratch} exists; refusing to reuse it")
    for f in FILES:
        os.makedirs(os.path.dirname(os.path.join(scratch, f)), exist_ok=True)
        shutil.copyfile(os.path.join(src, f), os.path.join(scratch, f))
    bp = os.path.join(scratch, B)
    pristine = open(bp, encoding="utf-8", newline="").read()
    rec = {"source_tree": src, "scratch": scratch, "python": sys.version.split()[0], "control": None,
           "mutations": []}
    rec["control"] = run_suite(scratch)
    for mid, what, anchor, repl in MUTATIONS:
        n = pristine.count(anchor)
        row = {"id": mid, "reintroduces": what, "anchor_hits": n}
        if n != 1:
            row["verdict"] = "NOT-APPLIED"
        else:
            with open(bp, "w", encoding="utf-8", newline="") as fh:
                fh.write(pristine.replace(anchor, repl))
            res = run_suite(scratch)
            row.update(res)
            row["verdict"] = "CAUGHT" if res["failed"] > 0 else "ESCAPED"
            with open(bp, "w", encoding="utf-8", newline="") as fh:
                fh.write(pristine)
        rec["mutations"].append(row)
        print(f"{mid} {row['verdict']:11s} {row.get('failed', '-')} failed  {what}", flush=True)
    rec["control_after"] = run_suite(scratch)
    ok = (rec["control"]["failed"] == 0 and rec["control_after"]["failed"] == 0
          and all(r["verdict"] == "CAUGHT" for r in rec["mutations"]))
    rec["verdict"] = "PASS" if ok else "FAIL"
    with open(out_json, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"ZZMUT-{rec['verdict']}-{sum(r['verdict'] == 'CAUGHT' for r in rec['mutations'])}"
          f"-{len(rec['mutations'])}ZZ")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
