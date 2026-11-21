"""Mutation proof for P3's target pre-build (`stack/scripts/prebuild_p3_targets.py`).

Seven mutations, each reintroducing a real defect this programme has actually made:
the population boundary drifting from the scorer's, a truncation that keeps the wrong rows,
NO_LABEL collapsed into labelled-clear, an offset array that skips unlabelled windows, and —
twice — a cross-check that passes when it should refuse.

The scratch copy carries the pre-build, the scorer it must agree with, and the INNER `taniteval`
package the scorer imports. ⛔ Without that last one the control fails on an import and the proof
is void — MEASURED, it did exactly that on the scorer's own first run.

Usage: python stack/scripts/mutate_prebuild_p3.py [--out <json>]
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
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STACK = REPO / "stack"
MOD = "stack/scripts/prebuild_p3_targets.py"
SCORER = "taniteval/tools/box_quality.py"
TEST = "stack/tests/test_prebuild_p3_targets.py"

MUTATIONS = [
    ("M1_gate_boundary_becomes_EXCLUSIVE", MOD,
     "    return (cx >= 0.0) & (cx <= X_FWD_M) & (np.abs(cy) <= Y_HALF_M)\n",
     "    return (cx > 0.0) & (cx < X_FWD_M) & (np.abs(cy) < Y_HALF_M)\n",
     ["test_the_gate_population_is_the_scorer_s_population",
      "test_the_boundary_literals_match_box_quality_exactly"]),
    ("M2_gate_window_drifts_from_the_scorer_s", MOD,
     "X_FWD_M, Y_HALF_M = 60.0, 16.0\n",
     "X_FWD_M, Y_HALF_M = 100.0, 16.0  # MUTATION M2\n",
     ["test_the_boundary_literals_match_box_quality_exactly"]),
    ("M3_truncation_keeps_the_FARTHEST", MOD,
     "    return np.argsort(np.hypot(np.asarray(cx, dtype=np.float64),\n"
     "                               np.asarray(cy, dtype=np.float64)))[:int(n)]\n",
     "    return np.argsort(-np.hypot(np.asarray(cx, dtype=np.float64),\n"
     "                                np.asarray(cy, dtype=np.float64)))[:int(n)]\n",
     ["test_nearest_n_keeps_the_CLOSEST_and_drops_the_rest"]),
    ("M4_NO_LABEL_collapsed_into_labelled_clear", MOD,
     "        if ag is None:                                   # NO_LABEL ≠ labelled-clear\n"
     "            lab.append(False)\n",
     "        if ag is None:                                   # MUTATION M4\n"
     "            lab.append(True)\n",
     ["test_the_census_counts_NO_LABEL_and_labelled_clear_DIFFERENTLY"]),
    ("M5_offsets_skip_unlabelled_windows", MOD,
     "            sel_off.append(sel_off[-1])\n            continue\n",
     "            continue  # MUTATION M5: no offset for this window\n",
     ["test_the_banked_selection_indexes_the_JOIN_S_OWN_ROWS"]),
    ("M6_no_literal_compared_reads_OK", MOD,
     '    if n_checked == 0:\n        return checks, bad, "UNVERIFIED — no literal was compared; '
     'census only, do NOT bank"\n',
     "    pass  # MUTATION M6: an unchecked grid passes\n",
     ["test_NO_literal_compared_is_UNVERIFIED_never_OK"]),
    ("M7_a_mismatch_no_longer_refuses", MOD,
     "        if got != want:\n            bad.append(key)\n",
     "        pass  # MUTATION M7: mismatches are recorded but never refuse\n",
     ["test_ONE_mismatched_literal_is_INCONCLUSIVE_and_names_the_key"]),
]

#: ⛔ THE GUARD MUST COVER THE DEPENDENCIES, NOT ONLY THE MODULE UNDER TEST. MEASURED
#: 2026-09-20: an earlier version asserted only `prebuild_p3_targets.__file__` and the control
#: went RED anyway — `box_quality` pulls in `tanitad`, which the venv's EDITABLE install resolves
#: to the ABANDONED G: checkout, and that read died with `OSError: [Errno 22]`. The guard passed
#: while the run was reading another disk. Every module the proof depends on is asserted here.
CONFTEST = '''import sys
from pathlib import Path
_R = Path(__file__).resolve().parents[2]
for _p in ("stack", "stack/scripts", "taniteval", "taniteval/tools"):
    sys.path.insert(0, str(_R / _p))
import prebuild_p3_targets as _m
import box_quality as _b
import tanitad as _t
for _name, _mod in (("prebuild_p3_targets", _m), ("box_quality", _b), ("tanitad", _t)):
    _f = getattr(_mod, "__file__", None)
    if not _f or not str(Path(_f).resolve()).startswith(str(_R)):
        raise SystemExit("WRONG-DISK IMPORT: %s -> %s not under %s" % (_name, _f, _R))
'''


def _copy(dst: Path) -> None:
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    (dst / "stack" / "scripts").mkdir(parents=True)
    (dst / "stack" / "tests").mkdir(parents=True)
    (dst / "taniteval" / "tools").mkdir(parents=True)
    # ⛔ the `tanitad` package itself: without it the venv's editable install serves it from
    # the abandoned G: checkout and the control dies on an unrelated OSError (MEASURED).
    shutil.copytree(STACK / "tanitad", dst / "stack" / "tanitad", ignore=ign)
    shutil.copy2(REPO / MOD, dst / MOD)
    shutil.copy2(REPO / SCORER, dst / SCORER)
    shutil.copy2(REPO / TEST, dst / TEST)
    (dst / "stack" / "tests" / "conftest.py").write_text(CONFTEST, encoding="utf-8")
    shutil.copytree(REPO / "taniteval" / "taniteval", dst / "taniteval" / "taniteval",
                    ignore=ign)


def _apply(dst: Path, rel: str, old: str, new: str) -> None:
    p = dst / rel
    raw = p.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")
    if s.count(old) != 1:
        raise SystemExit("anchor count %d in %s: %r" % (s.count(old), rel, old[:70]))
    s = s.replace(old, new, 1)
    p.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))


def _run(dst: Path) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               PYTHONIOENCODING="utf-8",
               PYTHONPATH=os.pathsep.join(str(dst / p) for p in
                                          ("stack", "stack/scripts", "taniteval",
                                           "taniteval/tools")))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", TEST,
                        "-p", "no:cacheprovider"], cwd=str(dst), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = r.stdout + r.stderr
    if "WRONG-DISK IMPORT" in out:
        raise SystemExit("⛔ wrong-disk import — no verdict:\n" + out[-1200:])
    return {"rc": r.returncode,
            "failed": sorted(set(re.findall(r"(?:FAILED|ERROR) \S+::(\w+)", out))),
            "n_failed": len(set(re.findall(r"(?:FAILED|ERROR) \S+::(\w+)", out))),
            "n_passed": len(set(re.findall(r"PASSED \S+::(\w+)", out))),
            "tail": out[-400:]}


# ⛔ A CHECKER MUST NOT DIE ON ITS OWN OUTPUT. MEASURED 2026-09-20: three separate
# readouts crashed with a cp1252 `UnicodeEncodeError` on this box mid-print -- one of
# them after reporting "lines lost = 1" but BEFORE naming the line, i.e. it had verified
# nothing while looking like it had. Relying on the caller to export PYTHONIOENCODING is
# a habit; this is a guard. `errors="replace"` means the print degrades instead of
# raising even if the stream cannot take utf-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 -- a stream that cannot be reconfigured is not fatal
    pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = {}
    with tempfile.TemporaryDirectory(prefix="p3_mut_") as tmp:
        c = Path(tmp) / "M0"
        _copy(c)
        res["M0_control"] = _run(c)
        for name, rel, old, new, red in MUTATIONS:
            d = Path(tmp) / name
            _copy(d)
            _apply(d, rel, old, new)
            r = _run(d)
            r["must_go_red"], r["caught"] = red, all(t in r["failed"] for t in red)
            res[name] = r
    ok = (res["M0_control"]["rc"] == 0 and res["M0_control"]["n_failed"] == 0
          and all(v["caught"] for k, v in res.items() if k != "M0_control"))
    for k, v in res.items():
        tag = ("GREEN (control)" if k == "M0_control" and v["n_failed"] == 0 else
               "CAUGHT" if v.get("caught") else "⛔ NOT CAUGHT")
        print("%-42s %s (%d passed / %d failed)" % (k, tag, v["n_passed"], v["n_failed"]))
        if k == "M0_control" and v["n_failed"]:
            print("    ⛔ CONTROL IS RED — the proof is VOID:\n", v["tail"])
        if "caught" in v and not v["caught"]:
            print("    must go red:", v["must_go_red"], "\n    went red:", v["failed"])
    print("\nMUTATION VERDICT:", "ALL CAUGHT, control green" if ok else "⛔ FAILED")
    if a.out:
        Path(a.out).write_text(json.dumps({"ok": ok, "results": res}, indent=1),
                               encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
