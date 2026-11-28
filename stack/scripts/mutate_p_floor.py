"""Mutation proof for the P-panel floor reporter (`stack/scripts/p_floor.py`).

⛔ The reporting rule was committed BEFORE the numbers existed so it could not drift afterwards.
A rule that only lives in prose drifts anyway, so each clause is broken on purpose here:

  M1  the "floor" is computed against A8 instead of the replicate  -> it stops being a seed floor
  M2  the upper-bound label is dropped                             -> two blocks read as two floors
  M3  one replicate is accepted as a floor                         -> H-ESTIM-SEED-1 violated
  M4  the families are pooled into one score                       -> hides the trade-off
  M5  a non-finite value reads as a zero delta                     -> "broken" reads as "no change"
  M6  a missing metric is silently dropped                         -> the omission the rule forbids

`p_floor` imports nothing from the stack, so the scratch copy is the module, its test and a
conftest that refuses a wrong-disk import of EITHER.

Usage: python stack/scripts/mutate_p_floor.py [--out <json>]
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
MOD = "stack/scripts/p_floor.py"
TEST = "stack/tests/test_p_floor.py"

MUTATIONS = [
    ("M1_the_floor_is_computed_against_A8", MOD,
     "    floor = families(p0, p0b)\n",
     "    floor = families(p0, a8 if a8 is not None else p0b)  # MUTATION M1\n",
     ["test_the_floor_is_P0_vs_P0b_and_is_LABELLED_as_such",
      "test_the_code_delta_gap_is_reported_because_it_IS_the_code_delta"]),
    # ⛔ THE DEMOTION MUTATION: the population label still EXISTS but is moved away from the
    # number it qualifies. An existence-predicate test passes this; an adjacency test must not.
    # This is the same shape as the landing guard's "appears anywhere" limit and as the
    # doc-checker's M4 — a claim about PLACE cannot be guarded by a check for PRESENCE.
    ("M_pop_label_DEMOTED_to_the_end_of_the_report", MOD,
     '             "  population: %s" % HEAD_POPULATION,' + chr(10) +
     '             "  seeds: %s" % (sf.get("seeds") or {})]' + chr(10),
     '             "  seeds: %s" % (sf.get("seeds") or {}),' + chr(10) +
     '             "  population: %s" % HEAD_POPULATION]  # MUTATION: demoted' + chr(10),
     ["test_the_headline_carries_its_POPULATION_in_json_and_text"]),
    ("M2_the_upper_bound_label_is_dropped", MOD,
     '            "admissible_as": "UPPER BOUND ONLY - never quote this as the floor",\n',
     '            "admissible_as": "a difference",  # MUTATION M2\n',
     ["test_the_A8_comparison_is_an_UPPER_BOUND_and_is_never_the_floor"]),
    ("M3_one_replicate_is_accepted_as_a_floor", MOD,
     "    if p0 is None or p0b is None:\n",
     "    if p0 is None:  # MUTATION M3: a single replicate now passes\n",
     ["test_ONE_replicate_is_NOT_a_floor"]),
    ("M4_the_families_are_POOLED_into_one_score", MOD,
     '            "by_family": floor,\n',
     '            "by_family": floor,\n            "total": sum(\n'
     '                v["abs_delta"] or 0.0 for d in floor.values() for v in d.values()),\n',
     ["test_every_family_is_reported_SEPARATELY_and_never_pooled"]),
    ("M5_a_non_finite_value_reads_as_a_ZERO_delta", MOD,
     "        if not (math.isfinite(fa) and math.isfinite(fb)):\n"
     '            out[k] = {"abs_delta": None, "reason": "non-finite"}\n            continue\n',
     "        pass  # MUTATION M5: non-finite falls through to the subtraction\n",
     ["test_non_finite_or_non_numeric_is_NEVER_a_zero_delta"]),
    ("M6_a_missing_metric_is_silently_dropped", MOD,
     '        if va is None or vb is None:\n            out[k] = {"abs_delta": None, "reason": "absent in %s" % (\n'
     '                "both" if va is None and vb is None else ("A" if va is None else "B"))}\n            continue\n',
     "        if va is None or vb is None:\n            continue  # MUTATION M6: dropped\n",
     ["test_a_missing_metric_is_REPORTED_with_its_reason_not_dropped"]),
]

CONFTEST = '''import sys
from pathlib import Path
_R = Path(__file__).resolve().parents[2]
for _p in ("stack", "stack/scripts"):
    sys.path.insert(0, str(_R / _p))
import p_floor as _m
_f = getattr(_m, "__file__", None)
if not _f or not str(Path(_f).resolve()).startswith(str(_R)):
    raise SystemExit("WRONG-DISK IMPORT: p_floor -> %s not under %s" % (_f, _R))
'''


def _copy(dst: Path) -> None:
    (dst / "stack" / "scripts").mkdir(parents=True)
    (dst / "stack" / "tests").mkdir(parents=True)
    shutil.copy2(REPO / MOD, dst / MOD)
    shutil.copy2(REPO / TEST, dst / TEST)
    (dst / "stack" / "tests" / "conftest.py").write_text(CONFTEST, encoding="utf-8")


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
               PYTHONPATH=os.pathsep.join(str(dst / p) for p in ("stack", "stack/scripts")))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", TEST,
                        "-p", "no:cacheprovider"], cwd=str(dst), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = r.stdout + r.stderr
    if "WRONG-DISK IMPORT" in out:
        raise SystemExit("⛔ wrong-disk import — no verdict:\n" + out[-1200:])
    failed = sorted(set(re.findall(r"(?:FAILED|ERROR) \S+::(\w+)", out)))
    return {"rc": r.returncode, "failed": failed, "n_failed": len(failed),
            "n_passed": len(set(re.findall(r"PASSED \S+::(\w+)", out))), "tail": out[-400:]}


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
    with tempfile.TemporaryDirectory(prefix="pf_mut_") as tmp:
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
        Path(a.out).write_text(json.dumps({"ok": ok, "results": res}, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
