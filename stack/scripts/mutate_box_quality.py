"""Mutation proof for the P-arm scorer (`taniteval/tools/box_quality.py`), 2026-09-20.

⛔ A scorer that shares the defect it checks for is green forever, so each load-bearing line is
broken in a SCRATCH COPY and the named tests must turn RED; the unmutated control must be GREEN.
M1-M3 are the three the Master Mind required; M4-M5 cover the two remaining ways the instrument
could flatter an arm.

⛔⛔ WRONG-DISK GUARD: the scratch `conftest.py` aborts unless `box_quality` imports from the
scratch copy — an unmutated import would make every mutation look "uncaught".

Usage: python stack/scripts/mutate_box_quality.py [--out <json>]
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

STACK = Path(__file__).resolve().parents[1]
REPO = STACK.parent
TEST = "stack/tests/test_box_quality.py"
MOD = "taniteval/tools/box_quality.py"

MUTATIONS = [
    ("M1_matcher_pairs_one_fewer", MOD,
     "    for i, j in zip(rows, cols):\n",
     "    for i, j in zip(rows[:-1], cols[:-1]):  # MUTATION M1\n",
     ["test_match_pairs_uses_the_TRAINING_matcher_and_pairs_every_target"]),
    ("M2_near_forward_mask_leaks_everything", MOD,
     '    if which == "near_forward":\n'
     "        return (gx >= NEAR_X[0]) & (gx <= NEAR_X[1]) & (np.abs(gy) <= NEAR_Y_HALF)\n",
     '    if which == "near_forward":\n'
     "        return np.ones(gx.shape, dtype=bool)  # MUTATION M2\n",
     ["test_population_geometry_reads_KNOWN_points",
      "test_improving_ONLY_far_behind_must_NOT_move_the_near_forward_number",
      "test_BOTH_populations_are_always_emitted_and_the_primary_is_the_gate_s"]),
    ("M3_velocity_floor_taken_from_the_PREDICTION", MOD,
     "            vf.append(float(vg.norm()))\n",
     '            vf.append(float(pred["rates"][0, i, :2].float().norm()))  # MUTATION M3\n',
     ["test_match_pairs_uses_the_TRAINING_matcher_and_pairs_every_target"]),
    ("M4_only_the_primary_population_is_emitted", MOD,
     '    for pop in POPULATIONS + ("far_or_behind",):\n',
     "    for pop in (POPULATIONS[0],):  # MUTATION M4\n",
     ["test_BOTH_populations_are_always_emitted_and_the_primary_is_the_gate_s",
      "test_headline_REFUSES_one_population_without_the_other"]),
    ("M5_the_bar_is_read_ten_times_too_loose", MOD,
     '    return {"n": int(mask.sum()), "centre_l1_m": round(l1, 4),\n',
     '    return {"n": int(mask.sum()), "centre_l1_m": round(l1, 4),  # MUTATION M5 below\n',
     ["test_the_bar_is_read_at_2m_on_the_PRIMARY"]),
]
#: M5 needs a second edit in the same file (the bar itself), applied with it.
M5_EXTRA = ('            "meets_bar": bool(l1 < BAR_M), "bar_m": BAR_M}\n',
            '            "meets_bar": bool(l1 < 10 * BAR_M), "bar_m": BAR_M}\n')

CONFTEST = '''import sys
from pathlib import Path
_R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_R / "stack"))
sys.path.insert(0, str(_R / "taniteval" / "tools"))
sys.path.insert(0, str(_R / "taniteval"))
import box_quality as _b
if not str(Path(_b.__file__).resolve()).startswith(str(_R)):
    raise SystemExit("WRONG-DISK IMPORT: %s not under %s" % (_b.__file__, _R))
'''


def _copy(dst: Path) -> None:
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    shutil.copytree(STACK / "tanitad", dst / "stack" / "tanitad", ignore=ign)
    (dst / "stack" / "tests").mkdir(parents=True)
    shutil.copy2(REPO / TEST, dst / TEST)
    (dst / "stack" / "tests" / "conftest.py").write_text(CONFTEST, encoding="utf-8")
    (dst / "taniteval" / "tools").mkdir(parents=True)
    shutil.copy2(REPO / MOD, dst / MOD)
    # the bootstrap the scorer calls lives in the INNER taniteval package; without it the
    # control fails on an import and the whole proof is void (MEASURED: it did).
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
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(dst / "stack"),
                                                        str(dst / "taniteval" / "tools"),
                                                        str(dst / "taniteval")]),
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", TEST, "-p",
                        "no:cacheprovider"], cwd=str(dst), env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    out = r.stdout + r.stderr
    if "WRONG-DISK IMPORT" in out:
        raise SystemExit("⛔ wrong-disk import -- no verdict:\n" + out[-1200:])
    return {"rc": r.returncode,
            "failed": sorted(set(re.findall(r"(?:FAILED|ERROR) \S+::(\w+)", out))),
            "n_passed": len(set(re.findall(r"PASSED \S+::(\w+)", out))),
            "tail": out[-400:]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = {}
    with tempfile.TemporaryDirectory(prefix="bq_mut_") as tmp:
        c = Path(tmp) / "M0"
        _copy(c)
        res["M0_control"] = _run(c)
        res["M0_control"]["n_failed"] = len(res["M0_control"]["failed"])
        for name, rel, old, new, red in MUTATIONS:
            d = Path(tmp) / name
            _copy(d)
            _apply(d, rel, old, new)
            if name.startswith("M5"):
                _apply(d, rel, *M5_EXTRA)
            r = _run(d)
            r["n_failed"] = len(r["failed"])
            r["must_go_red"], r["caught"] = red, all(t in r["failed"] for t in red)
            res[name] = r
    ok = (res["M0_control"]["rc"] == 0 and res["M0_control"]["n_failed"] == 0
          and all(v["caught"] for k, v in res.items() if k != "M0_control"))
    for k, v in res.items():
        tag = ("GREEN (control)" if k == "M0_control" and v["n_failed"] == 0 else
               "CAUGHT" if v.get("caught") else "⛔ NOT CAUGHT")
        print("%-42s %s (%d passed / %d failed)" % (k, tag, v["n_passed"], v["n_failed"]))
        if "caught" in v and not v["caught"]:
            print("    must go red:", v["must_go_red"], "\n    went red:", v["failed"])
    print("\nMUTATION VERDICT:", "ALL CAUGHT, control green" if ok else "⛔ FAILED")
    if a.out:
        Path(a.out).write_text(json.dumps({"ok": ok, "results": res}, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
