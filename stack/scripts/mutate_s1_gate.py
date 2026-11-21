"""Mutation proof for the S1 collision-gate harness core (PREREG_S1 S1A.6, 2026-09-19).

Each mutation reintroduces ONE defect in a scratch copy of ``taniteval/tools/s1_gate.py``, and
the named tests in ``stack/tests/test_s1_gate.py`` must turn RED; the unmutated control must be
fully GREEN. M-a..M-d are the four pre-registered in S1A.6; M-f and M-g reintroduce two defects
actually met while building it (a PRED velocity that forgets the ego's own v0, and a hard-coded
candidate count -- the ERRATUM-1 class).

⛔ Wrong-disk guard: the scratch ``conftest.py`` aborts unless ``s1_gate`` AND
``tanitad.rl.pdm_proxy`` import from the scratch copy -- an unmutated import would make every
mutation look "uncaught" (the MSYS/editable-install trap).

Usage:  python stack/scripts/mutate_s1_gate.py [--out <json>]
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
TEST = "stack/tests/test_s1_gate.py"
MOD = "taniteval/tools/s1_gate.py"

MUTATIONS = [
    ("M-a_gate_mask_deleted", MOD,
     '    rg = r.masked_fill(~gate_ok.bool(), float("-inf"))\n',
     "    rg = r  # MUTATION M-a\n",
     ["test_GATE_takes_the_free_candidate_and_CONST_changes_NOTHING",
      "test_all_blocked_falls_back_to_BASE",
      "test_the_MODEL_mask_is_honoured_before_the_gate"]),
    ("M-b_scored_against_the_GATE_tracks", MOD,
     "    true = score_true(states, human, tracks_rec, route, dac_cand, dac_human, cfg)\n",
     '    true = score_true(states, human, gate_tracks.get("GATE_PRED", tracks_rec), route,'
     " dac_cand, dac_human, cfg)  # MUTATION M-b\n",
     ["test_every_pick_is_SCORED_against_the_RECORDED_future"]),
    ("M-c_checker_stubbed_to_all_free", MOD,
     "    return P.no_at_fault_collision(states, tracks, cfg) == 1\n",
     "    return torch.ones(states.shape[0], dtype=torch.bool)  # MUTATION M-c\n",
     ["test_GATE_takes_the_free_candidate_and_CONST_changes_NOTHING"]),
    ("M-d_spline_v0_boundary_broken", MOD,
     "                      bc_type=((1, np.full(m, float(v0))), (2, np.zeros(m))))\n",
     "                      bc_type=((1, np.zeros(m)), (2, np.zeros(m))))  # MUTATION M-d\n",
     ["test_spline_STRAIGHT_constant_velocity_is_EXACT",
      "test_candidate_states_equal_the_HUMAN_constructor_on_the_same_path"]),
    ("M-f_pred_velocity_forgets_v0", MOD,
     '                    "vx": vrx + float(v0), "vy": vry})\n',
     '                    "vx": vrx, "vy": vry})  # MUTATION M-f\n',
     ["test_PRED_agents_threshold_classes_and_add_v0"]),
    ("M-g_hardcoded_N_117", MOD,
     '    out["RANDOM"] = {k: float(true[k].double().mean()) for k in keys}\n',
     '    out["RANDOM"] = {k: float(true[k][:117].double().mean()) for k in keys}'
     "  # MUTATION M-g\n",
     ["test_RANDOM_is_the_EXACT_fan_mean_over_N_read_from_the_fan"]),
]

CONFTEST = '''import sys
from pathlib import Path
_R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_R / "stack"))
sys.path.insert(0, str(_R / "taniteval" / "tools"))
import s1_gate as _s
import tanitad.rl.pdm_proxy as _p
for _m in (_s, _p):
    if not str(Path(_m.__file__).resolve()).startswith(str(_R)):
        raise SystemExit("WRONG-DISK IMPORT: %s from %s, not %s" % (_m.__name__, _m.__file__, _R))
'''


def _copy(dst: Path) -> None:
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    shutil.copytree(STACK / "tanitad", dst / "stack" / "tanitad", ignore=ign)
    (dst / "stack" / "tests").mkdir(parents=True)
    shutil.copy2(REPO / TEST, dst / TEST)
    (dst / "stack" / "tests" / "conftest.py").write_text(CONFTEST, encoding="utf-8")
    (dst / "taniteval" / "tools").mkdir(parents=True)
    shutil.copy2(REPO / MOD, dst / MOD)


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
                                                        str(dst / "taniteval" / "tools")]),
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", TEST, "-p",
                        "no:cacheprovider"], cwd=str(dst), env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    out = r.stdout + r.stderr
    if "WRONG-DISK IMPORT" in out:
        raise SystemExit("⛔ wrong-disk import -- no verdict:\n" + out[-1500:])
    failed = sorted(set(re.findall(r"(?:FAILED|ERROR) \S+::(\w+)", out)))
    passed = sorted(set(re.findall(r"PASSED \S+::(\w+)", out)))
    return {"rc": r.returncode, "failed": failed, "n_failed": len(failed),
            "n_passed": len(passed), "tail": out[-500:]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = {}
    with tempfile.TemporaryDirectory(prefix="s1_mut_") as tmp:
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
        print("%-40s %s (%d passed / %d failed)" % (k, tag, v["n_passed"], v["n_failed"]))
        if "caught" in v and not v["caught"]:
            print("    must go red:", v["must_go_red"], "\n    went red:", v["failed"])
    print("\nMUTATION VERDICT:", "ALL CAUGHT, control green" if ok else "⛔ FAILED")
    if a.out:
        Path(a.out).write_text(json.dumps({"ok": ok, "results": res}, indent=1),
                               encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
