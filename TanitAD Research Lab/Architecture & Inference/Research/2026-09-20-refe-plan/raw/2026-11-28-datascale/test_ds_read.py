#!/usr/bin/env python3
"""Tests of ds_read.py (the read-side support for `--lever ds_sh|ds_ep`): the lever table, `_ctrl_cv` (incl. CTRL-CV(e), recomputing cv5m from cv_select.json) and its mutations, and the
`_run_head` select check.  SYNTHETIC only.  NOT covered (UNVERIFIED): the full `read` / `confirm` path under the new levers.

  python test_ds_read.py        prints ZZDSREAD_PASS <n> (or ZZDSREAD_FAIL)
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFE = HERE.parent.parent / "refe"
sys.path.insert(0, str(REFE))
sys.path.insert(0, str(HERE))
_argv, sys.argv = sys.argv, [sys.argv[0]]

import numpy as np  # noqa: E402
import ds_read as DR  # noqa: E402
import l2next_read as L2  # noqa: E402
import safehead_data as D  # noqa: E402
import test_safehead as TS  # noqa: E402

N = 0


def ok(name):
    global N
    N += 1
    print(f"ok {N:02d} {name}", flush=True)


def make_run(tmp, fam, perm=None, tag=""):
    w = TS.make_world(tmp / f"w_{fam}{tag}", n_tr_logs=6, n_ho_logs=3, per_log=24, seed=5)
    out = tmp / f"run_{fam}{tag}"
    a = TS.argv_for(w, out, perm=perm, extra=["--smoke-steps", "40"])
    a = TS._swap(a, "--save-steps", "0,20,40") + ["--select", "cv5m"]
    if fam == "ds_ep":
        a = TS._swap(a, "--rows", "2")
    r = TS.trainer(a)
    assert r.returncode == 0, r.stderr[-600:]
    tc = [r_ for r_ in D.read_index(w["sealed"]) if r_["split"] in D.SPLITS_NAVTRAIN]
    return out, tc


def args(**kw):
    return types.SimpleNamespace(**{"n_train_tokens": None, "n_train_logs": None, **kw})


def test_levers():
    DR.install()
    assert set(DR.DS_LEVERS) <= set(L2.LEVERS) and {"shmlp", "ephead"} <= set(L2.LEVERS)
    for k, lv in DR.DS_LEVERS.items():
        assert L2.LEVERS[k]["cv"] and L2.LEVERS[k]["select_file"] == "cv_select.json" and L2.LEVERS[k]["null"] == "NULL-CV" and set(lv["next"]) == {"NULL-CV", "ADVERSE", "ADOPT", "SEPARATED-SMALL", "NULL"}
    assert L2.LEVERS["ds_sh"]["rows"] == (0, 1, 3, 5) and L2.LEVERS["ds_ep"]["rows"] == (2,) and L2.LEVERS["ds_ep"]["prefix"] == "refe_ds_ep"
    assert L2._ctrl_cv is DR._ctrl_cv_ds and L2._run_head is DR._run_head_ds
    DR.install()                                                                             # idempotent
    o = L2.outcome({"s0": {"chosen_step": 0}}, True, 0, "ds_ep")
    assert o["outcome"] == "NULL-CV" and "learning curve" in o["next_arm"]
    ok("levers: ds_sh / ds_ep registered beside shmlp / ephead; install is idempotent; a step-0 choice reads NULL-CV with the card's learning-curve next arm")


def test_ctrl_cv(tmp, fam):
    run, tc = make_run(tmp, fam)
    sel = json.loads((run / "cv_select.json").read_text())
    r = DR._ctrl_cv_ds(D, run, tc, sel["chosen_step"], args())
    assert r["pass"] and r["a_folds_ok"] and r["b_counts_ok"] and r["c_all_tokens_fit"] and r["d_chosen_consistent"] and r["e_chosen_equals_cv5m_recomputed"] and r["n_tokens"] == 216 and r["n_logs"] == 9, r
    assert DR._ctrl_cv_ds(D, run, tc, sel["chosen_step"], args(n_train_tokens=216, n_train_logs=9))["pass"]
    assert not DR._ctrl_cv_ds(D, run, tc, sel["chosen_step"], args(n_train_tokens=2272, n_train_logs=35))["b_counts_ok"], "the parent's 2,272 / 35 must not pass for this cache"
    # mutations: each of the five clauses can FAIL
    def mutate(fn, key):
        d = tmp / f"mut_{fam}_{key}"
        d.mkdir()
        for p in run.iterdir():
            (d / p.name).write_bytes(p.read_bytes())
        s = json.loads((d / "cv_select.json").read_text())
        b = json.loads((d / "built.json").read_text())
        rj = json.loads((d / "run.json").read_text())
        fn(s, b, rj)
        (d / "cv_select.json").write_text(json.dumps(s))
        (d / "built.json").write_text(json.dumps(b))
        (d / "run.json").write_text(json.dumps(rj))
        return DR._ctrl_cv_ds(D, d, tc, sel["chosen_step"], args())
    other = [s for s in (0, 20, 40) if s != sel["chosen_step"]][-1]
    cases = {
        "a_folds": (lambda s, b, rj: b["cv_fold_of_log"].__setitem__(sorted(b["cv_fold_of_log"])[0], [4]), "a_folds_ok"),
        "b_counts": (lambda s, b, rj: s.__setitem__("n_logs", 8), "b_counts_ok"),
        "c_fit": (lambda s, b, rj: b.__setitem__("n_train", 100), "c_all_tokens_fit"),
        "d_chosen": (lambda s, b, rj: rj.__setitem__("chosen_step", other), "d_chosen_consistent"),
        "e_table": (lambda s, b, rj: s["per_log"][sorted(s["per_log"])[0]]["per_step"][str(other)].__setitem__("pdms", 1e3), "e_chosen_equals_cv5m_recomputed"),
        "e_rule_margin": (lambda s, b, rj: s["margin_rule"].__setitem__("margin", "point_gain_gt_0"), "e_chosen_equals_cv5m_recomputed"),
        "e_built_rule": (lambda s, b, rj: b["cv_rule"].__setitem__("guard", ["nc", "dac", "ttc", "c"]), "e_chosen_equals_cv5m_recomputed"),
        "e_chosen_edit": (lambda s, b, rj: (s.__setitem__("chosen_step", other), rj.__setitem__("chosen_step", other)), "d_chosen_consistent"),
    }
    for k, (fn, clause) in cases.items():
        m = mutate(fn, k)
        assert not m["pass"] and not m[clause], (fam, k, m)
    ok(f"[{fam}] CTRL-CV (ds): PASS on a real cv5m run dir (216 tokens / 9 logs derived from the sealed index, never the parent's 2,272 / 35); each of {len(cases)} mutations (a moved fold, wrong counts, a short "
       "final fit, a edited chosen step, a perturbed per-log table, a changed margin rule, a changed built rule) turns its clause RED")


def test_run_head_select(tmp):
    run, tc = make_run(tmp, "ds_ep", tag="_hd")
    st = json.loads((run / "cv_select.json").read_text())["chosen_step"]
    r = DR._run_head_ds(D, run, "s0", "ds_ep")
    assert not any("head select" in p for p in r["problems"]), r["problems"]                 # the cv5m head is accepted by the select check
    assert any("SMOKE" in p for p in r["problems"]), "the other checks still run (the synthetic head is a gate smoke head)"
    ok(f"_run_head: a cv5m head is not flagged for its select (the original would flag it); every other check of l2next_read still runs (smoke head refused, step {st})")


def main() -> int:
    tmp = TS.mkdtemp()
    try:
        test_levers()
        for fam in ("ds_sh", "ds_ep"):
            test_ctrl_cv(tmp, fam)
        test_run_head_select(tmp)
    except Exception as e:                                                                  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"ZZDSREAD_FAIL {type(e).__name__}: {str(e)[:300]}")
        return 1
    print(f"ZZDSREAD_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
