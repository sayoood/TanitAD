#!/usr/bin/env python3
"""Measure 5r (eval/PREREG_MEASURE5R.md) gate G-R1 -- THE REPAIR MECHANISM, LABEL-FREE.

The two labelling functions of `refe/onpolicy_label_v4.label_one` -- the teacher's `Scorer.score` and NAVSIM's
`navsim_dac.navsim_dac_and_comfort` -- are replaced by STUBS that record their inputs and return dummy labels, so NO label
is computed. On REAL props lines (M5's queue, the first 32 samples) the gate asserts, with --repair-last-heading ON:
  every REFe candidate at BOTH inputs (the 64 originals, every copy, STOP):
    x, y bit-identical to the unrepaired candidate on all 20 poses; heading[0..18] bit-identical;
    heading[19] == the unrepaired heading[18] EXACTLY (the expectation is written here as a literal index operation,
    independently of planner.repair_last_heading);
  the reference candidate (teacher) bit-identical; the SERVED traj / yaw of the written line bit-identical to flag OFF;
  the line carries label_version 5 and "repair": "last_heading_hold".
with the flag OFF: every input is the unrepaired candidate exactly.
DELIBERATE-REGRESSION ARMS, each must turn the gate RED: m1 repair_wrong_index, m2 repair_served, m3 repair_skip_copies.
A control that must read a KNOWN value: on these real proposals heading[19] != heading[18] for most candidates (else the
gate could not tell a repair from its absence) -- the count is reported and must be > 0.

    python eval/selftest_v4r_repair.py        -> eval/raw/m5r/selftest_repair.json
"""
from __future__ import annotations

import glob
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFE = os.path.join(os.path.dirname(HERE), "refe")
sys.path.insert(0, HERE)
sys.path.insert(0, REFE)
QUEUE = "D:/Projects/TanitAD/data/refe_m5/queue"
OUT = os.path.join(HERE, "raw", "m5r", "selftest_repair.json")
N_SAMPLES = 32


def load_rows():
    rows = []
    for p in sorted(glob.glob(os.path.join(QUEUE, "props_r0_*.jsonl.done_*")))[:2]:
        for line in open(p, encoding="utf-8"):
            rows.append(json.loads(line))
    return rows[:N_SAMPLES]


class StubScorer:
    """Records every candidate the teacher would roll out; returns varying dummy values (so ndiff >= 3)."""

    def __init__(self, L4, OL):
        self.L4, self.OL, self.stride, self.seen = L4, OL, 2, None

    def score(self, _sc, _r, cands):
        self.seen = {nm: (xy.numpy().copy(), yw.numpy().copy()) for nm, xy, yw in cands}
        got = {}
        for i, (nm, _xy, _yw) in enumerate(cands):
            got[nm] = {k: float((i * (j + 3)) % 7) / 7.0 for j, k in enumerate(self.OL.KEEP)}
        nd = self.L4.ndiff_over(got, ["teacher"] + [c for c, _, _ in cands if not str(c).startswith("s")])
        return got, nd


def expected_repaired(a):
    q = np.array(a, dtype=np.float64, copy=True)
    q[..., 19, 2] = q[..., 18, 2]                  # the literal Amendment 7 rule, NOT the function under test
    return q


def run_arm(L4, OL, rows, repair, mutate, spec):
    nav_inputs = []

    def nav_stub(P, _ego, _map, grid="refe20"):
        nav_inputs.append(np.array(P, dtype=np.float64, copy=True))
        n = np.asarray(P).shape[0]
        return np.zeros(n), np.ones(n)
    L4.ND.navsim_dac_and_comfort = nav_stub
    S = StubScorer(L4, OL)
    sc = types.SimpleNamespace(get_ego_state_at_iteration=lambda i: None, map_api=None)
    recs = []
    for r in rows:
        nav_inputs.clear()
        res = L4.label_one(S, sc, r, spec, "selftest", mutate, repair=repair)
        recs.append({"r": r, "seen": dict(S.seen), "nav": list(nav_inputs), "res": res})
    return recs


def check(L4, recs, spec, repair: bool, off_recs=None) -> dict:
    import torch  # noqa: F401  (OL._t's dtype)
    fails, n_cand, n_diff19 = [], 0, 0
    for idx, x in enumerate(recs):
        r, seen, nav, res = x["r"], x["seen"], x["nav"], x["res"]
        P = np.asarray(r["props"], dtype=float)
        tr = np.asarray(r["teacher"], dtype=float)
        key = (r["log_name"], r["token"], int(r["step"]), int(r["rank"]))
        plan = L4.plan_slow(key, P.shape[0], r.get("logits"), spec) if spec is not None else None
        C = [] if plan is None else [L4.copy_traj(P, s, f) for s, f in zip(plan["src"], plan["factor"])]
        cands = {k: P[k] for k in range(P.shape[0])}
        cands.update({f"s{i}": c for i, c in enumerate(C)})
        # the reference candidate: never repaired
        t_xy, t_yw = seen["teacher"]
        if not (np.array_equal(t_xy, np.float32(tr[:, :2])) and np.array_equal(t_yw, np.float32(tr[:, 2]))):
            fails.append(f"sample {idx}: the teacher candidate changed")
        for nm, raw in cands.items():
            n_cand += 1
            n_diff19 += int(raw[19, 2] != raw[18, 2])
            exp = expected_repaired(raw) if repair else raw
            xy, yw = seen[nm]
            if not np.array_equal(xy, np.float32(exp[:, :2])):
                fails.append(f"sample {idx} cand {nm}: teacher-input positions differ")
            if not np.array_equal(yw, np.float32(exp[:, 2])):
                fails.append(f"sample {idx} cand {nm}: teacher-input headings differ from the expected "
                             f"({'repaired' if repair else 'unrepaired'})")
        # the NAVSIM inputs: first call = the 64 originals, second = the copies
        if not nav or not np.array_equal(nav[0], expected_repaired(P) if repair else P):
            fails.append(f"sample {idx}: NAVSIM input (originals) is not the expected array")
        if C:
            Cs = np.stack(C)
            if len(nav) < 2 or not np.array_equal(nav[1], expected_repaired(Cs) if repair else Cs):
                fails.append(f"sample {idx}: NAVSIM input (copies) is not the expected array")
        if res["line"] is None:
            fails.append(f"sample {idx}: no line written ({res['status']}: {res.get('error')})")
            continue
        d = json.loads(res["line"])
        if repair and (d.get("label_version") != 5 or d.get("repair") != "last_heading_hold"):
            fails.append(f"sample {idx}: the line does not declare label_version 5 / repair")
        if off_recs is not None:
            o = json.loads(off_recs[idx]["res"]["line"])
            if d["traj"] != o["traj"] or d["yaw"] != o["yaw"]:
                fails.append(f"sample {idx}: the SERVED traj/yaw differ from flag OFF")
    return {"pass": not fails, "n_fail": len(fails), "first_fails": fails[:6], "candidates_checked": n_cand,
            "control_candidates_with_heading19_ne_heading18": n_diff19}


def main() -> int:
    import onpolicy_label as OL
    import onpolicy_label_v4 as L4
    rows = load_rows()
    spec = L4.SlowSpec(factors=(0.75,), frac=1.0)            # every sample carries its copies (as the M5 labelling)
    res = {"_label": "G-R1 (PREREG_MEASURE5R §5): repair mechanism, label-free (labelling functions stubbed)",
           "rows": len(rows), "queue": QUEUE, "labeller": os.path.abspath(L4.__file__)}
    off = run_arm(L4, OL, rows, False, "", spec)
    res["flag_off"] = check(L4, off, spec, repair=False)
    on = run_arm(L4, OL, rows, True, "", spec)
    res["flag_on"] = check(L4, on, spec, repair=True, off_recs=off)
    pure = run_arm(L4, OL, rows[:8], True, "", None)                    # the pure-set path (no copies) with the flag
    res["flag_on_pure_sets"] = check(L4, pure, None, repair=True)
    res["mutations"] = {}
    for m in ("repair_wrong_index", "repair_served", "repair_skip_copies"):
        mr = run_arm(L4, OL, rows, True, m, spec)
        c = check(L4, mr, spec, repair=True, off_recs=off)
        res["mutations"][m] = {"gate_red": not c["pass"], "n_fail": c["n_fail"], "first_fails": c["first_fails"][:2]}
    res["G_R1_PASS"] = (res["flag_off"]["pass"] and res["flag_on"]["pass"] and res["flag_on_pure_sets"]["pass"]
                        and all(v["gate_red"] for v in res["mutations"].values())
                        and res["flag_on"]["control_candidates_with_heading19_ne_heading18"] > 0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "first_fails"})
                      for k, v in res.items()}, indent=1))
    print(f"ZZM5R_GR1_{'PASS' if res['G_R1_PASS'] else 'FAIL'}")
    return 0 if res["G_R1_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
