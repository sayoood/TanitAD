"""EGOWM, the planner half of the card's read (PREREG_V8F_EGOWM.md, frozen sha256 ddd6fc9b...f446e7). Master Mind, 2026-11-10 ~23:15,
written before any speed arm was planned. Tier T0.

A THIN wrapper over ``ladder_plantest/report_lp.py``: its ``read_family`` (PLANTEST on ALL windows of the form, PLANTEST2 on the SCORE half
at the TUNE-chosen lambda; G-TRUE minus HOLD and minus G-RAND per family, LONGITUDINAL and LATERAL, paired episode-cluster bootstrap B 2,000,
P.family_label, seeds_label) is called unchanged. Only two module globals are replaced for the call:
  * ``ctx_of``: a speed arm is prepared by ``lp_speed.prepare_arm_speed`` (BUILT speed mode = argv = the expected one), R35H by lp_lib as before;
  * ``planners_of``: FLAT1 only (TD3 is not computable on strides (3, 5), card).
R35H is read from the ladder's units (F:/TanitAD-scratch/ladder_plantest/runs), the speed arms from F:/TanitAD-scratch/egowm/plan_runs.
Deciding (card): an arm is PLANNABLE in a family when that family's label is separated-better than HOLD and than G-RAND at both seeds.
Beside, no bar: the arm's G-TRUE minus R35H's G-TRUE, same seed, same SCORE windows, paired bootstrap.
Usage: python report_speed.py [--out RESULT_EGOWM_PLAN.json]
"""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lp_speed as S  # noqa: E402
import report_lp as RL  # noqa: E402  (lp_speed put ladder_plantest on sys.path)
import numpy as np  # noqa: E402

PULLS = "C:/Users/Admin/t2v8f_pulls"
SPEED_RUNS = "F:/TanitAD-scratch/egowm/plan_runs"
LADDER_RUNS = RL.RUNS
MODE = {"E-KIN": "kin", "E-NOW": "now"}
_C = {}


def ctx_of(name):
    if name not in _C:
        fam = name.rsplit("_s", 1)[0]
        _C[name] = S.prepare_arm_speed(name, f"{PULLS}/egowm_{name}", expect_mode=MODE[fam]) if fam in MODE else S.LP.prepare_arm(name)
    return _C[name]


def beside_vs_r35h(out):
    """G-TRUE of the arm minus G-TRUE of R35H, same seed and form, on the SCORE windows, at each arm's own chosen lambda. Decides nothing."""
    import plantest_lib as P
    rows = {}
    for fam in MODE:
        for form in RL.FORMS:
            for seed in (0, 1):
                a, b = f"{fam}_s{seed}", f"R35H_s{seed}"
                ca, cb = ctx_of(a), ctx_of(b)
                sc = S.LP.sel_arrays(ca, form)["score"]
                assert np.array_equal(sc, S.LP.sel_arrays(cb, form)["score"]), "SCORE windows differ between the arm and R35H"
                uniq, Wt, pos = P.boot_setup(ca.eid[sc])
                for famname, q in RL.FAM.items():
                    pa, pb = os.path.join(SPEED_RUNS, f"choice_{a}_{form}_flat1.json"), os.path.join(LADDER_RUNS, f"choice_{b}_{form}_flat1.json")
                    if not (os.path.exists(pa) and os.path.exists(pb)):
                        continue
                    la, lb = json.load(open(pa))["choice"][famname], json.load(open(pb))["choice"][famname]
                    ra, rb = RL.ld(SPEED_RUNS, "score", a, form, "flat1", "true", f"L{RL.lam_s(la)}"), RL.ld(LADDER_RUNS, "score", b, form, "flat1", "true", f"L{RL.lam_s(lb)}")
                    if ra is None or rb is None:
                        continue
                    ma = P.window_metrics(ra["plan_raw"].astype(np.float64), ca.A[sc], ca.v0[sc])[q]
                    mb = P.window_metrics(rb["plan_raw"].astype(np.float64), cb.A[sc], cb.v0[sc])[q]
                    d = P.boot_mean_diff(ma, mb, pos, Wt)
                    rows[f"{fam}|{form}|{famname}|s{seed}"] = dict(lam_arm=la, lam_r35h=lb, arm_minus_r35h=d, sep=P.sep_label(*d), unit=RL.UNIT[q])
    out["beside_vs_R35H"] = rows


def main(argv):
    path = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(HERE, "RESULT_EGOWM_PLAN.json")
    RL.ctx_of = ctx_of
    RL.planners_of = lambda fam: ("flat1",)
    out = dict(card="PREREG_V8F_EGOWM.md", card_sha256_prefix="ddd6fc9b", tier="T0", tactical="n = 0, no decision output (card)",
               strategic="PAUSED (PI item 25)", families={})
    for fam in ("R35H", "E-KIN", "E-NOW"):
        runs = LADDER_RUNS if fam == "R35H" else SPEED_RUNS
        for form in RL.FORMS:
            try:
                RL.read_family(runs, fam, form, out["families"])
            except Exception as e:  # an arm or form whose units are not all there yet
                out["families"].setdefault(fam, {})[form] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    try:
        beside_vs_r35h(out)
    except Exception as e:
        out["beside_vs_R35H_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    out["plannable"] = {}
    for fam in ("E-KIN", "E-NOW"):
        for form in RL.FORMS:
            r = out["families"].get(fam, {}).get(form, {})
            for card in ("plantest", "plantest2"):
                rows = r.get(card, {}).get("rows", {})
                for famname in RL.FAM:
                    lab = rows.get(f"flat1|{famname}|label")
                    out["plannable"][f"{fam}|{form}|{card}|{famname}"] = lab
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    for k, v in out["plannable"].items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
