"""CPU tests of the YF3 estimator + committed decision rule (no GPU). Run: python test_yf3_analyze.py   (PY_D or PY_T)
Every REGRESSION arm must flip the verdict the way the PREREG says. Banked-data tests reproduce numbers that already exist in result.json / sft4_B_u400_families.json."""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yf3_common as C  # noqa: E402
import yf3_stats as S  # noqa: E402

RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def csvmap(p):
    import csv
    return {r["token"]: float(r["score"]) * 100.0 for r in csv.DictReader(open(p, encoding="utf-8")) if r.get("token") not in (None, "", "average")}


def main():
    # 1. estimator reproduces the banked SFT-4 primary (mean +3.0419, CI [1.2095, 4.8651]) from the two banked CSVs
    tk = C.jload(C.A5_TOKENS)
    toks = sorted(tk["tokens"])
    a0, dep = csvmap(C.BANKED_CSV), csvmap(C.BANKED_DEPLOYED_CSV)
    bt = S.Boot([tk["token_log"][t] for t in toks])
    d = np.asarray([a0[t] - dep[t] for t in toks])
    st = bt.stat(d)
    lo, hi = S.ci(st)
    res = C.jload(C.BANKED_RESULT)["primary"]
    check("estimator == proxy_eval.boot on the banked SFT-4 contrast (mean, lo, hi to 1e-9)", abs(float(d.mean()) - res["mean"]) < 1e-9 and abs(lo - res["ci95"][0]) < 1e-9 and abs(hi - res["ci95"][1]) < 1e-9,
          f"mean {d.mean():.6f} CI [{lo:.6f}, {hi:.6f}] vs banked {res['mean']:.6f} [{res['ci95'][0]:.6f}, {res['ci95'][1]:.6f}]")
    # 2. per-window curvature reproduces the banked families6 value (needs the PY_T artifact written by yf3_perwindow.py on the banked seam)
    pwp = HERE / "_scratch" / "banked_A0_pw.npz"
    if pwp.exists():
        pw = np.load(pwp)
        fam = C.jload(C.BANKED_FAMILIES)["families"]["refcv6"]["lateral"]
        cm = float(pw["curv_abs_sum"].sum() / pw["curv_n"].sum())
        check("per-window curvature ratio == banked families6 curvature_mae_1pm (0.01882) and n_steps_curvature", round(cm, 6) == fam["curvature_mae_1pm"] and int(pw["curv_n"].sum()) == fam["n_steps_curvature"],
              f"{cm:.6f} vs {fam['curvature_mae_1pm']}, n {int(pw['curv_n'].sum())} vs {fam['n_steps_curvature']}")
        import math
        check("per-window heading / yaw-rate / cross == banked", round(math.degrees(pw["head_abs_sum_rad"].sum() / pw["head_n"].sum()), 4) == fam["heading_mae_deg"]
              and round(math.degrees(pw["yaw_abs_sum_radps"].sum() / pw["yaw_n"].sum()), 4) == fam["yaw_rate_mae_degps"] and round(float(pw["cross_abs_sum"].sum() / pw["cross_n"].sum()), 4) == fam["cross_mae_m"],
              f"{math.degrees(pw['head_abs_sum_rad'].sum() / pw['head_n'].sum()):.4f}/{math.degrees(pw['yaw_abs_sum_radps'].sum() / pw['yaw_n'].sum()):.4f}/{float(pw['cross_abs_sum'].sum() / pw['cross_n'].sum()):.4f}")
        # 3. synthetic contrasts on the REAL A0 per-window sums: identical arm -> exactly 0; planted improvement -> ADOPT-grade; swapped -> worse
        num, den = pw["curv_abs_sum"], pw["curv_n"]
        logs = [tk["token_log"][t] for t in pw["token"]]
        b2 = S.Boot(logs)
        same = S.contrast(b2, [(num, den), (num, den)], [(num, den), (num, den)])
        check("contrast(arm, same arm) == exactly 0, CI [0, 0] (the C == A0 identity as the analysis sees it)", same["point"] == 0.0 and same["ci95"] == (0.0, 0.0) and same["F_seed_floor"] == 0.0)
        rng = np.random.default_rng(1)
        better = [(num * 0.7, den), (num * 0.72, den)]
        c = S.contrast(b2, better, [(num, den), (num, den)])
        pd = {"ci95": (-0.2, 0.3), "point": 0.0}
        v = S.decide(gates_ok=True, gates_red=[], prim=c, ref_mae=float(num.sum() / den.sum()), pdms=pd, adverse_by_seed=[[], []])
        check("PLANTED -29 % curvature error, 2 seeds, PDMS ok, no adverse -> ADOPT", v["verdict"] == "ADOPT", f"delta {c['point']:+.5f} CI [{c['ci95'][0]:+.5f},{c['ci95'][1]:+.5f}] F {c['F_seed_floor']:.5f}")
        worse = S.contrast(b2, [(num, den), (num, den)], better)
        v = S.decide(gates_ok=True, gates_red=[], prim=worse, ref_mae=float(num.sum() / den.sum()), pdms=pd, adverse_by_seed=[[], []])
        check("RED-AS-REQUIRED: B and Ref swapped (B is 29 % WORSE) -> REFUTED (CI entirely above 0)", v["verdict"] == "REFUTED", v["text"])
        weak = [(num * 0.97, den), (num * 0.97, den)]
        c = S.contrast(b2, weak, [(num, den), (num, den)])
        v = S.decide(gates_ok=True, gates_red=[], prim=c, ref_mae=float(num.sum() / den.sum()), pdms=pd, adverse_by_seed=[[], []])
        check("PLANTED -3 % (below the 5 % bar), CI < 0 -> NOT PROVEN", v["verdict"] == "NOT PROVEN" and not v["conditions"]["relative_improvement_ge_5pct"], f"rel {v['relative_improvement']:.3f}")
        split = [(num * 0.70, den), (num * 1.02, den)]
        c = S.contrast(b2, split, [(num, den), (num, den)])
        v = S.decide(gates_ok=True, gates_red=[], prim=c, ref_mae=float(num.sum() / den.sum()), pdms=pd, adverse_by_seed=[[], []])
        check("seeds DISAGREE in sign -> NOT PROVEN", v["verdict"] == "NOT PROVEN" and not v["conditions"]["both_seeds_agree_in_sign"])
        wide = [(num * 0.60, den), (num * 0.82, den)]
        c = S.contrast(b2, wide, [(num, den), (num, den)])
        v = S.decide(gates_ok=True, gates_red=[], prim=c, ref_mae=float(num.sum() / den.sum()), pdms=pd, adverse_by_seed=[[], []])
        check("same sign but seed floor F >= |mean Delta| / 2 -> NOT PROVEN", v["verdict"] == "NOT PROVEN" and not v["conditions"]["F_lt_half_abs_mean_delta"], f"F {c['F_seed_floor']:.5f} |mean|/2 {abs(c['point']) / 2:.5f}")
    else:
        check("banked per-window artifact present (run yf3_perwindow.py on the banked seam first)", False, str(pwp))
    # 4. the remaining branches of the rule on literal numbers
    prim = {"point": -0.002, "ci95": (-0.003, -0.001), "per_seed_point": [-0.0019, -0.0021], "F_seed_floor": 0.0002}
    ok_pd = {"ci95": (-0.5, 0.4)}
    base = dict(gates_ok=True, gates_red=[], prim=prim, ref_mae=0.0188, pdms=ok_pd, adverse_by_seed=[[], []])
    check("literal ADOPT case (-10.6 %, both seeds, F small)", S.decide(**base)["verdict"] == "ADOPT")
    check("RED-AS-REQUIRED: PDMS CI lower bound -1.2 (below the -1.0 margin) -> NOT PROVEN, not ADOPT", S.decide(**{**base, "pdms": {"ci95": (-1.2, 0.1)}})["verdict"] == "NOT PROVEN")
    check("RED-AS-REQUIRED: PDMS CI upper bound -1.1 -> REFUTED", S.decide(**{**base, "pdms": {"ci95": (-2.0, -1.1)}})["verdict"] == "REFUTED")
    check("RED-AS-REQUIRED: one adverse lateral component -> REFUTED even with a better primary", S.decide(**{**base, "adverse_by_seed": [[{"metric": "heading_mae_deg"}], []]})["verdict"] == "REFUTED")
    check("RED-AS-REQUIRED: a red gate -> VOID, never a statistic", S.decide(**{**base, "gates_ok": False, "gates_red": ["G7_x"]})["verdict"] == "VOID")
    check("primary CI straddles 0 -> NOT PROVEN", S.decide(**{**base, "prim": {**prim, "ci95": (-0.003, 0.001)}})["verdict"] == "NOT PROVEN")
    n_fail = RES.count(False)
    print(f"\n{len(RES) - n_fail}/{len(RES)} checks as required")
    print("ZZYF3_STATS_TESTS_OK" if n_fail == 0 else "ZZYF3_STATS_TESTS_FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
