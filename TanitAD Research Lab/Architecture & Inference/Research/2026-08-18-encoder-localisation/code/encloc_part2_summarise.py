"""PART 2 summariser - the rung table and the PAIRED stage deltas.

Pairs every Part-2 latent arm against rung 1 (the raw refa1f features through
the s16 pool+projection, preds_refa1f.pkl banked by chain 2), and rung3 against
rung2_trained (the predictor step's own cost), with the paired episode-cluster
bootstrap on element-wise-asserted identical rows. A dispersion (ridge-seed or
init-seed spread) is reported AS a spread, never as a CI (C109).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from encloc_ladder import paired_delta_r2c  # noqa: E402

P2_ARMS = ["rung2_trained", "rung2_rand0", "rung2_rand1", "rung2_rand2",
           "rung3_h1", "rung3_h4"]
TARGETS = ["ego_v0", "lead_gap", "lead_closing"]


def load_arm_json(raw: Path, name: str):
    f = raw / f"encloc_p2_{name}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text("utf-8"))
    return d["arms"]["cells"]["targets"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    raw = Path(a.raw)

    res = {arm: load_arm_json(raw, arm) for arm in P2_ARMS}
    # ridge-seed variants for the decision arms
    rs = {}
    for arm in ("rung2_trained", "rung3_h1"):
        rs[arm] = {}
        for r in (1, 2):
            f = raw / f"encloc_p2_{arm}_rs{r}.json"
            if f.exists():
                rs[arm][r] = json.loads(
                    f.read_text("utf-8"))["arms"]["cells"]["targets"]

    # rung 1 reference (chain 2 dump)
    p1 = raw / "preds_refa1f.pkl"
    rung1 = pickle.loads(p1.read_bytes()) if p1.exists() else None
    preds = {}
    for arm in P2_ARMS:
        f = raw / f"preds_p2_{arm}.pkl"
        if f.exists():
            preds[arm] = pickle.loads(f.read_bytes())

    table = {}
    for t in TARGETS:
        table[t] = {}
        for arm in P2_ARMS:
            if res[arm] is None or t not in res[arm]:
                continue
            n = res[arm][t]
            ps = n["per_seed"]
            k0 = sorted(ps)[0]
            r2s = [n["r2_ceiling_mean"]]
            for r, tt in rs.get(arm, {}).items():
                r2s.append(tt[t]["r2_ceiling_mean"])
            pv0 = ps[k0].get("r2_ceiling_partial_v0")
            table[t][arm] = {
                "r2": n["r2_ceiling_mean"],
                "r2_ridge_seed_spread": ([round(float(x), 5) for x in r2s]
                                         if len(r2s) > 1 else None),
                "partial_v0_r2": pv0,
                "corr_partial_v0": ps[k0].get("corr_partial_v0"),
                "K1_delta": n["K1_delta_mean"],
                "K1_PASSES": ps[k0]["K1_PASSES"],
                "pred_sd_over_gt_sd": ps[k0]["pred_sd_over_gt_sd"],
                "n_eval": n["n_eval"],
                "n_eval_clusters": n["n_eval_clusters"]}

    # ---- paired deltas ----------------------------------------------------
    def pair(pa, pb, label):
        outd = {}
        for t in TARGETS:
            if t not in pa["scored"] or t not in pb["scored"]:
                continue
            ya, ea, v0a = pa["scored"][t]
            yb, eb, v0b = pb["scored"][t]
            if ya.shape != yb.shape or not np.allclose(ya, yb):
                outd[t] = {"REFUSED": "eval targets differ - not pairable"}
                continue
            if list(map(str, ea)) != list(map(str, eb)):
                outd[t] = {"REFUSED": "episode ids differ - not pairable"}
                continue
            ka = sorted(pa["preds"][t])[0]
            kb = sorted(pb["preds"][t])[0]
            sa = sorted(pa["preds"][t][ka])[0]
            sb = sorted(pb["preds"][t][kb])[0]
            d1 = paired_delta_r2c(pa["preds"][t][ka][sa],
                                  pb["preds"][t][kb][sb], yb, eb,
                                  a.n_boot, seed=0)
            d2 = paired_delta_r2c(pa["preds"][t][ka][sa],
                                  pb["preds"][t][kb][sb], yb, eb,
                                  a.n_boot, seed=0, z=v0b)
            outd[t] = {"delta_r2": d1["delta"], "ci": [d1["lo"], d1["hi"]],
                       "separated": d1["separated"],
                       "delta_r2_partial_v0": d2["delta"],
                       "ci_partial_v0": [d2["lo"], d2["hi"]],
                       "separated_partial_v0": d2["separated"],
                       "estimator": d1["estimator"]}
        return {"label": label, "targets": outd}

    deltas = {}
    if rung1 is not None:
        for arm in P2_ARMS:
            if arm in preds:
                deltas[f"{arm}_vs_rung1"] = pair(preds[arm], rung1,
                                                 f"{arm} - rung1(refa1f raw)")
    if "rung3_h1" in preds and "rung2_trained" in preds:
        deltas["rung3_h1_vs_rung2_trained"] = pair(
            preds["rung3_h1"], preds["rung2_trained"],
            "rung3_h1 - rung2_trained (the predictor step)")
    if "rung2_trained" in preds and "rung2_rand0" in preds:
        deltas["rung2_trained_vs_rand0"] = pair(
            preds["rung2_trained"], preds["rung2_rand0"],
            "rung2_trained - rung2_rand0 (what TRAINING changed)")

    build_meta = {}
    bm = None
    for cand in (Path("C:/Users/Admin/tanitad-caches/encloc-20260818/part2"
                      "/part2_build_meta.json"),):
        if cand.exists():
            bm = json.loads(cand.read_text("utf-8"))
    if bm:
        build_meta = {"ckpt_sha256": bm["ckpt_sha256"],
                      "ckpt_step": bm["ckpt_step"],
                      "collapse_check": bm["collapse_check"],
                      "n_pad_rows": bm["n_pad_rows"],
                      "sanity_corr_min": bm["sanity_corr_min"]}

    out = {"_evidence_class": "MEASURED (ours; PART 2 - trained refa-dinov2-4b "
                              "latents, same banked windows)",
           "eval_tier": "T0-DIAGNOSTIC (a linear readout, NEVER driving "
                        "performance)",
           "experiment": "ENCODER_LOCALISATION.md PART 2 (SS8 pre-registration)",
           "estimator": "paired episode-cluster bootstrap; seed spreads are "
                        "spreads, never CIs (C109)",
           "forbidden": "overlapping_holdout_se",
           "build": build_meta, "table": table, "paired_deltas": deltas}
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), "utf-8")
    print(f"[p2sum] wrote {a.out}")

    for t in TARGETS:
        print(f"\n== {t} ==")
        print("  %-14s %9s %9s %8s %6s" % ("arm", "r2", "partialv0",
                                           "psd/gsd", "K1"))
        for arm in P2_ARMS:
            r = table.get(t, {}).get(arm)
            if not r:
                continue
            print("  %-14s %9.5f %9s %8.3f %6s%s" %
                  (arm, r["r2"],
                   ("%.5f" % r["partial_v0_r2"])
                   if r["partial_v0_r2"] is not None else "-",
                   r["pred_sd_over_gt_sd"],
                   "PASS" if r["K1_PASSES"] else "fail",
                   ("  rs-spread %s" % r["r2_ridge_seed_spread"])
                   if r["r2_ridge_seed_spread"] else ""))
    for k, dd in deltas.items():
        print(f"\n-- {dd['label']} --")
        for t, v in dd["targets"].items():
            if "REFUSED" in v:
                print(f"  {t}: REFUSED {v['REFUSED']}")
            else:
                print("  %-13s d=%+.5f [%+.5f,%+.5f] sep=%s | pv0 d=%+.5f "
                      "[%+.5f,%+.5f] sep=%s" %
                      (t, v["delta_r2"], v["ci"][0], v["ci"][1],
                       v["separated"], v["delta_r2_partial_v0"],
                       v["ci_partial_v0"][0], v["ci_partial_v0"][1],
                       v["separated_partial_v0"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
