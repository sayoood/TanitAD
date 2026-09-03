"""E-ARCH-EGOZERO-1 (c) — is the LONGITUDINAL column worse than the LATERAL one
in the live refcv3 run? Era-split, in COMPARABLE units.

⛔ Raw CE columns are NOT comparable across heads. `lat`/`lon` are the core aux
over the kin3 3-class vocabulary (`refc_v3_train.py:496-497`, chance ln3 =
1.0986); `lat_tac`/`lon_tac` are the z_tac heads over the v7.2 EIGHT-class
vocabulary (`v7_labels.HEADS['tac_lat'|'tac_lon'] = 8`, chance ln8 = 2.0794).
Their class priors also differ, so even lat-vs-lon at the same K is unfair raw.

The admissible statistic is the share of the NO-INFORMATION loss removed:

        skill = 1 - CE / H(label marginal)

which reads EXACTLY 0 for a constant/prior-only predictor (the control the
validate-design skill requires) and 1 for a perfect one, in the same units for
any vocabulary. H is MEASURED here from the eval corpus' own labels
(label_shift.json), not assumed.

⛔ NO CONFIDENCE INTERVAL IS COMPUTABLE FROM THIS FILE by any estimator: every
value is already a pooled mean over 8 x 20 = 160 windows, with no per-window
rows and no episode ids (recorded in the 2026-09-03 10:05 programme report).
Numbers below are point reads only.

Eras (2026-09-03 10:05 report): A <= 16,500 (pre-nav-switch); 16,500-17,500 the
in-training eval leaked held-out label marginals into the tactical prior
(C-REFCV3-EVAL-PRIOR-LEAK, fixed 2026-09-02); C >= 18,000 is the clean era.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ERAS = [("A_pre_nav_switch", 0, 16500), ("B_leaked", 16500, 18000),
        ("C_clean", 18000, 10 ** 9)]


def main():
    shift = json.load(open(os.path.join(HERE, "label_shift.json"),
                           encoding="utf-8"))
    H = {"lat": shift["eval"]["lat"]["marginal"]["entropy_nats"],
         "lon": shift["eval"]["lon"]["marginal"]["entropy_nats"]}
    rows = []
    with open(os.path.join(HERE, "metrics.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    ev = [r for r in rows if "eval_loss" in r and "step" in r]
    tr = [r for r in rows if "loss" in r and "step" in r]
    res = {"n_rows": len(rows), "n_eval_rows": len(ev), "n_train_rows": len(tr),
           "step_range": [min(r["step"] for r in rows),
                          max(r["step"] for r in rows)],
           "marginal_entropy_nats_eval_corpus": H,
           "vocab": {"lat": 3, "lon": 3, "lat_tac": 8, "lon_tac": 8},
           "chance_CE_nats": {"lat": float(np.log(3)), "lon": float(np.log(3)),
                              "lat_tac": float(np.log(8)),
                              "lon_tac": float(np.log(8))},
           "ci": "NOT COMPUTABLE from this file — each row is already a pooled "
                 "mean over 160 windows, no per-window rows, no episode ids",
           "eras": {}}
    for name, lo, hi in ERAS:
        sub = [r for r in ev if lo <= r["step"] < hi]
        if not sub:
            continue
        d = {"n_eval_rows": len(sub),
             "step_range": [sub[0]["step"], sub[-1]["step"]]}
        for col in ("eval_lat", "eval_lon", "eval_lat_tac", "eval_lon_tac",
                    "eval_goal2s_err_m", "eval_traj", "eval_anchor_acc"):
            v = np.array([r[col] for r in sub if col in r], dtype=np.float64)
            if v.size == 0:
                continue
            e = {"n": int(v.size), "mean": float(v.mean()),
                 "first": float(v[0]), "last": float(v[-1]),
                 "best": float(v.min()) if "acc" not in col else float(v.max()),
                 "last10_mean": float(v[-10:].mean())}
            base = col.replace("eval_", "").replace("_tac", "")
            if base in H and not col.endswith("_tac"):
                e["skill_vs_marginal_last10"] = 1.0 - e["last10_mean"] / H[base]
                e["skill_vs_marginal_mean"] = 1.0 - e["mean"] / H[base]
                e["H_marginal"] = H[base]
            d[col] = e
        # the paired lateral-vs-longitudinal read, same rows, same vocabulary
        la = np.array([r["eval_lat"] for r in sub], dtype=np.float64)
        lo_ = np.array([r["eval_lon"] for r in sub], dtype=np.float64)
        sk_lat = 1.0 - la / H["lat"]
        sk_lon = 1.0 - lo_ / H["lon"]
        d["skill_core_kin3"] = {
            "lat_last10": float(sk_lat[-10:].mean()),
            "lon_last10": float(sk_lon[-10:].mean()),
            "lat_minus_lon_last10": float(sk_lat[-10:].mean()
                                          - sk_lon[-10:].mean()),
            "lat_mean": float(sk_lat.mean()), "lon_mean": float(sk_lon.mean()),
            "lat_minus_lon_mean": float(sk_lat.mean() - sk_lon.mean()),
            "n_rows_lat_beats_lon": int((sk_lat > sk_lon).sum()),
            "note": "skill = 1 - CE/H(marginal); 0 = prior-only control, "
                    "same units for both heads. Paired over identical rows.",
        }
        # raw z_tac columns: same 8-class vocabulary as each other, so their
        # RATIO is at least well posed even without their marginals
        lt = np.array([r["eval_lat_tac"] for r in sub if "eval_lat_tac" in r])
        ln_ = np.array([r["eval_lon_tac"] for r in sub if "eval_lon_tac" in r])
        if lt.size and ln_.size:
            d["z_tac_raw_CE_8class"] = {
                "lat_last10": float(lt[-10:].mean()),
                "lon_last10": float(ln_[-10:].mean()),
                "lon_over_lat_last10": float(ln_[-10:].mean()
                                             / lt[-10:].mean()),
                "n_rows_lon_worse": int((ln_ > lt).sum()), "n": int(lt.size),
                "caveat": "marginals for the v7.2 8-class heads were NOT "
                          "measured here, so this is a raw-CE ratio, not a "
                          "skill comparison. 5 of 16 tactical classes have "
                          "zero support (R40), so the two priors differ.",
            }
        res["eras"][name] = d
    with open(os.path.join(HERE, "metrics_lat_lon.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)

    print(f"rows={len(rows)} eval={len(ev)} steps {res['step_range']}")
    print(f"H(marginal) eval corpus: lat {H['lat']:.4f}  lon {H['lon']:.4f} nats")
    for name in res["eras"]:
        d = res["eras"][name]
        s = d["skill_core_kin3"]
        z = d.get("z_tac_raw_CE_8class", {})
        print(f"\n[{name}] n_eval={d['n_eval_rows']} steps {d['step_range']}")
        print(f"  core kin3 SKILL (last10): lat {s['lat_last10']:+.4f} | "
              f"lon {s['lon_last10']:+.4f} | lat-lon {s['lat_minus_lon_last10']:+.4f}"
              f"  (lat beats lon on {s['n_rows_lat_beats_lon']}/{d['n_eval_rows']} rows)")
        if z:
            print(f"  z_tac raw CE  (last10): lat {z['lat_last10']:.4f} | "
                  f"lon {z['lon_last10']:.4f} | ratio {z['lon_over_lat_last10']:.3f}"
                  f"  (lon worse on {z['n_rows_lon_worse']}/{z['n']})")
        g = d.get("eval_goal2s_err_m", {})
        if g:
            print(f"  goal2s_err_m: mean {g['mean']:.4f} last10 {g['last10_mean']:.4f} best {g['best']:.4f}")
    print("\nwrote metrics_lat_lon.json")


if __name__ == "__main__":
    main()
