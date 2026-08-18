"""Collate the C104 falsifier battery into one table.

Reads every `fals_*.json` the chain wrote and emits, per ARM and TARGET, the
`r2_ceiling` mean and the SEED SPREAD — because C103 established that a
stability claim measured under a defect is not inherited by the repaired
instrument, so a spread is reported, never assumed to be zero.

⛔ Every ratio here is vs the `ours` baseline produced by the SAME command on
the SAME rows, never vs a number copied out of C104.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["ours", "dino1f", "dinorand", "randenc_s0", "randenc_s1", "randenc_s2"]
TARGETS = ["ego_v0", "lead_gap", "lead_closing"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = Path(a.raw)

    rows: dict[str, dict] = {}
    for f in sorted(raw.glob("fals_*.json")):
        tag = f.stem[len("fals_"):]
        d = json.loads(f.read_text("utf-8"))
        arm = d["arms"].get("p40")
        if arm is None:
            continue
        rec = {"label": d["arm_label"], "d_model": d["d_model"],
               "n_raw_features": arm["n_raw_features"],
               "n_fit_features": arm["n_fit_features"],
               "rp_dim": d["rp_dim"], "proj_seeds": d["proj_seeds"],
               "intercept_col": d.get("ridge_intercept_col"),
               "legacy_penalised_intercept": d.get(
                   "legacy_penalised_intercept"),
               "n_train": None, "n_eval": None, "n_eval_clusters": None,
               "targets": {}}
        for t, td in arm["targets"].items():
            per = td.get("per_seed", {})
            vals = [float(v["r2_ceiling"]) for v in per.values()]
            k1s = [bool(v.get("K1_PASSES")) for v in per.values()]
            rec["n_train"] = td.get("n_train")
            rec["n_eval"] = td.get("n_eval")
            rec["n_eval_clusters"] = td.get("n_eval_clusters")
            rec["targets"][t] = {
                "r2_mean": round(float(td["r2_ceiling_mean"]), 6),
                "r2_min": round(min(vals), 6) if vals else None,
                "r2_max": round(max(vals), 6) if vals else None,
                "r2_seed_spread": round(max(vals) - min(vals), 6) if vals else None,
                "n_seeds": len(vals),
                "K1_PASSES_n": int(sum(k1s)),
                "gt_sd": td.get("gt_sd"),
                "pred_sd_over_gt_sd": [
                    round(float(v.get("pred_sd_over_gt_sd", float("nan"))), 6)
                    for v in per.values()],
            }
        rows[tag] = rec

    base = rows.get("ours")
    table = []
    for tag in ORDER + [k for k in rows if k not in ORDER]:
        if tag not in rows:
            continue
        r = rows[tag]
        line = {"arm": tag, "label": r["label"], "token_width": r["d_model"],
                "n_raw_features": r["n_raw_features"]}
        for t in TARGETS:
            td = r["targets"].get(t)
            if td is None:
                continue
            line[f"{t}_r2"] = td["r2_mean"]
            line[f"{t}_spread"] = td["r2_seed_spread"]
            if base and t in base["targets"]:
                b = base["targets"][t]["r2_mean"]
                line[f"{t}_x_vs_ours"] = (round(td["r2_mean"] / b, 2)
                                          if b > 0 else None)
        table.append(line)

    out = {
        "_evidence_class": "MEASURED (ours; identical harness, identical rows, "
                           "identical deployed pool AvgPool2d((4,10)))",
        "eval_tier": "T0-DIAGNOSTIC — a frozen-latent linear readout. NEVER "
                     "driving performance (EVAL_DOCTRINE.md).",
        "estimator": "episode-cluster bootstrap (taniteval/ci.py); ⛔ NEVER "
                     "overlapping_holdout_se",
        "pool": "the DEPLOYED AvgPool2d((4,10)) only — E-R1-0 settled the "
                "pooling axis (C104), so the other rungs buy nothing here",
        "read_me_first": (
            "⚠️ `dino1f` is width-matched to ours (768) and is the CONCATENATION "
            "control: read `lead_gap` there (STATIC geometry, one frame "
            "suffices). `ego_v0`/`lead_closing` are MOTION quantities that NO "
            "one-frame arm can read, so a drop there is uninformative."),
        "arms": rows,
        "table": table,
    }
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), "utf-8")
    hdr = f"{'arm':<14}{'width':>6}{'raw_feat':>10}" + "".join(
        f"{t:>16}" for t in TARGETS)
    print(hdr)
    for line in table:
        s = (f"{line['arm']:<14}{line['token_width']:>6}"
             f"{line['n_raw_features']:>10}")
        for t in TARGETS:
            v = line.get(f"{t}_r2")
            x = line.get(f"{t}_x_vs_ours")
            s += f"{v:>10.5f}" + (f" {x:>4.1f}x" if x is not None else "      ")
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
