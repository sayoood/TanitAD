"""The four BINDING metric families for the SHARED-RE-TUNE candidate cells.

CLAUDE.md, binding 2026-08-02: an eval that reports a ranking metric alone is INCOMPLETE.
LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC, per situation, never pooled, each with the paired
episode-cluster bootstrap, and where a family cannot be computed it carries ITS REASON AND ITS n.

⛔ Strata are drawn with the **CAUSAL** ego block (`sitclf_b4_substrate.ego_causal.npz`), never the
quarantined legacy block — which is run beside it so the difference is MEASURED. The parent
measured that the legacy block moved `intersection` LATERAL-turning from -0.0604 to -0.0282, a 2.1x
change in a point estimate, so which block a number came from is stated, not assumed.

⛔ VISION-ONLY at inference: ego appears ONLY as the stratification variable, never as a model input.

Pure re-analysis of the banked score columns: 0 GPU, no refitting.

usage:
  python four_families_retune.py --out results_four_families.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "2026-08-03-sitclf-horizon"
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

from tanitad.eval.sitclf_deploy import ScoreBundle, four_family_report   # noqa: E402

#: the candidate SHARED cells, against the deployed frozen cell
FROZEN = "CELL_w8_L3.0"
CANDIDATES = ("CELL_w8_L1.0", "CELL_w32_L1.0")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(PARENT / "results_horizon_ps.scores.npz"))
    ap.add_argument("--ego-causal",
                    default=r"C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.ego_causal.npz")
    ap.add_argument("--ego-legacy",
                    default=r"C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.npz")
    ap.add_argument("--out", default="results_four_families.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--strata-n-boot", type=int, default=500)
    a = ap.parse_args()

    z = np.load(a.scores)
    sits = [str(s) for s in z["situations"]]
    y, valid, cc = z["y_lead3"], z["eval_rows"], z["clip_cluster"]
    ego_c = np.load(a.ego_causal)["E_causal"]
    ego_l = np.load(a.ego_legacy)["E"]
    arms = (FROZEN,) + CANDIDATES
    log(f"{y.shape[0]:,} rows, {len(np.unique(cc))} clusters, arms {arms}")

    out = {"_what": "the four BINDING families for the SHARED-re-tune candidates vs the frozen cell",
           "_binding": "CLAUDE.md 2026-08-02 — a ranking metric alone is an INCOMPLETE result",
           "_ego_block": ("CAUSAL rebuild (sitclf_b4_substrate.ego_causal.npz). The quarantined "
                          "legacy leaky block is also run, as ::legacy_ego, so the difference is "
                          "measured not asserted."),
           "_vision_only": "ego is the STRATIFICATION variable only; no arm reads it as an input",
           "_powered": ("verdicts are issued for lane_change (55 pos clusters) and intersection "
                        "(216) only. roundabout has 37 against a bar of 40 -> UNDERPOWERED_C_POW, "
                        "reported with its n, bar NOT lowered."),
           "n_boot": a.n_boot, "strata_n_boot": a.strata_n_boot, "situations": {}}

    for ego, tag in ((ego_c, "causal"), (ego_l, "legacy")):
        bundle = ScoreBundle(situations=np.array(sits), arms=list(arms), y=y, valid=valid,
                             clip_cluster=cc, scores={k: z[k] for k in arms}, ego=ego,
                             source=str(a.scores))
        for j, s in enumerate(sits):
            # the legacy pass exists only to MEASURE the stratum-boundary shift, so it runs on
            # the primary candidate alone rather than doubling the cost.
            for arm in (CANDIDATES if tag == "causal" else (CANDIDATES[0],)):
                rep = four_family_report(
                    bundle, s, fused=z[arm][:, j].astype(np.float64),
                    baseline=z[FROZEN][:, j].astype(np.float64),
                    baseline_name=FROZEN, fused_name=arm,
                    n_boot=a.n_boot, strata_n_boot=a.strata_n_boot)
                key = f"{s}::{arm}" + ("" if tag == "causal" else "::legacy_ego")
                out["situations"][key] = rep
                t = rep["families"]["TACTICAL"]
                d = t["paired_delta_ap_lift"]
                op_f, op_b = t["operating_point_5pct"][arm], t["operating_point_5pct"][FROZEN]
                log(f"{key}: TACTICAL ap-lift {t['ap_lift'][arm]:.4f} vs "
                    f"{t['ap_lift'][FROZEN]:.4f}  paired {d['delta']:+.4f} "
                    f"[{d['lo']:+.4f},{d['hi']:+.4f}]{' SEP' if d['separated'] else ''} | "
                    f"P@5% {op_f['precision']:.4f}/R {op_f['recall']:.4f} "
                    f"(fires {op_f['n_alarm']}, true {op_f['n_pos']}) vs "
                    f"{op_b['precision']:.4f}/R {op_b['recall']:.4f} | "
                    f"lead {t['anticipation_lead'][arm]['median_lead_s']}s vs "
                    f"{t['anticipation_lead'][FROZEN]['median_lead_s']}s")
                Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
            if tag == "causal":
                base = out["situations"][f"{s}::{CANDIDATES[0]}"]["families"]
                for fam in ("LONGITUDINAL", "LATERAL"):
                    b = base[fam]
                    if b.get("_status") == "UNAVAILABLE":
                        log(f"  {fam}: UNAVAILABLE — {b['_reason']}")
                        continue
                    ok = [k for k, v in b["strata"].items() if v.get("_status") != "UNPOWERED"]
                    up = [k for k, v in b["strata"].items() if v.get("_status") == "UNPOWERED"]
                    log(f"  {fam}: {len(ok)} powered {ok} | {len(up)} UNPOWERED {up}")
                log(f"  STRATEGIC: {base['STRATEGIC']['_status']}")
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
