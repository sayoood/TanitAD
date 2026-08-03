"""The four BINDING metric families for the per-situation-horizon arms.

CLAUDE.md, binding 2026-08-02: *"Any future eval must include these metrics."* An eval that reports
a ranking metric alone is INCOMPLETE. This produces LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC
per situation, never pooled, each with the paired episode-cluster bootstrap and its CI, on the same
rows as the study — and where a family genuinely cannot be computed it is reported WITH ITS REASON
AND ITS n, never silently dropped.

⭐ DIFFERENCE FROM THE SIBLING STREAM'S RUN: the LONGITUDINAL/LATERAL regime strata here are drawn
with the **CAUSAL** ego block rebuilt by `rebuild_causal_ego.py`, not the B4 substrate's legacy
block. The sibling had to disclose that its stratum boundaries were drawn with a channel reading
0.1 s past t; this run does not, because P4 fixed it. Both are reported so the difference is
visible rather than asserted.

Pure re-analysis of the banked score columns: 0 GPU, no refitting.

usage:
  python four_families_ps.py --scores results_horizon_ps.scores.npz \
      --ego-causal C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.ego_causal.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

from tanitad.eval.sitclf_deploy import (ScoreBundle,                  # noqa: E402
                                        four_family_report)

ARMS = ("FROZEN", "PS_SEL", "C_GLOBAL", "C_ORACLE_PS")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results_horizon_ps.scores.npz")
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
    y = z["y_lead3"]
    valid = z["eval_rows"]
    cc = z["clip_cluster"]
    ego_c = np.load(a.ego_causal)["E_causal"]
    ego_l = np.load(a.ego_legacy)["E"]
    log(f"{y.shape[0]:,} rows, {len(np.unique(cc))} clusters, arms {ARMS}")

    out = {"_what": "the four BINDING metric families, PS_SEL and controls vs the FROZEN setting",
           "_binding": "CLAUDE.md 2026-08-02 — a ranking metric alone is an INCOMPLETE result",
           "_ego_block": ("CAUSAL rebuild (rebuild_causal_ego.py). The legacy leaky block is also "
                          "run, as `strata_legacy_ego`, so the difference is measured not asserted."),
           "n_boot": a.n_boot, "strata_n_boot": a.strata_n_boot, "situations": {}}

    for ego, tag in ((ego_c, "causal"), (ego_l, "legacy")):
        bundle = ScoreBundle(situations=np.array(sits), arms=list(ARMS), y=y, valid=valid,
                             clip_cluster=cc, scores={k: z[k] for k in ARMS}, ego=ego,
                             source=str(a.scores))
        for j, s in enumerate(sits):
            for arm in ("PS_SEL", "C_GLOBAL", "C_ORACLE_PS"):
                rep = four_family_report(
                    bundle, s, fused=z[arm][:, j].astype(np.float64),
                    baseline=z["FROZEN"][:, j].astype(np.float64),
                    baseline_name="FROZEN", fused_name=arm,
                    n_boot=a.n_boot, strata_n_boot=a.strata_n_boot)
                key = f"{s}::{arm}" + ("" if tag == "causal" else "::legacy_ego")
                out["situations"][key] = rep
                t = rep["families"]["TACTICAL"]
                d = t["paired_delta_ap_lift"]
                op_f, op_b = (t["operating_point_5pct"][arm],
                              t["operating_point_5pct"]["FROZEN"])
                log(f"{key}: TACTICAL ap-lift {t['ap_lift'][arm]:.4f} vs "
                    f"{t['ap_lift']['FROZEN']:.4f}  paired {d['delta']:+.4f} "
                    f"[{d['lo']:+.4f},{d['hi']:+.4f}]{' SEP' if d['separated'] else ''} | "
                    f"P@5% {op_f['precision']:.4f}/R {op_f['recall']:.4f} vs "
                    f"{op_b['precision']:.4f}/R {op_b['recall']:.4f} "
                    f"(fires {op_f['n_alarm']}, true {op_f['n_pos']}) | "
                    f"lead {t['anticipation_lead'][arm]['median_lead_s']}s vs "
                    f"{t['anticipation_lead']['FROZEN']['median_lead_s']}s")
                Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
            if tag == "causal":
                for fam in ("LONGITUDINAL", "LATERAL"):
                    b = out["situations"][f"{s}::PS_SEL"]["families"][fam]
                    if b.get("_status") == "UNAVAILABLE":
                        log(f"  {fam}: UNAVAILABLE — {b['_reason']}")
                        continue
                    ok = [k for k, v in b["strata"].items() if v.get("_status") != "UNPOWERED"]
                    up = [k for k, v in b["strata"].items() if v.get("_status") == "UNPOWERED"]
                    log(f"  {fam}: {len(ok)} powered {ok} | {len(up)} UNPOWERED {up}")
                log(f"  STRATEGIC: {out['situations'][f'{s}::PS_SEL']['families']['STRATEGIC']['_status']}")
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
