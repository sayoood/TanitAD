"""The four BINDING metric families for this ladder, from the banked scores.

CLAUDE.md, binding 2026-08-02: *"Any future eval must include these metrics."* An eval that reports
AP alone is INCOMPLETE. This produces LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC per situation,
each with the paired episode-cluster bootstrap and its CI, on the same rows as the ladder — and
where a family genuinely cannot be computed it is reported **with its reason and its n**, never
silently dropped.

It reads only `results_temporal.scores.npz` + `results_fast.json` (for the peak arm), so it does not
wait on the slow run and is a pure re-analysis: **0 GPU, no refitting**.

⚠️ The ego block used for the LONGITUDINAL/LATERAL regime strata is the substrate's, which was built
BEFORE the 2026-08-03 causality fix and therefore carries the legacy centred channels (measured:
bit-identical to `kinematics(..., causal_pre=False)`). Stratification is not a model input and a
paired within-stratum contrast stays valid, but the stratum BOUNDARIES are drawn with a channel that
reads 0.1 s past t. Disclosed rather than assumed away.

usage:
  python four_families.py --scores results_temporal.scores.npz --fast results_fast.json \
      --out results_four_families.json
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


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results_temporal.scores.npz")
    ap.add_argument("--fast", default="results_fast.json")
    ap.add_argument("--out", default="results_four_families.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--strata-n-boot", type=int, default=500)
    a = ap.parse_args()

    z = np.load(a.scores)
    F = json.loads(Path(a.fast).read_text(encoding="utf-8"))
    ref = F["reference_arm"]
    sits = [str(s) for s in z["situations"]]
    arms = [k for k in z.files if k not in ("clip_cluster", "y", "valid", "ego",
                                            "cache_tag", "folds", "situations", "t")]
    real = [k for k in arms if not k.startswith("NEG_")]
    bundle = ScoreBundle(situations=np.array(sits), arms=real, y=z["y"],
                         valid=z["valid"], clip_cluster=z["clip_cluster"],
                         scores={k: z[k] for k in real}, ego=z["ego"],
                         source=str(a.scores))
    out = {"_what": "the four BINDING metric families, peak deployable arm vs the reference",
           "_binding": "CLAUDE.md 2026-08-02 — ADE/AP alone is an INCOMPLETE result",
           "_ego_caveat": ("the regime strata read the substrate's ego block, which predates the "
                           "2026-08-03 causality fix (legacy centred channels). Stratification is "
                           "not a model input; the stratum boundaries read 0.1 s past t."),
           "reference_arm": ref, "n_boot": a.n_boot,
           "strata_n_boot": a.strata_n_boot, "situations": {}}

    for j, s in enumerate(sits):
        peak = F["per_situation"][s]["peak_arm"]
        t0 = time.time()
        rep = four_family_report(
            bundle, s,
            fused=z[peak][:, j].astype(np.float64),
            baseline=z[ref][:, j].astype(np.float64),
            baseline_name=ref, fused_name=f"PEAK::{peak}",
            n_boot=a.n_boot, strata_n_boot=a.strata_n_boot)
        rep["_verdict_from_ladder"] = F["per_situation"][s]["VERDICT"]
        rep["_peak_arm"] = peak
        out["situations"][s] = rep
        t = rep["families"]["TACTICAL"]
        log(f"{s}: peak={peak} | TACTICAL ap-lift "
            f"{t['ap_lift'][f'PEAK::{peak}']:.3f} vs {t['ap_lift'][ref]:.3f} "
            f"paired {t['paired_delta_ap_lift']['delta']:+.3f} "
            f"[{t['paired_delta_ap_lift']['lo']:+.3f},{t['paired_delta_ap_lift']['hi']:+.3f}]"
            f"{' SEP' if t['paired_delta_ap_lift']['separated'] else ''} "
            f"| lead {t['anticipation_lead'][f'PEAK::{peak}']['median_lead_s']}s vs "
            f"{t['anticipation_lead'][ref]['median_lead_s']}s ({time.time()-t0:.0f}s)")
        for fam in ("LONGITUDINAL", "LATERAL"):
            b = rep["families"][fam]
            if b.get("_status") == "UNAVAILABLE":
                log(f"  {fam}: UNAVAILABLE — {b['_reason']}")
                continue
            ok = [k for k, v in b["strata"].items() if v.get("_status") != "UNPOWERED"]
            up = [k for k, v in b["strata"].items() if v.get("_status") == "UNPOWERED"]
            log(f"  {fam}: {len(ok)} powered strata {ok}, {len(up)} UNPOWERED {up}")
        log(f"  STRATEGIC: {rep['families']['STRATEGIC']['_status']}")
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
