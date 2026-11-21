"""E-BE-DRIFT-1 — what our "drift" metric actually measures, and where its blind spot is.

⛔ WHY THIS EXISTS. Backlog row **FS-1** proposes porting WorldRoamBench's
SEGMENT-BASED drift metric into `taniteval` and re-scoring the banked rollout
dumps, on this premise:

    "Our drift is read START-VS-END, which by their construction cannot see
     non-monotonic mid-sequence collapse."

Before spending the port, this audit establishes **what our drift metric is**,
from its own source rather than from its name.

⭐ IT IS NOT A START-VS-END TRAJECTORY MEASURE. `mm-e19-probes/latentmotion.py`
(E-DEC-59) states its own target:

    "TARGETS -- dz = z_{t+k} - z_t projected on its top PCA directions"
    "COLUMNS: z_t -- the DRIFT baseline, and the POSITIVE CONTROL"

⇒ "drift" is the **r of a k-fold-fit linear probe predicting dz from z_t** -- a
per-window PREDICTABILITY statistic at a single fixed k. There is no sequence
being reduced to its endpoints, so the specific defect FS-1 names cannot apply.

⭐⭐ BUT FS-1's UNDERLYING CONCERN SURVIVES, ON A DIFFERENT AXIS. Our statistic is
a single number at ONE k. A summary at one k cannot see non-monotone structure
**in k** -- which is the same class of blind spot, relocated. This script measures
the k axis on everything already banked.

⛔ CONTROLS, carried from the source reads and re-asserted here rather than
assumed:
  * `constant (control)` must read EXACTLY 0.0 on every arm and every read. A
    panel whose no-information column drifts off zero is not scored.
  * `z_t` is itself the POSITIVE control (E-DEC-59: a panel that cannot reproduce
    it "is broken and must not be read").
  * `n_rows` is printed with every number -- k=60 halves the usable windows, and a
    comparison that hides that is comparing two different sample sizes.

Usage:
    python drift_axis_audit.py --out ../raw/drift_axis.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
from datetime import datetime, timezone

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
BASE = os.path.join(REPO, "TanitAD Research Lab", "Architecture & Inference",
                    "Research")
SOURCES = {
    "k60_reread_2026-09-02": os.path.join(
        BASE, "2026-08-31-mm-e19-k60-horizon", "raw", "reread-2026-09-02"),
    "k8_attribution": os.path.join(
        BASE, "2026-09-01-mm-e19-k8-attribution", "raw"),
}
DRIFT_COL = "z_t (DRIFT / POSITIVE CONTROL)"
CONST_COL = "constant (control)"

#: ⛔ the k8 read's files store their arm under the internal key
#: "k60clip05p30k" -- the `--arm-name` default documented in the attribution
#: package (§6a), resolved only by checkpoint md5. Named here so the audit cannot
#: silently swap two arms.
INTERNAL_KEY_IS_ACTUALLY = {
    "k8_attribution": {"k60clip05p30k": "k8clip05p30k"},
}


def collect() -> tuple[dict, list[str]]:
    rows, problems = {}, []
    for label, d in SOURCES.items():
        if not os.path.isdir(d):
            problems.append(f"missing source dir: {d}")
            continue
        for f in sorted(os.listdir(d)):
            if "latentmotion" not in f or not f.endswith(".json"):
                continue
            j = json.load(io.open(os.path.join(d, f), encoding="utf-8"))
            for arm, v in j["arms"].items():
                true_arm = INTERNAL_KEY_IS_ACTUALLY.get(label, {}).get(arm, arm)
                cols = v["columns"]
                const = cols[CONST_COL]["r"]
                if const != 0.0:
                    problems.append(
                        f"{label}/{f}/{true_arm}: constant control reads {const}, "
                        f"required exactly 0.0 -- NOT SCORED")
                    continue
                rows.setdefault(true_arm, []).append({
                    "source": label, "file": f,
                    "k": j["k"], "pca_band": j["pca_band"],
                    "split": j.get("split"),
                    "n_clips": v["n_clips"], "n_rows": v["n_rows"],
                    "drift_r": cols[DRIFT_COL]["r"],
                    "drift_t": cols[DRIFT_COL]["t"],
                    "constant_control_r": const,
                })
    return rows, problems


def k_axis(rows: dict) -> dict:
    """For each (arm, band), how much does drift move between the banked k values?"""
    out = {}
    for arm, rs in rows.items():
        for band in ({tuple(r["pca_band"]) for r in rs}):
            sel = sorted((r for r in rs if tuple(r["pca_band"]) == band),
                         key=lambda r: r["k"])
            # prefer the most recent source when a (k, band) appears twice
            best: dict[int, dict] = {}
            for r in sel:
                if r["k"] not in best or r["source"] == "k60_reread_2026-09-02":
                    best[r["k"]] = r
            ks = sorted(best)
            if len(ks) < 2:
                continue
            lo, hi = best[ks[0]], best[ks[-1]]
            rel = (hi["drift_r"] - lo["drift_r"]) / lo["drift_r"]
            out[f"{arm} band{band[0]}_{band[1]}"] = {
                "k_low": ks[0], "k_high": ks[-1],
                "horizon_ratio": ks[-1] / ks[0],
                "drift_r_at_k_low": lo["drift_r"],
                "drift_r_at_k_high": hi["drift_r"],
                "relative_change": rel,
                "n_rows_low": lo["n_rows"], "n_rows_high": hi["n_rows"],
                "endpoints_equal_within_2pct": abs(rel) < 0.02,
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows, problems = collect()
    ax = k_axis(rows)
    res = {
        "_experiment": "E-BE-DRIFT-1",
        "_date": "2026-09-02",
        "_generated_utc": datetime.now(timezone.utc).isoformat(),
        "_evidence_class": "MEASURED (ours; re-analysis of banked latentmotion reads)",
        "_tier": "T0-DIAGNOSTIC (inherited from the source reads) -- NOT a driving claim",
        "_what_drift_is": (
            "r of a k-fold-fit linear probe predicting dz = z_{t+k} - z_t, "
            "projected on ITS OWN top PCA directions, from z_t. Source: "
            "mm-e19-probes/latentmotion.py (E-DEC-59). ⛔ NOT a start-vs-end "
            "trajectory measure."),
        "_caveat_pca_basis": (
            "⚠️ the target is dz projected on the top PCA directions OF THAT k, so "
            "k=4 and k=60 are not literally the same target vector -- the claim is "
            "about the predictability of each k's own leading dz structure, not "
            "about one fixed quantity."),
        "controls": {
            "constant_control_required": 0.0,
            "reads_not_scored_due_to_control_failure": problems,
            "positive_control": "z_t is itself the drift column (E-DEC-59)",
        },
        "reads": rows,
        "k_axis": ax,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)

    print("E-BE-DRIFT-1 — drift across the k axis (banked reads)")
    for k, v in sorted(ax.items()):
        print(f"  {k:34s} k {v['k_low']:>2}->{v['k_high']:<3} "
              f"({v['horizon_ratio']:.0f}x)  r {v['drift_r_at_k_low']:.4f} -> "
              f"{v['drift_r_at_k_high']:.4f}  ({v['relative_change']*100:+.2f}%)"
              f"  n {v['n_rows_low']}->{v['n_rows_high']}")
    print(f"\n  control failures: {len(problems)}")
    for p in problems:
        print("   ", p)
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
