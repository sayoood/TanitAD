"""LL3 — render the ladder. Reads every ``raw/ll_*.json`` and emits the tables.

⛔ NOTHING IS COMPUTED HERE. Every number is read from a banked per-arm JSON, so
the tables cannot disagree with the artifacts. The only arithmetic is min/max
over seeds, which is labelled as such.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LADDER = ["ego_v0", "ego_accel", "ego_yawrate", "ego_curv",
          "n_agents_grid", "n_agents_all", "lead_present",
          "nearest_any", "lead_gap", "lead_closing", "lead_inv_ttc"]

# display order: the four columns that decide every row
ARMS = [("ll_s11250.json", "v6 @11250"),
        ("ll_nullmatched.json", "NULL"),
        ("ll_proxyv0.json", "C-V0"),
        ("ll_orcdir.json", "ORACLE")]
CKPT = [("ll_s02000.json", "@2000"), ("ll_s09000.json", "@9000"),
        ("ll_s09250.json", "@9250"), ("ll_s10000.json", "@10000"),
        ("ll_s11250.json", "@11250")]
EGOORC = [("ll_egoorc_n0.1.json", "0.10x"), ("ll_egoorc_n1.json", "1.0x"),
          ("ll_egoorc_n3.json", "3.0x"), ("ll_egoorc_n10.json", "10x")]
REPAIR = [("ll_rep_s11250.json", "v6 @11250"),
          ("ll_rep_nullmatched.json", "NULL"),
          ("ll_rep_proxyv0.json", "C-V0"),
          ("ll_rep_orcdir.json", "ORACLE")]


def load(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def cell(blob, t, field, seed="0", fmt="%+.3f"):
    if blob is None or t not in blob["targets"]:
        return "--"
    v = blob["targets"][t]["per_seed"][seed].get(field)
    return "--" if v is None else (fmt % v)


def sep(blob, t, seed="0"):
    if blob is None or t not in blob["targets"]:
        return "--"
    p = blob["targets"][t]["per_seed"][seed]
    if p["K1_PASSES"]:
        return "**PASS**"
    return "fail" if p["K1_separated"] else "ns"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    d = Path(a.raw)
    B = {n: load(d, n) for n, _ in ARMS + CKPT + EGOORC + REPAIR}
    L = []

    def w(s=""):
        L.append(s)

    ref = B["ll_s11250.json"]
    w("# THE LATENT LINEAR LADDER — rendered tables")
    w()
    w("**Eval tier: T0-DIAGNOSTIC.** A frozen-latent readout is a world-model "
      "diagnostic and is NEVER driving performance.")
    w()
    if ref:
        w(f"Estimator `{ref['estimator']}`, n_boot {ref['n_boot']}, "
          f"seeds {ref['seeds']}, fit_mode `{ref['fit_mode']}`. "
          f"Features: {ref['features']} -> {ref['n_features']} incl. bias.")
        g = ref.get("pc6_equivalence_gate")
        if g:
            w()
            w(f"**pc6 equivalence gate: "
              f"{'PASSED' if g['PASSED'] else 'FAILED'}** "
              f"(against `{Path(g['against']).name}`) — `lead_gap` at seed 0 "
              "reproduces the banked ridge numbers, which is what makes every "
              "other row of this ladder comparable to the precedent.")
        al = ref.get("pose_alignment_check")
        if al:
            w()
            w(f"**Pose-grid binding:** `{al['rule']}` with offset "
              f"**{al['pose_index_offset']}** ({al['n_stack_note']}), accepted "
              f"on {al['accepted_on']} over {al['n_rows_checked']} rows "
              f"(max mismatch {al['max_abs_v0_mismatch']}). Scan: "
              f"{al['scan_max_abs_mismatch']}")
    w()
    w("## 1. THE LADDER — correlation with the truth, arm vs its controls")
    w()
    w("`r` = corr(prediction, truth). `r_wep` = the same AFTER both are "
      "demeaned by their own eval episode — the ANTI-EPISODE-IDENTITY "
      "statistic. `r_pv0` = partial correlation with EGO SPEED partialled out "
      "— the TRIVIAL-PROXY test.")
    w()
    w("| target | rung | unit | n | **v6 r** | **NULL r** | C-V0 r | ORACLE r "
      "| v6 r_wep | v6 r_pv0 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for t in LADDER:
        if ref is None or t not in ref["targets"]:
            continue
        m = ref["targets"][t]
        w("| `%s` | %s | %s | %d/%d | **%s** | %s | %s | %s | %s | %s |"
          % (t, m["rung"], m["unit"], m["n_eval"], m["n_eval_clusters"],
             cell(B["ll_s11250.json"], t, "corr"),
             cell(B["ll_nullmatched.json"], t, "corr"),
             cell(B["ll_proxyv0.json"], t, "corr"),
             cell(B["ll_orcdir.json"], t, "corr"),
             cell(B["ll_s11250.json"], t, "corr_within_ep"),
             cell(B["ll_s11250.json"], t, "corr_partial_v0")))
    w()
    w("## 2. THE SAME LADDER IN ERROR AND K1 — every row with its null")
    w()
    w("Positive K1 = the arm is WORSE than the constant. `PASS` = separated "
      "and negative.")
    w()
    w("| target | unit | C-CONST | v6 err | v6 K1 | NULL err | NULL K1 "
      "| C-V0 err | C-V0 K1 | ORACLE err | ORACLE K1 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in LADDER:
        if ref is None or t not in ref["targets"]:
            continue
        m = ref["targets"][t]
        row = ["`%s`" % t, m["unit"],
               cell(B["ll_s11250.json"], t, "c_const_err", fmt="%.4f")]
        for n, _ in ARMS:
            row += [cell(B[n], t, "err", fmt="%.4f"),
                    "%s %s" % (cell(B[n], t, "K1_delta"), sep(B[n], t))]
        w("| " + " | ".join(row) + " |")
    w()
    w("## 3. R^2, and the CEILING an optimally-rescaled linear readout reaches")
    w()
    w("⚠️ The fit is OVER-DISPERSED — it emits close to full variance at low "
      "correlation — so MAE can lose to a constant while `r` is positive. "
      "`r2_ceiling` = r^2 is the variance a perfectly-rescaled version of the "
      "SAME readout would explain. It is the fairest single number per rung.")
    w()
    w("| target | v6 R2 | **v6 r2_ceiling** | NULL r2_ceiling | "
      "C-V0 r2_ceiling | ORACLE r2_ceiling |")
    w("|---|---|---|---|---|---|")
    for t in LADDER:
        if ref is None or t not in ref["targets"]:
            continue
        w("| `%s` | %s | **%s** | %s | %s | %s |"
          % (t, cell(B["ll_s11250.json"], t, "R2"),
             cell(B["ll_s11250.json"], t, "r2_ceiling", fmt="%.4f"),
             cell(B["ll_nullmatched.json"], t, "r2_ceiling", fmt="%.4f"),
             cell(B["ll_proxyv0.json"], t, "r2_ceiling", fmt="%.4f"),
             cell(B["ll_orcdir.json"], t, "r2_ceiling", fmt="%.4f")))
    w()
    w("## 4. SEED SPREAD — between-condition vs between-seed")
    w()
    w("| target | v6 err seed-range | v6 K1 seed-range | v6 err | "
      "|v6 - NULL| err gap |")
    w("|---|---|---|---|---|")
    for t in LADDER:
        if ref is None or t not in ref["targets"]:
            continue
        m = ref["targets"][t]
        nb = B["ll_nullmatched.json"]
        gap = "--"
        if nb and t in nb["targets"]:
            gap = "%.4f" % abs(m["per_seed"]["0"]["err"]
                               - nb["targets"][t]["per_seed"]["0"]["err"])
        w("| `%s` | %.4f | %.4f | %.4f | %s |"
          % (t, m["seed_err_range"], m["seed_K1_range"],
             m["per_seed"]["0"]["err"], gap))
    w()
    if B.get("ll_egoorc_n0.1.json"):
        w("## 5. ⭐ THE READOUT'S OWN POSITIVE CONTROL FOR THE ANCHOR")
        w()
        w("`EGO-ORACLE` = the real cache with `cells` replaced by a "
          "DISTRIBUTED random projection of the window's own `v0`, at four "
          "noise levels (x the real cells' std). This is what the ladder's "
          "anchor row looks like when ego speed IS linearly present.")
        w()
        w("| noise | ego_v0 err (m/s) | K1 | r | r_wep | R2 |")
        w("|---|---|---|---|---|---|")
        for n, lab in EGOORC:
            if B.get(n) is None:
                continue
            w("| **%s** | %s | %s %s | **%s** | %s | %s |"
              % (lab, cell(B[n], "ego_v0", "err", fmt="%.4f"),
                 cell(B[n], "ego_v0", "K1_delta"), sep(B[n], "ego_v0"),
                 cell(B[n], "ego_v0", "corr"),
                 cell(B[n], "ego_v0", "corr_within_ep"),
                 cell(B[n], "ego_v0", "R2")))
        w("| *v6 @11250 (the real arm)* | %s | %s %s | **%s** | %s | %s |"
          % (cell(B["ll_s11250.json"], "ego_v0", "err", fmt="%.4f"),
             cell(B["ll_s11250.json"], "ego_v0", "K1_delta"),
             sep(B["ll_s11250.json"], "ego_v0"),
             cell(B["ll_s11250.json"], "ego_v0", "corr"),
             cell(B["ll_s11250.json"], "ego_v0", "corr_within_ep"),
             cell(B["ll_s11250.json"], "ego_v0", "R2")))
        w()
    if B.get("ll_s09000.json"):
        w("## 6. THE CHECKPOINT TRAJECTORY — `r` per rung")
        w()
        w("| target | " + " | ".join(l for _, l in CKPT) + " |")
        w("|---|" + "---|" * len(CKPT))
        for t in LADDER:
            if ref is None or t not in ref["targets"]:
                continue
            w("| `%s` | " % t + " | ".join(cell(B[n], t, "corr")
                                           for n, _ in CKPT) + " |")
        w()
    if B.get("ll_rep_s11250.json"):
        w("## 7. ⛔ THE INTERCEPT REPAIR — pc6 penalises its own bias term")
        w()
        w("pc6's `ridge_fit` puts the appended ones-column INSIDE "
          "`alpha * np.eye(d)`, so as alpha grows the prediction collapses "
          "toward ZERO rather than toward the mean: the readout is "
          "structurally unable to fall back to the very constant K1 scores it "
          "against. `centred` mode centres `y` and leaves the intercept "
          "unpenalised. Same caches, same split, same estimator, same seeds.")
        w()
        w("| target | v6 K1 pc6 | **v6 K1 repaired** | NULL K1 repaired | "
          "ORACLE K1 repaired | v6 r repaired |")
        w("|---|---|---|---|---|---|")
        for t in LADDER:
            if ref is None or t not in ref["targets"]:
                continue
            w("| `%s` | %s %s | **%s %s** | %s %s | %s %s | %s |"
              % (t, cell(B["ll_s11250.json"], t, "K1_delta"),
                 sep(B["ll_s11250.json"], t),
                 cell(B["ll_rep_s11250.json"], t, "K1_delta"),
                 sep(B["ll_rep_s11250.json"], t),
                 cell(B["ll_rep_nullmatched.json"], t, "K1_delta"),
                 sep(B["ll_rep_nullmatched.json"], t),
                 cell(B["ll_rep_orcdir.json"], t, "K1_delta"),
                 sep(B["ll_rep_orcdir.json"], t),
                 cell(B["ll_rep_s11250.json"], t, "corr")))
        w()
    Path(a.out_md).write_text("\n".join(L) + "\n", "utf-8")
    Path(a.out_json).write_text(json.dumps(
        {"_evidence_class": "MEASURED (ours; assembled from the banked "
                            "per-arm ll_*.json, no recomputation)",
         "eval_tier": "T0-DIAGNOSTIC",
         "arms": {n: (B[n]["arm"] if B[n] else None)
                  for n in B}, "blobs": B}, indent=1), "utf-8")
    print(f"[ll3] wrote {a.out_md} and {a.out_json} "
          f"({sum(1 for v in B.values() if v)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
