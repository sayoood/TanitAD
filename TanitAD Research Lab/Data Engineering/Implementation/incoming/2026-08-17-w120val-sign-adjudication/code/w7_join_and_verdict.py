"""STEP 7 — join the BLIND verdicts to the withheld metadata and decide the
pre-registered outcome.

⛔ **THE PRECISION MATHS IS THE STUDY'S OWN CODE, CALLED UNCHANGED.**
`r5_precision.py` computes the point estimate, the **episode-cluster bootstrap
over CLIPS** (`taniteval.ci`, 2 000 draws — never `overlapping_holdout_se`,
never a binomial), the bands and the sweep. `r8_g1_verdict_join.py` computes the
uniform-vs-max-area head-to-head and the box-area mechanism. Writing a second
implementation of either would reopen the confound this package exists to close.

What this file adds is only what the study's scripts do not compute:

  1. **the G1 SUBCLASS RATE** — `G1_RESULT.md:17`'s *"no sign visible in the crop
     at all"* — over ALL adjudicated cells, with its own clip-clustered CI, so it
     is directly comparable to G1's ~22/31 ≈ 0.71 and the study's 4/96 ≈ 0.042;
  2. **the cross-arm table** A (w120val uniform) / B (pilot-50, G1's rule) /
     C (G1's own tiles, re-read);
  3. **the pre-registered decision**, evaluated mechanically against
     `PREREG.md` §3 rather than argued for after the fact.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
STUDY = os.path.join(REPO, "TanitAD Research Hub", "Data Engineering",
                     "Implementation", "incoming",
                     "2026-08-16-sam3-concept-reliability", "code")
sys.path.insert(0, os.path.join(REPO, "taniteval"))
sys.path.insert(0, STUDY)

AUG120 = {"precision_resolvable": 0.880, "lo": 0.795, "hi": 0.958, "n": 64,
          "empty_box_rate": 4 / 96,
          "source": "…/2026-08-16-sam3-concept-reliability/raw/precision.json "
                    "+ its §4.1 (MEASURED, corpus aug120)"}
G1 = {"empty_box_rate": 22 / 31, "n": 31,
      "source": "Project Steering/G1_RESULT.md (MEASURED 2026-08-14), rows of "
                "the 50-clip pilot"}


def boot(ind, clips, seed=0):
    from taniteval.ci import episode_cluster_bootstrap
    if not ind:
        return None
    r = episode_cluster_bootstrap(ind, clips, reduce="mean", seed=seed)
    return {"point": round(r["mean"], 4), "lo": round(r["lo"], 4),
            "hi": round(r["hi"], 4), "n_cells": r["n_windows"],
            "n_clips": r["n_episodes"], "estimator": r["estimator"]}


def subclass_rate(sample_path, verdict_path, key="detections",
                  idkey="clip_id"):
    """G1's subclass 1, over ALL adjudicated cells, clip-clustered."""
    s = json.load(open(sample_path, encoding="utf-8"))
    v = json.load(open(verdict_path, encoding="utf-8"))
    rows = {int(d["idx"]): d for d in s[key]}
    empt = {str(i) for i in (v.get("g1_subclass_1_empty_box") or {})
            .get("indices", [])}
    ind, cl = [], []
    for k in v["verdicts"]:
        ind.append(1.0 if k in empt else 0.0)
        cl.append(rows[int(k)].get(idkey) or f"_tile{k}")
    return boot(ind, cl), len(ind), int(sum(ind))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("w7_join_and_verdict")
    ap.add_argument("--sample-a", required=True)
    ap.add_argument("--verdicts-a", required=True)
    ap.add_argument("--dist", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--sample-b", default=None)
    ap.add_argument("--verdicts-b", default=None)
    ap.add_argument("--sample-c", default=None)
    ap.add_argument("--verdicts-c", default=None)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    # ---- 1. the study's own precision maths, unchanged --------------------
    import r5_precision
    p_a = os.path.join(a.out_dir, "precision_A_w120val.json")
    rc = r5_precision.main(["--sample", a.sample_a, "--verdicts", a.verdicts_a,
                            "--dist", a.dist, "--cache", a.cache,
                            "--out", p_a])
    if rc:
        return rc
    prec_a = json.load(open(p_a, encoding="utf-8"))["concepts"]["traffic sign"]

    out = {
      "evidence_class": "MEASURED (blind human adjudication of rendered crops "
                        "+ the reliability study's own estimator code)",
      "estimator": "episode-cluster bootstrap over CLIPS (taniteval.ci), "
                   "2000 draws — never overlapping_holdout_se",
      "reference_numbers": {"aug120_reliability_study": AUG120, "G1": G1},
      "arms": {}}

    sc_a, n_a, k_a = subclass_rate(a.sample_a, a.verdicts_a)
    out["arms"]["A_w120val_uniform"] = {
        "corpus": "w120val (ph0_prod4) — 596 clips, 4 048 `traffic sign` "
                  "detections over 440 clips",
        "selection": "uniform at random within `traffic sign`",
        "rendering": "6x-box context window from native 640, WITH the box "
                     "outlined in gold",
        "n": prec_a["n_adjudicated"], "n_correct": prec_a["n_correct"],
        "n_wrong": prec_a["n_wrong"], "n_unclear": prec_a["n_unclear"],
        "unclear_rate": prec_a["unclear_rate"],
        "precision_resolvable_only": prec_a["precision_resolvable_only"],
        "precision_unclear_as_wrong": prec_a["precision_unclear_as_wrong"],
        "g1_subclass_no_sign_at_all": {"n": k_a, "of": n_a,
                                       "rate": round(k_a / n_a, 4),
                                       "ci": sc_a},
        "scores_of_false_positives": prec_a["score_wrong"],
        "bands": prec_a["bands"], "threshold_sweep": prec_a["threshold_sweep"]}

    # ---- 2. arm B, via the study's own head-to-head joiner ----------------
    if a.sample_b and os.path.exists(a.verdicts_b or ""):
        import r8_g1_verdict_join
        p_b = os.path.join(a.out_dir, "g1_headtohead_B.json")
        rc = r8_g1_verdict_join.main([
            "--recon-sample", a.sample_b, "--recon-verdicts", a.verdicts_b,
            "--sample", a.sample_a, "--verdicts", a.verdicts_a, "--out", p_b])
        if rc:
            return rc
        hh = json.load(open(p_b, encoding="utf-8"))
        sc_b, n_b, k_b = subclass_rate(a.sample_b, a.verdicts_b)
        out["arms"]["B_pilot50_g1_maxarea"] = {
            "corpus": "pilot-50 — G1's OWN clips, a strict subset of w120val",
            "selection": "G1's: largest-area `traffic sign` per clip, CENSUS",
            "rendering": "BOTH — G1's tight 4x LANCZOS and the 6x context "
                         "window WITH a gold box outline",
            **{k: hh["arm_maxarea_g1_selection"][k] for k in
               ("n", "n_correct", "n_wrong", "n_unclear",
                "precision_resolvable", "median_box_area_px")},
            "g1_subclass_no_sign_at_all": {"n": k_b, "of": n_b,
                                           "rate": round(k_b / n_b, 4),
                                           "ci": sc_b},
            "head_to_head_file": os.path.basename(p_b)}

    # ---- 3. arm C: G1's own tiles ----------------------------------------
    if a.sample_c and os.path.exists(a.verdicts_c or ""):
        sc = json.load(open(a.sample_c, encoding="utf-8"))
        vc = json.load(open(a.verdicts_c, encoding="utf-8"))
        vs = vc["verdicts"]
        nw = sum(1 for v in vs.values() if v == "wrong")
        nc = sum(1 for v in vs.values() if v == "correct")
        nu = sum(1 for v in vs.values() if v == "unclear")
        # every tile is its own cluster here: the 54 tiles span ~30 clips but
        # the tile->clip map is G1's, and the row->clip resolution is by
        # PREFIX, so the honest cluster is the ROW (a row = one clip).
        rowof = {t["idx"]: t["row_file"].replace("b.jpg", "").replace(".jpg", "")
                 for t in sc["tiles"]}
        ci = boot([1.0 if vs[str(i)] == "wrong" else 0.0 for i in rowof],
                  [rowof[i] for i in rowof])
        out["arms"]["C_g1_own_tiles_reread"] = {
            "corpus": "G1's exact banked crops (g1_evidence/crops/*.jpg)",
            "selection": "G1's", "rendering": "G1's — the identical JPEG bytes",
            "n": len(vs), "n_correct": nc, "n_wrong": nw, "n_unclear": nu,
            "no_sign_at_all_rate": round(nw / len(vs), 4),
            "ci_clustered_by_row": ci,
            "what_it_isolates": "the ADJUDICATOR, and nothing else"}

    # ---- 4. the PRE-REGISTERED decision, evaluated mechanically -----------
    P = out["arms"]["A_w120val_uniform"]["precision_resolvable_only"]
    E = out["arms"]["A_w120val_uniform"]["g1_subclass_no_sign_at_all"]["rate"]
    o1 = (E >= 0.40) or (P["hi"] < 0.70)
    o2 = (E <= 0.15) and (P["hi"] >= AUG120["lo"] and P["lo"] <= AUG120["hi"])
    out["prereg_decision"] = {
        "rule": "PREREG.md §3, evaluated mechanically",
        "E_val_no_sign_at_all_rate": E,
        "P_val": {"point": P["point"], "lo": P["lo"], "hi": P["hi"]},
        "outcome_1_corpora_differ_triggered": bool(o1),
        "outcome_2_g1_does_not_generalise_triggered": bool(o2),
        "outcome": ("1 — THE CORPORA GENUINELY DIFFER" if o1 and not o2 else
                    "2 — G1's READING DOES NOT GENERALISE TO A UNIFORM DRAW"
                    if o2 and not o1 else
                    "3 — NEITHER CLEANLY"),
        "note": "⚠️ PREREG.md §3 also binds: if the arms disagree with each "
                "other, outcome 3 is the answer regardless of arm A alone. "
                "That check is a JUDGEMENT and is made in the report, not "
                "here."}
    o = os.path.join(a.out_dir, "w120val_sign_verdict.json")
    json.dump(out, open(o, "w", encoding="utf-8"), indent=1)
    print("\n=== ARMS ===")
    for k, v in out["arms"].items():
        pr = (v.get("precision_resolvable_only")
              or v.get("precision_resolvable") or {})
        print(f"  {k:<28} n={v['n']:>3} ok={v['n_correct']:>3} "
              f"bad={v['n_wrong']:>3} unclear={v['n_unclear']:>2} "
              f"P={pr.get('point')} [{pr.get('lo')},{pr.get('hi')}] "
              f"no-sign-at-all="
              f"{(v.get('g1_subclass_no_sign_at_all') or {}).get('rate', v.get('no_sign_at_all_rate'))}")
    print(f"\n=== PRE-REGISTERED OUTCOME: {out['prereg_decision']['outcome']}")
    print("JOIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
