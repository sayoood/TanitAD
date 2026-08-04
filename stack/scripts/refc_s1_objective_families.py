"""The FOUR FAMILIES for the objective diagnostic — per family, never pooled.

POST-HOC, like the diagnostic it decomposes. It exists because the binding rule
(Sayed, 2026-08-02) is that any eval reports LONGITUDINAL / LATERAL / TACTICAL /
STRATEGIC alongside ADE, and the objective diagnostic's load-bearing row —
`soft-ADE minus the SAME features under the registered CE` — was published as an
ADE delta only. An ADE row is one row of five.

The claim being decomposed: swapping the objective from "listwise CE against the
oracle INDEX" (what `refc_train.loss_rcls` uses) to "expected ADE under the score's
own softmax" recovers essentially the whole deficit the CE-fitted rankers showed.
If that recovery is LONGITUDINAL it matters — 87.6-89.9 % of the selection gap is.
If it is lateral it cannot pay for itself here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

for _p in (Path.home() / "TanitAD" / "taniteval", Path.home() / "TanitAD" / "stack",
           Path.home() / "TanitAD" / "stack" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import refc_sel_probe as P                                        # noqa: E402
import refc_s1_climbout_probe as C                                # noqa: E402

SETS = ("B-both", "D-lon+scores", "C-lon")


def run(bank: str, arm: str, out_dir: str) -> dict:
    t0 = time.time()
    d = P.load_fan(bank)
    eid = list(d["eid"])
    de_all = P.candidate_ade(d["fan"], d["gt"])
    de_or = de_all.min(1).values
    keep, _ = C.survivor_mask(d)
    feats = C.build_features(d, keep)
    shipped_idx = d["sel"]
    shipped = P.ranker_block(de_all, de_or, shipped_idx, eid, tag="shipped")

    out = {}
    for name in SETS:
        names = C.FEATURE_SETS[name]
        ce_s, _ = C.loeo_scores(feats, names, keep, de_all, eid, objective="ce")
        sa_s, _ = C.loeo_scores(feats, names, keep, de_all, eid,
                                objective="softade")
        i_ce = C.argmax_over_survivors(ce_s, keep)
        i_sa = C.argmax_over_survivors(sa_s, keep)
        b_ce = P.ranker_block(de_all, de_or, i_ce, eid, tag=f"{name}-ce")
        b_sa = P.ranker_block(de_all, de_or, i_sa, eid, tag=f"{name}-softade")
        out[name] = {
            "features": names,
            "ade_ce": round(float(b_ce["_per_window_ade"].mean()), 6),
            "ade_softade": round(float(b_sa["_per_window_ade"].mean()), 6),
            "tactical": {
                "ce": {k: b_ce[k] for k in
                       ("rank_acc", "sel_gap", "frac_sel_2x_worse")},
                "softade": {k: b_sa[k] for k in
                            ("rank_acc", "sel_gap", "frac_sel_2x_worse")},
                "_note": ("the goal/anchor-SELECTION half of TACTICAL. The "
                          "manoeuvre-decision half needs decoded manoeuvre "
                          "logits, which a fan bank does not store — reported "
                          "UNAVAILABLE with n below, never silently dropped.")},
            "families_softade_minus_ce": P.family_paired(
                d, i_sa, i_ce, eid, tag=f"{name}: softADE - CE"),
            "families_softade_minus_shipped": P.family_paired(
                d, i_sa, shipped_idx, eid, tag=f"{name}: softADE - shipped"),
            "paired_ade_softade_minus_ce": P._paired(
                b_sa["_per_window_ade"].numpy(),
                b_ce["_per_window_ade"].numpy(), eid),
            "paired_ade_softade_minus_shipped": P._paired(
                b_sa["_per_window_ade"].numpy(),
                shipped["_per_window_ade"].numpy(), eid),
        }
    res = {
        "experiment": ("four-family decomposition of the OBJECTIVE diagnostic "
                       "(POST-HOC)"),
        "arm": arm, "n_windows": int(d["fan"].shape[0]),
        "n_episodes": len(set(eid)),
        "status": ("POST-HOC. Decides no registered branch. Added because an ADE "
                   "delta alone is an incomplete result (binding rule, 2026-08-02)."),
        "unavailable_families": {
            "TACTICAL_manoeuvre_decision": {
                "n": 0, "reason": "a fan bank stores no decoded manoeuvre logits"},
            "STRATEGIC": {
                "n": 0,
                "reason": ("no route/goal label in a fan bank, and the decode used "
                           "nav_mode='follow_constant' so the route input was "
                           "never exercised (the C6 confound, inherited "
                           "deliberately so the contrast stays paired against the "
                           "published 0.4728 / 0.4714)")},
            "LONGITUDINAL_distance_keeping": {
                "n": 0,
                "reason": ("`taniteval/lead_source.py` (the obstacle.offline join) "
                           "is NOT present on this host and the repo mount was in "
                           "a whole-mount READ-FAILURE state for this whole run; "
                           "the val epcache carries only frames/actions/poses/"
                           "maneuvers, no obstacle tracks. This is a RUN, not a "
                           "work item, the moment either is reachable — coverage "
                           "is ~270 of these exact 881 windows.")},
        },
        "arms": out,
        "prereg_pin": C._pin(),
        "estimator": C.PREREG_THRESHOLDS["estimator"],
        "wall_s": round(time.time() - t0, 1),
    }
    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    (o / f"s1_objective_families_{arm}.json").write_text(
        json.dumps(P._clean(res), indent=2), encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = run(a.bank, a.arm, a.out)
    for name, v in r["arms"].items():
        print(f"== {name}  ade CE={v['ade_ce']:.4f} -> softADE={v['ade_softade']:.4f}"
              f"  paired vs CE {v['paired_ade_softade_minus_ce']['delta']:+.4f}"
              f" sep={v['paired_ade_softade_minus_ce']['separated']}")
        for fam in ("LONGITUDINAL", "LATERAL"):
            for k, x in v["families_softade_minus_ce"][fam].items():
                print(f"   {fam:12s} {k:24s} {x['delta']:+.4f} "
                      f"[{x['lo']:+.4f},{x['hi']:+.4f}] sep={x['separated']}")
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
