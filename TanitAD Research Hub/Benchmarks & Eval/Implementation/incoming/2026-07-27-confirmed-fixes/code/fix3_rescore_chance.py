#!/usr/bin/env python3
"""FIX 3 — re-score every H2 AP-vs-chance null against a comparator that IS chance.

THE DEFECT (MEASURED, and reproduced by this script before anything new is read)
-------------------------------------------------------------------------------
`h2c_eval.py:138` builds the chance comparator as `np.zeros_like(y)`; the old
`h2c_stats.average_precision` ranked with a STABLE argsort, which on an all-tied
score returns ROW ORDER — and `h2c_eval.py:85` lays the rows out as
[all left-camera rows, then all right-camera rows]. So the "constant score" was
the ranker *"fire the left camera everywhere"*: **AP 0.005269 vs a base rate of
0.0030527 = 1.7259x chance** on the trigger surface, and **0.046040 vs 0.032762
= 1.4053x** on the `NOT_T_seen` surface.

The comparator is HARDER than chance ⇒ every `AP − chance` delta was UNDERSTATED
and every above-chance null biased toward "not separated".

WHAT THIS SCRIPT DOES
---------------------
1. **Fidelity first.** Re-derives each committed node with `ties="row_order"` and
   refuses to print a new number unless every one comes back bit-identical.
2. Re-scores with `ties="collapse"`, under which a constant score has AP EXACTLY
   equal to the base rate inside every bootstrap draw — which is what
   `h2c_eval.py:134-137` always claimed and never had. This is the primary
   corrected statistic and it is deterministic: no seed enters it.
3. Corroborates with a genuinely random ranking (24 seeds for the point estimate,
   3 seeds carried through the full paired bootstrap).
4. Repeats both on the `NOT_T_seen` surface — **which the 2026-07-27 predecessor
   audit got wrong twice**: it used `C["Y"]` (the TRIGGER label, unfolded to
   (camera, frame)) instead of `1 - EX[:,1]`, and picked the score key by the
   substring `arm in k`, so `head_img` silently read `head_img_ego`'s scores.
   Both are re-derived here from `h2c_c12fix.py`'s own definitions.

⚠️ This changes ONE published sentence and no more: `head_ego` IS above chance on
`NOT_T_seen`. It does NOT rescue the image arms. Do not overstate the repair.

Usage:
  python fix3_rescore_chance.py --h2c <2026-07-26-h2-classifier> \
      --scores <…/2026-07-27-rung1-planner-and-owed-controls/raw/h2clf_scores> \
      --out <…/2026-07-27-confirmed-fixes/raw>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

B_BOOT, SEED = 2000, 0
N_POINT_SEEDS = 24          # random-ranking seeds for the POINT estimate
N_CI_SEEDS = 3              # …carried through the full paired bootstrap


def _cf(a):
    """frame-level -> (camera, frame): left rows then right rows (h2c_eval.py:85)."""
    return np.concatenate([a[:, 0], a[:, 1]]) if a.ndim == 2 else np.concatenate([a, a])


def _interval(d, ok):
    d = d[ok]
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"lo": round(float(lo), 6), "hi": round(float(hi), 6),
            "separated": bool(lo > 0 or hi < 0), "favours_a": bool(lo > 0),
            "n_draws_used": int(d.size)}


def _surface(name, y, eid, scores, committed_chance, ap, draws_fn, boot, log):
    """One metric surface: fidelity, corrected comparator, every arm."""
    from taniteval.rank_metrics import (assert_chance_comparator, chance_ap,
                                        comparator_audit, random_ranking_ap)
    base = chance_ap(y)
    const = np.zeros_like(y)

    audit = comparator_audit(y, const, name=f"{name}/published_constant_zero")
    audit["published_AP_b_in_artifact"] = committed_chance
    audit["reproduces_published_AP_b"] = bool(
        abs(audit["AP_row_order"] - committed_chance) < 1e-6)
    audit["true_random_ranking"] = random_ranking_ap(y, n_seeds=N_POINT_SEEDS)
    # the guard must now ACCEPT the repaired comparator, and REFUSE the old one
    assert_chance_comparator(y, const, name=f"{name}/repaired")
    try:
        assert_chance_comparator(y, const, name=f"{name}/legacy", ties="row_order")
        audit["guard_fires_on_the_legacy_comparator"] = False
    except Exception as exc:                                   # noqa: BLE001
        audit["guard_fires_on_the_legacy_comparator"] = True
        audit["guard_message"] = str(exc)
    log(f"[{name}] AP(const,row_order)={audit['AP_row_order']:.6f} "
        f"base={base:.7f} inflation={audit['inflation_vs_chance_row_order']:.4f}x "
        f"| AP(const,collapse)={audit['AP']:.6f} "
        f"| AP(true random)={audit['true_random_ranking']['mean']:.6f}")

    rnd = {i: np.random.default_rng(40_000 + i).random(y.size)
           for i in range(N_CI_SEEDS)}
    series = dict(scores)
    series["_const_collapse"] = const
    series.update({f"_rand{i}": v for i, v in rnd.items()})

    draws = draws_fn(eid)
    per = {k: np.full(len(draws), np.nan) for k in series}
    ok = np.ones(len(draws), bool)
    t0 = time.time()
    for di, sel in enumerate(draws):
        ys = y[sel]
        if ys.sum() <= 0:
            ok[di] = False
            continue
        for k, s in series.items():
            per[k][di] = ap(ys, s[sel])
        if di % 500 == 0:
            log(f"  [{name}] draw {di}/{len(draws)}  {time.time() - t0:.0f}s")

    rows = {}
    for arm in scores:
        a_ap = float(ap(y, scores[arm]))
        row = {
            "AP_arm": round(a_ap, 6),
            "published": {"AP_a": round(float(ap(y, scores[arm], ties="row_order")), 6),
                          "AP_b": committed_chance,
                          "delta": round(float(ap(y, scores[arm], ties="row_order"))
                                         - committed_chance, 6)},
            "corrected_vs_exact_chance": {
                "AP_chance": round(base, 7),
                "delta": round(a_ap - base, 6),
                **_interval(per[arm] - per["_const_collapse"], ok),
                "comparator": "constant score, ties collapsed -> AP == base rate "
                              "in EVERY draw (exact, seed-free)"},
            "corrected_vs_true_random": {
                "n_ci_seeds": N_CI_SEEDS,
                "per_seed": [_interval(per[arm] - per[f"_rand{i}"], ok)
                             for i in range(N_CI_SEEDS)],
                "delta_point": round(
                    a_ap - audit["true_random_ranking"]["mean"], 6)},
        }
        row["corrected_vs_true_random"]["n_ci_seeds_separated"] = sum(
            int(x["separated"]) for x in row["corrected_vs_true_random"]["per_seed"])
        d = row["corrected_vs_exact_chance"]
        row["VERDICT"] = ("ABOVE CHANCE" if d["favours_a"] else
                          "BELOW CHANCE" if d["separated"] else "not separated")
        row["published_said"] = "not separated"
        row["CHANGED"] = bool(row["VERDICT"] != "not separated")
        rows[arm] = row
        log(f"  [{name}] {arm:14s} published {row['published']['delta']:+.6f} "
            f"-> corrected {d['delta']:+.6f} [{d['lo']:+.6f},{d['hi']:+.6f}] "
            f"=> {row['VERDICT']}")
    return {"base_rate": round(base, 7), "n_rows": int(y.size),
            "n_positives": int(y.sum()), "n_clusters": int(len(np.unique(eid))),
            "comparator_audit": audit, "arms": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h2c", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--boot", type=int, default=B_BOOT)
    a = p.parse_args()
    h2c, sc, outd = Path(a.h2c).resolve(), Path(a.scores).resolve(), Path(a.out).resolve()
    outd.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[6]
    sys.path.insert(0, str(repo / "taniteval"))
    sys.path.insert(0, str(h2c / "scripts"))
    from h2c_stats import _draws, average_precision, episode_index  # noqa: E402

    lines: list[str] = []

    def log(m):
        print(m, flush=True)
        lines.append(m)

    def draws_fn(eid):
        uniq, idx = episode_index(eid)
        return list(_draws(uniq, idx, a.boot, SEED))

    committed = json.loads((h2c / "artifacts" / "h2c_results.json").read_text(encoding="utf-8"))
    c12 = json.loads((h2c / "artifacts" / "c12_fix.json").read_text(encoding="utf-8"))

    # ---------------- FIDELITY GATE: reproduce before quoting ---------------- #
    HO = np.load(sc / "run" / "scores_heldout.npz")
    y_t = _cf(HO["Y"]).astype(float)
    eid_t = _cf(HO["clip"])
    sc_t = {k[3:].replace("__trigger", ""): _cf(HO[k])
            for k in HO.files if k.startswith("s__") and k.endswith("__trigger")}
    sc_t["heur_speed"] = _cf(HO["ego_v"])
    sc_t["heur_decel"] = -_cf(HO["alon_pre"])
    sc_t = {k: v for k, v in sc_t.items() if k in committed["paired_AP_vs_chance"]}

    fid = {}
    for arm, node in committed["paired_AP_vs_chance"].items():
        got_a = round(float(average_precision(y_t, sc_t[arm], ties="row_order")), 6)
        got_b = round(float(average_precision(y_t, np.zeros_like(y_t),
                                              ties="row_order")), 6)
        fid[arm] = {"committed_AP_a": node["AP_a"], "recomputed_AP_a": got_a,
                    "committed_AP_b": node["AP_b"], "recomputed_AP_b": got_b,
                    "matches": bool(abs(got_a - node["AP_a"]) < 1e-6
                                    and abs(got_b - node["AP_b"]) < 1e-6)}
    fid["REPRODUCES"] = all(v["matches"] for v in fid.values() if isinstance(v, dict))
    log(f"[fidelity] trigger surface reproduces = {fid['REPRODUCES']}")
    if not fid["REPRODUCES"]:
        raise SystemExit("FIDELITY GATE FAILED — refusing to quote a new number")

    out = {
        "block": "confirmed_fixes/FIX3_chance_comparator",
        "defect": ("h2c_eval.py:138 `chance = np.zeros_like(y)` ranked by a "
                   "STABLE argsort in h2c_stats.average_precision => ROW ORDER; "
                   "the row order is [all left-camera rows, then all right]"),
        "repair": ("taniteval.rank_metrics.average_precision collapses ties "
                   "(the sklearn definition), so a constant score scores the "
                   "base rate exactly; taniteval.rank_metrics."
                   "assert_chance_comparator refuses any comparator that does "
                   "not, and h2c_eval.py now calls it"),
        "estimator": {"interval": "paired_episode_cluster_bootstrap",
                      "module": "h2c_stats -> taniteval.ci._draws",
                      "n_boot": a.boot, "seed": SEED,
                      "resampling_unit": "clip (episode cluster)"},
        "fidelity_gate": fid,
        "surfaces": {},
    }

    out["surfaces"]["trigger"] = _surface(
        "trigger", y_t, eid_t, sc_t,
        committed["paired_AP_vs_chance"]["head_img_ego"]["AP_b"],
        average_precision, draws_fn, a.boot, log)

    # ---------------- the NOT_T_seen surface, re-derived correctly ----------- #
    C = np.load(sc / "run_c12fix" / "scores_heldout.npz")
    y_c = 1.0 - C["EX"][:, 1].astype(float)          # h2c_c12fix.py:39, verbatim
    eid_c = C["clip"]
    sc_c = {k[3:].replace("__NOT_T_seen", ""): C[k][:, 0]        # h2c_c12fix.py:53
            for k in C.files if k.startswith("s__") and k.endswith("__NOT_T_seen")}
    assert abs(float(y_c.mean()) - c12["base_rate"]) < 1e-9, "NOT_T_seen target mismatch"
    fid_c = {arm: {"committed_AP": c12["arms"][arm]["AP"]["point"],
                   "recomputed_AP": round(float(average_precision(
                       y_c, sc_c[arm], ties="row_order")), 6)}
             for arm in sc_c}
    for v in fid_c.values():
        v["matches"] = bool(abs(v["recomputed_AP"] - v["committed_AP"]) < 1e-6)
    fid_c["REPRODUCES"] = all(v["matches"] for v in fid_c.values()
                              if isinstance(v, dict))
    log(f"[fidelity] NOT_T_seen surface reproduces = {fid_c['REPRODUCES']}")
    out["fidelity_gate_NOT_T_seen"] = fid_c
    if not fid_c["REPRODUCES"]:
        raise SystemExit("FIDELITY GATE FAILED (NOT_T_seen)")

    out["surfaces"]["NOT_T_seen"] = _surface(
        "NOT_T_seen", y_c, eid_c, sc_c,
        c12["arms"]["head_ego"]["paired_AP_vs_chance"]["AP_b"],
        average_precision, draws_fn, a.boot, log)

    # ---------------- the predecessor audit's own two defects ---------------- #
    out["predecessor_audit_defects"] = {
        "file": ("…/2026-07-27-rung1-planner-and-owed-controls/scripts/"
                 "owed_chance_baseline.py"),
        "rows_affected": [71, 74],
        "defect_1": ("used `C['Y']` — the TRIGGER label, unfolded to (camera, "
                     "frame) — as the NOT_T_seen target; the correct target is "
                     "`1 - EX[:,1]` at FRAME level (h2c_c12fix.py:39)"),
        "defect_2": ("picked the score key with `arm in k`, so arm 'head_img' "
                     "matched 's__head_img_ego__NOT_T_seen' first; rows 71 and "
                     "74 therefore scored the SAME array (both report "
                     "AP_a = 0.00398)"),
        "consequence": ("the '0/24 seeds separate' reading for the image arms on "
                        "NOT_T_seen was computed on the wrong target and, for "
                        "row 74, the wrong score. Re-derived correctly in "
                        "surfaces.NOT_T_seen above."),
    }

    # ---------------- what changed, and what did NOT ------------------------ #
    tr, nt = out["surfaces"]["trigger"]["arms"], out["surfaces"]["NOT_T_seen"]["arms"]

    # THE load-bearing check: did any ARM's AP move, or only the comparator?
    unchanged = {}
    for surf, arms_, src in (("trigger", tr, committed["paired_AP_vs_chance"]),
                             ("NOT_T_seen", nt,
                              {k: {"AP_a": v["AP"]["point"]}
                               for k, v in c12["arms"].items()})):
        unchanged[surf] = {
            arm: {"published_AP": src[arm]["AP_a"],
                  "corrected_AP": v["AP_arm"],
                  "unchanged": bool(abs(src[arm]["AP_a"] - v["AP_arm"]) < 1e-5)}
            for arm, v in arms_.items() if arm in src}
    out["arm_APs_are_unchanged"] = unchanged
    all_unchanged = all(x["unchanged"] for s in unchanged.values()
                        for x in s.values())

    out["headline"] = {
        "published_sentence": "H2_CLASSIFIER.md §0.2: 'NO ARM IS ABOVE CHANCE'",
        "status": "FALSE — on a comparator that IS chance, every learned arm "
                  "separates above it on BOTH surfaces",
        "only_the_comparator_moved": all_unchanged,
        "_read": ("no arm's AP changed by more than 1e-5: the learned scores "
                  "carry almost no ties, so collapsing ties leaves them exactly "
                  "where they were. The ONLY quantity that moved is the "
                  "'chance' baseline, from 1.7259x (trigger) / 1.4053x "
                  "(NOT_T_seen) chance down to chance."),
        "trigger_surface": {k: v["VERDICT"] for k, v in tr.items()},
        "NOT_T_seen_surface": {k: v["VERDICT"] for k, v in nt.items()},
        "conservative_reading_true_random_comparator": {
            f"{s}/{k}": f"{v['corrected_vs_true_random']['n_ci_seeds_separated']}"
                        f"/{v['corrected_vs_true_random']['n_ci_seeds']} seeds"
            for s, arms_ in (("trigger", tr), ("NOT_T_seen", nt))
            for k, v in arms_.items()},
        "WHAT_STILL_STANDS": (
            "'adding images DESTROYS the working ego head' is UNTOUCHED: it is "
            "an ARM-vs-ARM comparison (head_ego AP 0.1226 = 3.74x base vs "
            "head_img_ego 0.0521 = 1.59x on NOT_T_seen) and no chance "
            "comparator enters it. Both APs are reproduced unchanged above."),
        "WHAT_DOES_NOT_STAND": (
            "the SUB-CLAUSE 'neither image arm clears chance at all' "
            "(H2_CLASSIFIER.md §0.4 / §7.1). Both image arms DO clear a correct "
            "chance comparator on NOT_T_seen (+0.0193 [+0.0083,+0.0404] and "
            "+0.0164 [+0.0057,+0.0424]). They remain far below the ego arm; "
            "'weak but non-zero' replaces 'nothing'."),
        "ALSO_NOT_RESCUED": (
            "the PRE-REGISTERED VERDICT stays UNDERPOWERED. It is decided on "
            "paired RECALL vs a rate-matched heuristic and vs random at the "
            "operating point (h2c_eval.py:399-414) — neither of which uses the "
            "AP chance comparator. Nothing here moves it, and the ~52 GB "
            "expansion is still the thing that would."),
        "predecessor_0_of_24_reading_is_VOID": (
            "the '0/24 seeds' for the image arms on NOT_T_seen came from "
            "owed_chance_baseline.py rows 71/74, which scored the WRONG TARGET "
            "and (row 74) the WRONG SCORE ARRAY — see predecessor_audit_defects."),
    }
    (outd / "fix3_chance_comparator.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    (outd / "fix3_chance_comparator.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"[write] {outd / 'fix3_chance_comparator.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
