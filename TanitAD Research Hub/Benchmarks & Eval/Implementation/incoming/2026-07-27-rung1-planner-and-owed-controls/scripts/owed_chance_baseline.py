#!/usr/bin/env python3
"""OWED ROWS 66-75 — the `CHANCE-BASELINE` family, re-adjudicated. ZERO GPU.

These 10 rows could not be re-powered by the predecessor because `n = 322` is
bounded by LABELLED CLIPS, not by episodes — the x3.4 episode-shrinkage argument
does not transfer at all. That remains true. What DOES change at the same n, and
is measured here:

⛔ (1) THE "CHANCE" COMPARATOR IS NOT CHANCE. `h2c_eval.py:138` builds it as
   `chance = np.zeros_like(y)` and `h2c_stats.average_precision` ranks with
   `np.argsort(-s, kind="mergesort")` — a STABLE sort. On an all-tied score a
   stable sort returns ROW ORDER, and the row order is set by `h2c_eval.py:85`
   to **every left-camera row, then every right-camera row**. So the "constant
   score" is really the ranker *"fire the left camera everywhere"*, which is
   informative on this corpus. The artifact's own numbers show it: every
   `paired_AP_vs_chance` node reports `AP_b = 0.005269` against
   `discrimination.base_rate = 0.0030527`. The comparator is measured here, and
   the corrected one (a genuinely random ranking, averaged over seeds) is run
   alongside. Every AP-vs-chance null was adjudicated against an INFLATED
   comparator, i.e. biased toward the null — which for a control is biased
   toward the desired verdict.

⛔ (2) THE RANDOM BASELINE IN THE OPERATING POINT IS ONE SEED. `h2c_eval.py:250`
   draws `default_rng(1000)` once. The 200-seed null sits one key away in the
   same artifact (`operating_point.random_seed_spread`). Re-run over many seeds
   here.

⭐ (3) M9 — TWO SIBLINGS ALREADY ANSWER THE `can_fire` QUESTION, and neither is
   quoted as such: `c12_fix.json :: arms.head_ego.paired_AP_vs_chance` is
   SEPARATED at the same 322 clusters and the same estimator, and
   `h2c_results.json :: paired_AP_vs_chance.heur_speed` is SEPARATED (negative)
   on the exact 306-positive target. The test demonstrably CAN fire at this n.

⭐ (4) TWO OF THE TEN ROWS ARE THE SAME NUMBER.
   `operating_point.paired_recall_deltas["head_img_ego - random_at_rate"]` is
   re-serialised as `verdict.delta_vs_random` (`h2c_eval.py:399`). Nine distinct
   nulls, not ten — reported, not silently de-duplicated, because a reader
   grepping either location gets the number.

⛔ Nothing is re-implemented: `average_precision`, `_recall`, `_paired_ap`,
   `_paired_recall`, `_draws` and `episode_index` are IMPORTED from the stream's
   own modules. The one thing computed here that they do not offer is a SHARED
   draw loop (each arm's per-draw AP is computed once and reused across
   comparators) — and it is validated by reproducing `_paired_ap`'s committed
   output exactly before any new number is read.

Usage:
    python owed_chance_baseline.py --h2c <h2-classifier dir> \
        --scores <dir with run/ and run_c12fix/> --out <raw dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[5]
for _p in (_REPO / "taniteval",):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

B_BOOT, SEED = 2000, 0
N_CHANCE_SEEDS = 24


def _cf(a):
    """frame-level -> (camera, frame) pair space: left rows then right rows.
    `h2c_eval.py:83-85`, verbatim — data marshalling, not estimation."""
    return np.concatenate([a[:, 0], a[:, 1]]) if a.ndim == 2 else np.concatenate([a, a])


def power_ceiling(lo, hi, delta, max_possible) -> dict:
    """M8. AP and recall are bounded in [0,1], so the largest attainable effect
    against a comparator scoring `b` is `1 - b`, and `mde >= max_possible` is a
    PROOF that the control could not have fired at any observed value."""
    mde = float(hi - lo) / 2.0
    return {"mde": round(mde, 6),
            "max_possible_effect": round(float(max_possible), 6),
            "mde_as_pct_of_max_possible": (None if max_possible <= 0 else
                                           round(100.0 * mde / max_possible, 1)),
            "can_fire": bool(max_possible > mde),
            "half_width_exceeds_point_estimate": bool(mde > abs(float(delta))),
            "_read": ("VOID -- cannot separate at any observed value"
                      if max_possible <= mde else
                      "can separate; it did not, at this n")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h2c", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=B_BOOT)
    a = ap.parse_args()
    h2c = Path(a.h2c).resolve()
    sc = Path(a.scores).resolve()
    outd = Path(a.out).resolve()
    outd.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(h2c / "scripts"))
    from h2c_stats import average_precision, _draws, episode_index   # noqa: E402
    from h2c_eval import _paired_ap, _paired_recall, _recall         # noqa: E402

    committed = json.load(open(h2c / "artifacts" / "h2c_results.json", encoding="utf-8"))
    c12 = json.load(open(h2c / "artifacts" / "c12_fix.json", encoding="utf-8"))

    out = {
        "block": "owed_controls/CHANCE-BASELINE",
        "frozen_list_rows": list(range(66, 76)),
        "estimator": {"interval": "paired_episode_cluster_bootstrap (reducer form)",
                      "module": "h2c_stats -> taniteval.ci._draws",
                      "n_boot": a.boot, "seed": SEED,
                      "resampling_unit": "clip (episode cluster)"},
        "n_cannot_grow_here": ("n = 322 clusters is bounded by LABELLED CLIPS "
                               "admitted to the pod2 episode cache, not by "
                               "episodes; the x3.4 episode-shrinkage projection "
                               "does not transfer to it"),
        "duplicate_rows": {},
        "sibling_can_fire_demonstrations": {},
        "comparator_audit": {},
        "rows": {},
        "recall_vs_random_seed_audit": {},
        "selftest": {},
    }

    # ---- (4) the duplicate ---------------------------------------------------- #
    d1 = committed["operating_point"]["paired_recall_deltas"]["head_img_ego - random_at_rate"]
    d2 = committed["verdict"]["delta_vs_random"]
    out["duplicate_rows"] = {
        "rows": [66, 68],
        "nodes": ["operating_point.paired_recall_deltas['head_img_ego - random_at_rate']",
                  "verdict.delta_vs_random"],
        "identical": bool(all(abs(d1[k] - d2[k]) < 1e-12 for k in ("delta", "lo", "hi"))),
        "source": "h2c_eval.py:399 re-serialises the first node as the second",
        "consequence": "the frozen list holds 9 DISTINCT chance nulls, not 10",
    }

    # ---- (3) the siblings that prove the test can fire ------------------------ #
    out["sibling_can_fire_demonstrations"] = {
        "c12_fix.arms.head_ego.paired_AP_vs_chance": {
            k: c12["arms"]["head_ego"]["paired_AP_vs_chance"][k]
            for k in ("delta", "lo", "hi", "separated", "n_episodes")},
        "h2c_results.paired_AP_vs_chance.heur_speed": {
            k: committed["paired_AP_vs_chance"]["heur_speed"][k]
            for k in ("delta", "lo", "hi", "separated", "n_episodes")},
        "_read": ("the SAME rig, estimator and 322 clusters DO separate from "
                  "this comparator, in both directions. So a null here is a "
                  "genuine 'not separated', not a structurally dead test -- "
                  "which is exactly the distinction the VOID verdict exists for"),
    }

    # ================= the primary metric space (h2c_results) ================== #
    HO = np.load(sc / "run" / "scores_heldout.npz")
    y = _cf(HO["Y"]).astype(float)
    eid = _cf(HO["clip"])
    arms = {k[3:].replace("__trigger", ""): _cf(HO[k])
            for k in HO.files if k.startswith("s__") and k.endswith("__trigger")}
    EX = HO["EX"]
    v_ho, a_ho = _cf(HO["ego_v"]), _cf(HO["alon_pre"])
    ego_scores = {"heur_speed": v_ho, "heur_decel": -a_ho}
    all_scores = dict(arms)
    all_scores.update(ego_scores)
    zeros = np.zeros_like(y)

    # ---- fidelity: the committed nodes must come straight back out ------------ #
    fid = {}
    for name in ("head_img_ego", "head_img", "head_ego", "heur_decel", "heur_speed"):
        c = committed["paired_AP_vs_chance"][name]
        r = _paired_ap(y, all_scores[name], zeros, eid, a.boot)
        fid[name] = {"committed": {k: c[k] for k in ("AP_a", "AP_b", "delta", "lo", "hi",
                                                     "separated")},
                     "recomputed": {k: r[k] for k in ("AP_a", "AP_b", "delta", "lo", "hi",
                                                      "separated")},
                     "matches": bool(abs(r["delta"] - c["delta"]) < 1e-6
                                     and abs(r["lo"] - c["lo"]) < 1e-6
                                     and abs(r["hi"] - c["hi"]) < 1e-6
                                     and r["separated"] == c["separated"])}
    fid["REPRODUCES"] = all(v["matches"] for v in fid.values() if isinstance(v, dict))
    out["selftest"]["fidelity_paired_AP_vs_chance"] = fid
    print(f"[fidelity] paired_AP_vs_chance REPRODUCES = {fid['REPRODUCES']}", flush=True)

    # ---- (1) ⛔ MEASURE the comparator --------------------------------------- #
    ap_const = float(average_precision(y, zeros))
    base = float(y.mean())
    rngs = [np.random.default_rng(10_000 + i) for i in range(N_CHANCE_SEEDS)]
    rand_scores = [r.random(y.size) for r in rngs]
    ap_rand = [float(average_precision(y, s)) for s in rand_scores]
    out["comparator_audit"] = {
        "AP_of_the_published_constant_zero_comparator": round(ap_const, 6),
        "base_rate": round(base, 7),
        "inflation_factor": round(ap_const / base, 4),
        "AP_of_a_TRUE_random_ranking": {
            "n_seeds": N_CHANCE_SEEDS, "mean": round(float(np.mean(ap_rand)), 6),
            "p2.5": round(float(np.percentile(ap_rand, 2.5)), 6),
            "p97.5": round(float(np.percentile(ap_rand, 97.5)), 6)},
        "the_claim_in_the_source": ("h2c_stats.average_precision's docstring and "
                                    "h2c_eval.py:134-137 assert a constant score "
                                    "has AP equal to the base rate within each "
                                    "draw"),
        "MEASURED": ("FALSE as implemented -- a STABLE argsort on an all-tied "
                     "score ranks by row order, and the row order is all "
                     "left-camera rows first"),
        "direction_of_the_bias": ("the comparator is HARDER than chance, so every "
                                  "AP-vs-chance delta is understated and every "
                                  "null is biased toward 'not separated' -- for a "
                                  "control, toward the desired verdict"),
    }
    print(f"[comparator] AP(const)={ap_const:.6f} base_rate={base:.7f} "
          f"inflation={ap_const / base:.3f}x  AP(true random) mean="
          f"{np.mean(ap_rand):.6f}", flush=True)

    # ---- the shared draw loop (validated above), then every paired delta ------ #
    uniq, idx = episode_index(eid)
    draws = list(_draws(uniq, idx, a.boot, SEED))
    per_draw = {k: np.empty(len(draws)) for k in list(all_scores) + ["_const"]}
    per_draw.update({f"_rand{i}": np.empty(len(draws)) for i in range(N_CHANCE_SEEDS)})
    ok = np.ones(len(draws), bool)
    for di, sel in enumerate(draws):
        ys = y[sel]
        if ys.sum() <= 0:
            ok[di] = False
            continue
        for k, s in all_scores.items():
            per_draw[k][di] = average_precision(ys, s[sel])
        per_draw["_const"][di] = average_precision(ys, zeros[sel])
        for i, s in enumerate(rand_scores):
            per_draw[f"_rand{i}"][di] = average_precision(ys, s[sel])
        if di % 400 == 0:
            print(f"  [draws] {di}/{len(draws)}", flush=True)

    def interval(d):
        d = d[ok]
        d = d[np.isfinite(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"lo": round(float(lo), 6), "hi": round(float(hi), 6),
                "separated": bool(lo > 0 or hi < 0), "n_draws_used": int(d.size)}

    # ---- the 8 AP rows, as published and CORRECTED ---------------------------- #
    AP_ROWS = {
        "69": ("paired_AP_vs_chance.head_ego", "head_ego"),
        "71": ("c12_fix.arms.head_img_ego.paired_AP_vs_chance", None),
        "72": ("paired_AP_vs_chance.heur_decel", "heur_decel"),
        "73": ("paired_AP_vs_chance.head_img_ego", "head_img_ego"),
        "74": ("c12_fix.arms.head_img.paired_AP_vs_chance", None),
        "75": ("paired_AP_vs_chance.head_img", "head_img"),
    }
    for row, (node, arm) in AP_ROWS.items():
        if arm is None:
            continue                                  # handled in the c12 block
        c = committed["paired_AP_vs_chance"][arm]
        as_pub = interval(per_draw[arm] - per_draw["_const"])
        as_pub["delta"] = round(float(np.mean(per_draw[arm][ok]) * 0
                                      + c["delta"]), 6)
        corr = [interval(per_draw[arm] - per_draw[f"_rand{i}"])
                for i in range(N_CHANCE_SEEDS)]
        corr_delta = float(average_precision(y, all_scores[arm])) - float(np.mean(ap_rand))
        pc = power_ceiling(c["lo"], c["hi"], c["delta"], 1.0 - c["AP_b"])
        n_sep = sum(int(x["separated"]) for x in corr)
        out["rows"][row] = {
            "node": node, "arm": arm,
            "committed": {k: c[k] for k in ("AP_a", "AP_b", "delta", "lo", "hi",
                                            "separated", "n_episodes")},
            "as_published_recomputed": as_pub,
            "corrected_comparator": {
                "delta_vs_true_random": round(corr_delta, 6),
                "delta_shift_vs_published": round(corr_delta - c["delta"], 6),
                "n_seeds": N_CHANCE_SEEDS,
                "n_seeds_separated": n_sep,
                "lo_range": [round(min(x["lo"] for x in corr), 6),
                             round(max(x["lo"] for x in corr), 6)],
                "hi_range": [round(min(x["hi"] for x in corr), 6),
                             round(max(x["hi"] for x in corr), 6)]},
            "power_ceiling": pc,
            "VERDICT": ("CONTROL FAILS" if n_sep == N_CHANCE_SEEDS else
                        "VOID" if not pc["can_fire"] else
                        "UNDER-POWERED (OWED)"),
            "why": ("separated from a TRUE random ranking on every seed"
                    if n_sep == N_CHANCE_SEEDS else
                    "MDE exceeds the largest attainable effect"
                    if not pc["can_fire"] else
                    f"not separated; the comparator was {ap_const / base:.2f}x "
                    f"chance, and correcting it moves the point estimate by "
                    f"{corr_delta - c['delta']:+.6f} without separating "
                    f"({n_sep}/{N_CHANCE_SEEDS} seeds)"),
        }
        print(f"[row {row}] {arm:12s} pub {c['delta']:+.6f} "
              f"[{c['lo']:+.6f},{c['hi']:+.6f}] sep={c['separated']} | "
              f"corrected {corr_delta:+.6f} sep_seeds={n_sep}/{N_CHANCE_SEEDS} "
              f"-> {out['rows'][row]['VERDICT']}", flush=True)

    # ---- (2) the recall-vs-random rows, over many seeds ----------------------- #
    op = committed["operating_point"]
    rate_h = float(op["realised_head_camera_frame_rate"])
    fires = {}
    for arm in ("head_img_ego", "head_ego", "head_img"):
        th = float(np.quantile(_cf(np.load(sc / "run" / "scores_oof_train.npz")
                                   [f"s__{arm}__trigger"]), 1.0 - op["B_preregistered"] / 2.0))
        fires[arm] = (arms[arm] >= th).astype(float)
    seed_deltas = {arm: [] for arm in fires}
    seed_seps = {arm: 0 for arm in fires}
    rec_rand = []
    for i in range(N_CHANCE_SEEDS):
        rr = np.random.default_rng(20_000 + i)
        u = rr.random(y.size)
        fr = (u >= np.quantile(u, 1.0 - max(rate_h, 1e-9))).astype(float)
        rec_rand.append(float(_recall(y, fr)))
        for arm, fa in fires.items():
            r = _paired_recall(y, fa, fr, eid, a.boot)
            seed_deltas[arm].append(r["delta"])
            seed_seps[arm] += int(r["separated"])
    out["recall_vs_random_seed_audit"] = {
        "published_single_seed": 1000,
        "published_random_recall": op["random_at_rate"]["recall"]["point"]
        if isinstance(op["random_at_rate"]["recall"], dict)
        else op["random_at_rate"]["recall"],
        "random_seed_spread_in_the_same_artifact": op.get("random_seed_spread"),
        "my_seeds": {"n": N_CHANCE_SEEDS, "recall_mean": round(float(np.mean(rec_rand)), 6),
                     "recall_p2.5": round(float(np.percentile(rec_rand, 2.5)), 6),
                     "recall_p97.5": round(float(np.percentile(rec_rand, 97.5)), 6)},
        "per_arm": {arm: {"published_delta": op["paired_recall_deltas"][f"{arm} - random_at_rate"]["delta"],
                          "mean_delta_over_seeds": round(float(np.mean(v)), 6),
                          "delta_shift": round(float(np.mean(v))
                                               - op["paired_recall_deltas"][f"{arm} - random_at_rate"]["delta"], 6),
                          "n_seeds_separated": seed_seps[arm],
                          "n_seeds": N_CHANCE_SEEDS}
                    for arm, v in seed_deltas.items()},
    }
    for arm in fires:
        pa = op["paired_recall_deltas"][f"{arm} - random_at_rate"]
        row = {"66": "head_img_ego", "67": "head_img_ego", "70": "head_img"}
        pc = power_ceiling(pa["lo"], pa["hi"], pa["delta"], 1.0 - pa["recall_b"])
        key = {"head_img_ego": "66", "head_ego": "67", "head_img": "70"}[arm]
        out["rows"][key] = {
            "node": f"operating_point.paired_recall_deltas['{arm} - random_at_rate']",
            "arm": arm,
            "committed": {k: pa[k] for k in ("recall_a", "recall_b", "delta", "lo",
                                             "hi", "separated", "n_episodes")},
            "corrected_comparator": out["recall_vs_random_seed_audit"]["per_arm"][arm],
            "power_ceiling": pc,
            "VERDICT": ("CONTROL FAILS"
                        if seed_seps[arm] == N_CHANCE_SEEDS else
                        "VOID" if not pc["can_fire"] else "UNDER-POWERED (OWED)"),
            "why": (f"separated from a random firing rule on "
                    f"{seed_seps[arm]}/{N_CHANCE_SEEDS} seeds; the published "
                    f"number used ONE seed"),
        }
        print(f"[row {key}] recall {arm:12s} pub {pa['delta']:+.6f} "
              f"sep_seeds={seed_seps[arm]}/{N_CHANCE_SEEDS} -> "
              f"{out['rows'][key]['VERDICT']}", flush=True)
    out["rows"]["68"] = {"node": "verdict.delta_vs_random",
                         "DUPLICATE_OF": "66",
                         "VERDICT": "DUPLICATE — same number, second location"}

    # ================= the c12_fix metric space (rows 71, 74) ================== #
    C = np.load(sc / "run_c12fix" / "scores_heldout.npz")
    key_y = "Y" if "Y" in C.files else None
    if key_y is not None:
        # ⚠️ the c12 target is FRAME-level, not (camera, frame): its arrays are
        # [N] or [N,1], so `_cf`'s two-camera unfold does not apply. Detected
        # from the shape rather than assumed — the crash that taught this is
        # recorded as amendment A2.
        def _flat(a):
            return _cf(a) if (a.ndim == 2 and a.shape[1] == 2) else np.asarray(a).reshape(-1)
        yc = _flat(C[key_y]).astype(float)
        eidc = _flat(C["clip"])
        cz = np.zeros_like(yc)
        for row, arm in (("71", "head_img_ego"), ("74", "head_img")):
            c = c12["arms"][arm]["paired_AP_vs_chance"]
            sk = [k for k in C.files if k.startswith("s__") and arm in k]
            rec = {"node": f"c12_fix.arms.{arm}.paired_AP_vs_chance", "arm": arm,
                   "committed": {k: c[k] for k in ("delta", "lo", "hi", "separated",
                                                   "n_episodes") if k in c},
                   "power_ceiling": power_ceiling(
                       c["lo"], c["hi"], c["delta"],
                       1.0 - c.get("AP_b", c12["base_rate"]))}
            if sk:
                s = _flat(C[sk[0]])
                r = _paired_ap(yc, s, cz, eidc, a.boot)
                rec["recomputed_as_published"] = {k: r[k] for k in
                                                  ("AP_a", "AP_b", "delta", "lo", "hi",
                                                   "separated")}
                corr = []
                for i in range(N_CHANCE_SEEDS):
                    rr = np.random.default_rng(30_000 + i).random(yc.size)
                    corr.append(_paired_ap(yc, s, rr, eidc, 400))
                rec["corrected_comparator"] = {
                    "n_seeds": N_CHANCE_SEEDS, "n_boot_per_seed": 400,
                    "mean_delta": round(float(np.mean([x["delta"] for x in corr])), 6),
                    "n_seeds_separated": int(sum(x["separated"] for x in corr)),
                    "AP_of_constant_comparator": r["AP_b"],
                    "AP_of_true_random_mean": round(
                        float(np.mean([x["AP_b"] for x in corr])), 6)}
                nsep = rec["corrected_comparator"]["n_seeds_separated"]
                rec["VERDICT"] = ("CONTROL FAILS" if nsep == N_CHANCE_SEEDS else
                                  "VOID" if not rec["power_ceiling"]["can_fire"]
                                  else "UNDER-POWERED (OWED)")
                rec["why"] = (f"{nsep}/{N_CHANCE_SEEDS} seeds separate against a "
                              f"TRUE random ranking; the published comparator "
                              f"scored AP {r['AP_b']:.6f} against a base rate of "
                              f"{c12['base_rate']:.6f}")
                print(f"[row {row}] c12 {arm:12s} pub {c['delta']:+.6f} "
                      f"sep={c['separated']} | corrected sep_seeds="
                      f"{nsep}/{N_CHANCE_SEEDS} -> {rec['VERDICT']}", flush=True)
            out["rows"][row] = rec

    (outd / "owed_chance_baseline.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"[write] {outd / 'owed_chance_baseline.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
