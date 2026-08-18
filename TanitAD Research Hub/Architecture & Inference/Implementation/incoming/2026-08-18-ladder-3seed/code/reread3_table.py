#!/usr/bin/env python
"""THE 165-ROW RE-READ, RE-DERIVED AT THREE SEEDS (C103).

⛔ WHY. C100's inventory of the 87 banked separated-FAILs was computed at
**seed 0**, licensed by the ladder's own claim that *"seed spread is exactly zero
on 8 of 11 rungs"*. C103 falsified that claim — it was measured **under** the C92
defect, which had FROZEN the alpha sweep. Repaired, the sweep is un-truncated,
the inner-split seed moves the alpha choice, and the same arm's K1B moves 2.516
across seeds. ⇒ **A stability claim measured under a defect is not inherited by
the repaired instrument.**

⭐ WHAT THIS SCRIPT ADDS OVER `reread_table.py` (which reads ``per_seed["0"]``):

1. **Every row is reported at 3 seeds** — per-seed K1/K1B/guard/verdict, the
   3-seed MEAN, and the seed SPREAD (max − min).
2. ⛔ **A verdict that flips across seeds is reported as `SEED-UNSTABLE`, not as
   its mean.** The C100 buckets are re-derived with `SEED-UNSTABLE` as an
   explicit fifth bucket; per-seed bucket counts are printed beside it so the
   reader can see *how* unstable.
3. ⛔ **THE TRIVIAL-PROXY CONTROL ON EVERY ROW (C92).** Each arm's K1B is
   reported against the **single ego-speed scalar** fitted on the **same window
   set** — `proxyv0` for the 12 cells-window arms, `proxytok` for the 3
   tokens-window arms. K1B is negative-is-better, so ``margin = arm − cv0`` and
   **positive means the SCALAR WINS**. Rows whose window family has no C-V0 arm
   are stamped `paired_window_set: false` rather than compared anyway.
4. ⭐ **A REPRODUCTION GATE.** My route-A **seed 0** rows must reproduce the
   BANKED route-A seed-0 rows field for field: `fit_one` draws its own
   ``default_rng(seed)`` per call and `taniteval.ci` bootstraps at a fixed
   ``seed=0``, so adding seeds 1 and 2 must not perturb seed 0. If it did, the
   3-seed run would not be an extension of C100's run but a different one.
   Exit codes are not evidence — the gate counts compared fields and prints them.

⛔ THE TWO REPAIR ROUTES ARE NEVER POOLED (C100/C103). Route A = ``unpen``
(the module's repair; C100's route). Route B = ``centred`` (the locally
re-derived repair; the route LATENT_LINEAR_LADDER §5.3/§7/§8 was rendered from).
They are placed side by side in exactly one table whose purpose is to show that
they differ, and every other table names the one route it came from.

⛔ T0-DIAGNOSTIC. A frozen-latent linear readout is a world-model diagnostic and
is never driving performance.
⛔ Estimator: ``taniteval.ci.paired_episode_cluster_bootstrap``, n_boot 2000,
clustered on the 70 eval episodes. ``overlapping_holdout_se`` is never imported.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

INC = pathlib.Path(
    "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/TanitAD Research Hub/"
    "Architecture & Inference/Implementation/incoming")
LL = INC / "2026-08-17-latent-linear-ladder" / "raw"            # incumbent pc6
RRA = INC / "2026-08-18-k1-degeneracy-guard" / "raw" / "reread"  # banked route A, seed 0
OUTD = INC / "2026-08-18-ladder-3seed" / "raw"

LADDER = ["ego_v0", "ego_accel", "ego_yawrate", "ego_curv", "n_agents_grid",
          "n_agents_all", "lead_present", "nearest_any", "lead_gap",
          "lead_closing", "lead_inv_ttc"]
# the 15 arms of the C100 re-read, in its order
ARMS = ["s11250", "nullmatched", "orcdir", "proxyv0", "s09000", "s09250",
        "s10000", "s02000", "egoorc_n0.1", "egoorc_n1", "egoorc_n3",
        "egoorc_n10", "tok11250", "tok11250null", "cells_tokwin"]
# ⚠️ TWO WINDOW FAMILIES. `tok11250`/`tok11250null`/`cells_tokwin` are fitted on
# the TOKENS cache (n_eval 1103–1507); the other 12 on the cells cache
# (2221–3023). A trivial-proxy comparison across families would be UNPAIRED, so
# each family gets its own C-V0 arm.
TOK_ARMS = {"tok11250", "tok11250null", "cells_tokwin", "proxytok"}
EXTRA_ARMS = ["proxytok"]          # the tokens-window C-V0 control, new here
CV0_OF = {a: ("proxytok" if a in TOK_ARMS else "proxyv0")
          for a in ARMS + EXTRA_ARMS}
SEEDS = ["0", "1", "2"]
SUBSTANTIVE_REL = 0.02             # C100's reporting threshold, reported not gated


def load(p):
    return json.loads(pathlib.Path(p).read_text("utf-8"))


def verdict(s) -> str:
    """The audit's own predicate, verbatim, so old and new are commensurable."""
    if not s["K1_separated"]:
        return "not-separated"
    return "PASS" if s["K1_delta"] < 0 else "FAIL-separated"


def bucket(s) -> str:
    """C100's four buckets, for a row that was a banked separated-FAIL."""
    v = verdict(s)
    if v == "not-separated":
        return "die_at_repair"
    if v == "PASS":
        return "flip_to_PASS"
    if s["k1_guard"]["guard_verdict"] in ("CONSTANT-OFFSET-ONLY",
                                          "DEGENERATE-CONSTANT"):
        return "killed_by_guard"
    return "survive_both"


def unanimous(vals, unstable="SEED-UNSTABLE"):
    """⛔ A verdict that flips across seeds is NOT a verdict."""
    return vals[0] if len(set(vals)) == 1 else unstable


def spread(xs):
    return round(float(max(xs) - min(xs)), 6)


def mean(xs):
    return round(float(sum(xs) / len(xs)), 6)


# --------------------------------------------------------------------------
def reproduction_gate(new_dir: pathlib.Path) -> dict:
    """⭐ My route-A seed-0 rows vs the BANKED route-A seed-0 rows.

    `fit_one` draws ``default_rng(seed)`` per call and ``taniteval.ci``
    bootstraps at a fixed ``seed=0``, so seeds 1 and 2 must not perturb seed 0.
    Anything else means the 3-seed run is a different instrument, not an
    extension of C100's.
    """
    FIELDS = ("alpha_chosen", "err", "c_const_err", "K1_delta", "K1_lo",
              "K1_hi", "K1_separated", "K1_PASSES", "corr", "corr_within_ep",
              "pred_sd", "gt_sd", "R2", "r2_ceiling", "K5_delta")
    n_cmp = n_bad = 0
    bad = []
    for a in ARMS:
        b = load(RRA / f"llR_{a}.json")
        n = load(new_dir / f"ll3_{a}.json")
        for t in LADDER:
            tb, tn = b["targets"][t], n["targets"][t]
            assert tb["n_eval"] == tn["n_eval"], (
                f"⛔ window set moved for {a}:{t} — {tb['n_eval']} vs "
                f"{tn['n_eval']}; seeds change the SOLVE, never the windows")
            sb, sn = tb["per_seed"]["0"], tn["per_seed"]["0"]
            for f in FIELDS:
                n_cmp += 1
                if sb[f] != sn[f]:
                    n_bad += 1
                    bad.append({"arm": a, "target": t, "field": f,
                                "banked": sb[f], "mine": sn[f]})
            for f in ("K1B_delta", "K1B_lo", "K1B_hi", "K1B_separated",
                      "guard_verdict", "sd_ratio"):
                n_cmp += 1
                if sb["k1_guard"][f] != sn["k1_guard"][f]:
                    n_bad += 1
                    bad.append({"arm": a, "target": t, "field": "guard." + f,
                                "banked": sb["k1_guard"][f],
                                "mine": sn["k1_guard"][f]})
    return {"fields_compared": n_cmp, "fields_differing": n_bad,
            "GATE": "PASS" if n_bad == 0 else "FAIL",
            "differences": bad[:40]}


# --------------------------------------------------------------------------
def build_rows(route_dir: pathlib.Path, route_name: str,
               inc: dict, want: list[str]) -> list[dict]:
    new = {a: load(route_dir / f"ll3_{a}.json") for a in want}
    rows = []
    for a in want:
        for t in LADDER:
            tn = new[a]["targets"][t]
            ps = [tn["per_seed"][s] for s in SEEDS]
            gs = [p["k1_guard"] for p in ps]
            gt_sd = gs[0]["gt_sd"]
            r = {
                "arm": a, "label": new[a]["arm"], "target": t,
                "rung": tn.get("rung"), "unit": tn.get("unit"),
                "n_eval": tn["n_eval"],
                "n_eval_clusters": tn["n_eval_clusters"],
                "gt_sd": gt_sd, "route": route_name,
                "window_family": "tokens" if a in TOK_ARMS else "cells",
                # --- repaired, per seed -------------------------------------
                "new_K1_per_seed": [p["K1_delta"] for p in ps],
                "new_K1_ci_per_seed": [[p["K1_lo"], p["K1_hi"]] for p in ps],
                "new_alpha_per_seed": [p["alpha_chosen"] for p in ps],
                "new_alpha_at_grid_edge":
                    [p.get("alpha_at_grid_edge") for p in ps],
                "new_pred_sd_per_seed": [p["pred_sd"] for p in ps],
                "new_verdict_per_seed": [verdict(p) for p in ps],
                "K1B_per_seed": [g["K1B_delta"] for g in gs],
                "K1B_ci_per_seed": [[g["K1B_lo"], g["K1B_hi"]] for g in gs],
                "K1B_separated_per_seed": [g["K1B_separated"] for g in gs],
                "guard_per_seed": [g["guard_verdict"] for g in gs],
                "corr_per_seed": [p["corr"] for p in ps],
                "r_pv0_per_seed": [p.get("corr_partial_v0") for p in ps],
                # --- the 3-seed summaries -----------------------------------
                "new_K1_mean": mean([p["K1_delta"] for p in ps]),
                "new_K1_seed_spread": spread([p["K1_delta"] for p in ps]),
                "K1B_mean": mean([g["K1B_delta"] for g in gs]),
                "K1B_seed_spread": spread([g["K1B_delta"] for g in gs]),
                "corr_mean": mean([p["corr"] for p in ps]),
                # ⛔ THE VERDICT, and it is only a verdict if it holds on all 3
                "verdict_3seed": unanimous([verdict(p) for p in ps]),
                "guard_3seed": unanimous([g["guard_verdict"] for g in gs]),
                "verdict_seed_stable":
                    len({verdict(p) for p in ps}) == 1,
                "guard_seed_stable":
                    len({g["guard_verdict"] for g in gs}) == 1,
                "alpha_seed_stable":
                    len({p["alpha_chosen"] for p in ps}) == 1,
                # ⚠️ SCALE. The guard answers ATTRIBUTION, not MAGNITUDE.
                "K1B_mean_rel_gt_sd":
                    round(mean([g["K1B_delta"] for g in gs]) / gt_sd, 6)
                    if gt_sd else None,
            }
            # --- the incumbent side, for the arms C100 paired ---------------
            if a in inc:
                ti = inc[a]["targets"][t]
                assert ti["n_eval"] == tn["n_eval"], (
                    f"⛔ window set moved for {a}:{t}")
                pi = [ti["per_seed"][s] for s in SEEDS]
                r.update({
                    "old_K1_per_seed": [p["K1_delta"] for p in pi],
                    "old_verdict_per_seed": [verdict(p) for p in pi],
                    "old_verdict_seed0": verdict(pi[0]),
                    "old_verdict_3seed": unanimous([verdict(p) for p in pi]),
                    "old_verdict_seed_stable":
                        len({verdict(p) for p in pi}) == 1,
                    "old_alpha_per_seed": [p["alpha_chosen"] for p in pi],
                    "old_alpha_seed_stable":
                        len({p["alpha_chosen"] for p in pi}) == 1,
                    "old_K1_seed_spread": spread([p["K1_delta"] for p in pi]),
                })
                # C100's population is the SEED-0 incumbent separated-FAILs
                if r["old_verdict_seed0"] == "FAIL-separated":
                    bs = [bucket(p) for p in ps]
                    r["bucket_per_seed"] = bs
                    r["bucket_3seed"] = unanimous(bs)
                    r["bucket_seed_stable"] = len(set(bs)) == 1
                    r["substantive_3seed"] = bool(
                        r["bucket_3seed"] == "survive_both"
                        and r["K1B_mean_rel_gt_sd"] is not None
                        and abs(r["K1B_mean_rel_gt_sd"]) >= SUBSTANTIVE_REL)
            rows.append(r)
    return rows


def add_trivial_proxy(rows: list[dict]) -> None:
    """⛔ C92's mandatory control, on EVERY row: what does the arm beat once the
    single ego-speed scalar is accounted for? K1B is negative-is-better, so
    ``margin = arm − cv0`` and POSITIVE MEANS THE SCALAR WINS."""
    by = {(r["arm"], r["target"]): r for r in rows}
    for r in rows:
        cv0_arm = CV0_OF.get(r["arm"], "proxyv0")
        c = by.get((cv0_arm, r["target"]))
        r["cv0_arm"] = cv0_arm
        if c is None or c["n_eval"] != r["n_eval"]:
            r["paired_window_set"] = False
            r["cv0_K1B_mean"] = None
            r["margin_vs_cv0_K1B_mean"] = None
            r["scalar_wins_3seed_mean"] = None
            continue
        r["paired_window_set"] = True
        r["cv0_K1B_mean"] = c["K1B_mean"]
        r["cv0_K1B_per_seed"] = c["K1B_per_seed"]
        m = [a - b for a, b in zip(r["K1B_per_seed"], c["K1B_per_seed"])]
        r["margin_vs_cv0_K1B_per_seed"] = [round(x, 6) for x in m]
        r["margin_vs_cv0_K1B_mean"] = round(r["K1B_mean"] - c["K1B_mean"], 6)
        r["margin_vs_cv0_rel_gt_sd"] = (
            round((r["K1B_mean"] - c["K1B_mean"]) / r["gt_sd"], 6)
            if r["gt_sd"] else None)
        r["margin_sign_seed_stable"] = len({x > 0 for x in m}) == 1
        # POSITIVE margin = the scalar wins (K1B is negative-is-better)
        r["scalar_wins_3seed_mean"] = bool(r["margin_vs_cv0_K1B_mean"] >= 0)
        r["scalar_wins_per_seed"] = [bool(x >= 0) for x in m]


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-a", default=str(OUTD / "reread_unpen"))
    ap.add_argument("--route-b", default=str(OUTD / "reread_centred"))
    ap.add_argument("--out", default=str(OUTD / "reread3_table.json"))
    a = ap.parse_args(argv)
    ra, rb = pathlib.Path(a.route_a), pathlib.Path(a.route_b)

    inc = {x: load(LL / f"ll_{x}.json") for x in ARMS}
    have_extra = [x for x in EXTRA_ARMS if (ra / f"ll3_{x}.json").exists()]
    want = ARMS + have_extra

    out = {
        "_evidence_class": "MEASURED (ours; every side opened from JSON on "
                           "disk, C91; paired on identical window sets)",
        "eval_tier": "T0-DIAGNOSTIC",
        "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, n_boot "
                     "2000, 70 episode clusters",
        "forbidden": "overlapping_holdout_se",
        "seeds": [int(s) for s in SEEDS],
        "guard": "taniteval.degeneracy.k1_guard",
        "routes": {
            "A_unpen": "ridge_fit(..., intercept_col=-1) — the MODULE's repair; "
                       "C100's route",
            "B_centred": "the locally re-derived repair; the route "
                         "LATENT_LINEAR_LADDER §5.3/§7/§8 was rendered from",
            "POOLING": "⛔ BANNED (C100/C103). Every table names one route.",
        },
        "arms": want, "n_arms": len(want),
        "trivial_proxy_control": {
            "cells_window_family": "proxyv0 (C-V0-PROXY@11250, 1 feature)",
            "tokens_window_family":
                ("proxytok (C-V0-PROXY on the TOKENS window set) — NEW here; "
                 "C100 had no C-V0 arm on that window family, so its 3 "
                 "tokens-window arms carried no trivial-proxy control at all"
                 if "proxytok" in have_extra else
                 "⛔ ABSENT — the 3 tokens-window arms are UNPAIRED"),
            "sign_convention":
                "K1B is negative-is-better; margin = arm − cv0; POSITIVE MEANS "
                "THE SCALAR WINS",
        },
    }

    # ---- R0 the reproduction gate (route A only: it is C100's route) --------
    out["R0_reproduction_gate_vs_banked_seed0"] = reproduction_gate(ra)

    # ---- R1 the per-row 3-seed table ---------------------------------------
    rows_a = build_rows(ra, "A_unpen", inc, want)
    add_trivial_proxy(rows_a)
    out["R1_rows_route_A"] = rows_a
    c100_rows = [r for r in rows_a if r.get("old_verdict_seed0")]
    out["n_rows_route_A"] = len(rows_a)
    out["n_rows_C100_set"] = sum(1 for r in rows_a if r["arm"] in ARMS)

    # ---- R2 the C100 inventory, re-derived at 3 seeds -----------------------
    sepfail = [r for r in rows_a if r.get("bucket_3seed")]
    b3 = Counter(r["bucket_3seed"] for r in sepfail)
    per_seed_counts = [Counter(r["bucket_per_seed"][i] for r in sepfail)
                       for i in range(3)]
    subst = [r for r in sepfail if r.get("substantive_3seed")]
    # ⭐ THE SHARPEST FORM OF THE INVENTORY. A row can be SEED-UNSTABLE because
    # the MECHANISM that kills it varies while the DEATH does not — those are
    # not "no verdict about whether it is a finding", they are "dead, by one
    # route or the other". Separating them is what makes the count quotable.
    DEAD = ("die_at_repair", "killed_by_guard")
    dead_all = [r for r in sepfail if all(b in DEAD for b in r["bucket_per_seed"])]
    sometimes = [r for r in sepfail
                 if "survive_both" in r["bucket_per_seed"]
                 and r["bucket_3seed"] == "SEED-UNSTABLE"]
    out["R2_fail_inventory_3seed"] = {
        "route": "A_unpen",
        "population": "the 87 banked separated-FAILs, defined exactly as C100 "
                      "did: incumbent (pc6) verdict at SEED 0",
        "banked_separated_FAILs_seed0": len(sepfail),
        "counts_3seed_unanimous": dict(b3),
        "counts_per_seed": [dict(c) for c in per_seed_counts],
        "n_bucket_seed_stable": sum(r["bucket_seed_stable"] for r in sepfail),
        "n_bucket_SEED_UNSTABLE":
            sum(not r["bucket_seed_stable"] for r in sepfail),
        "SHARPEST_FORM": {
            "_note": "⭐ the quotable partition: a SEED-UNSTABLE row whose "
                     "buckets are all deaths is DEAD, only by a varying "
                     "mechanism. 4 disjoint classes summing to the 87.",
            "dead_on_all_three_seeds": len(dead_all),
            "of_which_mechanism_varies":
                sum(1 for r in dead_all if not r["bucket_seed_stable"]),
            "flip_to_PASS_unanimous":
                sum(1 for r in sepfail if r["bucket_3seed"] == "flip_to_PASS"),
            "survive_both_unanimous":
                sum(1 for r in sepfail if r["bucket_3seed"] == "survive_both"),
            "survive_on_some_seeds_not_all": len(sometimes),
            "substantive": len(subst),
        },
        "substantive_threshold": f"|K1B_mean|/gt_sd >= {SUBSTANTIVE_REL} "
                                 "(C100's reporting threshold, reported not "
                                 "gated on)",
        "substantive_3seed": [
            {k: r[k] for k in ("arm", "label", "target", "rung", "K1B_mean",
                               "K1B_per_seed", "K1B_mean_rel_gt_sd",
                               "margin_vs_cv0_K1B_mean",
                               "scalar_wins_3seed_mean", "guard_3seed")}
            for r in subst],
        "seed_unstable_rows": [
            {k: r[k] for k in ("arm", "target", "bucket_per_seed",
                               "new_K1_per_seed", "K1B_per_seed",
                               "guard_per_seed", "new_alpha_per_seed")}
            for r in sepfail if not r["bucket_seed_stable"]],
    }

    # ---- R2b the incumbent's own seed stability (free, from banked) ---------
    inc_rows = [r for r in rows_a if "old_verdict_per_seed" in r]
    out["R2b_incumbent_vs_repaired_seed_stability"] = {
        "note": "⭐ THE MECHANISM C103 NAMES, measured on the same 165 rows: the "
                "DEFECTIVE instrument is seed-stable because its truncated "
                "alpha sweep froze the choice; the REPAIRED one is not.",
        "n_rows": len(inc_rows),
        "incumbent_alpha_seed_stable":
            sum(r["old_alpha_seed_stable"] for r in inc_rows),
        "repaired_alpha_seed_stable":
            sum(r["alpha_seed_stable"] for r in inc_rows),
        "incumbent_verdict_seed_stable":
            sum(r["old_verdict_seed_stable"] for r in inc_rows),
        "repaired_verdict_seed_stable":
            sum(r["verdict_seed_stable"] for r in inc_rows),
        "incumbent_max_K1_seed_spread":
            max(r["old_K1_seed_spread"] for r in inc_rows),
        "repaired_max_K1_seed_spread":
            max(r["new_K1_seed_spread"] for r in inc_rows),
        "repaired_max_K1B_seed_spread":
            max(r["K1B_seed_spread"] for r in inc_rows),
    }

    # ---- R3 the trivial-proxy control, summarised --------------------------
    paired = [r for r in rows_a if r.get("paired_window_set")
              and r["arm"] not in ("proxyv0", "proxytok")]
    out["R3_trivial_proxy_control"] = {
        "route": "A_unpen",
        "n_rows_with_a_paired_C_V0": len(paired),
        "n_rows_where_the_SCALAR_wins_or_ties_on_the_3seed_mean":
            sum(bool(r["scalar_wins_3seed_mean"]) for r in paired),
        "n_rows_where_the_LATENT_beats_the_scalar":
            sum(not r["scalar_wins_3seed_mean"] for r in paired),
        "n_rows_whose_margin_SIGN_flips_across_seeds":
            sum(not r["margin_sign_seed_stable"] for r in paired),
        "n_rows_unpaired_no_C_V0":
            sum(1 for r in rows_a if not r.get("paired_window_set")),
        "by_arm": {
            arm: {
                "scalar_wins": sum(bool(r["scalar_wins_3seed_mean"])
                                   for r in paired if r["arm"] == arm),
                "latent_wins": sum(not r["scalar_wins_3seed_mean"]
                                   for r in paired if r["arm"] == arm),
                "margin_sign_unstable": sum(not r["margin_sign_seed_stable"]
                                            for r in paired if r["arm"] == arm),
            } for arm in sorted({r["arm"] for r in paired})},
    }

    # ---- R6 the rung PROFILE (r²) at 3 seeds -------------------------------
    # ⭐ The quantity every downstream citation of this ladder quotes. C92/C97
    # act on the fit's DISPERSION and correlation is scale-invariant, so at a
    # FIXED alpha neither defect can move r² — but they truncated the alpha
    # sweep, and the seed moves alpha, so r² moves with the seed too.
    by_a = {(r["arm"], r["target"]): r for r in rows_a}
    prof = []
    for t in LADDER:
        v, n = by_a[("s11250", t)], by_a[("nullmatched", t)]
        r2 = [c * c for c in v["corr_per_seed"]]
        n2 = [c * c for c in n["corr_per_seed"]]
        prof.append({
            "target": t, "rung": v["rung"], "gt_sd": v["gt_sd"],
            "v6_r_per_seed": v["corr_per_seed"],
            "v6_r2_per_seed": [round(x, 6) for x in r2],
            "v6_r2_seed0": round(r2[0], 6),
            "v6_r2_3seed_mean": round(sum(r2) / 3, 6),
            "v6_r2_seed_spread": round(max(r2) - min(r2), 6),
            "null_r2_3seed_mean": round(sum(n2) / 3, 6),
        })
    ordered = sorted(prof, key=lambda p: -p["v6_r2_3seed_mean"])
    out["R6_rung_profile_r2_3seed"] = {
        "route": "A_unpen", "arm": "v6F-SW-30k@11250",
        "note": "r² = r_ceiling², the variance an optimally-rescaled version of "
                "the same readout would explain. Reported at 3 seeds because "
                "the repaired alpha sweep is seed-sensitive (C103).",
        "ordering_by_3seed_mean": [p["target"] for p in ordered],
        "ordering_by_seed0": [p["target"] for p in
                              sorted(prof, key=lambda q: -q["v6_r2_seed0"])],
        "ordering_changed_vs_seed0":
            [p["target"] for p in ordered]
            != [p["target"] for p in sorted(prof, key=lambda q: -q["v6_r2_seed0"])],
        "rows": prof,
    }

    # ---- R5 what is QUOTABLE: PASS and guard OK on ALL THREE seeds ---------
    # ⛔ The only list a reader may take a positive claim from. A row that
    # PASSes on two seeds and not the third is SEED-UNSTABLE and appears here
    # nowhere. Every entry carries its trivial-proxy margin (C92).
    q = [r for r in rows_a
         if r["verdict_3seed"] == "PASS" and r["guard_3seed"] == "OK"]
    CONTROL_ARMS = {"proxyv0", "proxytok", "orcdir", "nullmatched",
                    "tok11250null", "egoorc_n0.1", "egoorc_n1", "egoorc_n3",
                    "egoorc_n10"}
    out["R5_quotable_guarded_PASS_3seed"] = {
        "route": "A_unpen",
        "criterion": "verdict PASS on all 3 seeds AND guard OK on all 3 seeds",
        "n_total": len(q),
        "n_on_a_v6_LATENT_arm": sum(r["arm"] not in CONTROL_ARMS for r in q),
        "n_v6_rows_the_SCALAR_still_wins":
            sum(bool(r["scalar_wins_3seed_mean"])
                for r in q if r["arm"] not in CONTROL_ARMS),
        "rows": [{k: r[k] for k in
                  ("arm", "label", "target", "rung", "K1B_mean",
                   "K1B_per_seed", "K1B_mean_rel_gt_sd", "cv0_arm",
                   "cv0_K1B_mean", "margin_vs_cv0_K1B_mean",
                   "margin_vs_cv0_K1B_per_seed", "margin_vs_cv0_rel_gt_sd",
                   "scalar_wins_3seed_mean", "scalar_wins_per_seed",
                   "margin_sign_seed_stable", "corr_mean")}
                 for r in q],
    }

    # ---- R4 the two routes side by side, at 3 seeds -------------------------
    if (rb / "ll3_s11250.json").exists():
        rows_b = build_rows(rb, "B_centred", inc, want)
        add_trivial_proxy(rows_b)
        out["R4_rows_route_B"] = rows_b
        bb = {(r["arm"], r["target"]): r for r in rows_b}
        cmp_rows = []
        for r in rows_a:
            q = bb.get((r["arm"], r["target"]))
            if q is None:
                continue
            cmp_rows.append({
                "arm": r["arm"], "target": r["target"],
                "A_alpha": r["new_alpha_per_seed"],
                "B_alpha": q["new_alpha_per_seed"],
                "alpha_differs": r["new_alpha_per_seed"] != q["new_alpha_per_seed"],
                "A_K1_mean": r["new_K1_mean"], "B_K1_mean": q["new_K1_mean"],
                "A_K1B_mean": r["K1B_mean"], "B_K1B_mean": q["K1B_mean"],
                "A_verdict_3seed": r["verdict_3seed"],
                "B_verdict_3seed": q["verdict_3seed"],
                "verdict_differs": r["verdict_3seed"] != q["verdict_3seed"],
                "A_guard_3seed": r["guard_3seed"],
                "B_guard_3seed": q["guard_3seed"],
                "guard_differs": r["guard_3seed"] != q["guard_3seed"],
                "abs_K1_mean_gap": round(abs(r["new_K1_mean"]
                                             - q["new_K1_mean"]), 6),
                "abs_K1B_mean_gap": round(abs(r["K1B_mean"]
                                              - q["K1B_mean"]), 6),
            })
        out["R4_two_routes_3seed"] = {
            "note": "⛔ SIDE BY SIDE ONLY — never pooled. C103 measured the "
                    "route comparison on 44 rows AT ONE SEED; this is the same "
                    "comparison at three, on every row.",
            "n_rows": len(cmp_rows),
            "n_alpha_vectors_differing":
                sum(r["alpha_differs"] for r in cmp_rows),
            "n_verdicts_3seed_differing":
                sum(r["verdict_differs"] for r in cmp_rows),
            "n_guard_verdicts_differing":
                sum(r["guard_differs"] for r in cmp_rows),
            "max_abs_K1_mean_gap": max((r["abs_K1_mean_gap"]
                                        for r in cmp_rows), default=None),
            "max_abs_K1B_mean_gap": max((r["abs_K1B_mean_gap"]
                                         for r in cmp_rows), default=None),
            "rows": cmp_rows,
        }
        # the C100 inventory on route B, so the difference is visible
        sf_b = [r for r in rows_b if r.get("bucket_3seed")]
        out["R4b_fail_inventory_3seed_route_B"] = {
            "route": "B_centred",
            "banked_separated_FAILs_seed0": len(sf_b),
            "counts_3seed_unanimous":
                dict(Counter(r["bucket_3seed"] for r in sf_b)),
            "substantive_3seed": [
                {k: r[k] for k in ("arm", "target", "K1B_mean",
                                   "K1B_mean_rel_gt_sd",
                                   "margin_vs_cv0_K1B_mean")}
                for r in sf_b if r.get("substantive_3seed")],
        }
    else:
        out["R4_two_routes_3seed"] = {"available": False}

    pathlib.Path(a.out).write_text(json.dumps(out, indent=1), "utf-8")

    # ---------------- console summary ---------------------------------------
    g = out["R0_reproduction_gate_vs_banked_seed0"]
    print(f"R0 REPRODUCTION GATE vs banked route-A seed 0: {g['GATE']} "
          f"({g['fields_compared'] - g['fields_differing']}/"
          f"{g['fields_compared']} fields identical)")
    if g["differences"]:
        for d in g["differences"][:10]:
            print("   ⛔", d)
    inv = out["R2_fail_inventory_3seed"]
    print(f"\nR2 — OF THE {inv['banked_separated_FAILs_seed0']} BANKED "
          f"SEPARATED-FAILs, at 3 seeds (route A):")
    for k in ("die_at_repair", "killed_by_guard", "flip_to_PASS",
              "survive_both", "SEED-UNSTABLE"):
        print(f"   {k:18}: {inv['counts_3seed_unanimous'].get(k, 0)}")
    print(f"   per-seed counts   : {inv['counts_per_seed']}")
    print(f"   ⭐ substantive     : {len(inv['substantive_3seed'])} "
          f"(|K1B_mean|/gt_sd >= {SUBSTANTIVE_REL})")
    for r in inv["substantive_3seed"]:
        print(f"      {r['arm']:14} {r['target']:14} K1B_mean {r['K1B_mean']:+.4f} "
              f"rel {r['K1B_mean_rel_gt_sd']:+.4f}  margin_vs_cv0 "
              f"{r['margin_vs_cv0_K1B_mean']} scalar_wins="
              f"{r['scalar_wins_3seed_mean']}")
    s = out["R2b_incumbent_vs_repaired_seed_stability"]
    print(f"\nR2b SEED STABILITY over {s['n_rows']} rows — "
          f"alpha stable: incumbent {s['incumbent_alpha_seed_stable']} vs "
          f"repaired {s['repaired_alpha_seed_stable']}; verdict stable: "
          f"{s['incumbent_verdict_seed_stable']} vs "
          f"{s['repaired_verdict_seed_stable']}; max K1 spread "
          f"{s['incumbent_max_K1_seed_spread']} vs "
          f"{s['repaired_max_K1_seed_spread']}")
    p = out["R3_trivial_proxy_control"]
    print(f"\nR3 TRIVIAL-PROXY CONTROL over {p['n_rows_with_a_paired_C_V0']} "
          f"paired rows: scalar wins/ties "
          f"{p['n_rows_where_the_SCALAR_wins_or_ties_on_the_3seed_mean']}, "
          f"latent wins {p['n_rows_where_the_LATENT_beats_the_scalar']}, "
          f"margin sign flips across seeds "
          f"{p['n_rows_whose_margin_SIGN_flips_across_seeds']}, unpaired "
          f"{p['n_rows_unpaired_no_C_V0']}")
    q5 = out["R5_quotable_guarded_PASS_3seed"]
    print(f"\nR5 QUOTABLE (PASS + guard OK on ALL 3 seeds): {q5['n_total']} rows, "
          f"of which {q5['n_on_a_v6_LATENT_arm']} on a v6 latent arm — and the "
          f"single ego-speed scalar still wins "
          f"{q5['n_v6_rows_the_SCALAR_still_wins']} of those on the 3-seed mean")
    r4 = out.get("R4_two_routes_3seed", {})
    if r4.get("n_rows"):
        print(f"\nR4 ROUTES A vs B at 3 seeds over {r4['n_rows']} rows: "
              f"alpha vectors differ {r4['n_alpha_vectors_differing']}, "
              f"3-seed verdicts differ {r4['n_verdicts_3seed_differing']}, "
              f"guard differs {r4['n_guard_verdicts_differing']}, "
              f"max |ΔK1 mean| {r4['max_abs_K1_mean_gap']}, "
              f"max |ΔK1B mean| {r4['max_abs_K1B_mean_gap']}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
