#!/usr/bin/env python3
"""⭐⭐ RANK CANDIDATE GATE STATISTICS BY A MUTATION TEST, BEFORE ANY OF THEM IS SCORED.

Executes clauses 3 and 4 of the Master Mind ruling of 2026-09-06, pre-registered in
``.../2026-09-06-refcv4b-rl-repair/PREREG_STATISTIC_RANKING.md`` (committed BEFORE this
file computed a number).

⛔ ``G-REWARD`` REMAINS FAILED ON ITS COMMITTED STATISTIC (THE RATE). Nothing here
re-scores it. Every statistic below is an INSTRUMENT candidate, never a gate verdict.

THE MUTATION, fixed in the pre-registration:

    PERTURB(d):  lead_track[i, 0] += d   for i in {1, 2, 3}
                 lead_track[0] and lead_track[4] held EXACTLY fixed

The panel's lead track is exactly ``[5, 2]`` (``GRID_S`` = 0.0 .. 2.0 s), so "three of
five lead samples move by 5 m" is LITERAL here, not an analogy -- it is the same
perturbation the unit test used to disqualify ``_headway``'s ``amin``.

⭐ The endpoint is pinned so ``progress``'s lead cap -- a position query on
``lead_path[..., -1, 0]`` -- is provably inert, isolating the distance-keeping channel.
⛔ The perturbation moves POSITIONS only. No closing rate is introduced; TTC stays a VETO.

⛔ THE DISQUALIFICATION CRITERION: a candidate whose value is BIT-IDENTICAL under
PERTURB(5.0) is OUT, regardless of its score. Survivors rank by RESPONSE-TO-NOISE RATIO
``|dS| / SD_boot(S)``; ``RNR < 1`` is reported UNDERPOWERED.

⛔ SPLIT DISCIPLINE. ``--split select`` is the ONLY split the ranking may read.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("TANITAD_REPO") or os.path.dirname(os.path.dirname(HERE))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("rl_refcv3_min_for_statrank",
          os.path.join(REPO, "stack", "scripts", "rl_refcv3_min.py"))
np = M.np
torch = M.torch

# --------------------------------------------------------------------------- #
# ⛔ EVERY CONSTANT BELOW IS FROM THE PRE-REGISTRATION, NOT CHOSEN AFTER A NUMBER #
# --------------------------------------------------------------------------- #
#: the three of five lead samples that move; 0 and 4 (the endpoint) are pinned
PERT_IDX = (1, 2, 3)
#: the same three PLUS the endpoint -- the DISCRIMINATING control for cap inertness
PERT_IDX_ENDPOINT = (1, 2, 3, 4)

#: (cell name, indices moved, delta_m, headway_reduce, progress_lead_cap)
#: ⛔ `nopert` never enters the clone-and-add path; `null` does, with 0.0 -- so the NULL
#: control proves the perturbation MACHINERY introduces no drift, rather than comparing
#: a cell with itself.
CELLS = (
    ("nopert",    (),                 0.0,  "q0.25", False),
    ("null",      PERT_IDX,           0.0,  "q0.25", False),
    ("p5",        PERT_IDX,           5.0,  "q0.25", False),
    ("p15",       PERT_IDX,          15.0,  "q0.25", False),
    ("nopert_min", (),                0.0,  "min",   False),
    ("p5_min",    PERT_IDX,           5.0,  "min",   False),
    ("nopert_cap", (),                0.0,  "q0.25", "lead"),
    ("p5_cap",    PERT_IDX,           5.0,  "q0.25", "lead"),
    ("p5end_cap", PERT_IDX_ENDPOINT,  5.0,  "q0.25", "lead"),
    # ⭐ UNPERTURBED REWARD CONFIGURATIONS, added for `H-RL-GATE-STAT-2`. They take no
    # part in the mutation ranking (which reads `nopert`/`p5`/`p15` only) and exist so a
    # gate statistic can be read on the reward an arm would ACTUALLY train on.
    #   `ship`     -- the SHIPPED defaults: progress_lead_cap True -> "achievable",
    #                 headway_reduce "min"
    #   `lead_min` -- the sensitivity variant of the cap, legacy reduction
    #   `both_lq`  -- both 2026-09-06 repairs together
    ("ship",      (),                 0.0,  "min",   "achievable"),
    ("lead_min",  (),                 0.0,  "min",   "lead"),
    ("both_lq",   (),                 0.0,  "q0.25", "lead"),
)
LADDER = (float("inf"), 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0)
WEIGHTS = {"progress": 0.3, "collision": 1.0, "headway": 0.3}
EPS = 1e-12


def split_of(clip_id: str) -> str:
    """⛔ FIXED IN THE PRE-REGISTRATION, and computed from the clip id so it does not
    depend on enumeration order. Even first byte -> SELECT, odd -> SCORE."""
    h = hashlib.sha256(str(clip_id).encode("utf-8")).digest()
    return "select" if (h[0] % 2 == 0) else "score"


# --------------------------------------------------------------------------- #
# the candidate statistics -- enumerated in the pre-registration BEFORE any ran   #
# --------------------------------------------------------------------------- #
def _rank(a):
    """Average ranks, ties shared (the Wilcoxon convention)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    vals = a[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def S1_rate(g):
    return float((g >= -1e-12).mean())


def S2_cliffs(g):
    return float((g > 0).mean() - (g < 0).mean())


def S3_median(g):
    return float(np.median(g))


def S4_q75(g):
    return float(np.quantile(g, 0.75))


def S5_mean(g):
    return float(g.mean())


def S6_wmr(g):
    """Win-magnitude ratio: the share of the TOTAL gap mass paid to hold_v0.
    ⭐ Same [0, 1] scale as the rate, same 0.5-neutral reading."""
    pos = float(np.maximum(g, 0.0).sum())
    tot = float(np.abs(g).sum())
    return pos / tot if tot > 0 else 0.5


def S7_signedrank(g):
    """Wilcoxon r+ : the share of |g|-rank mass carried by the windows hold_v0 wins."""
    r = _rank(np.abs(g))
    tot = float(r.sum())
    return float(r[g > 0].sum()) / tot if tot > 0 else 0.5


def S8_logmassratio(g):
    pos = float(np.maximum(g, 0.0).sum())
    neg = float(np.maximum(-g, 0.0).sum())
    return float(np.log(pos + EPS) - np.log(neg + EPS))


def S9_skewsplit(g):
    return float(np.median(g) - g.mean())


CANDIDATES = {
    "S1_rate": (S1_rate, "SIGN"),
    "S2_cliffs": (S2_cliffs, "SIGN"),
    "S3_median": (S3_median, "ORDER"),
    "S4_q75": (S4_q75, "ORDER"),
    "S5_mean": (S5_mean, "MASS"),
    "S6_wmr": (S6_wmr, "MASS"),
    "S7_signedrank": (S7_signedrank, "RANK"),
    "S8_logmassratio": (S8_logmassratio, "MASS"),
    "S9_skewsplit": (S9_skewsplit, "MIXED"),
}


def boot_stat(fn, g, eids, n_boot=4000, seed=0, alpha=0.05):
    """Episode-cluster bootstrap of an ARBITRARY statistic.
    ⛔ Answers ONE question: would another draw of EPISODES say this? -- not another
    training run (H-ESTIM-SEED-1) and not another inference run."""
    uniq = np.unique(eids)
    idx = {e: np.flatnonzero(eids == e) for e in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[e] for e in pick])
        draws[b] = fn(g[sel])
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(draws.std(ddof=1))


# --------------------------------------------------------------------------- #
def build_rows(a):
    """ONE pass over the corpus; every cell scored on the SAME window so nothing can
    differ except the perturbation, the reduction and the cap."""
    M.LEAD_MODE = "track"
    _model, cfg, _targs, prov = M.load(a, "cpu")
    corp = M.open_corpus(a.episodes, a.labels, cfg, prov, a.lru)
    lead, _lm, _li = M.load_lead_block(a.lead_block)
    if lead is None or lead.ts_rel is None:
        raise SystemExit("[statrank] a lead block with ts_rel_s is required")
    spec = M.RewardSpec(weights=dict(M.DEFAULT_WEIGHTS), dt=M.DT_REWARD_S)
    rows = []
    for wi in M.scoreable_windows(corp, lead):
        has, xy, _ln = M.lead_row(corp, lead, wi)
        if not has:
            continue
        e_i, t = corp.ds.index[wi]
        clip = corp.clip_ids[e_i]
        row = lead.idx.get((clip, int(t + corp.W - 1 + corp.raw_off)))
        human, v0 = M._human_future(corp, wi)
        xs = torch.tensor([v0 * s for s in M.GRID_S], dtype=torch.float32)
        hold = torch.stack([xs, torch.zeros_like(xs)], dim=-1).reshape(1, 1, 1, 5, 2)
        frozen = torch.zeros(1, 1, 1, 5, 2)
        lt0 = M.lead_track(corp, lead, wi)                              # [5, 2]
        rec = {"wi": int(wi), "clip": clip, "eid": int(e_i), "v0": v0,
               "split": split_of(clip),
               "gt_time_gap_min_s": (float(lead.gt_time_gap[row])
                                     if lead.gt_time_gap is not None else float("nan")),
               "lead_end_x_m": float(lt0[-1, 0])}
        for cname, idxs, d, hred, cap in CELLS:
            lt = lt0.clone()
            for i in idxs:
                lt[i, 0] = lt[i, 0] + float(d)
            batch = {"v0": torch.tensor([v0]),
                     "lead_xy": torch.tensor([xy], dtype=torch.float32),
                     "lead_track": lt[None]}
            base_ctx = M.reward_ctx(batch, S5=5)
            ctx = {**base_ctx, "progress_lead_cap": cap, "headway_reduce": hred}
            for name, traj in (("human", human), ("hold_v0", hold),
                               ("frozen", frozen)):
                parts = spec.per_component(traj, ctx)
                rec["%s__%s" % (name, cname)] = {
                    k: float(v.reshape(-1)[0]) for k, v in parts.items()}
        rows.append(rec)
    return rows


def composed(side):
    return sum(w * float(side.get(k, 0.0)) for k, w in WEIGHTS.items())


def gaps(rows, cell):
    h = np.array([composed(r["human__" + cell]) for r in rows])
    v = np.array([composed(r["hold_v0__" + cell]) for r in rows])
    return v - h


def comp(rows, side, cell, key):
    return np.array([float(r["%s__%s" % (side, cell)].get(key, 0.0)) for r in rows])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--expect-step", type=int, default=M.EXPECT_BASE_STEP)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--lead-block", required=True)
    ap.add_argument("--lru", type=int, default=6)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="select", choices=("select",),
                    help="⛔ the ranking may read the SELECT split ONLY")
    ap.add_argument("--rows-out", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    rows_all = build_rows(a)
    if a.rows_out:
        with open(a.rows_out, "w", encoding="utf-8") as fh:
            json.dump(rows_all, fh)
    n_sel = sum(1 for r in rows_all if r["split"] == "select")
    n_sco = len(rows_all) - n_sel
    print("[statrank] %d lead windows / %d episodes  (SELECT %d, SCORE %d)"
          % (len(rows_all), len({r["eid"] for r in rows_all}), n_sel, n_sco))
    rows = [r for r in rows_all if r["split"] == a.split]
    if not rows:
        raise SystemExit("[statrank] the %s split is empty" % a.split)
    eids = np.array([r["eid"] for r in rows])
    tg = np.array([float(r.get("gt_time_gap_min_s", np.nan)) for r in rows])
    print("[statrank] ranking on SELECT: %d windows / %d episodes"
          % (len(rows), len(np.unique(eids))))

    # ------------------------------------------------------------------ #
    # ⛔ CONTROLS THAT MUST READ KNOWN VALUES -- computed BEFORE the ranking #
    # ------------------------------------------------------------------ #
    ctrl = {}

    # C1 NULL: the clone-and-add path with d = 0.0 must reproduce `nopert` EXACTLY
    null_worst = 0.0
    for side in ("human", "hold_v0", "frozen"):
        for k in ("progress", "collision", "headway", "comfort", "feasibility"):
            null_worst = max(null_worst, float(np.abs(
                comp(rows, side, "null", k) - comp(rows, side, "nopert", k)).max()))
    ctrl["NULL_max_abs_component_delta"] = null_worst
    ctrl["NULL_pass"] = bool(null_worst == 0.0)

    # C3 CAP INERTNESS (non-vacuous: cap="lead", so `progress` DOES read the lead)
    cap_worst = 0.0
    for side in ("human", "hold_v0", "frozen"):
        cap_worst = max(cap_worst, float(np.abs(
            comp(rows, side, "p5_cap", "progress")
            - comp(rows, side, "nopert_cap", "progress")).max()))
    ctrl["CAP_INERTNESS_max_abs_dprogress"] = cap_worst
    ctrl["CAP_INERTNESS_pass"] = bool(cap_worst == 0.0)
    # ... and its DISCRIMINATING control: moving the ENDPOINT too MUST move `progress`
    cap_disc = float(np.abs(
        comp(rows, "hold_v0", "p5end_cap", "progress")
        - comp(rows, "hold_v0", "nopert_cap", "progress")).max())
    cap_disc_frac = float((np.abs(
        comp(rows, "hold_v0", "p5end_cap", "progress")
        - comp(rows, "hold_v0", "nopert_cap", "progress")) > 0).mean())
    ctrl["CAP_DISCRIMINATING_max_abs_dprogress"] = cap_disc
    ctrl["CAP_DISCRIMINATING_moved_frac"] = cap_disc_frac
    ctrl["CAP_DISCRIMINATING_pass"] = bool(cap_disc > 0.0)

    # C2 TERM BLINDNESS at panel scale
    for tag, (b_cell, p_cell) in {"min": ("nopert_min", "p5_min"),
                                  "q0.25": ("nopert", "p5")}.items():
        for side in ("hold_v0", "human"):
            d = np.abs(comp(rows, side, p_cell, "headway")
                       - comp(rows, side, b_cell, "headway"))
            ctrl["TERM_BLIND_%s_%s_headway_bit_identical_frac" % (tag, side)] = \
                float((d == 0.0).mean())
        gb, gp = gaps(rows, b_cell), gaps(rows, p_cell)
        ctrl["COMPOSED_GAP_bit_identical_frac_%s" % tag] = \
            float((np.abs(gp - gb) == 0.0).mean())

    movers = {}
    for k in ("progress", "collision", "headway", "comfort", "feasibility"):
        movers[k] = float((np.abs(comp(rows, "hold_v0", "p5", k)
                                  - comp(rows, "hold_v0", "nopert", k)) > 0).mean())
    ctrl["COMPONENT_moved_frac_holdv0_q0.25_5m"] = movers

    # ------------------------------------------------------------------ #
    # ⛔ THE MUTATION RANKING -- run BEFORE any candidate is scored          #
    # ------------------------------------------------------------------ #
    g0 = gaps(rows, "nopert")
    ranking = {}
    for cid, (fn, fam) in CANDIDATES.items():
        s0 = fn(g0)
        lo, hi, sd = boot_stat(fn, g0, eids, a.n_boot, a.seed)
        ent = {"family": fam, "value_unperturbed": s0,
               "ci": [lo, hi], "sd_boot": sd, "deltas": {}}
        for cname in ("null", "p5", "p15"):
            ent["deltas"][cname] = fn(gaps(rows, cname)) - s0
        d5, d15 = ent["deltas"]["p5"], ent["deltas"]["p15"]
        ent["NULL_pass"] = bool(ent["deltas"]["null"] == 0.0)
        ent["bit_identical_at_5m"] = bool(d5 == 0.0)
        ent["DISQUALIFIED"] = bool(d5 == 0.0)
        ent["RNR"] = (abs(d5) / sd) if sd > 0 else float("inf")
        ent["UNDERPOWERED"] = bool(ent["RNR"] < 1.0)
        ent["MONOTONE_pass"] = bool(abs(d15) >= abs(d5))
        ent["direction_pays_holdv0_more"] = bool(d5 > 0)
        ranking[cid] = ent

    survivors = [c for c in ranking if not ranking[c]["DISQUALIFIED"]]
    survivors.sort(key=lambda c: -ranking[c]["RNR"])
    disq = [c for c in ranking if ranking[c]["DISQUALIFIED"]]

    per_rung = {}
    for thr in LADDER:
        m = (np.isfinite(tg) & (tg <= thr)) if np.isfinite(thr) else np.ones(len(rows), bool)
        label = "all" if not np.isfinite(thr) else "<=%.1fs" % thr
        if int(m.sum()) == 0:
            continue
        per_rung[label] = {"n_windows": int(m.sum()),
                           "n_episodes": int(len(np.unique(eids[m]))),
                           **{cid: CANDIDATES[cid][0](g0[m]) for cid in CANDIDATES}}

    out = {
        "_what": "MUTATION RANKING of candidate gate statistics -- clause 3 of the "
                 "Master Mind ruling 2026-09-06",
        "_prereg": "TanitAD Research Lab/Architecture & Inference/Research/"
                   "2026-09-06-refcv4b-rl-repair/PREREG_STATISTIC_RANKING.md",
        "_not_a_gate": "G-REWARD remains FAILED on its committed statistic (the RATE). "
                       "Nothing here re-scores it.",
        "_evidence_class": "MEASURED (ours)",
        "_tier": "T0 instrument probe, NON-PARITY RL-fit windows",
        "_estimator": "episode-cluster bootstrap, n_boot %d -- answers only 'would "
                      "another draw of EPISODES say this?'" % a.n_boot,
        "mutation": {"indices": list(PERT_IDX), "pinned": [0, 4],
                     "cells": [{"name": c[0], "indices": list(c[1]), "delta_m": c[2],
                                "headway_reduce": c[3], "progress_lead_cap": c[4]}
                               for c in CELLS],
                     "axis": "x (along-track)"},
        "split": {"rule": "sha256(clip_id)[0] even -> select, odd -> score",
                  "read_here": a.split,
                  "n_windows_select": n_sel, "n_windows_score": n_sco},
        "n_windows_ranked": len(rows),
        "n_episodes_ranked": int(len(np.unique(eids))),
        "controls": ctrl,
        "ranking": ranking,
        "survivors_ranked_by_RNR": survivors,
        "disqualified": disq,
        "per_rung_select_unperturbed": per_rung,
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("\n== CONTROLS (must read known values)")
    print("   NULL   max|dcomponent| %.3e  pass=%s"
          % (ctrl["NULL_max_abs_component_delta"], ctrl["NULL_pass"]))
    print("   CAP INERTNESS (cap=lead, endpoint pinned) max|dprogress| %.3e  pass=%s"
          % (ctrl["CAP_INERTNESS_max_abs_dprogress"], ctrl["CAP_INERTNESS_pass"]))
    print("   CAP DISCRIMINATING (endpoint MOVED) max|dprogress| %.6f  moved %.4f  pass=%s"
          % (cap_disc, cap_disc_frac, ctrl["CAP_DISCRIMINATING_pass"]))
    for tag in ("min", "q0.25"):
        print("   TERM BLINDNESS[%-5s] headway bit-identical at 5 m: hold_v0 %.4f  "
              "human %.4f | composed gap %.4f"
              % (tag,
                 ctrl["TERM_BLIND_%s_hold_v0_headway_bit_identical_frac" % tag],
                 ctrl["TERM_BLIND_%s_human_headway_bit_identical_frac" % tag],
                 ctrl["COMPOSED_GAP_bit_identical_frac_%s" % tag]))
    print("   COMPONENTS moved (hold_v0, q0.25, 5 m): " +
          "  ".join("%s %.4f" % (k, v) for k, v in movers.items()))

    print("\n== MUTATION RANKING  (SELECT split, headway_reduce=q0.25)")
    print("%-16s %-6s %11s %10s %11s %11s %8s %-14s"
          % ("candidate", "family", "value", "SD_boot", "dS(5 m)", "dS(15 m)",
             "RNR", "verdict"))
    for cid in survivors + disq:
        e = ranking[cid]
        v = ("DISQUALIFIED" if e["DISQUALIFIED"]
             else ("UNDERPOWERED" if e["UNDERPOWERED"] else "SURVIVES"))
        if not e["MONOTONE_pass"] and not e["DISQUALIFIED"]:
            v += "*"
        print("%-16s %-6s %11.6f %10.6f %11.6f %11.6f %8.2f %-14s"
              % (cid, e["family"], e["value_unperturbed"], e["sd_boot"],
                 e["deltas"]["p5"], e["deltas"]["p15"],
                 e["RNR"] if e["RNR"] != float("inf") else -1.0, v))
    print("\n[statrank] survivors (ranked): %s" % ", ".join(survivors))
    print("[statrank] disqualified:        %s" % ", ".join(disq))
    print("[statrank] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
