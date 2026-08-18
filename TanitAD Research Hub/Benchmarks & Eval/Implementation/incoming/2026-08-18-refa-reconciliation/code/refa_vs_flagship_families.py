"""REF-A vs flagship — the PAIRED, per-family comparison the ADE-only argument was missing.

WHY THIS EXISTS
---------------
The whole REF-A-vs-flagship verdict (H4, registry D-A5) is argued on ONE number:
ADE@2s 2.1675 vs 0.4271. `CLAUDE.md`'s binding rule (Sayed, 2026-08-02) says an eval
that reports ADE alone is an INCOMPLETE result, and must ADD the four families
(LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC), per family, never pooled.

The data to close that gap was already banked and needed **zero GPU**: the two arms'
per-window dumps are on the SAME 881 windows / 40 episodes, with BIT-IDENTICAL `gt`
and `cv` tensors (verified below, and the run aborts if they are not). That makes the
**paired** episode-cluster bootstrap legal — strictly stronger than comparing the two
published single-arm intervals, which is what §6 of the registry currently invites.

WHAT IT DOES
------------
1. ALIGNMENT GATE — refuses to run unless `eid`, `gt`, `cv`, `speed`, `head_deg` and
   `wp_steps` are identical across the arms. A "paired" test on unaligned windows is
   the estimator error this program has already been burned by, in a new costume.
2. REPRODUCTION GATE — recomputes each arm's published headline means from the dumps
   with `taniteval.driving.per_window` and compares against the banked
   `driving_<arm>.json`. If a single-arm number does not reproduce, the paired delta
   built on the same code is not trustworthy either, so the run aborts.
3. The paired episode-cluster bootstrap (`taniteval.ci`) of REF-A minus flagship on
   every headline metric, plus the same on the LONGITUDINAL regime strata and the
   speed strata, so the four families are reported PER FAMILY.

TIER: every number here is **T0** (teacher-forced). The dumps come from the
`taniteval.driving/tier0` block. NEVER quotable as driving performance.

TACTICAL and STRATEGIC are reported as UNAVAILABLE-with-reason-and-n, which the binding
rule requires — the tier-0 dumps carry waypoints only, no decoded manoeuvre or route.
See `taniteval/taniteval/four_families.py` for the same refusal on the same grounds.

Usage:
    PYTHONPATH=<repo>/taniteval python refa_vs_flagship_families.py --out raw/
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from taniteval import ci as tci
from taniteval import driving as tdrv

# (arm-key-in-dump, registry name) — the ONLY two arms the H4 verdict compares.
ARMS = {
    "refa-dinov2": "REF-A DINOv2 4B (frozen encoder)",
    "flagship-30k": "flagship v1 (flagship4b-speedjerk-30k, trained encoder)",
    "refa-dynin-30k": "REF-A dyn-in 4B (the H4 final answer)",
}

# Which family each headline metric belongs to. Signed rows are DIAGNOSTICS: they are
# reported but never paired, because "closer to zero" is not "smaller", and a paired
# delta on a signed bias is not interpretable as a win.
FAMILY = {
    "ade_0_2s": "TRAJECTORY (ADE — the incumbent, kept, never the whole result)",
    "fde_2s": "TRAJECTORY",
    "miss_2m": "TRAJECTORY",
    "long_abs_2s_m": "LONGITUDINAL",
    "long_signed_2s_m": "LONGITUDINAL (signed diagnostic)",
    "speed_mae_mps": "LONGITUDINAL",
    "speed_bias_mps": "LONGITUDINAL (signed diagnostic)",
    "progress_abs_err_m": "LONGITUDINAL",
    "progress_signed_err_m": "LONGITUDINAL (signed diagnostic)",
    "lat_abs_2s_m": "LATERAL",
    "lat_signed_2s_m": "LATERAL (signed diagnostic)",
    "heading_mae_2s_deg": "LATERAL",
    "heading_med_2s_deg": "LATERAL",
    "heading_exceed_5deg": "LATERAL",
    "pathgeom_crosstrack_m": "LATERAL",
    "curv_sign_agree": "LATERAL",
}

SIGNED = {"long_signed_2s_m", "speed_bias_mps", "progress_signed_err_m",
          "lat_signed_2s_m"}


def load(results: pathlib.Path, arm: str) -> dict:
    return torch.load(results / f"windows_{arm}.pt", map_location="cpu",
                      weights_only=False)


def alignment_gate(dumps):
    """A paired test on unaligned windows is not a paired test."""
    keys = list(dumps)
    ref = dumps[keys[0]]
    report = {"reference_arm": keys[0], "checks": {}}
    ok = True
    for arm in keys[1:]:
        d = dumps[arm]
        c = {
            "eid_identical": d["eid"] == ref["eid"],
            "gt_bit_identical": bool(torch.equal(d["gt"], ref["gt"])),
            "cv_bit_identical": bool(torch.equal(d["cv"], ref["cv"])),
            "speed_bit_identical": bool(torch.equal(d["speed"], ref["speed"])),
            "head_deg_bit_identical": bool(torch.equal(d["head_deg"],
                                                       ref["head_deg"])),
            "wp_steps_identical": list(d["wp_steps"]) == list(ref["wp_steps"]),
            "n_windows": int(d["pred"].shape[0]),
        }
        c["all_pass"] = all(v for k, v in c.items() if k != "n_windows")
        ok &= c["all_pass"]
        report["checks"][arm] = c
    report["n_windows"] = int(ref["pred"].shape[0])
    report["n_episodes"] = len(set(ref["eid"]))
    report["passed"] = bool(ok)
    return report


def reproduction_gate(dumps, banked, tol=5e-4):
    """Exit codes are not evidence — reproduce the SINGLE-ARM numbers first.

    If this code cannot reproduce the banked headline means from the banked dumps,
    a paired delta computed with the same code is not admissible either.
    """
    out = {"tol": tol, "rows": [], "passed": True}
    for arm, d in dumps.items():
        pw = tdrv.per_window(d["pred"], d["gt"])
        pub = banked[arm]["headline"]
        for m in tdrv.HEADLINE:
            red = tdrv.REDUCE.get(m, "mean")
            got = tci.resolve_reducer(red)(np.asarray(pw[m], dtype=np.float64))
            want = pub[m]["mean"]
            delta = abs(got - want)
            row = {"arm": arm, "metric": m, "recomputed": round(float(got), 6),
                   "banked": want, "abs_diff": round(float(delta), 6),
                   "pass": bool(delta <= tol)}
            out["passed"] &= row["pass"]
            out["rows"].append(row)
    return out


def paired_table(dumps, a_arm, b_arm, n_boot=2000, seed=0):
    """REF-A minus flagship, per metric. Positive delta = REF-A's value is LARGER.

    Sign is stated per metric rather than assumed, because `curv_sign_agree`
    is higher-is-better while every error metric is lower-is-better. A single
    'favours' string computed from the sign alone would be wrong on that one row,
    which is exactly the class of defect this programme keeps finding.
    """
    higher_is_better = {"curv_sign_agree"}
    pa = tdrv.per_window(dumps[a_arm]["pred"], dumps[a_arm]["gt"])
    pb = tdrv.per_window(dumps[b_arm]["pred"], dumps[b_arm]["gt"])
    eid = dumps[a_arm]["eid"]
    rows = []
    for m in tdrv.HEADLINE:
        red = tdrv.REDUCE.get(m, "mean")
        entry = {"metric": m, "family": FAMILY[m],
                 "a_point": round(float(
                     tci.resolve_reducer(red)(np.asarray(pa[m], np.float64))), 4),
                 "b_point": round(float(
                     tci.resolve_reducer(red)(np.asarray(pb[m], np.float64))), 4),
                 "a_arm": a_arm, "b_arm": b_arm}
        if m in SIGNED:
            entry["paired"] = None
            entry["paired_refused"] = (
                "SIGNED DIAGNOSTIC — a paired delta on a signed bias is not a "
                "win/loss; closer-to-zero is the question and that is not what a "
                "difference of means measures. Reported as two point values, per "
                "the same rule that excludes biases from driving.PAIRED.")
        else:
            d = tci.paired_episode_cluster_bootstrap(
                np.asarray(pa[m], np.float64), np.asarray(pb[m], np.float64),
                eid, n_boot=n_boot, seed=seed, reduce=red)
            better = b_arm if (d["delta"] > 0) ^ (m in higher_is_better) else a_arm
            d["favours"] = better if d["separated"] else "neither (CI contains 0)"
            d["higher_is_better"] = m in higher_is_better
            entry["paired"] = d
        rows.append(entry)
    return rows


def stratified(dumps, a_arm, b_arm, n_boot=2000, seed=0):
    """The same paired delta inside the LONGITUDINAL regimes and the speed strata.

    Why: a pooled longitudinal number cannot distinguish 'cannot cruise' from
    'cannot brake', and 88.7 % of the programme's oracle gap is longitudinal.
    """
    d0 = dumps[a_arm]
    pa = tdrv.per_window(d0["pred"], d0["gt"])
    pb = tdrv.per_window(dumps[b_arm]["pred"], dumps[b_arm]["gt"])
    eid = np.asarray([str(x) for x in d0["eid"]])
    reg = tdrv.regimes(d0["gt"], d0["speed"])
    out = {"longitudinal_regime": {}, "speed_strata": {}}
    for name in ("brake", "steady", "accel"):
        idx = np.flatnonzero(reg == name)
        cell = {"n_windows": int(idx.size),
                "n_episodes": int(len(set(eid[idx].tolist())))}
        for m in ("ade_0_2s", "speed_mae_mps", "long_abs_2s_m", "lat_abs_2s_m"):
            cell[m] = tci.paired_episode_cluster_bootstrap(
                np.asarray(pa[m], np.float64)[idx],
                np.asarray(pb[m], np.float64)[idx],
                eid[idx], n_boot=n_boot, seed=seed)
        out["longitudinal_regime"][name] = cell
    strata, _ = tdrv.speed_strata(d0["speed"])
    for name, mask in strata.items():
        idx = np.flatnonzero(np.asarray(mask))
        if idx.size == 0:
            continue
        cell = {"n_windows": int(idx.size),
                "n_episodes": int(len(set(eid[idx].tolist())))}
        for m in ("ade_0_2s", "speed_mae_mps", "lat_abs_2s_m"):
            cell[m] = tci.paired_episode_cluster_bootstrap(
                np.asarray(pa[m], np.float64)[idx],
                np.asarray(pb[m], np.float64)[idx],
                eid[idx], n_boot=n_boot, seed=seed)
        out["speed_strata"][name] = cell
    return out


UNAVAILABLE = {
    "TACTICAL": {
        "status": "UNAVAILABLE",
        "reason": ("the tier-0 window dumps carry waypoints only (pred/gt/cv, 4 "
                   "waypoints 0.5 s apart). No decoded manoeuvre, no tactical goal, "
                   "no anchor selection was persisted for EITHER arm, so a "
                   "manoeuvre-confusion matrix cannot be built from banked data for "
                   "either side of this comparison."),
        "n": 0,
        "what_would_close_it": ("a decode-traversing pass (`run_one` reports 'the "
                                "scored pass did NOT traverse the hierarchy') over "
                                "the same 881 windows for both arms — GPU, not free"),
    },
    "STRATEGIC": {
        "status": "UNAVAILABLE",
        "reason": ("same dump limitation, plus: PhysicalAI-AV carries no map, lane "
                   "graph or route/goal signal, and the flagship's route head was "
                   "measured to be an exact bijection of the nav input it is fed — so "
                   "even a persisted strategic decision would be scoring an echo, not "
                   "a capability."),
        "n": 0,
        "what_would_close_it": ("a route/goal signal that is not the ego's own future "
                                "path; blocked on corpus, not on compute"),
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None,
                    help="taniteval/results dir (default: infer from repo root)")
    ap.add_argument("--out", default=".")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    here = pathlib.Path(__file__).resolve()
    root = args.results
    if root is None:
        for p in here.parents:
            cand = p / "taniteval" / "results"
            if cand.is_dir():
                root = cand
                break
    root = pathlib.Path(root)
    dumps = {a: load(root, a) for a in ARMS}
    banked = {a: json.loads((root / f"driving_{a}.json").read_text(encoding="utf-8"))
              for a in ARMS}

    align = alignment_gate(dumps)
    if not align["passed"]:
        print("ALIGNMENT GATE FAILED - refusing to emit a paired number",
              file=sys.stderr)
        print(json.dumps(align, indent=2), file=sys.stderr)
        return 2

    repro = reproduction_gate(dumps, banked)
    if not repro["passed"]:
        bad = [r for r in repro["rows"] if not r["pass"]]
        print(f"REPRODUCTION GATE FAILED on {len(bad)} rows", file=sys.stderr)
        print(json.dumps(bad, indent=2), file=sys.stderr)
        return 3

    out = {
        "block": "refa_reconciliation/paired_four_families",
        "tier": "T0",
        "tier_note": ("teacher-forced. Source dumps are taniteval.driving/tier0. "
                      "NEVER quotable as driving performance (EVAL_DOCTRINE)."),
        "estimator": "paired_episode_cluster_bootstrap",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "arms": ARMS,
        "alignment_gate": align,
        "reproduction_gate": {"tol": repro["tol"], "passed": repro["passed"],
                              "n_rows": len(repro["rows"]),
                              "max_abs_diff": max(r["abs_diff"]
                                                  for r in repro["rows"])},
        "paired_refa_dinov2_minus_flagship": paired_table(
            dumps, "refa-dinov2", "flagship-30k", args.n_boot, args.seed),
        "paired_refa_dynin_minus_flagship": paired_table(
            dumps, "refa-dynin-30k", "flagship-30k", args.n_boot, args.seed),
        "strata_refa_dinov2_minus_flagship": stratified(
            dumps, "refa-dinov2", "flagship-30k", args.n_boot, args.seed),
        "families_not_computable": UNAVAILABLE,
        "sources": {
            "dumps": [f"taniteval/results/windows_{a}.pt" for a in ARMS],
            "banked": [f"taniteval/results/driving_{a}.json" for a in ARMS],
            "metric_defs": "taniteval/taniteval/driving.py:per_window",
            "estimator": "taniteval/taniteval/ci.py:paired_episode_cluster_bootstrap",
        },
    }
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "refa_vs_flagship_families.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    (outdir / "reproduction_gate_rows.json").write_text(
        json.dumps(repro, indent=1), encoding="utf-8")
    print(json.dumps({"alignment": align["passed"],
                      "reproduction": repro["passed"],
                      "max_abs_diff": out["reproduction_gate"]["max_abs_diff"],
                      "wrote": str(outdir / "refa_vs_flagship_families.json")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
