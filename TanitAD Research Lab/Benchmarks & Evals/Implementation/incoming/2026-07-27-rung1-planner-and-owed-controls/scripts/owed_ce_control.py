#!/usr/bin/env python3
"""OWED ROWS 52-65 — the `CE-CONTROL` family, re-adjudicated. ZERO GPU.

⭐ THE FINDING IS STRUCTURAL AND IT COMES OUT OF THE WHOLE ARTIFACT (M9), NOT THE
MATCHING NODE. `_folds_detail` records, per arm and per fold, the early-stopping
outcome. On several folds it is `best_step: 0` with
`finetune_helped_inner_val: false` — meaning the fine-tune was rolled back to its
initial state, so **on every window of those folds the arm IS the as-trained head,
bit for bit**. Folds are episode-disjoint (8 val episodes each), so those windows
contribute an EXACTLY ZERO paired delta to the contrast by construction.

Consequences, and they apply to all 14 rows:

  * `ce_control_minus_as_trained` is diluted by whole episode clusters of
    structural zeros. Its point estimate is attenuated by exactly the zero
    fraction, and its EFFECTIVE cluster count is smaller than the 40 the node
    reports — the interval was computed over 40 clusters of which several can
    only ever contribute 0.
  * `regret_minus_ce_control_ISOLATES_THE_LOSS` is worse: where BOTH arms
    early-stopped at 0 the contrast is identically zero, so the node that names
    itself as isolating the loss is partly measuring nothing at all.

That is the "a guard that cannot fail" class, reached by a route an MDE screen
cannot see, and it is measurable from the committed JSON with no GPU.

⛔ WHY THESE ROWS CANNOT BE RE-POWERED IN THIS TASK, MEASURED not asserted:
`_cache.bytes_on_gpu = 4,373,151,744` (4.07 GiB) for 6,844 windows over 40 val
episodes. The 600-episode val is 15x that = **61.1 GiB against an A40's 46 GiB**,
so the cached-feature design does not fit. The largest episode count that fits
with headroom is computed here. The rescorer additionally lives on the eval pod
(`/root/v4eval/stack`, `_stack_root`), which this task is forbidden to touch, and
going 40 -> N val episodes changes the CORPUS as well as n (registry §1.2a).

Usage:
    python owed_ce_control.py --bara <bar-a-selector dir> \
        --frozen <control-readjudication raw/frozen_list.json> --out <raw dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

A40_GIB = 46.0
HEADROOM_GIB = 6.0
N_VAL_FULL = 600


def resolve(js: dict, path: str):
    """`paired_intervals.ce_control_minus_as_trained.ade_0_2s` -> the node."""
    cur = js
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part.strip("[]"))]
        else:
            cur = cur[part]
    return cur


def zero_folds(fd: dict, arm: str):
    """Folds where `best_step == 0` — the arm is the as-trained head there."""
    return [r for r in fd.get(arm, []) if int(r["best_step"]) == 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bara", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bara = Path(a.bara).resolve()
    outd = Path(a.out).resolve()
    outd.mkdir(parents=True, exist_ok=True)

    frozen = json.load(open(a.frozen, encoding="utf-8"))
    # published §1.1 numbers the CE family 52-65 in prox-descending order, which
    # is the order `frozen_list.json` stores them in (asserted, not assumed).
    _ce = [r for r in frozen if r["family"].startswith("CE-CONTROL")]
    assert all(_ce[i]["prox"] >= _ce[i + 1]["prox"] for i in range(len(_ce) - 1)),         "frozen_list CE rows are not prox-descending; the published row numbers "         "cannot be recovered by position"
    rows = [(52 + i, r) for i, r in enumerate(_ce)]
    src = {"produced": json.load(open(bara / "raw" / "bar_a_produced.json",
                                      encoding="utf-8")),
           "oracle": json.load(open(bara / "raw" / "bar_a_oracle.json",
                                    encoding="utf-8"))}

    out = {
        "block": "owed_controls/CE-CONTROL",
        "frozen_list_rows": [i for i, _ in rows],
        "estimator": {"interval": "paired_episode_cluster_bootstrap",
                      "module": "taniteval.ci (as recorded on every node)",
                      "n_boot": 2000, "resampling_unit": "val episode"},
        "structural_zeros": {},
        "repower_blocker": {},
        "rows": {},
        "fidelity_vs_committed": {"all_nodes_resolve": True, "mismatches": []},
    }

    # ---- ⛔ the re-power blocker, MEASURED ----------------------------------- #
    cache = src["produced"]["_cache"]
    gib40 = cache["bytes_on_gpu"] / 2 ** 30
    out["repower_blocker"] = {
        "measured_bytes_on_gpu_at_40_val_episodes": cache["bytes_on_gpu"],
        "gib_at_40_episodes": round(gib40, 3),
        "n_cache_windows_at_40_episodes": cache["n_windows"],
        "stride": cache["stride"],
        "cache_build_seconds_per_goal_mode": cache["wallclock_s"],
        "projected_gib_at_600_episodes": round(gib40 * N_VAL_FULL / 40.0, 1),
        "a40_vram_gib": A40_GIB,
        "fits_at_600": bool(gib40 * N_VAL_FULL / 40.0 <= A40_GIB - HEADROOM_GIB),
        "max_episodes_that_fit": int((A40_GIB - HEADROOM_GIB) / gib40 * 40.0),
        "other_blockers": [
            "the rescorer's stack root is the EVAL POD (`_stack_root` = "
            "/root/v4eval/stack); this task's mandate is pod2 only",
            "the checkpoint is flagship-v4-fromscratch-30k, not the v1 this "
            "task's other job used",
            "going 40 -> N val episodes changes the CORPUS as well as n "
            "(MODEL_REGISTRY.md 1.2a: the 600-ep build is a measurably easier "
            "deployment, never substituted for the 40-ep canonical)",
        ],
        "_read": ("re-powering these 14 rows is a real, costed job on a "
                  "different host with a re-architected cache -- it is OWED, "
                  "and saying so is the honest verdict, not 'checked'"),
    }
    print(f"[repower] {gib40:.2f} GiB at 40 eps -> "
          f"{gib40 * 15:.1f} GiB at 600 (A40 has {A40_GIB}); "
          f"max fitting episodes ~{out['repower_blocker']['max_episodes_that_fit']}",
          flush=True)

    # ---- ⭐ the structural zeros -------------------------------------------- #
    for mode, js in src.items():
        fd = js["_folds_detail"]
        n_win_total = js["_cache"]["n_eval_windows"]
        blk = {}
        for arm in ("ce", "regret"):
            zf = zero_folds(fd, arm)
            nw = sum(int(r["n_test_windows"]) for r in zf)
            blk[arm] = {
                "folds_early_stopped_at_step_0": [int(r["fold"]) for r in zf],
                "n_folds_zero": len(zf), "n_folds": len(fd[arm]),
                "n_windows_structurally_as_trained": nw,
                "frac_windows": round(nw / n_win_total, 4),
                "n_episode_clusters_structurally_as_trained": 8 * len(zf),
                "effective_clusters": 40 - 8 * len(zf),
            }
        both = sorted(set(blk["ce"]["folds_early_stopped_at_step_0"])
                      & set(blk["regret"]["folds_early_stopped_at_step_0"]))
        nw_both = sum(int(r["n_test_windows"]) for r in fd["ce"]
                      if int(r["fold"]) in both)
        blk["both_arms_zero"] = {
            "folds": both, "n_windows_identically_zero": nw_both,
            "frac_windows": round(nw_both / n_win_total, 4),
            "n_episode_clusters": 8 * len(both),
            "effective_clusters_for_regret_minus_ce": 40 - 8 * len(both),
            "_read": ("on these windows `regret - ce_control` is EXACTLY 0 by "
                      "construction -- the node that names itself "
                      "ISOLATES_THE_LOSS is measuring nothing there"),
        }
        out["structural_zeros"][mode] = blk
        print(f"[zeros/{mode}] ce {blk['ce']['frac_windows']:.1%} windows, "
              f"regret {blk['regret']['frac_windows']:.1%}, both "
              f"{blk['both_arms_zero']['frac_windows']:.1%}", flush=True)

    # ---- the 14 rows --------------------------------------------------------- #
    for idx, r in rows:
        mode = "produced" if "bar_a_produced" in r["file"] else "oracle"
        js = src[mode]
        node = resolve(js, r["path"])
        if abs(float(node["delta"]) - float(r["effect"])) > 5e-4:
            out["fidelity_vs_committed"]["all_nodes_resolve"] = False
            out["fidelity_vs_committed"]["mismatches"].append(r["path"])
        group = r["path"].split(".")[1]
        metric = r["path"].split(".")[-1]
        z = out["structural_zeros"][mode]
        if group == "ce_control_minus_as_trained":
            f0 = z["ce"]["frac_windows"]
            eff = z["ce"]["effective_clusters"]
            # the effect this control exists NOT to be confused with:
            # the treatment contrast on the SAME metric
            twin = resolve(js, f"paired_intervals."
                               f"regret_minus_ce_control_ISOLATES_THE_LOSS.{metric}") \
                if metric in resolve(js, "paired_intervals."
                                         "regret_minus_ce_control_ISOLATES_THE_LOSS") else None
        else:
            f0 = z["both_arms_zero"]["frac_windows"]
            eff = z["both_arms_zero"]["effective_clusters_for_regret_minus_ce"]
            twin = resolve(js, f"paired_intervals.ce_control_minus_as_trained.{metric}") \
                if metric in resolve(js, "paired_intervals.ce_control_minus_as_trained") \
                else None
        mde = float(node["ci95"])
        eff_size = abs(float(twin["delta"])) if twin else None
        att = (round(float(node["delta"]) / (1.0 - f0), 6) if f0 < 1.0 else None)
        out["rows"][str(idx)] = {
            "goal_mode": mode, "node": r["path"], "metric": metric, "group": group,
            "committed": {k: node[k] for k in ("delta", "lo", "hi", "ci95",
                                               "separated", "n_windows",
                                               "n_episodes", "estimator")},
            "structural_zero_fraction_of_windows": f0,
            "effective_episode_clusters": eff,
            "reported_episode_clusters": int(node["n_episodes"]),
            "attenuation_corrected_point_estimate": att,
            "attenuation_corrected_shift": (None if att is None else
                                            round(att - float(node["delta"]), 6)),
            "_attenuation_note": ("EXACT arithmetic, not a projection: the mean "
                                  "over all windows equals (1 - f0) x the mean "
                                  "over the windows where the fine-tune actually "
                                  "differed, because the rest are identically "
                                  "zero. ⚠️ The INTERVAL cannot be recomputed "
                                  "without per-window data, which is on the eval "
                                  "pod -- so the corrected point estimate is "
                                  "MEASURED and its interval is OWED."),
            "power_ceiling": {
                "mde": round(mde, 6),
                "companion_contrast_it_must_be_distinguished_from": (
                    None if twin is None else
                    ("regret_minus_ce_control_ISOLATES_THE_LOSS"
                     if group == "ce_control_minus_as_trained"
                     else "ce_control_minus_as_trained")),
                "companion_effect_size": eff_size,
                "mde_as_pct_of_companion_effect": (
                    None if not eff_size else round(100.0 * mde / eff_size, 1)),
                "sharp_enough": (None if not eff_size else bool(eff_size > mde)),
            },
            "VERDICT": ("OWED — NOT RE-POWERABLE HERE"
                        if f0 == 0.0 else
                        "UNDER-POWERED (OWED) + STRUCTURALLY ATTENUATED"),
            "why": (f"{f0:.1%} of the windows ({40 - eff} of 40 episode "
                    f"clusters) are identically zero by construction because the "
                    f"fine-tune early-stopped at step 0 on those folds; the "
                    f"reported n_episodes = {int(node['n_episodes'])} overstates "
                    f"the clusters that can carry signal, which is "
                    f"{eff}"),
        }
        print(f"[row {idx:>2}] {mode:8s} {group[:28]:28s} {metric[:22]:22s} "
              f"delta={float(node['delta']):+.4f} f0={f0:.1%} eff_n={eff} "
              f"-> {out['rows'][str(idx)]['VERDICT']}", flush=True)

    (outd / "owed_ce_control.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"[write] {outd / 'owed_ce_control.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
