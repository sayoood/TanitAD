#!/usr/bin/env python3
"""RE-RUN every published closed-loop artifact through the decision-grade estimator.

WHY A RE-RUN AND NOT A RECOMPUTE
--------------------------------
`closedloop.run_and_save` never persisted the per-window paths — `collect()`
builds `closed_bike / closed_grnd / open_grnd / open_bike / cv / gt` as
`[N,4,2]` tensors and `analyze()` reduces them to scalars, but only the JSON of
scalars was written. The open-loop panel has `windows_<arm>.pt` dumps and can
therefore be re-scored offline; **the closed-loop panel has no such surface**.
So the deprecated statistic cannot be undone by arithmetic on the artifact — the
loop itself has to be re-driven.

That is why "the code was migrated but the artifacts were never re-run" was a
real, load-bearing gap and not a bookkeeping one.

THE SELF-VALIDATION THAT MAKES THE COMPARISON CLEAN
---------------------------------------------------
A re-run introduces a second difference besides the estimator: it is a NEW
forward pass. To prove the deltas below are the ESTIMATOR and not drift, the
migrated `analyze()` still emits the deprecated numbers under
`legacy_overlapping_holdout_se`, computed from the SAME split builder
(`gates.split_by_episode`) as the 2026-07-19 run. So for each arm we check:

    re-run's LEGACY block  ==  the PUBLISHED artifact      -> loop reproduced
    re-run's HEADLINE      vs  the PUBLISHED artifact      -> the correction

If the first holds, the second is attributable to the estimator alone.

WHAT THIS SCRIPT ALSO FIXES FOR THE FUTURE
------------------------------------------
It writes `clwin_<key>.pt` — the per-window closed-loop paths — so the next time
an estimator changes, this axis is a recompute (seconds, CPU) instead of a GPU
re-run. That absence is the reason this task existed.

Run:  OMP_NUM_THREADS=8 PYTHONPATH=/root/taniteval:/root/TanitAD/stack \
      python3 rerun_closedloop.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/root/taniteval")

from taniteval import closedloop as CL              # noqa: E402
from taniteval import data, loaders                 # noqa: E402
from taniteval.registry import MODELS               # noqa: E402

OUT = Path("/root/cl_rerun_20260726")
PUB = Path("/root/cl_rerun_20260726/published")     # the committed 2026-07-19 JSONs
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"

# The three arms with a COMMITTED closed-loop artifact.
ARMS = ["flagship-30k", "flagship-speed", "flagship-nospeed"]

# Headline scalars whose PUBLISHED value came out of the deprecated statistic.
# (block, key) -> where to read it in the analyze() result.
HEADLINES = [
    ("closed_bike", "ade_0_2s"), ("closed_bike", "fde@2s"),
    ("closed_bike", "ade@0.5s"), ("closed_bike", "ade@1s"),
    ("closed_bike", "ade@1.5s"),
    ("closed_grnd", "ade_0_2s"), ("open_grnd", "ade_0_2s"),
    ("open_bike", "ade_0_2s"), ("cv", "ade_0_2s"),
]


def collect_arm(key, device="cuda", episodes=40):
    """Mirror of `closedloop.run_and_save`'s collection half — but it RETURNS
    the window set instead of throwing it away."""
    entry = [m for m in MODELS if m["key"] == key]
    if not entry:
        raise SystemExit(f"unknown arm {key}")
    entry = entry[0]
    L = loaders.load(entry, device)
    model = L["model"]
    if not L["traj_capable"] or getattr(model, "tactical_policy", None) is None:
        raise SystemExit(f"{key}: not a 4-brain trajectory arm")
    files = data.list_val_episodes(VAL, episodes)
    if entry.get("train_ids"):
        from tanitad.data.mixing import load_episode
        tid = set(Path(entry["train_ids"]).read_text().split())
        files = [f for f in files
                 if str(load_episode(str(f), mmap=True).episode_id) not in tid]
    eps = (data.load_frames(files) if L["feed"] == "frames"
           else data.load_features(files, L["feed"], device))
    win = CL.collect(model, L["step_readout"], eps, device,
                     speed_input=bool(entry.get("speed_input")))
    return win, L, entry


def save_win(win, path):
    """Persist the per-window closed-loop surface (the thing that was missing)."""
    torch.save({k: (v if not torch.is_tensor(v) else v.cpu())
                for k, v in win.items()}, path)


def _get(res, block, key):
    return res["closedloop_ade_fde"]["heldout"][block][key]


def _get_legacy(res, block, key):
    return res[CL.LEGACY_BLOCK]["heldout"][block][key]


def compare(key, res):
    """published -> corrected -> ratio, plus the loop-reproduction check."""
    pubp = PUB / f"closedloop_{key}.json"
    pub = json.loads(pubp.read_text(encoding="utf-8")) if pubp.exists() else None
    rows, repro = [], []

    for block, mk in HEADLINES:
        new = _get(res, block, mk)
        leg = _get_legacy(res, block, mk)
        pv = (pub["closedloop_ade_fde"]["heldout"][block][mk]
              if pub else None)
        row = {
            "metric": f"{block}.{mk}",
            "published_mean": (round(pv["mean"], 4) if pv else None),
            "published_ci95": (round(pv["ci95"], 4) if pv else None),
            "published_estimator": "overlapping_holdout_se "
                                   "(labelled '8-split episode-disjoint jackknife')",
            "rerun_legacy_mean": leg["mean"], "rerun_legacy_ci95": leg["ci95"],
            "corrected_mean": new["mean"], "corrected_ci95": new["ci95"],
            "corrected_lo": new["lo"], "corrected_hi": new["hi"],
            "corrected_estimator": new["estimator"],
            "n_episodes": new["n_episodes"], "n_boot": new["n_boot"],
        }
        if pv:
            row["mean_shift_pct"] = round(
                100.0 * (pv["mean"] - new["mean"]) / max(1e-9, new["mean"]), 3)
            row["ci_width_ratio_corrected_over_published"] = (
                round(new["ci95"] / pv["ci95"], 3) if pv["ci95"] else None)
            # did the RE-RUN reproduce the 2026-07-19 loop?
            repro.append({
                "metric": row["metric"],
                "published": pv["mean"], "rerun_legacy": leg["mean"],
                "abs_diff": round(abs(pv["mean"] - leg["mean"]), 5),
                "reproduced": bool(abs(pv["mean"] - leg["mean"]) < 1e-3)})
        rows.append(row)

    # ---- deltas: compounding + divergence (PAIRED where two paths share windows)
    delta_rows = []
    for blk, pub_blk in (("compounding_error_grounded", "compounding_error_grounded"),
                         ("compounding_error_bicycle", "compounding_error_bicycle")):
        for h in ("0.5s", "1s", "1.5s", "2s"):
            k = f"delta@{h}"
            new = res[blk][k]
            leg = res[CL.LEGACY_BLOCK][blk][k]
            pv = pub[pub_blk][k] if pub else None
            r = {"metric": f"{blk}.{k}",
                 "published_mean": (pv["mean"] if pv else None),
                 "published_ci95": (pv["ci95"] if pv else None),
                 "published_separated": (pv.get("separated") if pv else None),
                 "rerun_legacy_mean": leg["mean"], "rerun_legacy_ci95": leg["ci95"],
                 "corrected_delta": new["delta"], "corrected_ci95": new["ci95"],
                 "corrected_lo": new["lo"], "corrected_hi": new["hi"],
                 "corrected_separated": new["separated"],
                 "corrected_estimator": new["estimator"],
                 "direction": new.get("direction")}
            if pv:
                r["mean_shift_pct"] = round(
                    100.0 * (pv["mean"] - new["delta"]) / max(1e-9, abs(new["delta"])), 3)
                r["ci_width_ratio"] = (round(new["ci95"] / pv["ci95"], 3)
                                       if pv["ci95"] else None)
                r["VERDICT_FLIP"] = bool(pv.get("separated") != new["separated"])
                repro.append({
                    "metric": r["metric"], "published": pv["mean"],
                    "rerun_legacy": leg["mean"],
                    "abs_diff": round(abs(pv["mean"] - leg["mean"]), 5),
                    "reproduced": bool(abs(pv["mean"] - leg["mean"]) < 1e-3)})
            delta_rows.append(r)

    dv_new = res["stability"]["divergence_rate_gt5m@2s"]
    # NOTE: in the legacy quarantine block the divergence rate is a TOP-LEVEL
    # key (`closedloop.py:772`), not nested under `stability` as it is in the
    # decision-grade block. Do not "tidy" this into a symmetric lookup.
    dv_leg = res[CL.LEGACY_BLOCK]["divergence_rate_gt5m@2s"]
    dv_pub = pub["stability"]["divergence_rate_gt5m@2s"] if pub else None
    div = {"metric": "stability.divergence_rate_gt5m@2s",
           "published_mean": (dv_pub["mean"] if dv_pub else None),
           "published_ci95": (dv_pub["ci95"] if dv_pub else None),
           "rerun_legacy_mean": dv_leg["mean"], "rerun_legacy_ci95": dv_leg["ci95"],
           "corrected_mean": dv_new["mean"], "corrected_ci95": dv_new["ci95"],
           "corrected_lo": dv_new["lo"], "corrected_hi": dv_new["hi"],
           "corrected_estimator": dv_new["estimator"]}
    if dv_pub:
        div["mean_shift_pct"] = round(
            100.0 * (dv_pub["mean"] - dv_new["mean"]) / max(1e-9, dv_new["mean"]), 3)
        div["ci_width_ratio"] = (round(dv_new["ci95"] / dv_pub["ci95"], 3)
                                 if dv_pub["ci95"] else None)
        repro.append({"metric": div["metric"], "published": dv_pub["mean"],
                      "rerun_legacy": dv_leg["mean"],
                      "abs_diff": round(abs(dv_pub["mean"] - dv_leg["mean"]), 5),
                      "reproduced": bool(abs(dv_pub["mean"] - dv_leg["mean"]) < 1e-3)})

    return {"arm": key, "published_artifact": str(pubp),
            "published_exists": pubp.exists(),
            "loop_reproduction": {
                "_what": "re-run's quarantined legacy block vs the 2026-07-19 "
                         "published numbers. If these match, the headline shift "
                         "below is the ESTIMATOR, not a new forward pass.",
                "all_reproduced": bool(repro and all(r["reproduced"] for r in repro)),
                "n_checked": len(repro),
                "n_reproduced": sum(1 for r in repro if r["reproduced"]),
                "rows": repro},
            "headline": rows, "deltas": delta_rows, "divergence": div}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allcmp = {}
    for key in ARMS:
        t0 = time.time()
        print(f"\n=== {key} ===", flush=True)
        wp = OUT / f"clwin_{key}.pt"
        if wp.exists():
            # The GPU pass is the expensive half and it is now PERSISTED, so a
            # re-analysis is free. This is the whole point of saving the dump.
            win = torch.load(wp, weights_only=False)
            L = {"step": None}
            entry = [m for m in MODELS if m["key"] == key][0]
            print(f"[{key}] reusing persisted clwin_{key}.pt "
                  f"({len(win['eid'])} windows) — no GPU pass", flush=True)
        else:
            win, L, entry = collect_arm(key)
            save_win(win, wp)
        res = CL.analyze(win)
        res["model"] = {k: entry.get(k) for k in
                        ("key", "name", "arch", "encoder", "speed_input")}
        res["ckpt_step"] = L["step"]
        res["wall_s"] = round(time.time() - t0, 1)
        res["_rerun"] = {
            "date": "2026-07-26",
            "reason": "the 2026-07-19 artifacts were produced by "
                      "overlapping_holdout_se (mislabelled '8-split "
                      "episode-disjoint jackknife'); the code was migrated "
                      "2026-07-25 but no closed-loop artifact had been re-run",
            "raw_surface_persisted": f"clwin_{key}.pt (per-window closed-loop "
                                     "paths — NOT persisted by the original run, "
                                     "which is why a GPU re-run was required)"}
        (OUT / f"closedloop_{key}.CORRECTED.json").write_text(
            json.dumps(res, indent=2, default=str), encoding="utf-8")
        c = compare(key, res)
        allcmp[key] = c
        lr = c["loop_reproduction"]
        cb = _get(res, "closed_bike", "ade_0_2s")
        print(f"[{key}] loop reproduced {lr['n_reproduced']}/{lr['n_checked']} "
              f"| closed_bike ade@2s corrected={cb['mean']} "
              f"[{cb['lo']},{cb['hi']}] ({cb['estimator']}) "
              f"| {res['wall_s']}s", flush=True)
    (OUT / "published_vs_corrected.json").write_text(
        json.dumps(allcmp, indent=2), encoding="utf-8")
    print("\nwrote published_vs_corrected.json")


if __name__ == "__main__":
    main()
