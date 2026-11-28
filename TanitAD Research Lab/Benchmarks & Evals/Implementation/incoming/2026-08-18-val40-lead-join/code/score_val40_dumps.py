#!/usr/bin/env python3
"""Score DISTANCE-KEEPING on the banked val40 tier-0 dumps, via the landed wiring:
`taniteval.dump_lead_join.attach_lead` (val40 agents join + poses + clip map) ->
`win["lead"]` -> `taniteval.four_families` / `taniteval.lead_metrics`.

⛔ TIER STAMP ON EVERY NUMBER: **T0 — teacher-forced WM diagnostic, NEVER "driving
performance"** (EVAL_DOCTRINE.md). These dumps are open-loop, true-future-conditioned.

Per dump (= per arm):
  * attach_lead -> lead block (saved durable, with its coverage.json — per-episode
    status/reason, the binding per-family absence reporting)
  * distance_keeping for THREE paths on the same lead block: the arm's `pred`, the
    human `gt`, and the dump's banked canonical `cv` floor (baseline_waypoints
    constant_velocity — the same floor every driving_*.json headline uses)
  * marginal CIs: episode-cluster bootstrap per metric (taniteval.ci) — NEVER
    overlapping_holdout_se
  * paired deltas on jointly-valid windows: arm-CV, GT-arm, GT-CV
    (lead_metrics.paired_distance_keeping -> paired_episode_cluster_bootstrap)
  * the FULL binding four-family block for the arm (four_families.all_families with
    win["lead"], tier="T0") banked per arm — LONGITUDINAL carries distance_keeping
    + by_speed strata; TACTICAL/STRATEGIC report their reasons on this surface.
Cross-arm: any two dumps whose `gt`/`eid` are bit-identical share the surface; the
requested pairs are scored with the PAIRED estimator (e.g. flagship-30k vs refc-base-30k).

CPU-only; touches no episode cache; reads dumps + join + poses, writes JSON + lead .pt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
for p in (REPO / "taniteval", REPO / "stack", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TIER = ("T0 — teacher-forced WM diagnostic (open-loop, true-future-conditioned). "
        "NEVER 'driving performance'.")


def _san(x, drop=("_per_window",)):
    """JSON-safe deep copy: numpy/torch scalars -> py, big arrays dropped by name."""
    import torch
    if isinstance(x, dict):
        return {k: _san(v, drop) for k, v in x.items() if k not in drop}
    if isinstance(x, (list, tuple)):
        return [_san(v, drop) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 64 else {"_array_dropped": list(x.shape)}
    if torch.is_tensor(x):
        return x.tolist() if x.numel() <= 64 else {"_tensor_dropped": list(x.shape)}
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _dk_summary(dk: dict) -> dict:
    keep = ("status", "reason", "n", "n_windows", "dt_s", "mean_headway_min_m",
            "mean_time_gap_min_s", "n_time_gap", "mean_min_ttc_s", "n_closing",
            "gap_convention", "ttc_cap_s")
    return {k: dk[k] for k in keep if k in dk}


def _marginal_cis(dk: dict, eid, n_boot, seed) -> dict:
    from taniteval.ci import episode_cluster_bootstrap
    out = {}
    for key in ("headway_min_m", "time_gap_min_s", "min_ttc_s"):
        v = np.asarray(dk[key], dtype=np.float64)
        ok = np.isfinite(v)
        if not ok.any():
            out[key] = {"n": 0, "status": "NOT-APPLICABLE"}
            continue
        r = episode_cluster_bootstrap(v[ok], list(np.asarray(eid, dtype=object)[ok]),
                                      reduce="mean", n_boot=n_boot, seed=seed)
        r["n"] = int(ok.sum())
        out[key] = r
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("score_val40_dumps")
    ap.add_argument("--results-dir", default=str(REPO / "taniteval" / "results"))
    ap.add_argument("--agents", default="C:/Users/Admin/tanitad-caches/"
                    "val40-obstacle-20260818/join/val40_agents.jsonl")
    ap.add_argument("--epdir", default="C:/Users/Admin/tanitad-caches/"
                    "val40-poses-20260818/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--clip-map", default=str(Path(__file__).resolve().parents[1]
                                              / "raw" / "val40_clipmap.json"))
    ap.add_argument("--lead-out", default="C:/Users/Admin/tanitad-caches/"
                    "val40-obstacle-20260818/leadblocks")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keys", nargs="*", default=None,
                    help="dump keys (default: every windows_*.pt)")
    ap.add_argument("--reference", default="flagship-30k",
                    help="dump whose gt/eid define the shared surface")
    ap.add_argument("--pairs", nargs="*",
                    default=["flagship-30k:refc-base-30k"],
                    help="cross-dump paired contrasts a:b (pred_a - pred_b)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-families", action="store_true",
                    help="panel only (no all_families block per arm)")
    a = ap.parse_args(argv)
    t0 = time.time()

    import torch
    from taniteval import dump_lead_join as dlj
    from taniteval import four_families as ff
    from taniteval import lead_source as ls
    from taniteval.lead_metrics import distance_keeping, paired_distance_keeping
    from taniteval.rollout import load_windows

    out_dir = Path(a.out_dir)
    (out_dir / "families").mkdir(parents=True, exist_ok=True)
    lead_out = Path(a.lead_out)
    lead_out.mkdir(parents=True, exist_ok=True)

    results = Path(a.results_dir)
    keys = a.keys or sorted(p.name[len("windows_"):-len(".pt")]
                            for p in results.glob("windows_*.pt"))
    if a.reference in keys:                       # reference first, then the rest
        keys = [a.reference] + [k for k in keys if k != a.reference]

    episodes = dlj.episodes_from_epdir(a.epdir)
    cm = json.loads(Path(a.clip_map).read_text(encoding="utf-8"))
    for kk, cid in cm.items():
        if int(kk) in episodes:
            episodes[int(kk)]["clip_id"] = str(cid)
    joins = dlj.read_agents_jsonl(a.agents)
    print(f"[score] join clips {len(joins)} · episodes {len(episodes)} · "
          f"dumps {len(keys)}", flush=True)

    panel: dict[str, dict] = {}
    ref_gt = ref_eid = None
    ref_dks: dict[str, dict] = {}      # key -> dk of the arm's pred, for cross pairs
    ref_shared: dict[str, bool] = {}

    for key in keys:
        t1 = time.time()
        win = load_windows(results / f"windows_{key}.pt")
        lead = dlj.attach_lead(win, episodes, joins,
                               n_boot=a.n_boot, seed=a.seed)
        cov = lead["coverage"]
        blk_p = lead_out / f"lead_{key}.pt"
        torch.save(lead, str(blk_p))
        Path(str(blk_p) + ".coverage.json").write_text(
            json.dumps({"coverage": _san(cov), "counts": _san(lead["counts"]),
                        "conventions": lead["conventions"], "tier": TIER},
                       indent=1, default=str), encoding="utf-8")

        eid = list(win["eid"])
        n_w = len(eid)
        dt_dk = float(lead["dt_s"]) if lead["dt_s"] else 0.5
        gt = torch.as_tensor(win["gt"]).float().numpy()
        pred = torch.as_tensor(win["pred"]).float().numpy()
        cv = (torch.as_tensor(win["cv"]).float().numpy()
              if win.get("cv") is not None else None)

        arms = {"pred": pred, "gt": gt}
        if cv is not None:
            arms["cv"] = cv
        dks = {nm: distance_keeping(p, lead["leads"], lead["lead_lens"],
                                    lead["speeds"], dt_dk)
               for nm, p in arms.items()}

        # shared-surface bookkeeping vs the reference dump
        if key == a.reference:
            ref_gt, ref_eid = gt, eid
            shared = True
        else:
            shared = (ref_gt is not None and gt.shape == ref_gt.shape
                      and np.array_equal(gt, ref_gt) and eid == ref_eid)
        ref_shared[key] = shared
        ref_dks[key] = dks["pred"]

        ent = {
            "tier": TIER,
            "n_windows": n_w,
            "n_episodes": cov["n_episodes"],
            "n_episodes_ok": cov["n_episodes_ok"],
            "window_states": _san(lead["counts"]),
            "coverage_fraction_lead": round(lead["counts"][ls.LEAD] / n_w, 4),
            "dt_dk_s": dt_dk,
            "grid_dt_s_median": cov["grid_dt_s_median"],
            "gt_matches_reference": bool(shared),
            "speed_check_max_mps": max((c.get("speed_check_max_mps") or 0.0)
                                       for c in cov["episodes"].values()),
            "episodes_not_ok": {e: {"status": c.get("status"),
                                    "n_windows": c.get("n_windows")}
                                for e, c in cov["episodes"].items()
                                if c.get("status") != dlj.EP_OK},
            "arms": {}, "paired": {},
        }
        for nm, dk in dks.items():
            ent["arms"][nm] = {
                **_dk_summary(dk),
                "kept_lead_of_LEAD": (f"{dk['n']}/{lead['counts'][ls.LEAD]}"),
                "ci": _marginal_cis(dk, eid, a.n_boot, a.seed),
            }
        if cv is not None:
            ent["paired"]["pred_minus_cv"] = _san(paired_distance_keeping(
                dks["pred"], dks["cv"], eid, names=(key, "cv"),
                n_boot=a.n_boot, seed=a.seed))
            ent["paired"]["gt_minus_cv"] = _san(paired_distance_keeping(
                dks["gt"], dks["cv"], eid, names=("gt", "cv"),
                n_boot=a.n_boot, seed=a.seed))
        ent["paired"]["gt_minus_pred"] = _san(paired_distance_keeping(
            dks["gt"], dks["pred"], eid, names=("gt", key),
            n_boot=a.n_boot, seed=a.seed))

        if not a.skip_families:
            win["lead"] = lead
            fam = ff.all_families(win, tier="T0", n_boot=a.n_boot, seed=a.seed)
            (out_dir / "families" / f"families_{key}.json").write_text(
                json.dumps(_san(fam), indent=1, default=str), encoding="utf-8")
            dk_f = fam["longitudinal"]["distance_keeping"]
            ent["families_file"] = f"families/families_{key}.json"
            ent["four_families_dk_status"] = dk_f.get("status")
            ent["longitudinal_claim_admissible"] = fam.get(
                "_longitudinal_claim_admissible")
            # consistency: the four_families dk means must equal the panel's
            for k_chk in ("mean_headway_min_m", "mean_time_gap_min_s",
                          "mean_min_ttc_s"):
                pv, fv = ent["arms"]["pred"].get(k_chk), dk_f.get(k_chk)
                if pv is not None and fv is not None and \
                        abs(float(pv) - float(fv)) > 5e-3:
                    ent.setdefault("_warn", []).append(
                        f"four_families {k_chk} {fv} != panel {pv} "
                        f"(dt {dk_f.get('dt_s')} vs {dt_dk})")

        panel[key] = ent
        print(f"[score] {key}: LEAD {lead['counts'][ls.LEAD]} NO_LEAD "
              f"{lead['counts'][ls.NO_LEAD]} NO_LABEL {lead['counts'][ls.NO_LABEL]}"
              f" · pred n {dks['pred']['n']} headway "
              f"{dks['pred'].get('mean_headway_min_m')} · {time.time()-t1:.1f}s",
              flush=True)

    # ---- requested cross-dump paired contrasts (same surface only) --------- #
    cross = {}
    for pair in a.pairs:
        an, bn = pair.split(":")
        if an not in ref_dks or bn not in ref_dks:
            cross[pair] = {"status": "SKIPPED", "reason": "dump not scored"}
            continue
        if not (ref_shared.get(an) and ref_shared.get(bn)):
            cross[pair] = {"status": "REFUSED",
                           "reason": "gt/eid not bit-identical across the two "
                                     "dumps — not the same surface, pairing "
                                     "would be invalid"}
            continue
        cross[pair] = _san(paired_distance_keeping(
            ref_dks[an], ref_dks[bn], ref_eid, names=(an, bn),
            n_boot=a.n_boot, seed=a.seed))
    out = {
        "_what": "val40 distance-keeping panel over the banked tier-0 dumps",
        "_tier": TIER,
        "_estimator": "episode_cluster_bootstrap / paired_episode_cluster_bootstrap "
                      "(taniteval.ci) — NEVER overlapping_holdout_se",
        "_lead_block": "attach_lead over the val40 agents join "
                       "(build_val40_join_local.py); QUERY_EPS_S fixture-pinned",
        "_inputs": {"agents": a.agents, "epdir": a.epdir, "clip_map": a.clip_map,
                    "n_boot": a.n_boot, "seed": a.seed},
        "arms": panel,
        "cross_arm_paired": cross,
        "wall_s": round(time.time() - t0, 1),
    }
    (out_dir / "val40_dk_panel.json").write_text(
        json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"[score] panel -> {out_dir / 'val40_dk_panel.json'} "
          f"({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
