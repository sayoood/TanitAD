#!/usr/bin/env python3
"""k_sweep_envelope.py — the K-SWEEP with the ENVELOPE GUARD (T1, §9.8).

WHAT THIS ANSWERS
-----------------
`POD2_EVAL_HOST.md` §4.5 recommends registering the next gate's closed-loop
co-primary at **K = 60 (6.0 s) primary / K = 70 hard maximum / K = 185
report-only-pooled**. That recommendation rests on MEASURED stratum yield. Its
*envelope* clause — "a 6 s rollout accumulates far less lateral drift than an
18.5 s one, so K = 60 is *likely* to sit inside the envelope" — is an explicitly
labelled **HYPOTHESIS**. This driver measures it.

THE INSTRUMENTS ARE THE PACKAGED ONES, ON PURPOSE
-------------------------------------------------
`taniteval.clhorizon.corridor_rollout` (K free) and `taniteval.ood.verdict`
(E1a's FULL disjunction). Escalation #8 of the standup was that these two
modules and the horizon recommendation are halves of one step and neither
stream knew about the other. This file is that step. Nothing is re-implemented:
`clhorizon.corridor_rollout` is pinned bit-identical to the `incoming/` driver
that produced the committed gate co-primary by
`taniteval/tests/test_clhorizon.py::test_port_is_tensor_identical_to_the_driver`.

DESIGN
------
* ONE process: one checkpoint load, one goal mint, one episode load, then K
  swept in **priority order** so a killed run still yields value:
      185 (reproduces the committed gate co-primary 0.6388 — harness check)
   -> 20  (reproduces the committed 0.0203               — harness check)
   -> 60, 70   (the recommendation and its hard maximum)
   -> 90, 120, 150 (the shape of the curve between them)
* Every K writes its JSON and its per-window tensors to disk IMMEDIATELY.
* Every K also emits the **truncation curves**: for every k <= K, the corridor
  departure rate and the out-of-envelope fractions over `lat[:, :k]`. Free (no
  GPU), and it gives the envelope-departure curve at 0.1 s resolution on a FIXED
  window set.
  WARNING, stated because it is a real difference and is measured in the report:
  the first k steps of a K-step rollout are NOT identical to a k-step rollout.
  The window SET differs (a k-rollout has more starts) and the nearest-reference
  search runs over `K+1` reference poses rather than `k+1`. The truncation curve
  is a within-rollout curve; the sweep's own K points are the measured ones.
* Nothing is pooled across surfaces. Every block is `surface="closed_loop"`.

EVIDENCE / ESTIMATOR
--------------------
episode_cluster_bootstrap (taniteval/ci.py) B=2000, unit = val EPISODE; paired
form for every K-vs-K delta. `overlapping_holdout_se` is used NOWHERE.

GOAL PROVENANCE
---------------
`--goal-mode oracle` matches the registered gate co-primary exactly, and is
stamped ORACLE per GATE_PROTOCOL §0.8. This number is therefore NOT a
deployed-capability claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def truncation_curves(pw, K, eid, halfwidths, env_lat, env_yaw, ks):
    """CDR + out-of-envelope fractions as a function of elapsed steps k <= K.

    Pure arithmetic on the per-window tensors already in hand — no GPU."""
    lat = pw["lat"].numpy() if torch.is_tensor(pw["lat"]) else np.asarray(pw["lat"])
    yaw = pw["yaw"].numpy() if torch.is_tensor(pw["yaw"]) else np.asarray(pw["yaw"])
    out_lat = lat > env_lat
    out_yaw = yaw > env_yaw
    out_any = out_lat | out_yaw
    rows = []
    for k in ks:
        if k > K:
            continue
        L, OA, OL, OY = lat[:, :k], out_any[:, :k], out_lat[:, :k], out_yaw[:, :k]
        row = {"k": int(k), "s": round(k * 0.1, 2),
               "frac_steps_lat_over_3m": round(float(OL.mean()), 5),
               "frac_steps_yaw_over_12deg": round(float(OY.mean()), 5),
               "frac_steps_any": round(float(OA.mean()), 5),
               "frac_windows_any_step_out_of_envelope":
                   round(float(OA.any(1).mean()), 4),
               "mean_peak_xte_m": round(float(L.max(1).mean()), 4),
               "p90_peak_xte_m": round(float(np.percentile(L.max(1), 90)), 4)}
        for t in halfwidths:
            row[f"cdr_{t:g}"] = round(float((L > t).mean(1).mean()), 5)
            row[f"windep_{t:g}"] = round(float((L > t).any(1).mean()), 5)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser("k_sweep_envelope")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--anchors-dense", required=True)
    ap.add_argument("--head-config", default=None)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--p1-json", required=True)
    ap.add_argument("--horizons", default="185,20,60,70,90,120,150")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--corridor-grid", default="1.0,1.75,2.5")
    ap.add_argument("--corridor-halfwidth", type=float, default=1.75)
    ap.add_argument("--junction-deg", type=float, default=10.0)
    ap.add_argument("--goal-mode", default="oracle")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stack", default="/root/TanitAD/stack")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    for p in (a.stack, str(Path(a.stack) / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from taniteval import clhorizon as CH
    from taniteval import corridor as _corr
    from taniteval import ci as _ci
    from taniteval import data as _data
    from taniteval import ood as _ood
    from tanitad.instruments.numerics import strict_numerics
    import goal_modes
    from eval_flagship_v4 import load_v4_from_ck

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    Ks = [int(x) for x in a.horizons.split(",")]
    grid = tuple(float(x) for x in a.corridor_grid.split(","))
    prim = float(a.corridor_halfwidth)
    assert prim in grid, (prim, grid)
    om = _ood.OODMap(a.p1_json)

    files = _data.list_val_episodes(a.val_dir, a.episodes)
    # ⚠️ `_data.load_raw`, NOT `_data.load_frames`. `load_frames` wraps each
    # episode in `RawEp`, which exposes the frames as `.feats` — while
    # `clhorizon._default_frames` reads `ep.frames`. `clhorizon.run_v4` pairs
    # `load_frames` with that default and therefore raises AttributeError on
    # the first rollout step (reported; defect D2 of this run). `load_raw` is
    # what the committed gate driver used, so this is also the surface-matching
    # choice, not merely the working one.
    episodes = _data.load_raw(files)
    Ts = [int(e.poses.shape[0]) for e in episodes]
    surv = {K: int(sum(1 for T in Ts if T - CH.W - K >= 1)) for K in Ks}
    nwin = {K: int(sum(len(range(0, T - CH.W - K, a.stride)) for T in Ts))
            for K in Ks}
    print(f"[ksweep] {len(episodes)} eps | T in [{min(Ts)},{max(Ts)}] | "
          f"K order {Ks} | dev {device}", flush=True)
    print(f"[ksweep] windows at stride {a.stride}: {nwin}", flush=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    world, grounding, head, step, hcfg, goal_head = load_v4_from_ck(
        ck, device,
        head_config_path=(a.head_config or Path(a.ckpt).parent / "config.json"),
        anchors_dense_path=a.anchors_dense)
    del ck
    planner = CH.V4Planner(world, head, goal_head, a.goal_mode, goal_modes)

    t0 = time.time()
    goals = CH._v4_goal_cache(episodes, stack_paths=(a.stack,
                                                     str(Path(a.stack) / "scripts")))
    for i in range(len(episodes)):
        goals._build(i)
    print(f"[ksweep] goal oracle minted for {goals.n_total} indices in "
          f"{time.time() - t0:.0f}s; refusals {goals.n_fail}", flush=True)

    head_json = outdir / "ksweep_header.json"
    res = {
        "_experiment": ("K-SWEEP with the ENVELOPE GUARD — closed-loop "
                        "corridor_departure_rate + E1a's FULL OOD disjunction, "
                        "K in " + str(Ks)),
        "_evidence_class": "MEASURED (ours; artifacts = this dir)",
        "_instruments": {
            "rollout": "taniteval.clhorizon.corridor_rollout (K free; pinned "
                       "bit-identical to the incoming/ driver that produced the "
                       "committed 30k gate co-primary)",
            "emitter": "taniteval.corridor.stratified via clhorizon.emit",
            "ood": "taniteval.ood.verdict — E1a's FULL disjunction (ratio > 1.5x "
                   "OR steps leave the MEASURED envelope), with "
                   "assert_envelope_verdict_consistent armed"},
        "_estimator": ("episode_cluster_bootstrap (taniteval/ci.py) B=%d, unit = "
                       "val EPISODE; PAIRED form for K-vs-K deltas. "
                       "overlapping_holdout_se is used NOWHERE." % a.n_boot),
        "_surface": "closed_loop (real-footage-in-the-loop). NEVER pooled with "
                    "the open-loop dense surface.",
        "ckpt": a.ckpt, "ckpt_md5": _md5(a.ckpt), "ckpt_step": step,
        "anchors_dense": a.anchors_dense, "anchors_md5": _md5(a.anchors_dense),
        "head_cfg": {"n_anchors": hcfg.n_anchors, "horizons": list(hcfg.horizons),
                     "factorised": hcfg.factorised,
                     "cond_vtarget": hcfg.cond_vtarget,
                     "cond_route": hcfg.cond_route,
                     "cond_imagination": hcfg.cond_imagination},
        "goal_provenance": goal_modes.provenance(a.goal_mode, cfg=head.cfg),
        "goal_index_policy": ("FOLLOW — the oracle goal is re-minted at EVERY "
                              "rollout step at the reference index the model "
                              "observes (t0 + mstar + W - 1)."),
        "goal_labeler_refusals": goals.n_fail,
        "val_dir": a.val_dir, "val_parity": _data.last_val_parity(),
        "n_episodes": len(episodes),
        "episode_T_min": min(Ts), "episode_T_max": max(Ts),
        "windows_at_each_horizon": nwin,
        "episodes_surviving_each_horizon": surv,
        "stride": a.stride, "batch": a.batch, "n_boot": a.n_boot,
        "corridor_thresholds_m": list(grid), "corridor_primary_m": prim,
        "junction_deg": a.junction_deg,
        "p1_envelope": a.p1_json, "p1_baseline_ade2s": om.base,
        "envelope_constants": {"lat_max_m": _ood.ENV_LAT_MAX,
                               "yaw_max_deg": _ood.ENV_YAW_MAX,
                               "provenance": "P1 MEASURED (lowood_flagship_ci"
                                             ".json) on the flagship v1 arm — "
                                             "NOT on v4"},
        "ratio_supremum_analysis": {
            "_what": ("the EXACT supremum of OODMap.ratio_arr over ALL possible "
                      "inputs, computed from the envelope JSON alone"),
            "sup_ratio": None, "lat_max_excess": None, "yaw_max_excess": None},
        "harness_checks_expected": {
            "K185_overall_cdr": 0.6388, "K185_n_win": 41,
            "K20_overall_cdr": 0.0203, "K20_n_win": 881,
            "_source": ("committed gate artifact coprimary/"
                        "corridor_v4_30k_K185.json, same ckpt md5")},
        "results": {},
    }
    ex_l = max(0.0, float((om.lat_y.max() - om.base) / om.base))
    ex_y = max(0.0, float((om.yaw_y.max() - om.base) / om.base))
    res["ratio_supremum_analysis"].update(
        sup_ratio=round(1.0 + ex_l + ex_y, 6),
        lat_max_excess=round(ex_l, 6), yaw_max_excess=round(ex_y, 6),
        can_ever_exceed_1p30=bool((1 + ex_l + ex_y) > 1.30),
        can_ever_exceed_1p50=bool((1 + ex_l + ex_y) > 1.50),
        _read=("OODMap.ratio_arr is 1 + clip((interp(lat)-base)/base,0,inf) + "
               "clip((interp(yaw)-base)/base,0,inf) and np.interp CLAMPS, so "
               "the ratio has a HARD CEILING. Any test of the form "
               "`ratio <= 1.30` or `ratio > 1.5` against this map is decided "
               "before the model runs."))
    head_json.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"[ksweep] ratio_arr SUPREMUM = {res['ratio_supremum_analysis']['sup_ratio']}"
          f"  (can exceed 1.30? "
          f"{res['ratio_supremum_analysis']['can_ever_exceed_1p30']}; "
          f"1.50? {res['ratio_supremum_analysis']['can_ever_exceed_1p50']})",
          flush=True)

    CURVE_KS = sorted(set(list(range(1, 21)) + list(range(20, 191, 5))))
    per_K = {}
    with strict_numerics():
        for K in Ks:
            t = time.time()
            pw = CH.corridor_rollout(planner, episodes, goals, device, K,
                                     stride=a.stride, batch=a.batch)
            if pw is None:
                res["results"][str(K)] = {"_skipped": "no window at this horizon"}
                continue
            blk = CH.emit(pw, K, ood_map=om, thresholds=grid, primary=prim,
                          junction_deg=a.junction_deg, surface="closed_loop",
                          n_boot=a.n_boot, seed=0)
            blk["truncation_curve"] = truncation_curves(
                pw, K, pw["eid"], grid, _ood.ENV_LAT_MAX, _ood.ENV_YAW_MAX,
                CURVE_KS)
            blk["wall_s"] = round(time.time() - t, 1)
            res["results"][str(K)] = blk
            per_K[K] = pw
            o, oo = blk["overall"], blk["ood"]["overall"]
            print(f"[ksweep] K={K:4d} ({K * 0.1:5.1f}s) n_win={o['n_windows']:5d} "
                  f"n_ep={o['n_episodes']:3d} "
                  f"CDR@{prim:g}={o['corridor_departure_rate']['mean']:.4f} "
                  f"[{o['corridor_departure_rate']['lo']:.4f},"
                  f"{o['corridor_departure_rate']['hi']:.4f}] "
                  f"peakXTE={o['peak_xte_m']['mean']:.3f} | OOD: "
                  f"steps_out={oo['EXTRAPOLATION_frac_steps_any']:.4f} "
                  f"win_out={oo['EXTRAPOLATION_frac_windows_any_step_out_of_envelope']:.4f} "
                  f"ratio={oo['ood_peak_ratio']['mean']:.4f} "
                  f"c1={oo['criterion_1_ratio_over_1p5']['fires']} "
                  f"c2={oo['criterion_2_steps_outside_measured_envelope']['fires']} "
                  f"-> {oo['EXTRAPOLATION_VERDICT'][:24]} ({blk['wall_s']:.0f}s)",
                  flush=True)
            (outdir / "ksweep_results.json").write_text(
                json.dumps(res, indent=2, default=str), encoding="utf-8")
            torch.save({k: pw[k] for k in ("lat", "yaw", "ade2s", "hd2s", "hdK",
                                           "speed", "eid", "t0", "epi",
                                           "de_fixed", "fixed_steps")},
                       str(outdir / f"perwindow_K{K}.pt"))

    # ---------------- COMMON-START PAIRED design across K --------------------
    key = lambda pw: [(int(x), int(y)) for x, y in zip(pw["epi"], pw["t0"])]  # noqa
    if len(per_K) >= 2:
        common = sorted(set.intersection(*[set(key(per_K[K])) for K in per_K]))
        node = {"_note": ("IDENTICAL windows at every horizon — the start set at "
                          "K_max is a subset of every smaller K's, so this "
                          "removes the window-composition confound. Deltas use "
                          "the PAIRED episode-cluster bootstrap."),
                "n_common_windows": len(common),
                "n_common_episodes": len({c[0] for c in common})}
        if len(common) >= 8:
            sel = {K: np.array([{c: i for i, c in enumerate(key(per_K[K]))}[c]
                                for c in common]) for K in per_K}
            for K in per_K:
                pw, s = per_K[K], sel[K]
                sub = {"eid": [pw["eid"][i] for i in s], "lat": pw["lat"][s],
                       "yaw": pw["yaw"][s], "ade2s": pw["ade2s"][s],
                       "hd2s": pw["hd2s"][s], "hdK": pw["hdK"][s],
                       "speed": pw["speed"][s], "de_fixed": pw["de_fixed"][s],
                       "fixed_steps": pw["fixed_steps"]}
                node[str(K)] = CH.emit(sub, K, ood_map=om, thresholds=grid,
                                       primary=prim,
                                       junction_deg=a.junction_deg,
                                       surface="closed_loop", n_boot=a.n_boot)
            ref = min(per_K)
            hd = per_K[ref]["hd2s"].numpy()[sel[ref]]
            spd = per_K[ref]["speed"].numpy()[sel[ref]]
            eidc = [per_K[ref]["eid"][i] for i in sel[ref]]
            st = _corr.strata(hd, spd, a.junction_deg)
            deltas = {}
            base_lat = per_K[ref]["lat"].numpy()[sel[ref]]
            for K in sorted(per_K):
                if K == ref:
                    continue
                kl = per_K[K]["lat"].numpy()[sel[K]]
                deltas[str(K)] = {}
                for sname, m in st.items():
                    ix = np.flatnonzero(m)
                    if len(ix) < 2 or len({eidc[i] for i in ix}) < 2:
                        deltas[str(K)][sname] = None
                        continue
                    d = _ci.paired_episode_cluster_bootstrap(
                        _corr.corridor_departure(kl[ix], prim),
                        _corr.corridor_departure(base_lat[ix], prim),
                        [eidc[i] for i in ix], n_boot=a.n_boot)
                    d["_orientation"] = (f"CDR(K={K}) - CDR(K={ref}) on the SAME "
                                         f"windows; POSITIVE = the longer "
                                         f"horizon departs more")
                    deltas[str(K)][sname] = d
            node["paired_delta_vs_K%d" % ref] = deltas
            node["reference_K"] = ref
        res["common_start_paired"] = node
    (outdir / "ksweep_results.json").write_text(
        json.dumps(res, indent=2, default=str), encoding="utf-8")
    print("KSWEEP_DONE -> " + str(outdir / "ksweep_results.json"), flush=True)


if __name__ == "__main__":
    main()
