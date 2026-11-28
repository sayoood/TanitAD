"""eval_v58f.py — T0 four-families-ready eval of the v5.8f assembly (881 grid).

WHAT IS SCORED. The v5.8f composition (``tanitad.models.v58f.V58F``): frozen
v5f 30k trunk+head -> W4 ``UnicycleEmission`` fan ((a, kappa) controls,
unicycle-integrated, feasible by construction) -> selection by
``--select-rule``. ⛔ The DEFAULT rule is ``from-gate``: the W4b gate record
(``w4b_gate.json``, PREREG_W4B_SELECTOR.md) decides at assembly time — G1
pass -> ``rescorer-argmax``; G2 -> ``rescorer-top8-kincost`` (top-8 pruner +
kinematic-cost pick, cost = mean|a| + 0.5*mean|jerk| FROM THE CONTROLS; the
W1 refutation applied to the old 97.6 %-infeasible fan only, registry §1.13).
``frozen-argmax`` is the W4 control arm (frozen selector, 0.7933 reference).

TIER: **T0** (teacher-forced diagnostic, conditioned on logged frames — NEVER
quotable as driving performance, EVAL_DOCTRINE.md). GOAL PROVENANCE: ORACLE —
the forward runs through ``frozen_forward`` (imported via
``V58F.plan_batch``), whose ``_goal_inputs`` mints vt_band/route from batch
labels: the SAME conditioning every banked W4/W4b gate number was measured
under, an upper bound, not a deployable capability.

GRID: episodes < 40, stride 8 over the v2 val cache — the banked 881-window
grid all reference numbers (0.1975 / 0.1077 / 0.7933 / the 0.45 G1 gate)
live on; ``matches_banked_grid`` is stamped and a mismatch is warned loudly.

FOUR FAMILIES (Sayed 2026-08-02, binding — ADE is ONE ROW, never "the
result"): LONGITUDINAL (speed + accel MAE, both waypoint-derived and
from-controls; headway/TTC = pod-side lead-block work item, stated per the
rule) · LATERAL (heading / yaw-rate / curvature MAE on the selected
trajectory) · TACTICAL (selector rank quality — this instrument's family) ·
STRATEGIC (n/a on PhysicalAI: no route/goal label exists — settled at five
probes, CLAUDE.md rule 2; stated with reason, not silently dropped).

PERSISTED:
  * ``windows_<arm>.pt`` — the EXACT ``collect_planner`` schema of
    ``eval_flagship_v4.py`` (pred/gt/cv/eid/speed/head_deg/wp_steps +
    pred_dense/gt_dense/dense_steps/dt_s/dense_provenance), so
    ``taniteval/tools/eval_four_families.py --windows-in`` and
    ``taniteval.driving.from_windows()`` consume it unchanged.
  * ``v58f_fan_windows_<arm>.pt`` — per-window per-candidate fan ADE +
    deployed scores + sel_idx + eid (the ``w4b_eval_windows.pt`` pattern),
    the input to the episode-cluster-bootstrap selgap rescore
    (``taniteval.selgap``) with no re-run.
  * ``v58f_eval_<arm>.json`` — the summary below.

ESTIMATOR: every number in the summary is a POINT ESTIMATE over the grid; the
decision-grade interval is the EPISODE-CLUSTER BOOTSTRAP (taniteval/ci.py) —
computed in-process when ``taniteval`` imports (best-effort, recorded either
way) and always recomputable from the banked arrays. Never
``overlapping_holdout_se``.

⚠️ POD-SIDE ONLY: this box has no GPU, no checkpoints, no v2 corpus. Runnable
here: ``python -m py_compile`` + the composition CPU tests
(``stack/tests/test_v58f.py``).

Usage (pod5; PYTHONPATH=/workspace/TanitAD/stack REQUIRED or `import tanitad`
dies; W4b must have FINISHED — the gate JSON that decides the default rule is
written at the END of the W4b run, and evaluating on a still-training pod is
forbidden; OMP_NUM_THREADS=6 before any multi-arm panel):

  OMP_NUM_THREADS=6 python3 eval_v58f.py \
      --ckpt /workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt \
      --w4-ckpt /workspace/experiments/w4-unicycle-head-c/unicycle_emission.pt \
      --w4b-ckpt /workspace/experiments/w4b-selector-feat/w4b_rescorer.pt \
      --select-rule from-gate \
      --anchors-dense /workspace/experiments/anchors/anchors_dense_1to20.pt \
      --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
      --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
      --v2-subframe 176x624 \
      --out /workspace/experiments/v58f-eval

(control arm: drop --w4b-ckpt and pass --select-rule frozen-argmax)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# stack root too, so `import tanitad` resolves even without the pod-side
# PYTHONPATH (harmless when PYTHONPATH is set — same directory).
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from tanitad.models.v58f import (SELECT_RULES, accel_mae_from_controls,  # noqa: E402
                                 kinematic_cost, load_v58f,
                                 select_rule_from_gate)
from train_v58f_unicycle_head import (A_MAX, accel_mae,  # noqa: E402
                                      unicycle_rollout)
from train_w4b_selector import (EXPECTED_GRID_WINDOWS, GATE_SELECTED_ADE,  # noqa: E402
                                REF_FROZEN_SELECTOR_NEW_FAN,
                                REF_OLD_SELECTOR_OLD_FAN, REF_W4_ORACLE, TOPK,
                                fan_ade, selected_family_sums,
                                topk_oracle_per_window)

ARM_PREFIX = "v58f"


def build_args(argv=None):
    ap = argparse.ArgumentParser("eval_v58f", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="v5f checkpoint "
                    "(keys: model, grounding, head[, goal_head])")
    ap.add_argument("--w4-ckpt", required=True,
                    help="trained W4 unicycle_emission.pt — loaded FROZEN")
    ap.add_argument("--w4b-ckpt", default=None,
                    help="trained W4b w4b_rescorer.pt — REQUIRED for the "
                         "rescorer rules, REFUSED with frozen-argmax (drop it "
                         "for the control arm)")
    ap.add_argument("--select-rule", default="from-gate",
                    choices=SELECT_RULES + ("from-gate",),
                    help="'from-gate' (DEFAULT): the W4b gate decides at "
                         "assembly time (G1 -> rescorer-argmax; G2 -> "
                         "rescorer-top8-kincost). An explicit rule OVERRIDES "
                         "the gate and is stamped as such.")
    ap.add_argument("--w4b-gate", default=None,
                    help="w4b_gate.json for --select-rule from-gate "
                         "(default: sibling of --w4b-ckpt)")
    ap.add_argument("--head-config", default=None,
                    help="run config.json (default: sibling of --ckpt)")
    ap.add_argument("--anchors-dense", default=None,
                    help="trained dense-anchor buffer (pass explicitly)")
    ap.add_argument("--probe-vocab", default=None,
                    help="probe_vocab.pt for cond_imagination heads "
                         "(default: sibling of --ckpt)")
    # corpus (v2 compressed only — the ONLY format v5 has); the W4/W4b seams
    ap.add_argument("--v2-val-cache", required=True, nargs="+",
                    help="v2 compressed VAL split dir(s)")
    ap.add_argument("--v2-lru", type=int, default=64)
    ap.add_argument("--v2-subframe", default=None, metavar="HxW",
                    help="centred sub-frame the model reads (e.g. 176x624) — "
                         "MUST match the run; cross-checked vs config.json")
    ap.add_argument("--require-parity", action="store_true")
    from tanitad.geometry import add_geometry_args
    add_geometry_args(ap)      # --frame-h/--frame-w/--frame-hfov/--projection
    # the banked grid (episodes<40, stride 8 -> 881 windows)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out", required=True, help="output DIR for "
                    "windows_<arm>.pt / v58f_fan_windows_<arm>.pt / "
                    "v58f_eval_<arm>.json")
    ap.add_argument("--key", default=None,
                    help="arm name override (default: v58f-<select_rule>)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = build_args(argv)
    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[v58f-eval] WARNING: cuda unavailable, falling back to cpu",
              flush=True)
        device = "cpu"
    amp_on = False if a.no_amp else None       # None -> loader decides by device
    os.makedirs(a.out, exist_ok=True)

    # ---- WHICH selection rule: the W4b gate decides the default -------------
    rule_rec = None
    if a.select_rule == "from-gate":
        gate_path = a.w4b_gate or (
            str(Path(a.w4b_ckpt).parent / "w4b_gate.json") if a.w4b_ckpt
            else None)
        if not gate_path or not Path(gate_path).exists():
            raise SystemExit(
                "[v58f-eval] --select-rule from-gate (the DEFAULT — the W4b "
                "gate decides the assembly rule, PREREG_W4B_SELECTOR.md) "
                "needs the gate record: pass --w4b-gate <w4b_gate.json> or "
                "--w4b-ckpt with a sibling w4b_gate.json. To OVERRIDE the "
                "gate, pass an explicit --select-rule.")
        rule, rule_rec = select_rule_from_gate(gate_path)
        decided_by = f"w4b_gate ({gate_path})"
        print(f"[v58f-eval] select_rule={rule} DECIDED BY THE W4b GATE: "
              f"{rule_rec['decision']}", flush=True)
        if "warning" in rule_rec:
            print(f"[v58f-eval] ⚠ {rule_rec['warning']}", flush=True)
    else:
        rule = a.select_rule
        decided_by = "explicit --select-rule flag (OVERRIDES the gate default)"
    arm = a.key or f"{ARM_PREFIX}-{rule}"

    from torch.utils.data import default_collate

    import driving_diagnostic as dd
    from eval_flagship_v4 import (WP_STEPS, _eval_cfg, _plan,
                                  build_v2_val_episodes, resolve_eval_frames)
    from flagship_v4_data import FlagshipV4Dataset
    from tanitad.data import parity
    from train_flagship_v4 import _to_device

    # ---- geometry FIRST, cross-checked against the run's own config.json ----
    cfg = _eval_cfg()
    cache_frame, model_frame = resolve_eval_frames(a, cfg, label="eval_v58f")
    plan = _plan(cfg)
    head_cfg_path = a.head_config or str(Path(a.ckpt).parent / "config.json")
    run_cfg = None
    if Path(head_cfg_path).exists():
        try:
            run_cfg = json.loads(Path(head_cfg_path).read_text())
        except Exception as ex:
            print(f"[v58f-eval] WARNING: could not parse {head_cfg_path}: "
                  f"{ex}", flush=True)
    frame_check = parity.assert_eval_frame_matches_run(
        run_cfg, model_frame, label="--ckpt vs v58f eval frame",
        cache_frame=cache_frame)
    if not frame_check["checked"]:
        print(f"[v58f-eval] ⚠ FRAME UNVERIFIED: {frame_check['note']}",
              flush=True)

    # ---- the assembly (loader = tanitad.models.v58f.load_v58f) --------------
    model, prov = load_v58f(
        a.ckpt, a.w4_ckpt, a.w4b_ckpt, select_rule=rule, device=device,
        frame=model_frame, head_config=a.head_config,
        anchors_dense=a.anchors_dense, probe_vocab=a.probe_vocab,
        amp_on=amp_on)
    horizons = model.horizons
    K = len(horizons)
    if not set(WP_STEPS) <= set(horizons):
        raise SystemExit(f"[v58f-eval] wp_steps {WP_STEPS} not a subset of "
                         f"the head's horizons {horizons}")
    wp_pos = [horizons.index(k) for k in WP_STEPS]

    # ---- data (the W4/W4b val seam) -----------------------------------------
    val_eps, val_prov = build_v2_val_episodes(
        a, cache_frame=cache_frame, train_frame=model_frame)
    ds_val = FlagshipV4Dataset(val_eps, window=cfg.predictor.window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
    print(f"[v58f-eval] val {len(val_eps)} eps / {len(ds_val)} windows",
          flush=True)
    grid = [i for i, (e, t) in enumerate(ds_val.index)
            if e < a.episodes and t % a.stride == 0]
    if not grid:
        raise SystemExit("[v58f-eval] grid selected 0 windows — check "
                         "--episodes/--stride against the val cache")

    # ---- the eval loop ------------------------------------------------------
    sums = {k: 0.0 for k in
            ("selected", "oracle", "frozen_selected", "winner_hit",
             "rank_pct", "sel_matches_frozen", "accel_mae_wp",
             "accel_mae_ctrl", "kincost_selected", "viol_frac", "wp4_ade",
             "wp4_oracle")}
    sums.update({f"top{k}": 0.0 for k in TOPK})
    fam = {k: 0.0 for k in ("speed_mae", "accel_mae", "heading_mae_rad",
                            "yaw_rate_mae_rads", "curvature_mae_1pm")}
    P, PD, GD, G, C, HDG, EID, SPD = [], [], [], [], [], [], [], []
    ERRS, SCORES, SELS = [], [], []
    pose_cache: dict[int, torch.Tensor] = {}
    n = 0
    n_cand = None
    t0 = time.time()
    for b0 in range(0, len(grid), a.batch):
        idx = grid[b0:b0 + a.batch]
        b = _to_device(default_collate([ds_val[i] for i in idx]), device)
        r = model.plan_batch(b, device)                     # frozen_forward path
        fan, tgt = r["fan"], r["tgt"].float()
        traj, sel, v0 = r["traj"], r["sel_idx"], r["v0"]
        a_ctl, kappa = r["controls"]["a"], r["controls"]["kappa"]
        scores = r["scores"]
        if scores is None:                                  # pragma: no cover
            raise SystemExit("[v58f-eval] no deployable scores surface — the "
                             "head emitted no refined_logits and no rescorer "
                             "is loaded; top-k oracles would be undefined")
        frozen_idx = r["head_out"]["sel_idx"]
        err = fan_ade(fan, tgt)                             # [B, N]
        bs = err.shape[0]
        n_cand = int(err.shape[1])
        ar = torch.arange(bs, device=err.device)
        e_sel = err[ar, sel]
        sums["selected"] += float(e_sel.sum())
        sums["oracle"] += float(err.min(dim=1).values.sum())
        sums["frozen_selected"] += float(err[ar, frozen_idx].sum())
        sums["winner_hit"] += float((sel == err.argmin(dim=1)).float().sum())
        sums["rank_pct"] += float(
            ((err < e_sel[:, None]).sum(dim=1).float()
             / max(err.shape[1] - 1, 1)).sum())
        sums["sel_matches_frozen"] += float((sel == frozen_idx).float().sum())
        for k in TOPK:
            sums[f"top{k}"] += float(
                topk_oracle_per_window(err, scores, k).sum())
        # LONGITUDINAL accel: BOTH derivations. wp = the W4 gate instrument
        # (waypoint finite differences, comparable to w4_gate.json's 0.774);
        # ctrl = the selected candidate's commanded accels (v5.8f's new
        # surface — exact up to the v>=0 clamp).
        a_sel = a_ctl[ar, sel]
        sums["accel_mae_wp"] += accel_mae(traj, tgt) * bs
        sums["accel_mae_ctrl"] += accel_mae_from_controls(a_sel, tgt) * bs
        sums["kincost_selected"] += float(kinematic_cost(a_sel).sum())
        # feasibility census on the emitted controls (W4 mini_eval block —
        # reported, not assumed, even though bounded activations make it ~0)
        _, v_pre = unicycle_rollout(a_ctl, kappa, v0)
        yaw_rate = kappa * v_pre
        viol = ((a_ctl.abs() > A_MAX + 1e-4)
                | (yaw_rate.abs() > 0.33 * v_pre + 0.05 + 1e-4))
        sums["viol_frac"] += float(viol.float().mean()) * bs
        for key, v in selected_family_sums(traj, tgt).items():
            fam[key] += v
        # 4-waypoint cross-check (the historical convention, steps 5/10/15/20)
        pred4, tgt4 = traj[:, wp_pos], tgt[:, wp_pos]
        fan4_err = (fan[:, :, wp_pos, :] - tgt4[:, None]).norm(dim=-1) \
            .mean(dim=-1)
        sums["wp4_ade"] += float((pred4 - tgt4).norm(dim=-1).mean(dim=-1)
                                 .sum())
        sums["wp4_oracle"] += float(fan4_err.min(dim=1).values.sum())
        # ---- persistence (the collect_planner schema, row for row) ----------
        P.append(pred4.float().cpu())
        PD.append(traj.float().cpu())
        GD.append(tgt.float().cpu())
        SPD.append(v0.float().cpu())
        ERRS.append(err.cpu())
        SCORES.append(scores.cpu())
        SELS.append(sel.cpu())
        for i in idx:
            e_i, t = ds_val.index[i]
            po = pose_cache.get(e_i)
            if po is None:
                po = torch.as_tensor(ds_val.episodes[e_i].poses,
                                     dtype=torch.float32)
                pose_cache[e_i] = po
            last = torch.tensor([t + ds_val.window - 1])
            G.append(dd.gt_ego_waypoints(po, last, wp_steps=WP_STEPS))
            C.append(dd.baseline_waypoints(
                po, last, wp_steps=WP_STEPS)["constant_velocity"])
            HDG.append(dd.net_heading_change_deg(po, last))
            EID.append(int(ds_val.episodes[e_i].episode_id))
        n += bs
        if b0 % (a.batch * 10) == 0:
            print(f"  [v58f-eval] {n}/{len(grid)} windows "
                  f"({time.time() - t0:.0f}s)", flush=True)
    model.close()

    # ---- windows_<arm>.pt: the eval_flagship_v4 collect_planner schema ------
    windows = {
        "pred": torch.cat(P), "gt": torch.cat(G).float(),
        "cv": torch.cat(C).float(), "eid": EID,
        "speed": torch.cat(SPD).float(),
        "head_deg": torch.cat(HDG).float(), "wp_steps": list(WP_STEPS),
        "pred_dense": torch.cat(PD), "gt_dense": torch.cat(GD),
        "dense_steps": list(horizons), "dt_s": 0.1,
        "dense_provenance": (
            "pred_dense = the v5.8f SELECTED candidate of the W4 unicycle fan "
            f"(select_rule={rule}) at the head's dense 1..{K} horizons; "
            "gt_dense = refb_labels.waypoint_targets via frozen_forward (NOT "
            "driving_diagnostic.gt_ego_waypoints, which built the sparse "
            "`gt`). pred[:, j] == pred_dense[:, wp_steps[j]-1] by "
            "construction."),
    }
    win_path = Path(a.out) / f"windows_{arm}.pt"
    torch.save(windows, win_path)
    print(f"[v58f-eval] windows -> {win_path} (consumable by "
          f"taniteval.driving.from_windows() and "
          f"tools/eval_four_families.py --windows-in)", flush=True)

    # ---- fan windows: the selgap-rescore input (w4b_eval_windows.pt pattern)
    err_all = torch.cat(ERRS)
    scr_all = torch.cat(SCORES)
    sel_all = torch.cat(SELS)
    fan_path = Path(a.out) / f"v58f_fan_windows_{arm}.pt"
    torch.save({"fan_err_ade": err_all, "scores": scr_all, "sel_idx": sel_all,
                "eid": torch.tensor(EID),
                "_read": ("per-window per-candidate dense ADE of the W4 "
                          "unicycle fan + the v5.8f deployed scores over the "
                          "eval grid — the input to the pod-side episode-"
                          "cluster bootstrap (taniteval.selgap)")}, fan_path)

    # best-effort selgap CI — stack does not DEPEND on taniteval (its rule).
    selgap_ci: dict | str
    try:
        root = Path(__file__).resolve().parents[2] / "taniteval"
        if str(root) not in sys.path:
            sys.path.append(str(root))
        from taniteval.selgap import selgap as _selgap
        selgap_ci = _selgap(err_all.numpy(), sel_all.numpy(), EID,
                            scores=scr_all.numpy(),
                            level=f"operative_{arm}")
    except Exception as ex:                                  # noqa: BLE001
        selgap_ci = (f"taniteval unavailable here ({type(ex).__name__}: {ex})"
                     f" — rescore pod-side from {fan_path.name}")

    # ---- summary ------------------------------------------------------------
    summary = {
        "arm": arm,
        "tier": "T0",
        "_tier_note": ("T0 teacher-forced diagnostic — conditioned on logged "
                       "frames; NEVER quotable as driving performance "
                       "(EVAL_DOCTRINE.md); the T1 instrument is "
                       "taniteval/tools/t1_eval.py"),
        "goal_provenance": (
            "ORACLE — frozen_forward/_goal_inputs mints vt_band/route from "
            "batch labels (ego future); the SAME conditioning the W4/W4b "
            "gate numbers were measured under. UPPER BOUND, not a deployable "
            "capability; report as 'T0, goal-oracle'."),
        "select_rule": rule,
        "select_rule_decided_by": decided_by,
        "w4b_gate_record": rule_rec,
        "n_windows": n,
        "n_candidates": n_cand,
        "grid": {"episodes": a.episodes, "stride": a.stride,
                 "batch": a.batch, "expected_n": EXPECTED_GRID_WINDOWS,
                 "matches_banked_grid": n == EXPECTED_GRID_WINDOWS},
        "selected_ade": round(sums["selected"] / n, 6),
        "oracle_ade": round(sums["oracle"] / n, 6),
        "sel_gap": round((sums["selected"] - sums["oracle"]) / n, 6),
        "oracle_topk": {str(k): round(sums[f"top{k}"] / n, 6) for k in TOPK},
        "frozen_selected_ade": round(sums["frozen_selected"] / n, 6),
        "winner_hit_frac": round(sums["winner_hit"] / n, 6),
        "sel_rank_pct_mean": round(sums["rank_pct"] / n, 6),
        "sel_matches_frozen_frac": round(sums["sel_matches_frozen"] / n, 6),
        "accel_mae_selected_wp_ms2": round(sums["accel_mae_wp"] / n, 6),
        "accel_mae_selected_from_controls_ms2":
            round(sums["accel_mae_ctrl"] / n, 6),
        "kincost_selected_mean": round(sums["kincost_selected"] / n, 6),
        "viol_frac": round(sums["viol_frac"] / n, 6),
        "wp4": {"ade_0_2s_selfcomputed": round(sums["wp4_ade"] / n, 6),
                "oracle_ade_0_2s": round(sums["wp4_oracle"] / n, 6),
                "note": "historical 4-waypoint convention (steps 5/10/15/20) "
                        "— cross-checkable against taniteval.driving on the "
                        "persisted windows"},
        "families": {
            "LONGITUDINAL": {
                "speed_mae_ms": round(fam["speed_mae"] / n, 6),
                "accel_mae_wp_ms2": round(sums["accel_mae_wp"] / n, 6),
                "accel_mae_from_controls_ms2":
                    round(sums["accel_mae_ctrl"] / n, 6),
                "headway_ttc": ("not computable here: no lead-agent channel "
                                "in this instrument — pod-side "
                                "tools/build_lead_block.py + "
                                "eval_four_families --windows-in "
                                f"{win_path.name} --lead <block>, stated per "
                                "the 2026-08-02 rule"),
            },
            "LATERAL": {
                "heading_mae_rad": round(fam["heading_mae_rad"] / n, 6),
                "yaw_rate_mae_rads": round(fam["yaw_rate_mae_rads"] / n, 6),
                "curvature_mae_1pm": round(fam["curvature_mae_1pm"] / n, 6),
                "note": "waypoint-derived adjuncts on the SELECTED trajectory "
                        "(atan2 of finite differences; noisy near standstill)",
            },
            "TACTICAL": {
                "note": "selector rank quality IS this instrument's family; "
                        "manoeuvre-class confusion lives in the hierarchy "
                        "harness (eval_four_families hierarchy pass)",
                "winner_hit_frac": round(sums["winner_hit"] / n, 6),
                "sel_rank_pct_mean": round(sums["rank_pct"] / n, 6),
                "sel_matches_frozen_frac":
                    round(sums["sel_matches_frozen"] / n, 6),
            },
            "STRATEGIC": ("n/a: no route/goal label exists on PhysicalAI-AV "
                          "(settled, five probes — CLAUDE.md rule 2); stated "
                          "per the 2026-08-02 rule"),
        },
        "selgap_ci": selgap_ci,
        "reference": {
            "v5f_oracle_ade_registry": 0.1975,
            "w4_oracle_ade_new_fan": REF_W4_ORACLE,
            "frozen_selector_selected_ade_new_fan":
                REF_FROZEN_SELECTOR_NEW_FAN,
            "old_selector_selected_ade_old_fan": REF_OLD_SELECTOR_OLD_FAN,
            "w4b_g1_threshold_m": GATE_SELECTED_ADE,
            "w4_accel_mae_selected": 0.774,
            "_read": "registry §1.13 / w4_gate.json / PREREG_W4B_SELECTOR.md",
        },
        "provenance": {
            **prov,
            "corpus": {
                "format": "v2 compressed (<clip_id>.v2ep.pt)",
                "val_cache": list(a.v2_val_cache),
                "require_parity": bool(a.require_parity),
                "eval_frame": model_frame.report(),
                "cache_frame": cache_frame.report(),
                "v2_subframe_arg": a.v2_subframe,
                "frame_matches_checkpoint": frame_check,
                "val_parity": val_prov["val_parity"],
                "geometry_binding": val_prov["geometry_binding"],
            },
        },
        "artifacts": {"windows_pt": str(win_path),
                      "fan_windows_pt": str(fan_path)},
        "_estimator_note": ("POINT ESTIMATES over the eval grid (episodes<"
                            f"{a.episodes}, stride {a.stride}). The decision-"
                            "grade interval for any registry claim is the "
                            "EPISODE-CLUSTER BOOTSTRAP (taniteval.selgap / "
                            "taniteval/ci.py) on the banked per-window arrays "
                            "— never overlapping_holdout_se."),
        "_evidence_class": "MEASURED (ours; artifact = this JSON)",
        "wallclock_s": round(time.time() - t0, 1),
    }
    if not summary["grid"]["matches_banked_grid"]:
        print(f"[v58f-eval] ⚠ grid has {n} windows, banked references are on "
              f"{EXPECTED_GRID_WINDOWS} — comparisons to 0.1975/0.1077/"
              f"0.7933/0.45 are cross-grid; say so wherever quoted",
              flush=True)
    out_json = Path(a.out) / f"v58f_eval_{arm}.json"
    out_json.write_text(json.dumps(summary, indent=1, default=str),
                        encoding="utf-8")
    print(f"\n[V58F EVAL SUMMARY] {json.dumps(summary, indent=1, default=str)}",
          flush=True)
    print(f"[v58f-eval] -> {out_json}", flush=True)
    print(f"[v58f-eval] selected {summary['selected_ade']:.4f} / oracle "
          f"{summary['oracle_ade']:.4f} / accel MAE (wp) "
          f"{summary['accel_mae_selected_wp_ms2']:.4f} / viol "
          f"{summary['viol_frac']:.4f} — rule {rule} ({decided_by})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
