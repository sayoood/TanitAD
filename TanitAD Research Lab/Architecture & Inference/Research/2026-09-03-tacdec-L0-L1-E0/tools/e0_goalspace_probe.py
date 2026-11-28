#!/usr/bin/env python
"""E0 — THE ARM THAT CAN REFUTE THE WHOLE TACTICAL-DECODER LINE, FOR ~0 GPU.

Pre-registered in `Project Steering/PREREG_TACTICAL_DECODER.md` §8, quoted
verbatim in this package's SPEC.md:

  Population: ep2's 25 windows whose imagined goal already carries curvature.
  Procedure : re-score those windows' named seeds, in FLOAT64, under the
              2x2x2 cross {seed as shipped, seed = goal's control} x
              {1-cos, chord} x {w_kappa as shipped, w_kappa scaled}, plus the
              C-REG regression arm.
  Q1        : does the tactical 6 s goal field prefer a turn AT ALL, once the
              seed is the goal's own control? If the goal-term advantage stays
              <= 0 on a MAJORITY of the 25 even at seed/goal identity and in
              f64, the goal SPACE cannot represent a manoeuvre and the whole
              decoder-then-cost line is REFUTED.
  Q2        : if it does prefer the turn, by how much, and what w_kappa /
              w_jerk would let it win.
  §8.4      : E0 must ALSO report the RAW-INPUT FLOOR -- the same windows with
              the goal replaced by the GROUND-TRUTH future field. If the
              oracle-goal cost also fails to prefer the turn, the defect is the
              COST; if it succeeds, the defect is the IMAGINED GOAL.

⭐ WHY Q1 IS NOT A TAUTOLOGY. At exact seed/goal identity the turn's goal term
is 0 by construction, so the turn's advantage over constant velocity IS
``cv_c_goal = 1 - cos(z_cv, z_goal)``. Q1 therefore asks *"is the cv rollout
distinguishable from the goal rollout at all?"* -- exactly the goal-space
question, and it can come back 0.

THE ARMS (each x {units "kappa" = convention A, "steer" = convention B})

  shipped         goal = the 6 s imagined field; candidate = the shipped seed
                  ``ctrl[:plan_steps]`` fed DENSE. Reproduces the banked panel.
  identity_full   goal = the SAME 6 s field; candidate feed = the goal's own
                  tactical actions.  -> Q1 on the SHIPPED goal space.
  identity_plan   goal = re-rolled over the PLAN window (`refa_v1.py`
                  ``goal_time_grid="plan"``, L0); candidate = the shipped seed.
                  -> what L0 as implemented actually buys.
  C_REG           the DELIBERATE REGRESSION: shipped seed with the SATURATING
                  tactical regrid ``[0,3,6,9,9,...]``. Must NOT read identity.
  oracle_floor    goal = the ground-truth 6 s future field (§8.4).

CONTROLS THAT MUST READ KNOWN VALUES (a failure VOIDS the panel)
  K1  identity arms' turn goal term == 0 to <= 4*spacing(1f) = 4.77e-07 (f32)
      and <= 1e-12 (f64) -- asserted, never assumed.
  K2  ZERO-GOAL stratum: a window whose canonical control is all-zero has
      goal rollout == cv rollout, so cv_c_goal must be EXACTLY 0.0.
  K3  C_REG must NOT satisfy K1 on curved tokens.
  K4  every fp32 difference carries its ULP and `resolvable_in_fp32`.
  K5  n and d printed; the 25/140 denominator never mixed with the 27/140
      geometric stratum.
  K6  banked agreement: the `shipped` arm reproduces the banked
      `2026-09-03-refav1-cost-surface/raw/cost_surface_ep2.json` per window
      (`named_A_goal_canonical_c_goal` / `named_A_cv_c_goal`), MATCHED ON
      (ep_name, t) -- never on t alone: the 25 curved windows take only SIX
      distinct t values, so a t-keyed join compares different episodes and
      manufactures a disagreement (measured, on this tool's first pass).
      The decoded (lat, lon) token must agree on every matched window too.

TIER: T0 (open-loop, banked checkpoint; a cost/WM diagnostic, never a driving
claim). EVIDENCE CLASS: MEASURED. GPU: one encode + ~16 tactical rollouts per
window on the dev-box 4060, forward-only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

W_JERK = 0.02           #: refa_v1.py `_cost_chunk`
W_KAPPA = 0.05          #: refa_v1.py `_cost_chunk`
COS32_TOL = 4.0 * float(np.spacing(np.float32(1.0)))     # 4.768e-07
COS64_TOL = 1e-12


def ulp32(x: float) -> float:
    x = abs(float(x))
    if x == 0.0:
        return float(np.spacing(np.float32(0.0)))
    return float(np.spacing(np.float32(x)))


def _cos_term(a, b, dtype):
    """``1 - cosine_similarity(a.flatten, b.flatten)`` at the requested dtype.
    fp32 is what the shipped cost computes; fp64 is what E0 is registered to
    report, on the SAME fp32 fields."""
    import torch
    x = a.flatten().to(dtype)
    y = b.flatten().to(dtype)
    c = torch.dot(x, y) / (x.norm() * y.norm())
    return float(1.0 - c)


# --------------------------------------------------------------------------- #
def collect(a) -> dict:
    import torch

    sys.path.insert(0, a.arm_tool_dir)
    import refav1_arm as ARM                                   # noqa: N812
    from tanitad.models.v6 import (tactical_lat_actions,
                                   tactical_lon_actions_v)
    from tanitad.refs.refa_v1 import canonical_controls

    dev = a.device
    model, cfg, prov = ARM.load_model(a.ckpt, a.config, dev, False)
    k_wm = int(cfg.op_steps)
    names = ARM.episode_names(a.cache)
    if a.episodes_n:
        names = names[:int(a.episodes_n)]
    ld = ARM.build_loader(a, cfg, k_wm, names)
    vv = getattr(cfg, "tac_vocab_version", "v6.0")
    lat_names = list(tactical_lat_actions(vv))
    lon_names = list(tactical_lon_actions_v(vv))

    stride = int(round(cfg.tac_dt / cfg.op_dt))
    H = int(cfg.plan_steps)
    T = int(cfg.tac_steps)
    dt = float(cfg.op_dt)
    #: the saturating regrid, `refa_v1.py` `tac_idx` under cost_time_grid
    #: "tactical" -- the C-REG arm's feed
    reg_idx = torch.tensor([min(j * stride, H - 1) for j in range(T)],
                           dtype=torch.long, device=dev)

    sel = [(wi, ei, t) for wi, (ei, t) in enumerate(ld.windows)
           if (t - (ld.W - 1)) % int(a.window_stride) == 0]
    print(f"[e0] {len(sel)} windows / {len(ld.names)} episodes; vocab={vv}; "
          f"op_steps={cfg.op_steps} tac_steps={T} stride={stride} "
          f"plan_steps={H} plan_level={cfg.plan_level}", flush=True)

    by_ep: dict[int, list] = {}
    for i, (wi, ei, t) in enumerate(sel):
        by_ep.setdefault(ei, []).append((i, wi, t))

    rows: list[dict] = []
    t0 = time.time()
    for fi, ei in enumerate(sorted(by_ep)):
        for (i, wi, t) in by_ep[ei]:
            ld._order, ld._cursor = [wi], 0
            b = ld.batch(1)
            feats = b["feats"].to(dev)
            fut = b["future_feats"].to(dev)
            v0 = float(b["v0"][0])
            v0_t = torch.tensor([v0], dtype=torch.float32, device=dev)
            nav_i = int(b["nav_cmd"][0]) if b.get("nav_cmd") is not None else 0
            nav_t = torch.tensor([nav_i], device=dev)

            with torch.no_grad():
                field = model.encode(feats)
                last = model._last_state(field)
                z_tac0 = model._tac_field(last)
                brains = model._run_brains(field.mean(dim=-2), nav_t)
                if brains is None:
                    raise SystemExit("no hierarchy on this checkpoint — E0 is "
                                     "a question about the TACTICAL goal")
                intent = brains["intent"]
                lat = lat_names[int(model.lat_head(intent).argmax(-1))]
                lon = lon_names[int(model.lon_head(intent).argmax(-1))]
                ctrl = canonical_controls(lat, lon, v0, cfg.op_steps,
                                          cfg.op_dt).to(last)          # [30,2]
                seed = ctrl[:H][None]                                  # [1,H,2]
                packed = ctrl[::stride][:T][None]                      # [1,T,2]
                cv = torch.zeros_like(seed)

                row = {"ep_index": fi, "ep": ld.names[ei], "t": int(t),
                       "v0": v0, "nav": nav_i, "lat": lat, "lon": lon,
                       "goal_kappa_max": float(ctrl[:, 1].abs().max()),
                       "goal_accel_absmax": float(ctrl[:, 0].abs().max()),
                       "arms": {}}

                for units, conv in (("kappa", "A"), ("steer", "B")):
                    # ---- the feeds, built EXACTLY as refa_v1 builds them --- #
                    f_goal6 = model._model_actions(
                        ctrl[None], v0_t, units)[:, ::stride][:, :T]
                    f_seed = model._model_actions(seed, v0_t, units)
                    f_cv = model._model_actions(cv, v0_t, units)
                    f_creg = f_seed.index_select(1, reg_idx)
                    f_cvreg = f_cv.index_select(1, reg_idx)
                    f_pack = model._model_actions(packed, v0_t, units)

                    def _roll(fa):
                        return model.tactical.rollout(z_tac0, fa,
                                                      intent=intent,
                                                      last_only=True)

                    z_goal6 = _roll(f_goal6)
                    z_seed = _roll(f_seed)
                    z_cv = _roll(f_cv)
                    z_creg = _roll(f_creg)
                    z_cvreg = _roll(f_cvreg)
                    z_pack = _roll(f_pack)
                    # §8.4 RAW-INPUT FLOOR: the GROUND-TRUTH 6 s field, pooled
                    # into the tactical query space by the model's own tac_pool
                    # — the same pooling `plan(goal_field=...)` applies.
                    z_orc = model._tac_field(
                        model.adapter(model.std(fut[:, :cfg.op_steps]))[
                            :, cfg.op_steps - 1])

                    arms = {
                        # goal, turn-candidate terminal field, turn controls
                        "shipped": (z_goal6, z_seed, seed, z_cv),
                        "identity_full": (z_goal6, z_pack, packed, z_cv),
                        "identity_plan": (z_seed, z_seed, seed, z_cv),
                        "C_REG": (z_goal6, z_creg, seed, z_cvreg),
                        "oracle_floor": (z_orc, z_pack, packed, z_cv),
                        "oracle_floor_shipped_seed": (z_orc, z_seed, seed,
                                                      z_cv),
                    }
                    for nm, (zg, zt, ct, zc) in arms.items():
                        e = {}
                        for tag, dtp in (("f32", torch.float32),
                                         ("f64", torch.float64)):
                            e[f"turn_c_goal_{tag}"] = _cos_term(zt, zg, dtp)
                            e[f"cv_c_goal_{tag}"] = _cos_term(zc, zg, dtp)
                        # the explicit penalties, on the CONTROL (refa_v1
                        # charges them on `controls`, never on the feed)
                        jt = float(((ct[:, 1:, 0] - ct[:, :-1, 0]) / dt)
                                   .pow(2).mean())
                        jc = float(((cv[:, 1:, 0] - cv[:, :-1, 0]) / dt)
                                   .pow(2).mean())
                        e["charge_jerk"] = W_JERK * (jt - jc)
                        e["charge_kappa"] = W_KAPPA * (
                            float(ct[..., 1].pow(2).mean())
                            - float(cv[..., 1].pow(2).mean()))
                        e["turn_kappa_rms"] = float(
                            ct[..., 1].pow(2).mean().sqrt())
                        row["arms"][f"{nm}__{conv}"] = e
            rows.append(row)
        print(f"  ep{fi:03d} {ld.names[ei]} ({time.time()-t0:.1f}s)",
              flush=True)

    return {"rows": rows,
            "meta": {"ckpt": a.ckpt, "config": a.config,
                     "step": prov.get("step"),
                     "config_source": prov.get("config_source"),
                     "n_windows": len(rows), "n_episodes": len(by_ep),
                     "window_stride": int(a.window_stride),
                     "op_steps": int(cfg.op_steps), "tac_steps": T,
                     "stride": stride, "plan_steps": H,
                     "plan_level": cfg.plan_level,
                     "tac_vocab_version": vv,
                     "w_jerk": W_JERK, "w_kappa": W_KAPPA,
                     "wallclock_s": round(time.time() - t0, 1)}}


# --------------------------------------------------------------------------- #
def _q(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if not v.size:
        return None
    return {"n": int(v.size), "median": float(np.median(v)),
            "mean": float(v.mean()), "min": float(v.min()),
            "max": float(v.max()), "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75))}


def analyse(d: dict, banked: str | None) -> dict:
    rows = d["rows"]
    curved = [r for r in rows if r["goal_kappa_max"] > 0.0]
    zero = [r for r in rows if r["goal_kappa_max"] == 0.0
            and r["goal_accel_absmax"] == 0.0]
    out = {"n_windows": len(rows), "n_curved_goal": len(curved),
           "n_zero_goal_control": len(zero),
           "denominator_note": (
               "the 25/140 CURVED-GOAL stratum is a property of the DECODER's "
               "output; it is NOT the 27/140 geometric turn stratum "
               "(gt_turn_deg >= 5) and no count mixes the two"),
           "controls": {}, "arms": {}}

    # ---- K2: the ZERO-GOAL control must read EXACTLY 0.0 ------------------- #
    k2 = []
    for r in zero:
        for key in ("shipped__A", "shipped__B"):
            k2.append(abs(r["arms"][key]["cv_c_goal_f64"]))
    out["controls"]["K2_zero_goal_cv_c_goal_max_abs_f64"] = (
        max(k2) if k2 else None)
    out["controls"]["K2_pass"] = bool(k2) and max(k2) <= COS64_TOL

    for conv in ("A", "B"):
        for nm in ("shipped", "identity_full", "identity_plan", "C_REG",
                   "oracle_floor", "oracle_floor_shipped_seed"):
            key = f"{nm}__{conv}"
            adv32, adv64, ck, cj = [], [], [], []
            turn32, turn64, cv64 = [], [], []
            wins, wins_chord, res32 = [], [], []
            need, need_chord, lev = [], [], []
            for r in curved:
                e = r["arms"][key]
                a32 = e["cv_c_goal_f32"] - e["turn_c_goal_f32"]
                a64 = e["cv_c_goal_f64"] - e["turn_c_goal_f64"]
                charge = e["charge_kappa"] + e["charge_jerk"]
                gc = math.sqrt(2.0 * max(e["turn_c_goal_f64"], 0.0))
                cc = math.sqrt(2.0 * max(e["cv_c_goal_f64"], 0.0))
                a_ch = cc - gc
                u = max(ulp32(e["turn_c_goal_f32"]),
                        ulp32(e["cv_c_goal_f32"]))
                turn32.append(e["turn_c_goal_f32"])
                turn64.append(e["turn_c_goal_f64"])
                cv64.append(e["cv_c_goal_f64"])
                adv32.append(a32)
                adv64.append(a64)
                ck.append(e["charge_kappa"])
                cj.append(e["charge_jerk"])
                wins.append(1.0 if a64 > charge else 0.0)
                wins_chord.append(1.0 if a_ch > charge else 0.0)
                res32.append(1.0 if abs(a32) > 2 * u else 0.0)
                lev.append(a_ch / a64 if a64 not in (0.0,) else float("nan"))
                base = (e["charge_kappa"] / W_KAPPA
                        + e["charge_jerk"] / W_JERK)
                need.append(a64 / base if base > 0 else float("nan"))
                need_chord.append(a_ch / base if base > 0 else float("nan"))
            scale = _q(need_chord)
            w_scaled = (W_KAPPA * (scale["median"] if scale else float("nan")))
            out["arms"][key] = {
                "n": len(curved),
                "turn_c_goal_f32": _q(turn32),
                "turn_c_goal_f64": _q(turn64),
                "cv_c_goal_f64": _q(cv64),
                "goal_advantage_f64": _q(adv64),
                "goal_advantage_f32": _q(adv32),
                "n_goal_advantage_positive_f64":
                    int(sum(1 for x in adv64 if x > 0)),
                "n_goal_advantage_positive_f32":
                    int(sum(1 for x in adv32 if x > 0)),
                "n_resolvable_in_fp32": int(sum(res32)),
                "charge_kappa": _q(ck), "charge_jerk": _q(cj),
                # the 2x2x2 cross's win counts
                "n_turn_wins__1mcos__w_kappa_shipped": int(sum(wins)),
                "n_turn_wins__chord__w_kappa_shipped": int(sum(wins_chord)),
                "penalty_scale_that_would_tip_it__1mcos": _q(need),
                "penalty_scale_that_would_tip_it__chord": scale,
                "w_kappa_scaled_to_match_leverage": w_scaled,
                "chord_leverage_gain_x": _q(lev),
                # K1/K3: does this arm actually READ identity?
                "max_abs_turn_c_goal_f32": (max(abs(x) for x in turn32)
                                            if turn32 else None),
                "max_abs_turn_c_goal_f64": (max(abs(x) for x in turn64)
                                            if turn64 else None),
                "reads_identity_f32": bool(
                    turn32 and max(abs(x) for x in turn32) <= COS32_TOL),
                "reads_identity_f64": bool(
                    turn64 and max(abs(x) for x in turn64) <= COS64_TOL),
            }
    # ⭐ AT SCALED WEIGHTS the win count is by definition >= half by
    # construction of the median, so it is reported as a SIZING number
    # (Q2), never as evidence. The decision number is the shipped-weight one.
    out["controls"]["K1_identity_full_reads_identity"] = all(
        out["arms"][f"identity_full__{c}"]["reads_identity_f32"]
        for c in ("A", "B"))
    out["controls"]["K1_identity_plan_reads_identity"] = all(
        out["arms"][f"identity_plan__{c}"]["reads_identity_f32"]
        for c in ("A", "B"))
    out["controls"]["K3_C_REG_does_NOT_read_identity"] = not any(
        out["arms"][f"C_REG__{c}"]["reads_identity_f32"] for c in ("A", "B"))

    # ---- K6: agreement with the banked COST SURFACE ----------------------- #
    # ⛔ MATCHED ON (ep_name, t), NEVER ON t ALONE. `t` repeats across
    # episodes (the 25 curved windows take only 6 distinct t values), so a
    # t-keyed dict silently compares a window with a DIFFERENT episode's row
    # and manufactures a disagreement. Measured: it did, on the first pass.
    if banked and os.path.exists(banked):
        with open(banked, encoding="utf-8") as f:
            bj = json.load(f)
        brows = {(str(w["ep_name"]), int(w["t"])): w for w in bj["rows"]}
        d_turn, d_cv, matched, tok_ok = [], [], 0, 0
        for r in curved:
            w = brows.get((str(r["ep"]), int(r["t"])))
            if w is None:
                continue
            matched += 1
            tok_ok += int(w.get("goal_lat") == r["lat"]
                          and w.get("goal_lon") == r["lon"])
            d_turn.append(abs(r["arms"]["shipped__A"]["turn_c_goal_f32"]
                              - float(w["named_A_goal_canonical_c_goal"])))
            d_cv.append(abs(r["arms"]["shipped__A"]["cv_c_goal_f32"]
                            - float(w["named_A_cv_c_goal"])))
        out["controls"]["K6_banked_agreement"] = {
            "source": banked, "n_matched_on_ep_and_t": matched,
            "n_banked_rows": len(bj["rows"]),
            "n_decoded_token_agrees": tok_ok,
            "max_abs_diff_turn_c_goal_f32": max(d_turn) if d_turn else None,
            "max_abs_diff_cv_c_goal_f32": max(d_cv) if d_cv else None,
            "pass": bool(matched and tok_ok == matched
                         and max(d_turn) <= COS32_TOL
                         and max(d_cv) <= COS32_TOL),
            "note": ("convention A only -- the banked cost surface pre-dates "
                     "4139203's goal-side units crossing, so B is not "
                     "expected to match and is not asserted. The tolerance is "
                     "the float32 cosine resolution, which is the resolution "
                     "the banked numbers themselves were quantised to."),
        }
        # ---- the §8.4 STRATIFICATION the floor cannot be read without ----- #
        # ⚠️ these 25 windows are where the DECODER asked for a turn, NOT
        # where the CAR turned. Reading the oracle floor without this split
        # confuses "the true future is straight" with "the cost cannot rank".
        gts = {}
        for r in curved:
            w = brows.get((str(r["ep"]), int(r["t"])))
            if w is not None:
                gts[(r["ep"], r["t"])] = float(w["gt_turn_deg"])
        turners = [r for r in curved
                   if gts.get((r["ep"], r["t"]), 0.0) >= 5.0]
        out["strata"] = {
            "gt_turn_deg_source": "cost_surface_ep2.json rows (banked)",
            "n_curved_goal": len(curved),
            "n_curved_goal_AND_gt_turn_ge_5deg": len(turners),
            "gt_turn_deg_of_the_curved_25": sorted(round(v, 3)
                                                   for v in gts.values()),
            "by_arm_on_the_gt_turners": {},
            "note": ("⛔ the 25 curved-goal windows and the geometric turn "
                     "stratum are DIFFERENT denominators and no count mixes "
                     "them; this sub-table exists only so the §8.4 oracle "
                     "floor can be read on windows where the ground truth "
                     "ACTUALLY turns"),
        }
        for key in ("identity_full__A", "identity_full__B",
                    "oracle_floor__A", "oracle_floor__B",
                    "oracle_floor_shipped_seed__A", "shipped__A"):
            nm = key.rsplit("__", 1)
            adv = [r["arms"][key]["cv_c_goal_f64"]
                   - r["arms"][key]["turn_c_goal_f64"] for r in turners]
            out["strata"]["by_arm_on_the_gt_turners"][key] = {
                "n": len(turners),
                "n_goal_advantage_positive_f64":
                    int(sum(1 for x in adv if x > 0)),
                "goal_advantage_f64": _q(adv),
            }

    # ---- THE VERDICT, against the committed branch ------------------------ #
    maj = (len(curved) // 2) + 1
    n_pos = {c: out["arms"][f"identity_full__{c}"][
        "n_goal_advantage_positive_f64"] for c in ("A", "B")}
    n_pos_orc = {c: out["arms"][f"oracle_floor__{c}"][
        "n_goal_advantage_positive_f64"] for c in ("A", "B")}
    refuted = all(n_pos[c] < maj for c in ("A", "B"))
    out["verdict"] = {
        "majority_threshold": maj,
        "n_curved": len(curved),
        "Q1_n_goal_advantage_positive_at_identity_f64": n_pos,
        "Q1_oracle_floor_n_positive_f64": n_pos_orc,
        "REFUTED": bool(refuted),
        "committed_branch_verbatim": (
            "PREREG_TACTICAL_DECODER.md 8.2 Q1: 'If the goal-term advantage "
            "stays <= 0 on a majority of the 25 even at seed/goal identity "
            "and in f64, then the goal SPACE -- not the decoder, not the "
            "metric, not the weights -- is what cannot represent a manoeuvre, "
            "and the whole decoder-then-cost line is REFUTED.'"),
        "section_8_4_floor_verbatim": (
            "'If the oracle-goal cost also fails to prefer the turn, the "
            "defect is the cost; if it succeeds, the defect is the imagined "
            "goal.'"),
    }
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--episodes", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--nav", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes-n", type=int, default=20)
    ap.add_argument("--window-stride", type=int, default=10)
    ap.add_argument("--lru", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm-tool-dir", default=None,
                    help="dir holding refav1_arm.py (taniteval/tools)")
    ap.add_argument("--banked-cost-surface", default=None,
                    help="cost_surface_<arm>.json — the K6 agreement gate and "
                         "the source of gt_turn_deg for the §8.4 strata")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-out", default=None)
    ap.add_argument("--from-rows", default=None,
                    help="⭐ RE-ANALYSE banked rows with ZERO GPU. The "
                         "compute is already paid for; an analysis fix must "
                         "never re-roll it (the `t1_eval` --analyze-only "
                         "lesson).")
    a = ap.parse_args()

    if a.from_rows:
        with open(a.from_rows, encoding="utf-8") as f:
            d = json.load(f)
        print(f"[e0] re-analysing {len(d['rows'])} banked rows from "
              f"{a.from_rows} — no GPU", flush=True)
    else:
        for req in ("ckpt", "cache", "episodes", "arm_tool_dir"):
            if getattr(a, req) is None:
                raise SystemExit(f"--{req.replace('_', '-')} is required "
                                 f"unless --from-rows is given")
        d = collect(a)
    res = analyse(d, a.banked_cost_surface)
    res["tool"] = "e0_goalspace_probe.py"
    res["tier"] = "T0"
    res["evidence_class"] = "MEASURED"
    res["arm"] = a.name
    res["meta"] = d["meta"]
    res["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    if a.raw_out:
        with open(a.raw_out, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
    print(json.dumps({"meta": res["meta"], "controls": res["controls"],
                      "verdict": res["verdict"]}, indent=1))
    for k in ("shipped__A", "identity_full__A", "identity_plan__A",
              "C_REG__A", "oracle_floor__A"):
        print(f"--- {k}\n" + json.dumps(res["arms"][k], indent=1)[:1800])
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
