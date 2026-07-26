"""E1c SHARED CODE — the held-out probe, the frontier statistics and the
PRE-REGISTERED verdict logic.

This module is imported UNCHANGED by all three consumers, which is the point:

  * ``e1c_clsft.py``  — the in-training gate (guard on HELD-OUT, never on the
                        replayed training corpus: the one defect E1b's own data
                        localised).
  * ``e1c_eval.py``   — the frontier evaluator.
  * ``e1c_selftest.py`` — drives the SAME statistics + verdict functions with
                        synthetic per-window arrays built to fail each
                        registered guardrail ONE AT A TIME, and asserts the
                        failing verdict is rendered.

E1b's shipped evaluator would have reported SUCCESS on a BOUND run because its
verdict string never consulted the guardrails.  Here the guardrail evaluation
IS the verdict function, there is exactly one implementation of it, and it is
executed against deliberately-failing synthetic input before the deciding run.

ESTIMATOR: paired episode-cluster bootstrap (taniteval/ci.py), B=2000,
resampling the held-out EPISODES.  ``overlapping_holdout_se`` is used NOWHERE.
"""
from __future__ import annotations

import numpy as np

B_BOOT = 2000

# --- frozen at pre-registration time (PRE_REGISTRATION_E1C.md §3) ------------ #
CHECKPOINT_STEPS = [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000,
                    2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000]
N_FRONTIER_POINTS = len(CHECKPOINT_STEPS) + 1          # + base (step 0) = 18
BONFERRONI_ALPHA = 0.05 / N_FRONTIER_POINTS            # 0.002778 (§4.4)
OOD_BAND = 1.30                                        # Gc (E1a measured band)
NONINF_TOL_M = 0.05                                    # descriptor only (§4.4)
TIE_TOL = 0.005                                        # selection-rule tie band


# --------------------------------------------------------------------------- #
# estimator wrappers                                                           #
# --------------------------------------------------------------------------- #
def make_stat(ci_mod, n_boot=B_BOOT):
    """Bind the estimator once so every consumer provably uses the same one."""
    def boot(v, eid, reduce="mean"):
        return ci_mod.episode_cluster_bootstrap(np.asarray(v, float), eid,
                                                reduce=reduce, n_boot=n_boot)

    def paired(a, b, eid, reduce="mean"):
        return ci_mod.paired_episode_cluster_bootstrap(
            np.asarray(a, float), np.asarray(b, float), eid,
            n_boot=n_boot, reduce=reduce)
    return boot, paired


# --------------------------------------------------------------------------- #
# OPEN LOOP on HELD-OUT — guardrail (a) ADE@2s + guardrail (b) anchor block     #
#                                                                              #
# VERBATIM from e1b_eval.openloop_full (which is e1a.openloop_canary's loop     #
# plus refc_train.compute_losses' anchor block, lines 268-276, with the GT      #
# target).  Kept identical so the in-training gate and the final evaluator      #
# measure literally the same thing.                                            #
# --------------------------------------------------------------------------- #
def openloop_full(model, episodes, device, W, WP_STEPS, gt_ego_waypoints,
                  stride=8, batch=16):
    import torch
    import torch.nn.functional as F
    P, G, ade, acc, ce, l1, eids, keys, spd = [], [], [], [], [], [], [], [], []
    anchors = model.decoder.anchors.detach().float()          # [Na,4,2]
    with torch.no_grad():
        for ei, ep in enumerate(episodes):
            fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
                else ep.frames.float()
            poses = ep.poses.float()
            T = fr.shape[0]
            starts = list(range(0, T - W - max(WP_STEPS), stride))
            for bi in range(0, len(starts), batch):
                ch = starts[bi:bi + batch]
                frames = torch.stack([fr[t0:t0 + W] for t0 in ch]).to(device)
                last = torch.tensor([t0 + W - 1 for t0 in ch])
                v0 = poses[last, 3].to(device)
                out = model(frames, nav_cmd=None, v0=v0, steps=2)
                pred = out["traj"].float()                     # [B,4,2]
                gt = gt_ego_waypoints(poses, last).to(device).float()
                b = pred.shape[0]
                ade.append(torch.linalg.norm(pred - gt, dim=-1).mean(1).cpu())
                dist = ((gt[:, None] - anchors[None]) ** 2).sum(dim=(-1, -2))
                a_star = dist.argmin(dim=1)
                logits = out["anchor_logits"].float()
                acc.append((logits.argmax(dim=1) == a_star).float().cpu())
                ce.append(F.cross_entropy(logits, a_star, reduction="none").cpu())
                recon = out["anchor_traj"].float()[
                    torch.arange(b, device=device), a_star]
                l1.append((recon - gt).abs().mean(dim=(-1, -2)).cpu())
                P.append(pred.cpu()); G.append(gt.cpu())
                spd.append(poses[last, 3].clone())
                eids += [str(ei)] * b
                keys += [(ei, int(t)) for t in ch]
    return {"key": keys, "eid": eids,
            "ade2s": torch.cat(ade).numpy(),
            "anchor_acc": torch.cat(acc).numpy(),
            "anchor_ce": torch.cat(ce).numpy(),
            "anchor_traj_l1": torch.cat(l1).numpy(),
            "speed": torch.cat(spd).numpy(),
            "pred": torch.cat(P), "gt": torch.cat(G)}


# --------------------------------------------------------------------------- #
# pairing helpers                                                              #
# --------------------------------------------------------------------------- #
def common_index(a_map, b_map, mask_name=None):
    """(idx_a, idx_b, eid) on the windows both arms produced.

    The stratum mask is taken from BASE so both arms see the identical split.
    """
    bk = {k: i for i, k in enumerate(b_map["key"])}
    com = [(i, bk[k]) for i, k in enumerate(a_map["key"]) if k in bk]
    ia = np.array([i for i, _ in com], dtype=int)
    ib = np.array([j for _, j in com], dtype=int)
    if mask_name is not None:
        m = np.asarray(b_map[mask_name])[ib]
        ia, ib = ia[m], ib[m]
    return ia, ib, [b_map["eid"][j] for j in ib]


def paired_field(a_map, b_map, field, paired, mask_name=None, reduce="mean"):
    ia, ib, eid = common_index(a_map, b_map, mask_name)
    if len(ia) < 2:
        return {"n": int(len(ia)), "note": "too few common windows",
                "separated": False}
    return paired(np.asarray(a_map[field])[ia], np.asarray(b_map[field])[ib],
                  eid, reduce=reduce)


def arm_field(m, field, boot, mask_name=None, reduce="mean"):
    idx = (np.flatnonzero(np.asarray(m[mask_name])) if mask_name is not None
           else np.arange(len(m["eid"])))
    if len(idx) < 2:
        return {"n": int(len(idx)), "note": "too few windows"}
    return boot(np.asarray(m[field])[idx], [m["eid"][i] for i in idx],
                reduce=reduce)


# --------------------------------------------------------------------------- #
# the DECIDING statistics for one frontier point                               #
# --------------------------------------------------------------------------- #
CLOSED_FIELDS = ("dep", "win_dep", "peak_xte", "mean_xte", "peak_dpsi",
                 "ood_peak", "ood_mean", "out_env", "ade2s")
OPEN_FIELDS = ("ade2s", "anchor_acc", "anchor_ce", "anchor_traj_l1")
STRATA = {"overall": None, "junction": "junc", "longitudinal": "long"}


def frontier_point_stats(step, ft_cl, base_cl, ft_ol, base_ol, boot, paired,
                         label=""):
    """Every number the pre-registered verdict consumes, for ONE checkpoint.

    ``*_cl`` are per_window() dicts at K=185, ``*_ol`` are openloop_full()
    dicts.  Both arms are on identical windows, so every delta is PAIRED.
    """
    out = {"step": int(step), "label": label,
           "estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py, "
                        "B=%d, resampling held-out EPISODES)" % B_BOOT,
           "closed_loop_K185": {}, "open_loop": {}}

    for nm, mk in STRATA.items():
        out["closed_loop_K185"][nm] = {
            f: {"base": arm_field(base_cl, f, boot, mk),
                "ft": arm_field(ft_cl, f, boot, mk),
                "paired_delta_ft_minus_base":
                    paired_field(ft_cl, base_cl, f, paired, mk)}
            for f in CLOSED_FIELDS}

    for f in OPEN_FIELDS:
        ia, ib, eid = common_index(ft_ol, base_ol)
        out["open_loop"][f] = {
            "base": boot(np.asarray(base_ol[f])[ib], eid),
            "ft": boot(np.asarray(ft_ol[f])[ia], eid),
            "paired_delta_ft_minus_base":
                paired(np.asarray(ft_ol[f])[ia], np.asarray(base_ol[f])[ib],
                       eid)}
    return out


# --------------------------------------------------------------------------- #
# THE PRE-REGISTERED PREDICATES (PRE_REGISTRATION_E1C.md §4.1)                  #
# --------------------------------------------------------------------------- #
def _sep_lower(d):
    """CI-separated LOWER: an improvement for a lower-is-better metric."""
    return bool(isinstance(d, dict) and d.get("separated") and d.get("hi", 0) < 0)


def _sep_higher(d):
    """CI-separated HIGHER."""
    return bool(isinstance(d, dict) and d.get("separated") and d.get("lo", 0) > 0)


def evaluate_point(st):
    """Apply the six frozen conditions to ONE frontier point's statistics.

    P1  corridor-departure@K185 OVERALL   separated LOWER          (must fire)
    P2  corridor-departure@K185 JUNCTION  separated LOWER          (must fire)
    Ga  open-loop ADE@2s        NOT separated-worse (higher)       (must hold)
    Gb1 open-loop anchor_acc    NOT separated-worse (lower)        (must hold)
    Gb2 open-loop anchor_traj_l1 NOT separated-worse (higher)      (must hold)
    Gc  closed-loop OOD peak ratio (ft, overall) <= 1.30           (must hold)
    """
    cl = st["closed_loop_K185"]
    ol = st["open_loop"]
    d_ov = cl["overall"]["dep"]["paired_delta_ft_minus_base"]
    d_ju = cl["junction"]["dep"]["paired_delta_ft_minus_base"]
    g_ade = ol["ade2s"]["paired_delta_ft_minus_base"]
    g_acc = ol["anchor_acc"]["paired_delta_ft_minus_base"]
    g_l1 = ol["anchor_traj_l1"]["paired_delta_ft_minus_base"]
    ood_ft = cl["overall"]["ood_peak"]["ft"].get("mean", float("inf"))

    v = {
        "step": int(st["step"]),
        "label": st.get("label", ""),
        "P1_dep_overall_separated_lower": _sep_lower(d_ov),
        "P2_dep_junction_separated_lower": _sep_lower(d_ju),
        "Ga_openloop_ade2s_ok": not _sep_higher(g_ade),
        "Gb1_anchor_acc_ok": not _sep_lower(g_acc),
        "Gb2_anchor_traj_l1_ok": not _sep_higher(g_l1),
        "Gc_ood_in_band": bool(ood_ft <= OOD_BAND + 1e-9),
        # --- reported, NON-DECIDING descriptors (§4.4) ---
        "_dep_overall_delta": d_ov.get("delta"),
        "_dep_junction_delta": d_ju.get("delta"),
        "_openloop_ade2s_delta": g_ade.get("delta"),
        "_openloop_ade2s_ci": [g_ade.get("lo"), g_ade.get("hi")],
        "_ood_peak_ft": ood_ft,
        "multiplicity_robust": bool(
            d_ov.get("p_delta_gt0", 1.0) < BONFERRONI_ALPHA and
            d_ju.get("p_delta_gt0", 1.0) < BONFERRONI_ALPHA and
            _sep_lower(d_ov) and _sep_lower(d_ju)),
        "noninferior_0p05": bool(g_ade.get("hi", 1e9) < NONINF_TOL_M),
    }
    v["primary_ok"] = bool(v["P1_dep_overall_separated_lower"] and
                           v["P2_dep_junction_separated_lower"])
    v["guardrails_ok"] = bool(v["Ga_openloop_ade2s_ok"] and
                              v["Gb1_anchor_acc_ok"] and
                              v["Gb2_anchor_traj_l1_ok"] and
                              v["Gc_ood_in_band"])
    v["SUCCESS_POINT"] = bool(v["primary_ok"] and v["guardrails_ok"])
    v["failed"] = sorted(k for k in
                         ("P1_dep_overall_separated_lower",
                          "P2_dep_junction_separated_lower",
                          "Ga_openloop_ade2s_ok", "Gb1_anchor_acc_ok",
                          "Gb2_anchor_traj_l1_ok", "Gc_ood_in_band")
                         if not v[k])
    return v


def select_winner(evals):
    """Frozen selection rule (§4.2): among SUCCESS points take the most negative
    paired Delta on corridor-departure@K185 OVERALL; ties (within TIE_TOL)
    broken by the smaller open-loop ADE@2s Delta; still tied -> the earlier step.
    """
    cands = [e for e in evals if e["SUCCESS_POINT"]]
    if not cands:
        return None
    best = min(c["_dep_overall_delta"] for c in cands)
    tied = [c for c in cands if c["_dep_overall_delta"] <= best + TIE_TOL]
    tied.sort(key=lambda c: (round(c["_openloop_ade2s_delta"], 6), c["step"]))
    return tied[0]


def render_verdict(evals):
    """The single verdict string.  It CONSUMES the guardrails — it cannot say
    SUCCESS while a guardrail has failed, which is exactly the defect that made
    E1b's shipped evaluator report SUCCESS on a BOUND run."""
    w = select_winner(evals)
    if w is None:
        n = len(evals)
        n_prim = sum(1 for e in evals if e["primary_ok"])
        n_guard = sum(1 for e in evals if e["guardrails_ok"])
        why = ("the primary never fired" if n_prim == 0 else
               "the held-out guardrails never held" if n_guard == 0 else
               "the two never held at the SAME checkpoint")
        return {
            "verdict": "BOUND",
            "winner_step": None,
            "text": ("BOUND: NO checkpoint on the %d-point frontier satisfies all "
                     "six pre-registered conditions. Primary (P1 and P2, "
                     "corridor-departure@K185 CI-separated LOWER on overall AND "
                     "junction) fired at %d/%d points; the held-out guardrails "
                     "(Ga, Gb1, Gb2, Gc) held at %d/%d; the intersection is EMPTY "
                     "— %s. Per PRE_REGISTRATION_E1C §4.2 this means the "
                     "closed-loop / open-loop trade is REAL and not an artifact "
                     "of the guard."
                     % (n, n_prim, n, n_guard, n, why)),
            "n_points": n,
            "n_primary_ok": n_prim,
            "n_guardrails_ok": n_guard,
            "n_success_points": 0,
            "primary_ok_steps": [e["step"] for e in evals if e["primary_ok"]],
            "guardrails_ok_steps": [e["step"] for e in evals if e["guardrails_ok"]],
        }
    n_succ = sum(1 for e in evals if e["SUCCESS_POINT"])
    return {
        "verdict": "SUCCESS",
        "winner_step": w["step"],
        "text": ("SUCCESS: checkpoint step %d satisfies all six pre-registered "
                 "conditions (corridor-departure@K185 overall Delta %s and "
                 "junction Delta %s, both CI-separated LOWER; open-loop ADE@2s "
                 "Delta %s CI %s not separated-worse; anchor block not "
                 "separated-worse; OOD peak %.4f <= %.2f). %d of %d frontier "
                 "points qualified; the frozen selection rule (§4.2) picks step "
                 "%d. multiplicity_robust=%s (Bonferroni p<%.6f)."
                 % (w["step"], w["_dep_overall_delta"], w["_dep_junction_delta"],
                    w["_openloop_ade2s_delta"], w["_openloop_ade2s_ci"],
                    w["_ood_peak_ft"], OOD_BAND, n_succ, len(evals), w["step"],
                    w["multiplicity_robust"], BONFERRONI_ALPHA)),
        "n_points": len(evals),
        "n_primary_ok": sum(1 for e in evals if e["primary_ok"]),
        "n_success_points": n_succ,
    }


# --------------------------------------------------------------------------- #
# the in-training HELD-OUT gate (§4.3)                                         #
# --------------------------------------------------------------------------- #
def heldout_gate_row(step, ft_ol, base_ol, boot, paired):
    """The corrected forgetting guard: paired Delta(current - base) on HELD-OUT
    open loop.  NEVER computed on the replay corpus."""
    ia, ib, eid = common_index(ft_ol, base_ol)
    row = {"step": int(step), "n_windows": int(len(ia)),
           "n_episodes": int(len(set(eid)))}
    for f in OPEN_FIELDS:
        d = paired(np.asarray(ft_ol[f])[ia], np.asarray(base_ol[f])[ib], eid)
        row[f] = {"ft_mean": round(float(np.asarray(ft_ol[f])[ia].mean()), 5),
                  "base_mean": round(float(np.asarray(base_ol[f])[ib].mean()), 5),
                  "paired_delta": d}
    row["Ga_openloop_ade2s_ok"] = not _sep_higher(row["ade2s"]["paired_delta"])
    row["Gb1_anchor_acc_ok"] = not _sep_lower(row["anchor_acc"]["paired_delta"])
    row["Gb2_anchor_traj_l1_ok"] = not _sep_higher(
        row["anchor_traj_l1"]["paired_delta"])
    row["gate_ok"] = bool(row["Ga_openloop_ade2s_ok"])
    return row
