"""TanitEval — hierarchy panel (H26): is the operative->tactical->strategic
stack LOAD-BEARING, or decorative?

TanitAD's headline claim is that a hierarchy with cross-layer alignment
(strategic ctx --FiLM--> tactical intent --FiLM--> operative) is efficient AND
dominant: each layer's conditioning measurably helps the layer it conditions,
and the layers stay mutually consistent. This panel PROVES or FALSIFIES that on
the canonical held-out val, for any WorldModel arm with trained tactical_policy
+ strategic_policy (flagship, REF-A). Three measurements:

1. CROSS-LAYER CONDITIONING ABLATION (the key proof). For each conditioning
   seam, run the downstream layer WITH the real upstream signal vs a
   mean/zero-replaced signal — SAME weights, ONLY the FiLM cond changes — and
   report the downstream-metric delta with an episode-jackknife CI (bench.py
   protocol). Orientation is "helps-positive": a positive, CI>0 delta means the
   real upstream signal makes the downstream layer better -> the seam is
   LOAD-BEARING. If the CI straddles 0 the seam is DECORATIVE (reported so).
     nav_cmd -> strategic     route acc: true command vs follow vs zeroed-nav
     strategic ctx -> tactical  maneuver acc, waypoint ADE, goal-latent cos
                                (real ctx vs mean/zero ctx)
     tactical intent -> operative  grounded rollout ADE@2s
                                (real intent vs mean/zero/none intent)

2. CROSS-LAYER CONSISTENCY / AGREEMENT. On each window, do the layers cohere?
   strategic route (L/S/R)  <->  tactical maneuver direction (turn_l/r/keep)
   tactical maneuver dir    <->  operative rolled-trajectory net-heading dir
   Reported as agreement rate (jackknife CI) + Cohen's kappa (beyond-chance),
   OVERALL and on the turn-active subset — the highway-follow corpus is
   straight-dominated, so raw agreement is inflated and kappa is the honest read.

3. PER-LAYER GROUNDED vs UNGROUNDED (H18). At the tactical GOAL: does the
   dynamics-grounded operative rollout endpoint (grounded, via the metric step-
   readout) beat the tactical waypoint head's direct 2 s regression (ungrounded)
   as an ego waypoint vs GT? delta + CI. Grounded winning supports scoring the
   layer's decision by the grounded rollout, not the head's confidence.

Reuses the EXACT grounded rollout machinery (metric_dynamics) and CI protocol
(episode-disjoint jackknife, tanitad.eval.gates.split_by_episode) as bench.py,
so the numbers are apples-to-apples with every gate/leaderboard figure. The only
new code is the ablation harness — the weights are never touched, only the FiLM
conditioning tensor fed into an already-loaded checkpoint."""
from __future__ import annotations

import contextlib
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
sys.path.insert(0, "/root/taniteval")

import refb_labels as rl  # noqa: E402
from driving_diagnostic import WP_STEPS, gt_ego_waypoints  # noqa: E402
from tanitad.eval.gates import split_by_episode  # noqa: E402
from tanitad.models.metric_dynamics import accumulate_se2  # noqa: E402
from tanitad.refs.refb import MANEUVER_CLASSES, ROUTE_CLASSES  # noqa: E402

SPEED_SCALE = 10.0
DT = 0.1                              # 10 Hz
YAW_SCALE = 1.0                       # yaw-rate normalizer (refa_train_plus)
WIN = 8
K_MAX = max(WP_STEPS)                 # 20 steps = 2 s @ 10 Hz
GOAL_H = K_MAX                        # maneuver + goal horizon (2 s)
IDX = [k - 1 for k in WP_STEPS]       # rollout-waypoint indices at WP_STEPS
OP_HORIZONS = (1, 2, 4)               # cfg.predictor.horizons (intent's regime)
N_SPLITS = 8
# Minimum PRACTICALLY-meaningful effect sizes for the load-bearing verdicts. With
# n~880 windows a CI can separate a delta of ~0.001 from 0 — statistically real,
# but too small to call a seam load-bearing. A seam counts only if its delta is
# BOTH CI-separated AND at least this large. Conservative, documented.
MIN_ACC = 0.02                        # maneuver / route accuracy (2 points)
MIN_ADE_M = 0.05                      # waypoint ADE (metres)
MIN_COS = 0.01                        # latent / goal cosine (1%)


def _ego_channels(ep, last, speed_input, yaw_input, dyn_input, device):
    """Ego-motion action channels appended for speed/dyn-input arms, matching
    refa_train_plus._append_ego / flagship exactly. Order [v0, yr0] (v0 first):
      v0  = pose_last.v / SPEED_SCALE                         (--speed-input)
      yr0 = wrap(yaw_last - yaw_{last-1}) / DT / YAW_SCALE    (--yaw/--dyn-input,
            OBSERVED-only, leakage-safe). Returns [b, n_ego] or None.
    Eval always feeds the true ego vector (the dyn-input ego-dropout guard is
    training-only)."""
    feed_yaw = yaw_input or dyn_input
    if not (speed_input or feed_yaw):
        return None
    chans = []
    if speed_input:
        chans.append(ep.poses[last, 3:4].float() / SPEED_SCALE)
    if feed_yaw:
        yr0 = (rl.wrap_to_pi(ep.poses[last, 2] - ep.poses[last - 1, 2]) / DT
               / YAW_SCALE).reshape(-1, 1)
        chans.append(yr0)
    return torch.cat(chans, dim=-1).to(device)

# maneuver class ids (refb_labels): 0 lane_keep 1 turn_left 2 turn_right
#                                   3 accelerate 4 brake_stop
# route/direction class ids (ROUTE_CLASSES): 0 left 1 straight 2 right
R_LEFT, R_STRAIGHT, R_RIGHT = 0, 1, 2
MAN2DIR = {0: R_STRAIGHT, 1: R_LEFT, 2: R_RIGHT, 3: R_STRAIGHT, 4: R_STRAIGHT}
DIR_YAW_RAD = rl.YAW_TURN_RAD         # 0.15 rad net-heading turn threshold (2 s)


# --------------------------------------------------------------------------- #
# Ablation primitives — only the FiLM cond changes, never a weight             #
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _zero_nav(strategic):
    """Temporarily zero the strategic nav-command embedding so the strategic
    FiLM cond is 0 for any command — the "no nav conditioning at all" control.
    Fully reversible (weights restored on exit)."""
    saved = strategic.nav_emb.weight.data.clone()
    strategic.nav_emb.weight.data.zero_()
    try:
        yield
    finally:
        strategic.nav_emb.weight.data.copy_(saved)


@torch.no_grad()
def _rollout_intent(predictor, states, actions, future_actions, step_readout, k,
                    intent):
    """rollout_decode with an explicit intent token threaded into the operative
    predictor's action FiLM. ``intent=None`` reproduces the canonical grounded
    rollout (rollout_decode) byte-for-byte. Returns (waypoints[b,k,2],
    step_dpose[b,k,3])."""
    win_s, win_a = states, actions
    dposes = []
    for j in range(k):
        z_hat = predictor(win_s, win_a, intent=intent)[1]     # 1-step head
        dposes.append(step_readout(win_s[:, -1], z_hat))
        if j < k - 1:
            a_next = (future_actions[:, j] if future_actions is not None
                      else win_a[:, -1])
            win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], dim=1)
            win_a = torch.cat([win_a[:, 1:], a_next.unsqueeze(1)], dim=1)
    step_dpose = torch.stack(dposes, dim=1)
    return accumulate_se2(step_dpose), step_dpose


def _dir_of(net_yaw):
    """Signed net heading (rad, +=CCW=left) -> direction class {L,S,R}."""
    d = torch.full(net_yaw.shape, R_STRAIGHT, dtype=torch.long,
                   device=net_yaw.device)
    d[net_yaw > DIR_YAW_RAD] = R_LEFT
    d[net_yaw < -DIR_YAW_RAD] = R_RIGHT
    return d


# --------------------------------------------------------------------------- #
# Episode-jackknife statistics (bench.py protocol)                             #
# --------------------------------------------------------------------------- #
def _jack(vals, eids, mask=None, n_splits=N_SPLITS):
    """Episode-disjoint jackknife of a per-window value array (mean of the 8
    split-means +- CI95), matching bench._agg. ``vals`` are already in the
    "helps-positive" orientation for a delta, or the raw quantity for a rate.
    Returns {mean, ci95, n, separated} — separated = (mean - ci95 > 0)."""
    v = np.asarray(vals, dtype=float)
    e = np.asarray([int(x) for x in eids])
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        v, e = v[m], e[m]
    n = int(v.size)
    if n == 0 or len(set(e.tolist())) < 2:
        return {"mean": (round(float(np.nanmean(v)), 4) if n else None),
                "ci95": None, "n": n, "separated": False}
    sm = []
    for s in range(n_splits):
        _tr, va = split_by_episode(e.tolist(), 0.2, s)
        if va:
            sm.append(float(np.nanmean(v[va])))
    sm = np.asarray(sm)
    mean = float(np.mean(sm))
    ci = float(1.96 * np.std(sm) / max(1, len(sm)) ** 0.5)
    return {"mean": round(mean, 4), "ci95": round(ci, 4), "n": n,
            "separated": bool(mean - ci > 0)}


def _meaningful(j, min_effect):
    """A delta is load-bearing only if CI-separated from 0 AND at least
    ``min_effect`` in magnitude (guards against negligible-but-significant)."""
    return bool(j.get("separated") and j.get("mean") is not None
                and abs(j["mean"]) >= min_effect)


def _mean(vals, mask=None):
    v = np.asarray(vals, dtype=float)
    if mask is not None:
        v = v[np.asarray(mask, dtype=bool)]
    return round(float(np.nanmean(v)), 4) if v.size else None


def _kappa(a, b, mask=None):
    """Cohen's kappa between two discrete label arrays over {0,1,2}."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        a, b = a[m], b[m]
    if a.size == 0:
        return None
    po = float((a == b).mean())
    pe = 0.0
    for c in (0, 1, 2):
        pe += float((a == c).mean()) * float((b == c).mean())
    return round((po - pe) / (1 - pe), 4) if (1 - pe) > 1e-9 else None


def _hist(vals):
    v = np.asarray(vals, dtype=int)
    return {ROUTE_CLASSES[c]: int((v == c).sum()) for c in (0, 1, 2)}


# --------------------------------------------------------------------------- #
# Panel                                                                        #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run(model, step_readout, episodes, device, speed_input=False, max_eps=40,
        stride=8, batch=16, yaw_input=False, dyn_input=False):
    if getattr(model, "tactical_policy", None) is None or \
       getattr(model, "strategic_policy", None) is None:
        return {"skipped": "no trained tactical/strategic policy brains"}
    model.eval()
    strat, tac_pol, pred = (model.strategic_policy, model.tactical_policy,
                            model.predictor)

    rec = defaultdict(list)                 # per-window scalars / discrete
    cache = []                              # per-batch cached tensors (phase 2)
    sum_ctx = sum_int = None
    n_tot = 0

    # ---- Phase 1: encode + real-cond heads + running cond means + cache -----
    for ep in episodes[:max_eps]:
        fr = ep.feats
        T = fr.shape[0]
        starts = list(range(0, T - WIN - K_MAX, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            b = len(ch)
            last = torch.tensor([t + WIN - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(fr[t:t + WIN]) for t in ch]
                             ).to(device).float()
            if fr.dtype == torch.uint8:
                fw = fw.div_(255.0)
            states = model.encode_window(fw)               # [b, W, S]

            aw = torch.stack([ep.actions[t:t + WIN] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t + WIN:t + WIN + K_MAX] for t in ch]
                             ).to(device)
            ego = _ego_channels(ep, last, speed_input, yaw_input, dyn_input, device)
            if ego is not None:                             # [v0(,yr0)] broadcast
                aw = torch.cat([aw, ego[:, None].expand(-1, aw.shape[1], -1)], -1)
                fa = torch.cat([fa, ego[:, None].expand(-1, fa.shape[1], -1)], -1)

            # --- labels (the trainer's refb_labels derivations) -------------
            fut = torch.stack([torch.as_tensor(ep.poses[t + WIN:t + WIN + GOAL_H])
                               for t in ch]).to(device).float()   # [b,GOAL_H,4]
            pl = ep.poses[last].to(device).float()                # [b,4]
            man_tgt = rl.classify_maneuver(pl[:, 2], fut[:, GOAL_H - 1, 2],
                                           pl[:, 3], fut[:, GOAL_H - 1, 3]).long()
            gtwp = gt_ego_waypoints(ep.poses.float(), last).to(device)  # [b,4,2]
            gt_net = rl.wrap_to_pi(fut[:, GOAL_H - 1, 2] - pl[:, 2])     # signed

            navs, rts, valids = [], [], []
            for t in ch:
                cmd, valid = rl.nav_command(ep.poses, t + WIN - 1)
                navs.append(cmd); rts.append(rl.route_target(cmd))
                valids.append(bool(valid))
            nav = torch.tensor(navs, device=device)
            follow = torch.zeros(b, dtype=torch.long, device=device)

            # --- strategic: real nav / follow / zeroed-nav ------------------
            sf = strat(states, follow)
            sn = strat(states, nav)
            with _zero_nav(strat):
                sz = strat(states, follow)
            ctx_real = sf["ctx"]                            # follow ctx (deploy)

            # --- tactical with REAL ctx -------------------------------------
            tacf = tac_pol(states, ctx_real)
            intent_real = tacf["intent"]
            man_pred = tacf["maneuver_logits"].argmax(-1)
            wp_head = torch.stack([tacf["waypoints"][k] for k in WP_STEPS], 1)
            # true future latents via the future window [last+1 .. last+K_MAX],
            # encoded the SAME way training builds fut_states (model.encode_window
            # -> temporal adapter for REF-A, per-frame for flagship). fut_lat[:,
            # k-1] is the state k steps ahead. This is training-faithful for the
            # intent-conditioned JEPA target (flagship_losses / refa_train_plus)
            # and identical to per-frame encode for the flagship path.
            ffut = torch.stack([torch.as_tensor(fr[t + WIN:t + WIN + K_MAX])
                                for t in ch]).to(device).float()
            if fr.dtype == torch.uint8:
                ffut = ffut.div_(255.0)
            fut_lat = model.encode_window(ffut)             # [b, K_MAX, S]
            z_goal = fut_lat[:, GOAL_H - 1]
            ztrue_h = {h: fut_lat[:, h - 1] for h in OP_HORIZONS}
            goal_cos_real = F.cosine_similarity(tacf["target_latent"], z_goal, -1)

            sum_ctx = ctx_real.sum(0) if sum_ctx is None else sum_ctx + ctx_real.sum(0)
            sum_int = intent_real.sum(0) if sum_int is None else sum_int + intent_real.sum(0)
            n_tot += b

            rec["eid"] += [ep.episode_id] * b
            rec["man_tgt"] += man_tgt.tolist()
            rec["man_pred"] += man_pred.tolist()
            rec["man_corr_real"] += (man_pred == man_tgt).float().cpu().tolist()
            rec["route_tgt"] += rts
            rec["valid"] += valids
            rec["route_nav"] += sn["route_logits"].argmax(-1).tolist()
            rec["route_follow"] += sf["route_logits"].argmax(-1).tolist()
            rec["route_zero"] += sz["route_logits"].argmax(-1).tolist()
            rec["wp_ade_head_real"] += torch.linalg.norm(
                wp_head - gtwp, dim=-1).mean(1).cpu().tolist()
            rec["goal_cos_real"] += goal_cos_real.cpu().tolist()
            rec["gt_dir"] += _dir_of(gt_net).cpu().tolist()

            cache.append(dict(states=states.half().cpu(), aw=aw.cpu(),
                              fa=fa.cpu(), gtwp=gtwp.cpu(), z_goal=z_goal.cpu(),
                              man_tgt=man_tgt.cpu(),
                              intent_real=intent_real.cpu(),
                              ztrue={h: ztrue_h[h].cpu() for h in OP_HORIZONS},
                              b=b))

    if n_tot == 0:
        return {"skipped": "no eligible windows (episodes too short)"}

    mean_ctx, zero_ctx = sum_ctx / n_tot, torch.zeros_like(sum_ctx)
    mean_int, zero_int = sum_int / n_tot, torch.zeros_like(sum_int)

    # ---- Phase 2: ablated tactical + operative rollouts on cached states ----
    for c in cache:
        states = c["states"].to(device).float()
        aw, fa = c["aw"].to(device), c["fa"].to(device)
        gtwp, z_goal = c["gtwp"].to(device), c["z_goal"].to(device)
        man_tgt = c["man_tgt"].to(device)
        b = c["b"]

        # strategic ctx -> tactical ablation (mean / zero ctx)
        for tag, ctx in (("mean", mean_ctx), ("zero", zero_ctx)):
            ta = tac_pol(states, ctx[None].expand(b, -1))
            wp = torch.stack([ta["waypoints"][k] for k in WP_STEPS], 1)
            rec[f"man_corr_{tag}ctx"] += (
                ta["maneuver_logits"].argmax(-1) == man_tgt).float().cpu().tolist()
            rec[f"wp_ade_head_{tag}ctx"] += torch.linalg.norm(
                wp - gtwp, dim=-1).mean(1).cpu().tolist()
            rec[f"goal_cos_{tag}ctx"] += F.cosine_similarity(
                ta["target_latent"], z_goal, -1).cpu().tolist()

        # tactical intent -> operative ablation. intents:
        #   real = the hierarchy's actual intent token for this window
        #   mean = the global-average intent (on-distribution, info removed)
        #   zero = a zeroed intent token
        #   none = no intent term at all (== canonical rollout / leaderboard)
        intents = {"real": c["intent_real"].to(device),
                   "mean": mean_int[None].expand(b, -1),
                   "zero": zero_int[None].expand(b, -1),
                   "none": None}
        ztrue = {h: c["ztrue"][h].to(device) for h in OP_HORIZONS}
        z_cur = states[:, -1]
        # mechanistic diagnostic: the intent's contribution to the predictor
        # FiLM cond vs the action embedding it is ADDED to (predictor.forward:
        # cond = act_emb(actions) + intent_proj(intent)). If intent swamps the
        # action term the operative prediction is corrupted.
        ir = c["intent_real"].to(device)
        rec["cond_ae_norm"] += pred.act_emb(aw)[:, -1].norm(dim=-1).cpu().tolist()
        rec["cond_intent_norm"] += pred.intent_proj(ir).norm(dim=-1).cpu().tolist()
        for tag, it in intents.items():
            # (a) IN-REGIME: intent-conditioned multi-horizon JEPA latent, cos +
            #     persistence-relative error vs the true future latent {1,2,4}.
            preds = pred(states, aw, intent=it)
            coss, rels = [], []
            for h in OP_HORIZONS:
                coss.append(F.cosine_similarity(preds[h], ztrue[h], -1))
                denom = (ztrue[h] - z_cur).norm(dim=-1).clamp_min(1e-6)
                rels.append((preds[h] - ztrue[h]).norm(dim=-1) / denom)
            rec[f"lat_cos_{tag}"] += torch.stack(coss).mean(0).cpu().tolist()
            rec[f"lat_rel_{tag}"] += torch.stack(rels).mean(0).cpu().tolist()
            # (b) DIAGNOSTIC: intent threaded into the 20-step grounded pose
            #     rollout (OUT-OF-REGIME: the readout is calibrated intent-free).
            wp_op, dp = _rollout_intent(pred, states, aw, fa, step_readout,
                                        K_MAX, it)
            rec[f"ade_op_{tag}"] += torch.linalg.norm(
                wp_op[:, IDX] - gtwp, dim=-1).mean(1).cpu().tolist()
            if tag == "none":     # canonical (intent-free) rollout = grounded traj
                rec["traj_dir"] += _dir_of(dp[..., 2].sum(1)).cpu().tolist()

    return _assemble(rec)


def _assemble(rec):
    eids = rec["eid"]
    N = len(eids)
    A = {k: np.asarray(v) for k, v in rec.items()}
    valid = np.asarray(rec["valid"], dtype=bool)

    # ---- SEAM 1: nav_cmd -> strategic (route acc; valid windows only) -------
    rc_nav = (A["route_nav"] == A["route_tgt"]).astype(float)
    rc_follow = (A["route_follow"] == A["route_tgt"]).astype(float)
    rc_zero = (A["route_zero"] == A["route_tgt"]).astype(float)
    # majority baseline: always-predict-straight (the honest bar, NOT 1/3 chance,
    # because the highway-follow corpus is straight-dominated).
    straight_rate = _mean((A["route_tgt"] == R_STRAIGHT).astype(float), valid)
    follow_hist = _hist(A["route_follow"][valid]) if valid.any() else {}
    seam_nav = {
        "route_acc_nav": _mean(rc_nav, valid),
        "route_acc_follow": _mean(rc_follow, valid),
        "route_acc_zeronav": _mean(rc_zero, valid),
        "majority_straight_rate": straight_rate,
        "chance_1_of_3": round(1 / 3, 4),
        "follow_pred_distribution": follow_hist,
        "delta_nav_vs_follow": _jack(rc_nav - rc_follow, eids, valid),
        "delta_nav_vs_zeronav": _jack(rc_nav - rc_zero, eids, valid),
        "delta_follow_vs_zeronav": _jack(rc_follow - rc_zero, eids, valid),
        "n_valid": int(valid.sum()),
    }
    # genuine vision-route understanding = follow-route acc ABOVE the majority
    # baseline (does the head infer route from the scene, or just say straight?)
    seam_nav["vision_route_beats_majority"] = bool(
        (seam_nav["route_acc_follow"] or 0) > (straight_rate or 0) + 0.03)
    seam_nav["load_bearing"] = _meaningful(seam_nav["delta_nav_vs_follow"], MIN_ACC)
    seam_nav["_note"] = ("route target is derived from the SAME future heading as "
                         "the nav command, so nav->route is load-bearing BY "
                         "CONSTRUCTION (the command propagates to the route head). "
                         "The genuine strategic-vision signal is whether "
                         "route_acc_follow beats majority_straight_rate — if the "
                         "follow head just predicts straight, it adds no vision.")

    # ---- SEAM 2: strategic ctx -> tactical ---------------------------------
    def seam_metric(real, meanc, zeroc, lower_better):
        # delta orientation: positive == real signal HELPS
        d_mean = (A[meanc] - A[real]) if lower_better else (A[real] - A[meanc])
        d_zero = (A[zeroc] - A[real]) if lower_better else (A[real] - A[zeroc])
        return {"real": _mean(A[real]), "mean_ctx": _mean(A[meanc]),
                "zero_ctx": _mean(A[zeroc]),
                "delta_real_vs_mean": _jack(d_mean, eids),
                "delta_real_vs_zero": _jack(d_zero, eids)}
    seam_ctx = {
        "maneuver_acc": seam_metric("man_corr_real", "man_corr_meanctx",
                                    "man_corr_zeroctx", lower_better=False),
        "wp_ade_2s": seam_metric("wp_ade_head_real", "wp_ade_head_meanctx",
                                 "wp_ade_head_zeroctx", lower_better=True),
        "goal_latent_cos": seam_metric("goal_cos_real", "goal_cos_meanctx",
                                       "goal_cos_zeroctx", lower_better=False),
    }
    # honest primary control = mean (on-distribution); zero is full-removal ref.
    # load-bearing needs a meaningful mean-control effect on >=1 tactical metric.
    seam_ctx["load_bearing"] = bool(
        _meaningful(seam_ctx["maneuver_acc"]["delta_real_vs_mean"], MIN_ACC)
        or _meaningful(seam_ctx["wp_ade_2s"]["delta_real_vs_mean"], MIN_ADE_M)
        or _meaningful(seam_ctx["goal_latent_cos"]["delta_real_vs_mean"], MIN_COS))
    seam_ctx["content_matters"] = seam_ctx["load_bearing"]

    # ---- SEAM 3: tactical intent -> operative (IN-REGIME latent fidelity) ---
    # intent conditions the multi-horizon JEPA latent prediction (horizons
    # 1/2/4, flagship_losses L169); the grounded pose rollout is intent-free by
    # design, so we measure intent where it actually acts: the predicted next
    # latent. rel = ||z_hat - z_true|| / ||z_true - z_t|| (persistence-relative;
    # lower better). cos = cosine(z_hat, z_true) (higher better).
    seam_int = {
        "_regime": ("intent conditions the intent-FiLM of the multi-horizon JEPA "
                    "latent prediction (horizons 1/2/4); the grounded pose "
                    "rollout is intent-free by design (eval_grounded_rollout_4b) "
                    "so it is NOT the seam's regime — see diagnostic below"),
        "latent_rel_err": {"real": _mean(A["lat_rel_real"]),
                           "mean_intent": _mean(A["lat_rel_mean"]),
                           "zero_intent": _mean(A["lat_rel_zero"]),
                           "none": _mean(A["lat_rel_none"])},
        "latent_cos": {"real": _mean(A["lat_cos_real"]),
                       "mean_intent": _mean(A["lat_cos_mean"]),
                       "zero_intent": _mean(A["lat_cos_zero"]),
                       "none": _mean(A["lat_cos_none"])},
        # helps-positive: rel lower=better -> mean-real ; cos higher=better -> real-mean
        "delta_rel_real_vs_mean": _jack(A["lat_rel_mean"] - A["lat_rel_real"], eids),
        "delta_rel_real_vs_none": _jack(A["lat_rel_none"] - A["lat_rel_real"], eids),
        "delta_cos_real_vs_mean": _jack(A["lat_cos_real"] - A["lat_cos_mean"], eids),
        "delta_cos_real_vs_none": _jack(A["lat_cos_real"] - A["lat_cos_none"], eids),
        "cond_norms": {
            "action_emb": _mean(A["cond_ae_norm"]),
            "intent_proj": _mean(A["cond_intent_norm"]),
            "_note": ("norms in the predictor FiLM-cond space; intent_proj(intent) "
                      ">= action_emb means the intent term SWAMPS the action "
                      "signal it is added to -> corrupts the operative prediction"),
        },
        "diagnostic_intent_in_grounded_rollout_OUT_OF_REGIME": {
            "op_ade_2s_real": _mean(A["ade_op_real"]),
            "op_ade_2s_mean": _mean(A["ade_op_mean"]),
            "op_ade_2s_zero": _mean(A["ade_op_zero"]),
            "op_ade_2s_none_canonical": _mean(A["ade_op_none"]),
            "_note": ("threading intent into the 20-step recursive grounded "
                      "rollout (readout calibrated intent-free) is off-manifold; "
                      "the ADE blow-up shows the intent FiLM DOES strongly "
                      "perturb the operative predictor, but this is not how the "
                      "seam is used"),
        },
    }
    # Two distinct reads (the frozen-encoder confound needs both):
    #  * per_window_content_helps: real beats MEAN intent (on-distribution) —
    #    does the tactical brain's SPECIFIC per-window intent steer the operative?
    #    This is the strict "conditioning helps the conditioned layer" test.
    #  * helps_vs_none / harmful_if_engaged: real vs NONE (no intent term) — does
    #    ENGAGING the intent pathway help or hurt vs an intent-free operative?
    #    A frozen-encoder operative can co-adapt to always having the term
    #    (real>none) WITHOUT using its per-window content (real==mean).
    seam_int["per_window_content_helps"] = _meaningful(
        seam_int["delta_cos_real_vs_mean"], MIN_COS)
    dcn = seam_int["delta_cos_real_vs_none"]            # real - none (helps+)
    seam_int["helps_vs_none"] = _meaningful(dcn, MIN_COS)
    seam_int["harmful_if_engaged"] = bool(
        dcn["mean"] is not None and dcn["ci95"] is not None
        and dcn["mean"] + dcn["ci95"] < 0 and abs(dcn["mean"]) >= MIN_COS)
    # strict load-bearing (thesis) = the tactical intent's per-window content
    # measurably improves the operative it conditions.
    seam_int["load_bearing"] = seam_int["per_window_content_helps"]
    seam_int["deployed_operative_is_intent_free"] = True   # rollout is intent-free

    # ---- CONSISTENCY / AGREEMENT (model-internal coherence) ----------------
    # directions in {0 L, 1 S, 2 R}. traj_dir is from the CANONICAL (intent-free)
    # grounded rollout = the model's actual driven trajectory heading.
    route_f = A["route_follow"]                 # follow head (deploy, no cmd)
    route_n = A["route_nav"]                     # commanded route head
    man_dir = np.array([MAN2DIR[int(m)] for m in A["man_pred"]])
    traj = A["traj_dir"]

    def agree(x, y):
        a = (x == y).astype(float)
        turn = (x != R_STRAIGHT) | (y != R_STRAIGHT)     # >=1 side signals a turn
        return {"agreement": _jack(a, eids), "kappa": _kappa(x, y),
                "agreement_turn_subset": _jack(a, eids, turn),
                "kappa_turn_subset": _kappa(x, y, turn),
                "n_turn_active": int(turn.sum())}
    consistency = {
        "maneuver_vs_trajectory": agree(man_dir, traj),     # same 2 s timescale — KEY
        "commanded_route_vs_maneuver": agree(route_n, man_dir),
        "commanded_route_vs_trajectory": agree(route_n, traj),
        "follow_route_head_degenerate": {
            "distribution": _hist(route_f),
            "_note": ("under follow (no route command) the strategic route head "
                      "collapses to constant-straight -> carries no directional "
                      "signal at deploy; use the commanded head for coherence")},
        "distributions": {"route_follow": _hist(route_f),
                          "route_commanded": _hist(route_n),
                          "maneuver_dir": _hist(man_dir),
                          "trajectory_dir": _hist(traj),
                          "gt_dir": _hist(A["gt_dir"])},
        "_note": ("route is a 15-25 s heading; maneuver/trajectory are 2 s — some "
                  "cross-timescale disagreement is CORRECT, not incoherence. "
                  "maneuver_vs_trajectory is the same-timescale coherence test."),
    }

    # ---- H18: grounded rollout vs ungrounded tactical head @ the goal ------
    # grounded = canonical intent-free operative rollout endpoint (the calibrated
    # grounded surface); ungrounded = tactical waypoint head direct regression.
    h18 = {
        "grounded_op_rollout_ade_2s": _mean(A["ade_op_none"]),
        "ungrounded_tactical_head_ade_2s": _mean(A["wp_ade_head_real"]),
        "delta_ungrounded_minus_grounded": _jack(
            A["wp_ade_head_real"] - A["ade_op_none"], eids),
    }
    h18["grounded_wins"] = _meaningful(
        h18["delta_ungrounded_minus_grounded"], MIN_ADE_M)

    out = {
        "n_windows": N,
        "protocol": {
            "cond_replaced": "FiLM conditioning tensor only; weights fixed",
            "ci": "8-split episode-disjoint jackknife (bench.py protocol)",
            "primary_control": "mean (on-distribution); zero/none = full-removal refs",
            "intent_conditioning": "follow-command hierarchy (deploy-realistic, no route cmd given)",
        },
        "seam_nav_to_strategic": seam_nav,
        "seam_ctx_to_tactical": seam_ctx,
        "seam_intent_to_operative": seam_int,
        "consistency": consistency,
        "h18_grounded_vs_ungrounded": h18,
    }
    out["verdict"] = _verdict(out)
    out["thesis_read"] = _thesis(out)
    return out


def _thesis(o):
    """Map the measurements onto the three parts of TanitAD's headline claim.
    Data-driven so it flips correctly for a future converged checkpoint."""
    nav, ctx, itn = (o["seam_nav_to_strategic"], o["seam_ctx_to_tactical"],
                     o["seam_intent_to_operative"])
    con, h18 = o["consistency"], o["h18_grounded_vs_ungrounded"]
    nav_beneficial = bool(nav["vision_route_beats_majority"])
    ctx_beneficial = bool(ctx["load_bearing"])
    int_beneficial = bool(itn["load_bearing"] and not itn["harmful_if_engaged"])
    n_help = sum((nav_beneficial, ctx_beneficial, int_beneficial))
    mt = con["maneuver_vs_trajectory"]
    coherent = bool(mt["kappa"] is not None and mt["kappa"] >= 0.2)
    goal_d = ctx["goal_latent_cos"]["delta_real_vs_mean"]["mean"]
    return {
        "claim": ("operative->tactical->strategic hierarchy with cross-layer "
                  "alignment is efficient AND dominant"),
        "A_conditioning_helps_conditioned_layer": {
            "nav->strategic": ("adds vision route inference"
                               if nav_beneficial else
                               "command echo only — load-bearing by construction, "
                               "no scene-based route inference (follow==majority)"),
            "ctx->tactical": (("load-bearing" if ctx_beneficial else "decorative")
                              + f"; best on-distribution effect goal-cos Δ{goal_d} "
                              "(maneuver & waypoint deltas not CI-separated)"),
            "intent->operative": (
                ("HARMFUL if engaged (real cos < intent-free)"
                 if itn["harmful_if_engaged"] else
                 "per-window content load-bearing" if itn["load_bearing"] else
                 "helps only as a co-adapted offset (real>none but real~=mean)"
                 if itn["helps_vs_none"] else "decorative (no effect vs none)")
                + f"; vs-none Δcos {itn['delta_cos_real_vs_none']['mean']}, "
                + f"per-window vs-mean Δcos {itn['delta_cos_real_vs_mean']['mean']}; "
                + "deployed operative rollout is intent-free"),
            "n_of_3_seams_beneficial": n_help,
            "verdict": (("SUPPORTED" if n_help >= 2 else "FALSIFIED")
                        + " — top-down conditioning "
                        + ("carries" if n_help >= 2 else "does NOT measurably carry")
                        + " downstream performance at this checkpoint"),
        },
        "B_layers_mutually_consistent": {
            "maneuver_vs_trajectory_kappa": mt["kappa"],
            "verdict": (("SUPPORTED" if coherent else "WEAK")
                        + " — same-timescale tactical maneuver & operative "
                        + ("trajectory cohere" if coherent
                           else "trajectory do not clearly cohere")),
        },
        "C_grounding_dominant_H18": {
            "grounded_ade_2s": h18["grounded_op_rollout_ade_2s"],
            "ungrounded_head_ade_2s": h18["ungrounded_tactical_head_ade_2s"],
            "verdict": ("SUPPORTED — grounded readout beats the ungrounded head"
                        if h18["grounded_wins"] else "NOT SHOWN"),
        },
    }


def _lb(flag):
    return "LOAD-BEARING" if flag else "decorative"


def _verdict(o):
    nav, ctx, itn = (o["seam_nav_to_strategic"], o["seam_ctx_to_tactical"],
                     o["seam_intent_to_operative"])
    h18, con = o["h18_grounded_vs_ungrounded"], o["consistency"]
    bits = [
        f"nav->strat {_lb(nav['load_bearing'])} by construction "
        f"(route {nav['route_acc_nav']} cmd / {nav['route_acc_follow']} follow vs "
        f"{nav['majority_straight_rate']} majority; vision route "
        f"{'ADDS' if nav['vision_route_beats_majority'] else 'NONE'})",
        f"ctx->tactical {_lb(ctx['load_bearing'])} "
        f"(vs-mean: man Δ{ctx['maneuver_acc']['delta_real_vs_mean']['mean']}, "
        f"wp Δ{ctx['wp_ade_2s']['delta_real_vs_mean']['mean']}, "
        f"goal-cos Δ{ctx['goal_latent_cos']['delta_real_vs_mean']['mean']})",
        f"intent->operative {'HARMFUL' if itn['harmful_if_engaged'] else _lb(itn['load_bearing'])} "
        f"(cos vs-none Δ{itn['delta_cos_real_vs_none']['mean']}, "
        f"per-window vs-mean Δ{itn['delta_cos_real_vs_mean']['mean']})",
        f"coherence man~traj kappa={con['maneuver_vs_trajectory']['kappa']} "
        f"(agree {con['maneuver_vs_trajectory']['agreement']['mean']})",
        f"H18 grounded {'BEATS' if h18['grounded_wins'] else '<='} ungrounded goal "
        f"({h18['grounded_op_rollout_ade_2s']} vs "
        f"{h18['ungrounded_tactical_head_ade_2s']} m)",
    ]
    n_lb = sum(s["load_bearing"] for s in (nav, ctx, itn))
    return f"hierarchy {n_lb}/3 seams load-bearing · " + " · ".join(bits)


def main():
    import argparse
    import json
    sys.path.insert(0, "/root/taniteval")
    from taniteval import data, loaders
    from taniteval.registry import MODELS
    ap = argparse.ArgumentParser("taniteval.hierarchy")
    ap.add_argument("--model", default="flagship-speed")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    a = ap.parse_args()
    e = [m for m in MODELS if m["key"] == a.model][0]
    L = loaders.load(e, "cuda")
    files = data.list_val_episodes(
        "/root/valdata/physicalai-val-0c5f7dac3b11", a.episodes)
    eps = (data.load_frames(files) if L["feed"] == "frames"
           else data.load_features(files, L["feed"], "cuda"))
    res = run(L["model"], L["step_readout"], eps, "cuda",
              speed_input=bool(e.get("speed_input")), max_eps=a.episodes,
              stride=a.stride, yaw_input=bool(e.get("yaw_input")),
              dyn_input=bool(e.get("dyn_input")))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
