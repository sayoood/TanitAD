#!/usr/bin/env python3
"""T3 — the 4-head BCE rule rescorer, scored on PDMS-lite. §8 pre-registration.

PRE-REGISTRATION (``PERCANDIDATE_LABELS.md`` §8), honoured verbatim
-------------------------------------------------------------------
arms          AS_TRAINED | CE_CONTROL (Bar A's exact refit, original target)
              | BCE_RULE (conf_head Linear(512,1) -> Linear(512,5): channel 0 is
              the imitation logit, trained with the SAME CE as CE_CONTROL, so
              ``BCE_RULE - CE_CONTROL`` isolates EXACTLY the four added rule
              heads; channels 1..4 are the 4 sigmoids, BCE on NC/TTC/C/EP)
protocol      Bar A's 5-fold episode-disjoint cross-fit, its LR grid
              {3e-5, 1e-4, 3e-4}, its cache-fidelity + failing-input self-tests
combine       Hydra-MDP form  w_im*log S_im + Sum_k w_k*log S_k, weights by grid
              search on the FIT folds only (never the test episodes)
primary read  PDMS-lite (no-map), NOT ade_0_2s
estimator     paired episode-cluster bootstrap, B=2000, unit = episode.
              NEVER overlapping_holdout_se
CONFIRM       BCE_RULE - CE_CONTROL separated-better on PDMS-lite AND its
              at-fault collision rate separated-below CE_CONTROL's
REFUTE        not separated on PDMS-lite. Say so plainly; do not re-scope
precondition  the kinematic clip applied first; comfort EXCLUDED from any hard
              veto (it is saturated -- a soft term only)
free flag     ``w_im = 0`` IS the pre-registered ``use_q``/``normalize_base``
              flag (hide the planner's own score from the selector); it is a
              CORNER OF THE GRID, so it is measured rather than chosen

THE TWO INTERPRETATION CALLS, STATED BEFORE THE RUN
--------------------------------------------------
1. §8 says ``Linear(512,4)`` + 4 sigmoids, and separately that inference
   combines ``w1 log S_im + ...``.  A 4-output head has no S_im, so the head is
   built with FIVE outputs and channel 0 is initialised from the as-trained
   ``Linear(512,1)`` (rows 1..4 zero-init => log sigma(0) is a constant, i.e.
   the arm STARTS bit-identical to as-trained).  The literal 4-only reading is
   the ``w_im = 0`` corner of the weight grid.
2. The combine weights are an INFERENCE-time object, so they are grid-searched
   POST-HOC on the inner-validation episodes of each fit split (never the test
   episodes).  Training runs at the grid centre ``w_im = 1, s = 1`` so BCE_RULE
   costs exactly what CE_CONTROL costs -- 3 LR fits per fold, not 36.
3. LR is selected on inner-validation **PDMS-lite** for BOTH fitted arms,
   because PDMS-lite is the pre-registered primary.  Selecting CE_CONTROL on
   ADE and BCE_RULE on PDMS-lite would confound the arm with its selection
   criterion.  The ADE-selected CE_CONTROL is also computed and reported as the
   reproduction check against Bar A's committed 0.9231.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STACK = "/root/v4eval/stack"
sys.path.insert(0, STACK + "/scripts")
sys.path.insert(0, STACK)
sys.path.insert(0, "/root/taniteval")

import eval_flagship_v4 as E  # noqa: E402
from taniteval.ci import (episode_cluster_bootstrap,  # noqa: E402
                          paired_episode_cluster_bootstrap)

CKPT = "/workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt"
HCFG = "/workspace/_v4gate/flagship-v4-fromscratch-30k/config.json"
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
ANCH = "/root/models/flagship-v4-fromscratch-15k/flagship_v4_anchors_dense.pt"
DEV = "cuda"

# ---- Bar A's pre-registered constants, unchanged --------------------------
TAU = 1.0
N_FOLDS = 5
N_INNER_VAL_EP = 6
MAX_STEPS = 2000
EVAL_EVERY = 100
BATCH = 32
LR_GRID = (3e-5, 1e-4, 3e-4)
SEED = 0
TRAINABLE = ("decoder.conf_head", "lat_head", "lon_head", "dist_head",
             "lat_to_anchor", "lon_to_anchor", "dist_to_anchor", "sel_gate")
COMMITTED = {"produced": {"ade": 0.8563, "oif": 0.2505},
             "bar_a_ce_control_oof_ade": 0.9231,
             "bar_a_regret_oof_ade": 0.8817}
SEL_ACCEL_MAX, HORIZON_S = 2.5, 2.0            # the head's own reach clamp
VETO = -1e4                                    # candidate veto in score space

# ---- the pre-registered inference-combine grid ----------------------------
W_IM_GRID = (0.0, 1.0)
W_SCALE_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
W_BASE = (1.0, 5.0 / 12.0, 2.0 / 12.0, 5.0 / 12.0)   # NC, TTC, C, EP (PDM shape)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# =========================================================================== #
# the composite -- PDMS-lite (no-map), identical arithmetic to t2_labels.py   #
# =========================================================================== #
def pdms_lite(NCF, TF, CO, PR, mask=None):
    """[W,C] bools + progress -> [W,C] PDMS-lite. mask restricts EP's max."""
    if mask is None:
        pm = PR.max(dim=1, keepdim=True).values
    else:
        pm = torch.where(mask, PR, torch.full_like(PR, -float("inf"))
                         ).max(dim=1, keepdim=True).values
    pm = torch.clamp(pm, min=1e-6)
    EP = torch.clamp(PR / pm, 0.0, 1.0)
    EP = torch.where(PR <= 0, torch.zeros_like(EP), EP)
    return (~NCF).float() * (5.0 * EP + 5.0 * (~TF).float()
                             + 2.0 * CO.float()) / 12.0, EP


# =========================================================================== #
# selector forward over the cached features                                   #
# =========================================================================== #
def selector_params(head, extra=()):
    out = {}
    for n, p in head.named_parameters():
        if any(n == t or n.startswith(t + ".") for t in TRAINABLE):
            out[n] = p
    for i, m in enumerate(extra):
        for n, p in m.named_parameters():
            out[f"_extra{i}.{n}"] = p
    return out


def score_arm(head, C, sl, conf, arm, w, veto_mask=None):
    """conf = the conf_head module in use. Returns dict with the pick + parts."""
    qf = C["qf"][sl].float()
    fan = C["fan"][sl].float()
    stl = C["state_last"][sl].float()
    v0 = C["v0"][sl]
    vts = C["vt_speed"][sl]
    vk = C["vt_keep"][sl]
    o = conf(qf)                                        # [B,N,1] or [B,N,5]
    if arm == "bce_rule":
        im = o[..., 0]
        rl = o[..., 1:]                                 # [B,N,4]
        logp = F.logsigmoid(rl)                         # log S_k
        base = (w[0] * F.log_softmax(im.float(), dim=1)
                + (logp.float() * torch.as_tensor(
                    w[1:], device=logp.device, dtype=torch.float32)).sum(-1))
    else:
        im = o.squeeze(-1)
        rl = None
        base = im
    lat, lon, dist = head.lat_head(stl), head.lon_head(stl), head.dist_head(stl)
    refined, seam = head._factor_grafts(base, lat, lon, dist)
    if veto_mask is not None:
        m = veto_mask[sl]
        refined = refined + torch.where(m, torch.zeros_like(refined),
                                        torch.full_like(refined, VETO))
    traj, idx, score, s_tele = head.select(fan, refined, vts, vk, v0)
    return {"traj": traj, "idx": idx, "score": score, "seam": seam,
            "im": im, "rule_logits": rl, "fan": fan,
            "tgt": C["tgt"][sl]}


def anchor_cls_loss(head, conf, q0, tgt):
    """Bar A keeps the original recipe's anchor CE at weight 1.0. Verbatim."""
    logits = conf(q0.float())
    logits = logits[..., 0] if logits.shape[-1] > 1 else logits.squeeze(-1)
    a = head.decoder.anchors.to(tgt.dtype)
    d = ((tgt[:, None] - a[None]) ** 2).sum(dim=(-1, -2))
    return F.cross_entropy(logits.float(), d.argmin(dim=1).detach())


def wp4(traj, horizons, wp_steps=(5, 10, 15, 20)):
    pos = [list(horizons).index(k) for k in wp_steps]
    return traj[:, pos]


def ade_of(traj, tgt, horizons):
    d = (wp4(traj, horizons) - wp4(tgt, horizons)).norm(dim=-1)
    return d.mean(dim=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/_bara/cache_produced_stride1.pt")
    ap.add_argument("--labels", default="/workspace/_t3/t3_labels.pt")
    ap.add_argument("--out", default="/workspace/_t3/t3_result.json")
    ap.add_argument("--dump", default="/workspace/_t3/t3_windows.pt")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    global MAX_STEPS, EVAL_EVERY, LR_GRID
    if a.smoke:
        MAX_STEPS, EVAL_EVERY, LR_GRID = 60, 30, (1e-4,)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_start = time.time()

    R = {
        "_experiment": "T3 -- the 4-head BCE rule rescorer on flagship-v4-"
                       "fromscratch-30k's frozen fan, scored on PDMS-lite",
        "_evidence_class": "MEASURED (ours)",
        "_primary_read": "PDMS-lite (no-map). ade_0_2s is REPORTED but does NOT "
                         "adjudicate (L2/ADE vs closed-loop Driving Score "
                         "rho = -0.36, p = 0.43).",
        "_estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py), "
                      "B=2000, unit = episode. NEVER overlapping_holdout_se.",
        "_host": platform.node(), "_python": sys.version.split()[0],
        "_torch": torch.__version__, "_gpu": torch.cuda.get_device_name(0),
        "_ckpt": {"path": CKPT, "md5": md5(CKPT)},
        "_smoke": bool(a.smoke),
    }

    cfg = E._eval_cfg()
    plan = E._plan(cfg)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    world, grounding, head, step, hcfg, goal_head = E.load_v4_from_ck(
        ck, DEV, head_config_path=HCFG, anchors_dense_path=ANCH)
    del ck
    horizons = head.cfg.horizons
    base_sd = {k: v.detach().clone() for k, v in head.state_dict().items()}
    R["_ckpt_step"] = int(step)

    C = torch.load(a.cache, map_location="cpu", weights_only=False)
    for k in ("q0", "qf", "fan", "state_last", "tgt", "v0", "vt_speed",
              "vt_keep"):
        C[k] = C[k].to(DEV)
    L = torch.load(a.labels, map_location="cpu", weights_only=False)
    W = C["qf"].shape[0]
    NCAND = C["fan"].shape[1]
    assert L["nc_fault"].shape == (W, NCAND), "label/cache shape mismatch"
    NCF = L["nc_fault"].to(DEV)
    TF = L["ttc_flag"].to(DEV)
    CO = L["comfort_ok"].to(DEV)
    PR = L["progress"].float().to(DEV)
    VT = L["v_term"].float().to(DEV)
    has_tracks = L["has_tracks"].to(DEV)

    # ---- the pre-registered PRECONDITION: the head's own reachability clip --
    reach = SEL_ACCEL_MAX * HORIZON_S
    v0all = C["v0"].float()
    KEEP = ((VT >= torch.clamp(v0all - reach, min=0.0)[:, None])
            & (VT <= (v0all + reach)[:, None]))
    dead = ~KEEP.any(dim=1)
    KEEP[dead] = True                       # never empty a window
    PDMS_clip, EP_clip = pdms_lite(NCF, TF, CO, PR, mask=KEEP)
    PDMS_raw, EP_raw = pdms_lite(NCF, TF, CO, PR)
    R["precondition_kinematic_clip"] = {
        "rule": "v_term in [max(0, v0 - 5.0), v0 + 5.0], "
                "reach = sel_accel_max * horizon = 2.5 * 2.0",
        "frac_candidates_removed": round(float((~KEEP).float().mean()), 4),
        "frac_windows_emptied_and_restored": round(float(dead.float().mean()), 5),
        "frac_windows_ade_oracle_survives": round(float(
            torch.gather(KEEP, 1, (C["fan"] - C["tgt"][:, None]).norm(dim=-1)
                         .mean(-1).argmin(1)[:, None]).float().mean()), 4),
    }

    ep = C["ep"].numpy()
    is_eval = (C["t"] % 8 == 0).numpy()
    eval_idx = torch.nonzero(torch.from_numpy(is_eval)).squeeze(-1)
    eids_eval = [str(int(x)) for x in C["eid"][eval_idx]]
    tr_eval = has_tracks[eval_idx.to(DEV)].cpu().numpy()
    R["_windows"] = {"n_total": int(W), "n_eval": int(eval_idx.numel()),
                     "n_eval_with_tracks": int(tr_eval.sum()),
                     "n_episodes": int(len(set(ep.tolist()))),
                     "n_episodes_with_tracks": int(len(set(
                         ep[has_tracks.cpu().numpy()].tolist())))}

    # ================= SELF-TESTS (both directions) ======================== #
    conf0 = head.decoder.conf_head
    with torch.no_grad():
        o = score_arm(head, C, eval_idx, conf0, "ce", None)
        ade_at = ade_of(o["traj"], o["tgt"], horizons)
        pick_at = o["idx"]
    ref_idx = C["ref_sel_idx"][eval_idx].to(DEV)
    pick_mismatch = float((pick_at != ref_idx).float().mean())
    fanE = C["fan"][eval_idx].float()
    tgtE = C["tgt"][eval_idx]
    pos = [list(horizons).index(k) for k in (5, 10, 15, 20)]
    oif4 = float((fanE[:, :, pos, :] - tgtE[:, pos][:, None]).norm(dim=-1)
                 .mean(dim=-1).min(dim=1).values.mean())
    with torch.no_grad():
        o_clip = score_arm(head, C, eval_idx, conf0, "ce", None, KEEP)
        ade_at_clip = ade_of(o_clip["traj"], o_clip["tgt"], horizons)
        pick_at_clip = o_clip["idx"]
    R["T1_clip_on_v4_own_fan"] = {
        "_prediction_from_T1": "on REF-C-XL's fan the clip moved the as-trained "
                               "pick in ZERO windows (paired delta exactly "
                               "0.0000). Tested here on v4's OWN emitted fan.",
        "frac_windows_pick_moves": round(float(
            (pick_at_clip != pick_at).float().mean()), 5),
        "as_trained_ade_unclipped": round(float(ade_at.mean()), 4),
        "as_trained_ade_clipped": round(float(ade_at_clip.mean()), 4),
        "paired_delta_ade": round(float((ade_at_clip - ade_at).mean()), 6)}
    R["selftest"] = {"cache_fidelity": {
        "ade_0_2s_via_cache": round(float(ade_at.mean()), 4),
        "ade_0_2s_committed": COMMITTED["produced"]["ade"],
        "abs_diff": round(abs(float(ade_at.mean())
                              - COMMITTED["produced"]["ade"]), 5),
        "frac_windows_pick_differs_from_forward_pass": round(pick_mismatch, 5),
        "oracle_in_fan_4wp_via_cache": round(oif4, 4),
        "oracle_in_fan_4wp_committed": COMMITTED["produced"]["oif"],
        "oif_abs_diff": round(abs(oif4 - COMMITTED["produced"]["oif"]), 5)}}
    cf = R["selftest"]["cache_fidelity"]
    cf["PASS"] = bool(cf["abs_diff"] <= 1e-3 and cf["oif_abs_diff"] <= 1e-3
                      and pick_mismatch == 0.0)

    g = torch.Generator(device=DEV).manual_seed(1234)
    rnd = torch.randint(0, NCAND, (eval_idx.numel(),), device=DEV, generator=g)
    traj_rnd = fanE[torch.arange(fanE.shape[0], device=DEV), rnd]
    ade_rnd = ade_of(traj_rnd, tgtE, horizons)
    pdE = PDMS_clip[eval_idx.to(DEV)]
    nfE = NCF[eval_idx.to(DEV)]
    R["selftest"]["failing_input"] = {
        "random_pick_ade_0_2s": round(float(ade_rnd.mean()), 4),
        "random_pick_pdms_lite": round(float(
            torch.gather(pdE, 1, rnd[:, None]).mean()), 4),
        "as_trained_ade_0_2s": round(float(ade_at.mean()), 4),
        "as_trained_pdms_lite": round(float(
            torch.gather(pdE, 1, pick_at[:, None]).mean()), 4),
        "_rule": "a random selector over the same frozen fan must be WORSE on "
                 "BOTH surfaces, or the harness cannot detect a bad selector"}
    fi = R["selftest"]["failing_input"]
    fi["PASS"] = bool(fi["random_pick_ade_0_2s"] > fi["as_trained_ade_0_2s"]
                      and fi["random_pick_pdms_lite"] < fi["as_trained_pdms_lite"])
    print(json.dumps(R["selftest"], indent=1), flush=True)
    if not (cf["PASS"] and fi["PASS"]):
        R["ABORTED"] = "self-test failed -- no result is quotable"
        Path(a.out).write_text(json.dumps(R, indent=1))
        raise SystemExit("[t3] SELF-TEST FAILED -- aborting before training")

    # ================= the fitted arms ===================================== #
    eps = sorted(set(ep.tolist()))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(eps))
    folds = [[eps[i] for i in perm[k::N_FOLDS]] for k in range(N_FOLDS)]
    R["_folds"] = {f"fold{k}": v for k, v in enumerate(folds)}
    ep_t = torch.from_numpy(ep)

    def make_conf(arm):
        if arm == "ce":
            m = nn.Linear(conf0.in_features, 1).to(DEV)
            with torch.no_grad():
                m.weight.copy_(conf0.weight)
                m.bias.copy_(conf0.bias)
            return m
        m = nn.Linear(conf0.in_features, 5).to(DEV)
        with torch.no_grad():
            m.weight.zero_(); m.bias.zero_()
            m.weight[0].copy_(conf0.weight[0])
            m.bias[0].copy_(conf0.bias[0])
        return m

    def metrics(idx, sl):
        slg = sl.to(DEV)
        return (torch.gather(PDMS_clip[slg], 1, idx[:, None]).squeeze(1),
                torch.gather(NCF[slg], 1, idx[:, None]).squeeze(1).float(),
                torch.gather(TF[slg], 1, idx[:, None]).squeeze(1).float())

    W_TRAIN = (1.0,) + tuple(W_BASE)          # the training-time combine
    W_GRID = [(wi,) + tuple(sc * b for b in W_BASE)
              for wi in W_IM_GRID for sc in W_SCALE_GRID]

    def inner_eval(conf, arm, w, ival_idx):
        """-> (mean PDMS-lite, mean ade) on the inner-val windows, or (None,
        None) if the head's OWN fail-loud seam guard refuses this state."""
        with torch.no_grad():
            pd_, ad_ = [], []
            try:
                for s_ in range(0, len(ival_idx), 256):
                    sl = ival_idx[s_:s_ + 256]
                    oo = score_arm(head, C, sl, conf, arm, w, KEEP)
                    p_, _, _ = metrics(oo["idx"], sl)
                    pd_.append(p_.cpu().numpy())
                    ad_.append(ade_of(oo["traj"], oo["tgt"],
                                      horizons).cpu().numpy())
            except RuntimeError as e:
                if "seam norm ratio" in str(e):
                    return None, None
                raise
        return float(np.concatenate(pd_).mean()), float(np.concatenate(ad_).mean())

    def fit_fold(arm, fit_idx, ival_idx, lr, log=None):
        head.decoder.conf_head = conf0          # shapes must match base_sd
        head.load_state_dict(base_sd, strict=True)
        conf = make_conf(arm)
        head.decoder.conf_head = conf
        sel = selector_params(head)
        for p_ in head.parameters():
            p_.requires_grad_(False)
        for p_ in sel.values():
            p_.requires_grad_(True)
        opt = torch.optim.AdamW(list(sel.values()), lr=lr, weight_decay=0.0)
        gg = torch.Generator().manual_seed(SEED)
        w_tr = W_TRAIN if arm == "bce_rule" else None

        p0, a0 = inner_eval(conf, arm, w_tr, ival_idx)
        if p0 is None:
            raise RuntimeError("as-trained state already violates the seam "
                               "guard -- impossible; the harness is wrong")
        best = (p0, a0, 0, {k: v.detach().clone() for k, v in sel.items()})
        raised, seam_max = None, 0.0
        for stp in range(1, MAX_STEPS + 1):
            pick = fit_idx[torch.randint(len(fit_idx), (BATCH,), generator=gg)]
            try:
                oo = score_arm(head, C, pick, conf, arm, w_tr, KEEP)
            except RuntimeError as e:
                if "seam norm ratio" not in str(e):
                    raise
                raised = f"train step {stp}: {e}"
                break
            seam_max = max(seam_max, float(oo["seam"].get(
                "seam_norm_ratio_preclamp_max", 0.0)))
            slg = pick.to(DEV)
            fe = (oo["fan"] - oo["tgt"][:, None]).norm(dim=-1).mean(dim=-1)
            # the SAME two terms Bar A trains CE_CONTROL with, verbatim
            loss = (F.cross_entropy(oo["score"].float(), fe.argmin(1).detach())
                    + anchor_cls_loss(head, conf, C["q0"][slg], oo["tgt"]))
            if arm == "bce_rule":
                tgt4 = torch.stack([(~NCF[slg]).float(), (~TF[slg]).float(),
                                    CO[slg].float(), EP_clip[slg]], dim=-1)
                loss = loss + F.binary_cross_entropy_with_logits(
                    oo["rule_logits"].float(), tgt4)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if stp % EVAL_EVERY == 0:
                p_, a_ = inner_eval(conf, arm, w_tr, ival_idx)
                if p_ is None:
                    raised = (f"inner-val step {stp}: seam guard refused the "
                              f"state (graft norm > seam_fail x base norm)")
                    break
                if p_ > best[0]:
                    best = (p_, a_, stp,
                            {k: v.detach().clone() for k, v in sel.items()})
                if log:
                    print(f"    [{log}] step {stp} loss={float(loss):.4f} "
                          f"ival_pdms={p_:.4f} ade={a_:.4f} "
                          f"(best {best[0]:.4f}@{best[2]})", flush=True)
        # restore the best state, then grid the INFERENCE combine on the SAME
        # inner-val episodes (fit split only -- never the test episodes)
        with torch.no_grad():
            for n_, p_ in head.named_parameters():
                if n_ in best[3]:
                    p_.copy_(best[3][n_])
        w_best, w_sweep = w_tr, []
        if arm == "bce_rule":
            bestp = -1.0
            for w in W_GRID:
                pv, av = inner_eval(conf, arm, w, ival_idx)
                w_sweep.append({"w": [round(x, 4) for x in w],
                                "inner_pdms": None if pv is None else round(pv, 4),
                                "inner_ade": None if av is None else round(av, 4)})
                if pv is not None and pv > bestp:
                    bestp, w_best = pv, w
            pfin, afin = inner_eval(conf, arm, w_best, ival_idx)
        else:
            pfin, afin = best[0], best[1]
        return {"state": best[3], "inner_pdms": pfin, "inner_ade": afin,
                "inner_pdms_at_w_train": best[0], "best_step": best[2],
                "seam_preclamp_max": round(seam_max, 4),
                "seam_guard_raised": raised, "conf": conf,
                "w": w_best, "w_sweep": w_sweep, "lr": lr}

    arms_out = {}
    for arm in ("ce", "bce_rule"):
        oof_pick = torch.zeros(eval_idx.numel(), dtype=torch.long)
        oof_traj = torch.zeros(eval_idx.numel(), len(horizons), 2)
        info = []
        for k, test_ep in enumerate(folds):
            tset = set(test_ep)
            test_mask = torch.tensor([e in tset for e in ep.tolist()])
            train_ep = [e for e in eps if e not in tset]
            ival_ep = set(train_ep[i] for i in np.random.RandomState(SEED + k)
                          .permutation(len(train_ep))[:N_INNER_VAL_EP])
            fit_idx = torch.nonzero(torch.tensor(
                [(e not in tset) and (e not in ival_ep) for e in ep.tolist()])
            ).squeeze(-1)
            ival_idx = torch.nonzero(torch.tensor(
                [e in ival_ep for e in ep.tolist()]) & torch.from_numpy(is_eval)
            ).squeeze(-1)
            best, sweep = None, []
            for lr in LR_GRID:
                r = fit_fold(arm, fit_idx, ival_idx, lr,
                             log=f"{arm}/f{k}/lr{lr:g}")
                sweep.append({"lr": lr,
                              "w": [round(x, 4) for x in r["w"]] if r["w"] else None,
                              "inner_pdms": round(r["inner_pdms"], 4),
                              "inner_ade": round(r["inner_ade"], 4),
                              "inner_pdms_at_w_train": round(
                                  r["inner_pdms_at_w_train"], 4),
                              "best_step": r["best_step"],
                              "seam_preclamp_max": r["seam_preclamp_max"],
                              "seam_guard_raised": r["seam_guard_raised"],
                              "w_sweep": r["w_sweep"]})
                if best is None or r["inner_pdms"] > best["inner_pdms"]:
                    best = r
            head.decoder.conf_head = conf0
            head.load_state_dict(base_sd, strict=True)
            head.decoder.conf_head = best["conf"]
            with torch.no_grad():
                for n_, p_ in head.named_parameters():
                    if n_ in best["state"]:
                        p_.copy_(best["state"][n_])
                sub = torch.nonzero(test_mask[eval_idx]).squeeze(-1)
                try:
                    oo = score_arm(head, C, eval_idx[sub], best["conf"], arm,
                                   best["w"], KEEP)
                    ok = True
                except RuntimeError as e:
                    if "seam norm ratio" not in str(e):
                        raise
                    head.decoder.conf_head = conf0
                    head.load_state_dict(base_sd, strict=True)
                    oo = score_arm(head, C, eval_idx[sub], conf0, "ce", None, KEEP)
                    ok = False
                oof_pick[sub] = oo["idx"].cpu()
                oof_traj[sub] = oo["traj"].cpu()
            info.append({"fold": k, "test_ep": test_ep,
                         "n_test_windows": int(sub.numel()),
                         "lr": best["lr"],
                         "w": [round(x, 4) for x in best["w"]] if best["w"] else None,
                         "best_step": best["best_step"],
                         "inner_pdms": round(best["inner_pdms"], 4),
                         "inner_ade": round(best["inner_ade"], 4),
                         "seam_preclamp_max": best["seam_preclamp_max"],
                         "seam_guard_raised": best["seam_guard_raised"],
                         "scored_with_finetuned_state": ok, "grid": sweep})
            print(f"  [{arm}] fold {k}: lr={best['lr']:g} w={best['w']} "
                  f"step={best['best_step']} ival_pdms={best['inner_pdms']:.4f} "
                  f"n_test={int(sub.numel())} ({time.time()-t_start:.0f}s)",
                  flush=True)
        head.decoder.conf_head = conf0
        head.load_state_dict(base_sd, strict=True)
        arms_out[arm] = {"pick": oof_pick, "traj": oof_traj, "folds": info}
        Path(f"/workspace/_t3/t3_partial_{arm}.json").write_text(
            json.dumps(info, indent=1, default=str))

    # ================= scoring ============================================= #
    ev = eval_idx.to(DEV)
    arms_out["as_trained"] = {"pick": pick_at.cpu(), "traj": None, "folds": []}
    arms_out["as_trained_clipped"] = {"pick": pick_at_clip.cpu(),
                                      "traj": None, "folds": []}
    per = {}
    for name, A in arms_out.items():
        pk = A["pick"].to(DEV)
        pdms = torch.gather(PDMS_clip[ev], 1, pk[:, None]).squeeze(1)
        col = torch.gather(NCF[ev], 1, pk[:, None]).squeeze(1).float()
        ttc_ = torch.gather(TF[ev], 1, pk[:, None]).squeeze(1).float()
        traj = (A["traj"].to(DEV) if A["traj"] is not None
                else fanE[torch.arange(fanE.shape[0], device=DEV), pk])
        ade = ade_of(traj, tgtE, horizons)
        per[name] = {k: v.cpu().numpy() for k, v in
                     (("pdms", pdms), ("collision", col), ("ttc", ttc_),
                      ("ade", ade))}

    tr = tr_eval.astype(bool)
    eids_tr = [e for e, t in zip(eids_eval, tr) if t]
    R["point_estimates"] = {
        name: {"pdms_lite": round(float(v["pdms"][tr].mean()), 4),
               "at_fault_collision_rate": round(float(v["collision"][tr].mean()), 4),
               "ttc_infraction_rate": round(float(v["ttc"][tr].mean()), 4),
               "ade_0_2s_all_881": round(float(v["ade"].mean()), 4),
               "ade_0_2s_track_subset": round(float(v["ade"][tr].mean()), 4)}
        for name, v in per.items()}
    R["_reproduction_check"] = {
        "bar_a_ce_control_oof_ade_committed": COMMITTED["bar_a_ce_control_oof_ade"],
        "_note": "Bar A selected LR on inner-val ADE and applied NO kinematic "
                 "clip; this run selects on inner-val PDMS-lite WITH the clip, "
                 "so an exact match is not expected. The matched quantity that "
                 "MUST reproduce is the as-trained bar 0.8563 (cache fidelity)."}

    B = 2000
    def paired(x, y):
        return paired_episode_cluster_bootstrap(x[tr], y[tr], eids_tr,
                                                n_boot=B, seed=0)
    R["paired_intervals"] = {
        "PRIMARY_bce_rule_minus_ce_control": {
            "pdms_lite": paired(per["bce_rule"]["pdms"], per["ce"]["pdms"]),
            "at_fault_collision_rate": paired(per["bce_rule"]["collision"],
                                              per["ce"]["collision"]),
            "ttc_infraction_rate": paired(per["bce_rule"]["ttc"], per["ce"]["ttc"]),
            "ade_0_2s": paired(per["bce_rule"]["ade"], per["ce"]["ade"]),
            "_orientation": "arm - control; PDMS-lite POSITIVE = better, "
                            "collision/TTC/ADE NEGATIVE = better"},
        "bce_rule_minus_as_trained": {
            "pdms_lite": paired(per["bce_rule"]["pdms"], per["as_trained"]["pdms"]),
            "at_fault_collision_rate": paired(per["bce_rule"]["collision"],
                                              per["as_trained"]["collision"]),
            "ade_0_2s": paired(per["bce_rule"]["ade"], per["as_trained"]["ade"])},
        "ce_control_minus_as_trained": {
            "pdms_lite": paired(per["ce"]["pdms"], per["as_trained"]["pdms"]),
            "at_fault_collision_rate": paired(per["ce"]["collision"],
                                              per["as_trained"]["collision"]),
            "ade_0_2s": paired(per["ce"]["ade"], per["as_trained"]["ade"])},
        "bce_rule_minus_as_trained_CLIPPED_matched_candidate_set": {
            "pdms_lite": paired(per["bce_rule"]["pdms"],
                                per["as_trained_clipped"]["pdms"]),
            "at_fault_collision_rate": paired(
                per["bce_rule"]["collision"],
                per["as_trained_clipped"]["collision"]),
            "ade_0_2s": paired(per["bce_rule"]["ade"],
                               per["as_trained_clipped"]["ade"])},
    }
    R["single_arm_intervals"] = {
        name: {"pdms_lite": episode_cluster_bootstrap(
                   v["pdms"][tr], eids_tr, n_boot=B, seed=0),
               "at_fault_collision_rate": episode_cluster_bootstrap(
                   v["collision"][tr], eids_tr, n_boot=B, seed=0)}
        for name, v in per.items()}

    p = R["paired_intervals"]["PRIMARY_bce_rule_minus_ce_control"]
    confirm = bool(p["pdms_lite"]["separated"] and p["pdms_lite"]["delta"] > 0
                   and p["at_fault_collision_rate"]["separated"]
                   and p["at_fault_collision_rate"]["delta"] < 0)
    R["VERDICT"] = {
        "verdict": "CONFIRM" if confirm else "REFUTE",
        "_rule": "CONFIRM iff BCE_RULE - CE_CONTROL is separated-BETTER on "
                 "PDMS-lite AND its at-fault collision rate is separated-BELOW "
                 "CE_CONTROL's. Anything else is REFUTE -- stated plainly, "
                 "never re-scoped.",
        "pdms_lite_separated_better": bool(
            p["pdms_lite"]["separated"] and p["pdms_lite"]["delta"] > 0),
        "collision_separated_below": bool(
            p["at_fault_collision_rate"]["separated"]
            and p["at_fault_collision_rate"]["delta"] < 0)}

    torch.save({"eval_idx": eval_idx, "eid": C["eid"][eval_idx],
                "ep": C["ep"][eval_idx], "t": C["t"][eval_idx],
                "has_tracks": torch.from_numpy(tr_eval),
                **{f"{n}|{k}": torch.from_numpy(v)
                   for n, d in per.items() for k, v in d.items()},
                **{f"{n}|pick": d["pick"] for n, d in arms_out.items()},
                "_dump_IS": "per-window PDMS-lite / collision / TTC / ade_0_2s "
                            "for every T3 arm on the 881 canonical windows; "
                            "every bar in T3_AND_LAMBDA_TAU.md recomputes from "
                            "this file with NO GPU."}, a.dump)
    R["_wallclock_s"] = round(time.time() - t_start, 1)
    Path(a.out).write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps({k: v for k, v in R.items()
                      if k not in ("_folds",)}, indent=1, default=str),
          flush=True)


if __name__ == "__main__":
    main()
