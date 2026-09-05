"""H-EGO-LIT-4 — score every withheld-bank panel arm on the v7-tiny rig.

⛔ RIG SCOPE. Every number is a property of a ~19 M-param screening rung on a
NON-PARITY corpus (48 train episodes, 2,000 steps, one seed). It validates the
DESIGN and the GATE; it is not a model claim and may not enter MODEL_REGISTRY.md
(H-SCALE-2). Pre-registration: SPEC.md next to this file (GOALS_AND_CLAIMS
`H-EGO-LIT-4`). Structure and windows: the H-ECHO-8 rig's `gate_score.py`
(`.../2026-09-03-refc-v4-design/raw/`), same 640-window draw (seed 20260904).

Per arm, EVERY window is forwarded in two regimes on the SAME weights:
  k  — ego block MEASURED (keep = 1)   : the eval / deployed regime
  w  — ego block WITHHELD (keep = 0)   : the training-time dropout regime, under
                                          the arm's OWN bank policy
and, for the `pred` arm, two more withheld forwards that separate the WEIGHTS
from the BANK: `w_fixed` (bank swapped back to the 10 m/s roll) and `w_shuf`
(the `pred` bank rolled at a PERMUTED row's predicted speed — the gain must
vanish; H-EGODROP-PRED's committed control).

Scored: gate 1 (echo_gate vs ha/ha0/ha0_ext/constant_only), gate 2 (structural)
and 2b (functional source ablation) — the H-ECHO-8 separation instrument —, the
four families on the SELECTED trajectory through `refav1_arm._components` (the
banked instrument) in both regimes paired vs A0, the goal-row families, the
tactical head vs the majority-class control, the withheld ceilings by v0 band
(fixed / own prediction / true v0 = LEAK bound), D3 (endpoint gradient) and D4
(plan predictability). Splits: FIT = first half of the windows (ridge lambdas on
an inner val carved from FIT), TEST = second half, never tuned on.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

STACK = Path(os.environ.get("WB_STACK", r"C:\Users\Admin\refcv4b_suite\stack"))
TE = Path(os.environ.get("WB_TANITEVAL", r"C:\Users\Admin\tanitad-wt\taniteval"))
for p in (str(STACK), str(STACK / "scripts"), str(TE), str(TE / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from refb_train import load_cached_episodes            # noqa: E402
import refc_v3_train as tr                              # noqa: E402
from refc_v3_train import V3Dataset                    # noqa: E402
import refb_labels                                     # noqa: E402  (scripts/)
from tanitad.eval import echo_gate as EG               # noqa: E402
from tanitad.refs import refc_v3 as v3                 # noqa: E402
from taniteval import ci as CI                         # noqa: E402
import refav1_arm as ra                                # noqa: E402

W = Path(os.environ.get("WB_ROOT", r"C:\Users\Admin\run_wbank"))
CORPUS = {
    "val_cache": "physicalai-val-bb543bdf7836 (epcache, 256x256, 40 episode-disjoint val episodes)",
    "train_subset": "physicalai-train-14231cd29c74[:48] -- a NON-parity RIG subset (48 of 401 cached "
                    "episodes); the parity corpus is physicalai-train-e438721ae894 (2376 ep, skip-hash "
                    "f09e44db) and is NOT what these arms trained on",
    "parity": "NOT the parity corpus (rig); cross-arm comparability holds because every arm "
              "trained on the identical 48-episode subset and is scored on the identical 640 windows",
}
EPCACHE = os.environ.get("WB_EPCACHE", r"C:\Users\Admin\tanitad-data\physicalai\_epcache")
ARMS = Path(os.environ.get("ARMS_DIR", str(W / "arms")))
OUT = Path(os.environ.get("PANEL_OUT", str(W / "panel_report.json")))
DUMPS = Path(os.environ.get("PANEL_DUMPS", str(W / "score_dumps")))
BATCH = 16
# WB_DEV=cpu forces CPU WITHOUT CUDA_VISIBLE_DEVICES="" -- that env var makes
# torch.backends.cudnn._init() die on `min([])` when an RNN module is moved
# (MEASURED 2026-09-05 on this box).
DEV = os.environ.get("WB_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
N_BOOT_FAM, N_BOOT_PROBE, SEED = 2000, 500, 0
SLOTS2S = [0, 1, 2, 3]          # 0.5, 1.0, 1.5, 2.0 s of the 8-slot plan (dt 0.5)
SLOTS6S = [1, 3, 4, 5, 6, 7]    # 1..6 s (dt 1.0)
T8 = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
BANDS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 99))
STRAIGHT_IDX = 67
REGIMES_BASE = ("k", "w")


def p(*a):
    print(*a, flush=True)


def paired(a, b, eid, n_boot=N_BOOT_FAM):
    """a - b, paired episode-cluster bootstrap; non-finite pairs dropped (counted)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return {"status": "UNAVAILABLE", "n_windows": int(ok.sum())}
    e = [x for x, k in zip(eid, ok) if k]
    r = CI.paired_episode_cluster_bootstrap(a[ok], b[ok], e, n_boot=n_boot, seed=SEED)
    r["n_dropped_nonfinite"] = int((~ok).sum())
    return r


def level(a, eid, n_boot=N_BOOT_FAM):
    a = np.asarray(a, np.float64)
    ok = np.isfinite(a)
    if ok.sum() < 8:
        return {"status": "UNAVAILABLE", "n_windows": int(ok.sum())}
    e = [x for x, k in zip(eid, ok) if k]
    r = CI.episode_cluster_bootstrap(a[ok], e, n_boot=n_boot, seed=SEED)
    r["n_dropped_nonfinite"] = int((~ok).sum())
    return r


# --------------------------------------------------------------------------- #
# model                                                                       #
# --------------------------------------------------------------------------- #
def build_model(arm_dir: Path):
    cfg_j = json.loads((arm_dir / "config.json").read_text(encoding="utf-8"))
    ego = cfg_j.get("ego", {})
    size = cfg_j.get("size", "tiny")
    assert ego.get("ego_state_inject"), "panel arms are v4 builds"
    cfg = v3.refc_v4_config(size, hier=True, echo_base=bool(ego.get("echo_base")))
    cfg.core.ego_dropout = float(ego.get("ego_dropout", 0.5))
    cfg.tac_vocab_version = cfg_j.get("tac_vocab_version", cfg.tac_vocab_version)
    anc = cfg_j.get("anchors", {}) or {}
    cfg.core.anchors.n_anchors = int(anc.get("shape", [117])[0])
    cfg.core.anchors.v0_conditioned = bool(anc.get("v0_conditioned", True))
    cfg.core.anchors.control_units = anc.get("control_units", "alat")
    sel = cfg_j.get("selection", {}) or {}
    if "sel_accel_max" in sel:
        cfg.core.sel_accel_max = float(sel["sel_accel_max"])
    m = v3.RefCV3Model(cfg)
    sd = torch.load(arm_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    missing, unexpected = m.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if "goal_gate" not in k]
    assert not real_missing, real_missing
    m.eval()
    wb = cfg_j.get("withheld_bank", {}) or {"mode": "fixed"}
    dec = m.core.decoder
    dec.anchor_withheld_speed_max = float(wb.get("speed_max_ms", 35.0))
    return m, cfg, cfg_j, wb, {"missing": len(missing), "unexpected": len(unexpected)}


def rebuild_random_pool(model, seed: int):
    """The `random` arm's marginal, rebuilt EXACTLY as the trainer built it
    (same 48 training episodes, same sub-sampling seed) through the trainer's
    own function — so the eval-time draw is from the distribution the arm
    trained on, not an approximation of it."""
    eps, _ = load_cached_episodes(EPCACHE, "*train*", 48)
    ns = argparse.Namespace(withheld_bank="random", withheld_bank_warmup=0,
                            withheld_speed_max=35.0, seed=seed)
    return tr._apply_withheld_bank(model, ns, eps, DEV)


# --------------------------------------------------------------------------- #
# the shared windows                                                          #
# --------------------------------------------------------------------------- #
def shared_windows(n_eps: int, n_win: int):
    eps, ddir = load_cached_episodes(EPCACHE, "*val*", n_eps)
    cfg0 = v3.refc_v3_sized_config("tiny")
    ds = V3Dataset(eps, window=cfg0.core.window, max_horizon=20, channels=9)
    g = torch.Generator().manual_seed(20260904)
    perm = torch.randperm(len(ds), generator=g)[:n_win].tolist()
    taus = [t * 0.1 for t in cfg0.goal_tau_steps]
    # the t0 kinematics, read exactly as gate_score.py reads them
    v0, a0, k0, vprev, steerprev, man = [], [], [], [], [], []
    for i in perm:
        e_i, t = ds.index[i]
        ep = ds.episodes[e_i]
        t0 = t + cfg0.core.window - 1
        po = torch.as_tensor(ep.poses[t0]).float()
        ac = torch.as_tensor(ep.actions[t0]).float()
        acp = torch.as_tensor(ep.actions[max(t0 - 1, 0)]).float()
        pop = torch.as_tensor(ep.poses[max(t0 - 1, 0)]).float()
        v0.append(po[3]); a0.append(ac[1])
        k0.append(torch.tan(ac[0]) / v3.WHEELBASE_CONST2P9)
        vprev.append(pop[3]); steerprev.append(acp[0])
        mn = getattr(ep, "maneuvers", None)
        man.append(int(mn[t0]) if mn is not None else -1)
    kin = {"v0": torch.stack(v0), "a0": torch.stack(a0), "k0": torch.stack(k0),
           "vprev": torch.stack(vprev), "steerprev": torch.stack(steerprev),
           "man": torch.tensor(man)}
    return eps, ds, perm, taus, cfg0, kin


def batches(ds, perm):
    for s in range(0, len(perm), BATCH):
        yield torch.utils.data.default_collate([ds[i] for i in perm[s:s + BATCH]])


def gt_tensors(ds, perm, cfg0):
    """GT for the shared windows: goal rows, 8-slot waypoints + validity, ids, last frame."""
    T, V, E, FR, PL, FX, AC, WP, SV, RT, NV = [], [], [], [], [], [], [], [], [], [], []
    hz = cfg0.core.trajectory.horizons
    for b in batches(ds, perm):
        T.append(b["goal_tac"].float()); V.append(b["goal_tac_valid"])
        E.append(b["episode_id"].reshape(-1))
        FR.append(b["frames"][:, -1, :3].clone())
        PL.append(b["pose_last"].float()); FX.append(b["future_poses_ext"][:, :20].float())
        AC.append(b["actions"])
        wp = refb_labels.waypoint_targets(b["pose_last"].float(),
                                          b["future_poses_ext"].float(), hz)
        WP.append(wp.float())
        SV.append(torch.stack([b["future_valid_ext"][:, h - 1] for h in hz], dim=1))
        RT.append(b["route_target"].reshape(-1).long()); NV.append(b["nav_valid"].reshape(-1).bool())
    return {"goal": torch.cat(T), "gval": torch.cat(V), "eid": torch.cat(E),
            "frlast": torch.cat(FR), "pose_last": torch.cat(PL), "fx": torch.cat(FX),
            "wp": torch.cat(WP), "sv": torch.cat(SV), "route_target": torch.cat(RT),
            "nav_valid": torch.cat(NV)}


# --------------------------------------------------------------------------- #
# forwards                                                                    #
# --------------------------------------------------------------------------- #
def forward_regime(model, cfg, ds, perm, regime: str, bank_mode: str,
                   ablate: bool, shuffle: bool = False, seed: int = 0):
    """One pass over the shared windows in one regime under one bank policy.
    regime 'k' = keep 1; 'w' = keep 0 (values zeroed by the model). `shuffle`
    = the pred bank at a PERMUTED row's predicted speed (derangement within
    the batch); the model's own prediction is read from a first pass."""
    dec = model.core.decoder
    dec.anchor_withheld_bank = bank_mode
    torch.manual_seed(seed)
    R = {k: [] for k in ("g_tac", "traj", "sel_idx", "bank", "bsp", "lat", "lon",
                         "keep", "route")}
    for b in batches(ds, perm):
        fr = b["frames"].to(DEV)
        if ablate:
            fr = torch.full_like(fr, float(fr.mean()))
        pose_last = b["pose_last"].to(DEV)
        ego = v3.ego_state_from_batch({"pose_last": pose_last,
                                       "actions": b["actions"]}, device=DEV)
        if regime == "w":
            ego = ego.clone(); ego[:, 4] = 0.0
        kw = dict(nav_cmd=b["nav_cmd"].to(DEV), v0=pose_last[:, 3],
                  steps=cfg.core.decoder.diffusion_steps, ego_state=ego)
        with torch.no_grad():
            if shuffle:
                first = model(fr, **kw)
                bsp = first["bank_speed_pred"]
                n = bsp.shape[0]
                perm_i = torch.roll(torch.arange(n, device=bsp.device), 1) if n > 1 \
                    else torch.arange(n, device=bsp.device)
                out = model(fr, withheld_speed=bsp[perm_i], **kw)
            else:
                out = model(fr, **kw)
        R["g_tac"].append(out["g_tac"].float().cpu())
        R["traj"].append(out["traj"].float().cpu())
        R["sel_idx"].append(out["sel_idx"].reshape(-1).cpu())
        R["bank"].append(out["anchor_bank"].float().cpu())
        R["bsp"].append(out["bank_speed_pred"].float().cpu())
        R["lat"].append(out["lat_logits_tac"].float().cpu())
        R["lon"].append(out["lon_logits_tac"].float().cpu())
        R["keep"].append(out["ego_keep"].float().cpu())
        R["route"].append(out["route_logits"].float().cpu())
    return {k: torch.cat(v) for k, v in R.items()}


def oiv(bank, wp, sv):
    """min over the bank of the slot-masked ADE to the GT 8-slot plan -> [N]."""
    d = torch.linalg.vector_norm(bank - wp[:, None], dim=-1)          # [N, M, 8]
    ade = (d * sv[:, None].float()).sum(-1) / sv.float().sum(-1).clamp_min(1)[:, None]
    return ade.min(1).values, ade.argmin(1)


def roll_at(dec, speeds):
    """the decoder's own roll at explicit per-row speeds (every row 'kept')."""
    v = torch.as_tensor(speeds, dtype=torch.float32, device=dec.anchors.device)
    keep = torch.ones(v.shape[0], dtype=torch.bool, device=v.device)
    prev = dec.anchor_withheld_bank
    dec.anchor_withheld_bank = "fixed"
    with torch.no_grad():
        out = torch.cat([dec.roll_bank(v[s:s + 64], keep[s:s + 64], int(v[s:s + 64].shape[0]),
                                       torch.float32).cpu()
                         for s in range(0, v.shape[0], 64)])
    dec.anchor_withheld_bank = prev
    return out


def by_band(v0, fn):
    rows = []
    v0 = np.asarray(v0)
    for lo, hi in BANDS:
        m = (v0 >= lo) & (v0 < hi)
        if m.sum() == 0:
            continue
        rows.append({"v0_band_ms": [lo, hi], "n": int(m.sum()), **fn(m)})
    return rows


# --------------------------------------------------------------------------- #
# the four-families artifact, one per arm x regime, in the registry's shape    #
# --------------------------------------------------------------------------- #
def _decision(pt, gt, dt):
    """trajectory-derived lat/lon decisions via the canonical labeller."""
    from taniteval import four_families as ff
    from tanitad.refs.refc_tactical import factor_from_kinematics
    dy_p, dv_p, v0p, v1p, _ = ff.maneuver_kinematics(pt, dt)
    dy_g, dv_g, v0g, v1g, _ = ff.maneuver_kinematics(gt, dt)
    lat_p, lon_p = factor_from_kinematics(dy_p, dv_p, v0p, v1p)
    lat_g, lon_g = factor_from_kinematics(dy_g, dv_g, v0g, v1g)
    return lat_p.long(), lon_p.long(), lat_g.long(), lon_g.long()


def _dec_block(pred, gt):
    k = int(max(int(pred.max()), int(gt.max()))) + 1
    conf = torch.zeros(k, k, dtype=torch.long)
    for g_, p_ in zip(gt.tolist(), pred.tolist()):
        conf[g_, p_] += 1
    cnt = torch.bincount(gt, minlength=k).float()
    return {"accuracy": float((pred == gt).float().mean()), "n": int(gt.numel()),
            "majority_class_control": float(cnt.max() / cnt.sum()),
            "confusion_gt_rows_pred_cols": conf.tolist()}


def write_family_artifact(path, arm, tag, comps2, P2, G2, r, Gt, eid, sel, a_star,
                          controls=None, corpus=None):
    """`four_families.*` in the CRITERIA_REGISTRY.json key layout, so
    `tools/criteria_check.py` can score PRESENT / REFUSED / ABSENT on it."""
    from taniteval import four_families as ff
    pt, gt = torch.as_tensor(P2).float(), torch.as_tensor(G2).float()
    Pg, Gg = ff._seq_geometry(pt, 0.5), ff._seq_geometry(gt, 0.5)
    prog = (Pg["along"][:, -1] - Gg["along"][:, -1]).abs().numpy()   # 2 s endpoint
    lat_p, lon_p, lat_g, lon_g = _decision(pt, gt, 0.5)
    rt, nv = Gt["route_target"], Gt["nav_valid"]
    rp = r["route"].argmax(-1)
    m = nv & (rt >= 0)
    strat = ({"decision_accuracy": float((rp[m] == rt[m]).float().mean()), "n": int(m.sum()),
              "majority_class_control": float(torch.bincount(rt[m]).float().max() / m.sum()),
              "_is": "3-way route head (v2.1 route target from nav) on nav-valid windows -- "
                     "the ONLY strategic label the epcache rig carries; g_str is unsupervised here"}
             if int(m.sum()) >= 32 else {"status": "TOO_FEW_LABELLED", "n": int(m.sum())})
    art = {
        "block": "taniteval.driving/tier1", "tier": "T1",
        "scope": "RIG (tiny rung, 48 non-parity train episodes, 2,000 steps, 1 seed) -- never a "
                 "model claim (H-SCALE-2); T1-style self-action open loop on the selected plan",
        "arm": arm, "regime": tag, "n_windows": int(len(eid)), "n_episodes": int(len(set(eid.tolist()))),
        "estimator": "paired_episode_cluster_bootstrap",
        "corpus": corpus or {},
        # ctrl.floor_comparison: every family metric paired against the trivial
        # controls on the SAME windows (negative delta = the arm beats the control)
        "controls": controls or {},
        # leak guards, stated for THIS regime rather than assumed
        "protocol": {
            "inference_inputs": (["frames (9-ch window)", "nav_cmd (v1 derivation)",
                                  "ego_state @ t0 = (v0, a_long, yaw_rate, curvature, keep=1) "
                                  "-- E11' (PI 2026-09-03), guarded by ego_dropout 0.5 + X15 + "
                                  "the anti-echo gate"] if tag == "k" else
                                 ["frames (9-ch window)", "nav_cmd (v1 derivation)",
                                  "ego_state WITHHELD (keep=0, values zeroed): vision + nav only"]),
            "vision_only": (tag != "k"),
            "vision_only_note": ("kept regime: the measured ego block at t0 is an admitted "
                                 "inference input under E11' (PI 2026-09-03); withheld regime: "
                                 "no ego channel reaches any node"),
            "goal_source": ("g_str / g_tac / goal_point_tac read from pooled frames (+ nav, + the "
                            "ego block at t0 in the kept regime); no situation-classifier output "
                            "enters any goal (RefCV3Model.provenance_roles: situation_output = [])"),
            "corpus": (corpus or {}).get("val_cache"),
        },
        "strategic": {
            "echo_test": {
                "route_target_is_a_function_of_the_nav_input": True,
                "reading": ("the v2.1 route target is DERIVED from nav_cmd (v1), so a route-head "
                            "accuracy near 1.0 here would be an echo of its own input (C6); the "
                            "strategic family above is read against the majority-class control "
                            "only and is not a strategic capability claim"),
            }
        },
        "four_families": {
            "longitudinal": {"speed_mae_mps": level(comps2["LON_speed_mae_mps"], eid),
                             "along_mae_m": level(comps2["LON_along_mae_m"], eid),
                             "accel_mae_mps2": level(comps2["LON_accel_mae_mps2"], eid),
                             "ego_progress": level(prog, eid)},
            "lateral": {"cross_mae_m": level(comps2["LAT_cross_mae_m"], eid),
                        "heading_mae_deg": level(comps2["LAT_heading_mae_deg"], eid),
                        "yaw_rate_mae_degps": level(comps2["LAT_yaw_rate_mae_radps"] * 180.0 / math.pi, eid)},
            "tactical": {"lateral_decision": _dec_block(lat_p, lat_g),
                         "longitudinal_decision": _dec_block(lon_p, lon_g),
                         "goal_setting": {"agrees_with_own_oracle_frac": float((sel == a_star).mean()),
                                          "straight_ahead_idx67_frac": float((sel == STRAIGHT_IDX).mean()),
                                          "n": int(sel.size)}},
            "strategic": strat,
        },
        "refused": {
            "headway_ttc_distance_keeping": {
                "reason": "no lead-agent join on the epcache rig: obstacle.offline is a pod-side join "
                          "and b1_eval_lead_block.npz covers the B1 eval corpus, not the val epcache "
                          "windows", "n": 0},
            "curvature_mae_at_this_resolution": {
                "reason": "the 2 s grid is 4 slots at dt 0.5 s; a finite-difference curvature on 4 "
                          "points is noise -- heading and yaw-rate carry the lateral family here",
                "n": int(len(eid))},
            "nav_compliance": {
                "reason": "no nav-shuffle / nav-zero control on this rig (v1 nav derivation, "
                          "`follow` + invalid on most windows); the route-head accuracy above is "
                          "the only strategic readout", "n": int(m.sum())},
        },
    }
    Path(path).write_text(json.dumps(art, indent=1, default=float), encoding="utf-8")
    return art


# --------------------------------------------------------------------------- #
# D3 / D4                                                                     #
# --------------------------------------------------------------------------- #
def d3_endpoint_gradient(model, cfg, ds, perm, ablate: bool):
    """‖∂(x_T, y_T)/∂ s_0‖ at the 6 s endpoint of the SELECTED trajectory,
    kept regime, w.r.t. (v0 through BOTH entry points, a_long, yaw_rate,
    curvature). Two backward passes per batch (x_T.sum(), y_T.sum()); rows are
    independent in eval mode, so the per-row rows of the batch gradient ARE
    the per-window partials. Mean and median over windows, plus the per-channel
    split so a v0-dominated reading can be seen for what it is."""
    dec = model.core.decoder
    dec.anchor_withheld_bank = "fixed"
    norms, per_ch = [], []
    for b in batches(ds, perm):
        fr = b["frames"].to(DEV)
        if ablate:
            fr = torch.full_like(fr, float(fr.mean()))
        pose_last = b["pose_last"].to(DEV)
        ego = v3.ego_state_from_batch({"pose_last": pose_last,
                                       "actions": b["actions"]}, device=DEV)
        ego = ego.detach().clone().requires_grad_(True)
        v0 = pose_last[:, 3].detach().clone().requires_grad_(True)
        out = model(fr, nav_cmd=b["nav_cmd"].to(DEV), v0=v0,
                    steps=cfg.core.decoder.diffusion_steps, ego_state=ego)
        end = out["traj"][:, -1, :2]                                   # [B, 2]
        J = []
        for c in range(2):
            g_ego, g_v0 = torch.autograd.grad(end[:, c].sum(), [ego, v0],
                                              retain_graph=(c == 0), allow_unused=True)
            g_ego = torch.zeros_like(ego) if g_ego is None else g_ego
            g_v0 = torch.zeros_like(v0) if g_v0 is None else g_v0
            row = g_ego[:, :4].clone()
            row[:, 0] = row[:, 0] + g_v0                    # v0 enters twice
            J.append(row)
        J = torch.stack(J, dim=1)                          # [B, 2, 4]
        norms.append(torch.linalg.matrix_norm(J).detach().cpu())
        per_ch.append(torch.linalg.vector_norm(J, dim=1).detach().cpu())   # [B, 4]
    n = torch.cat(norms); c = torch.cat(per_ch)
    return {"mean": float(n.mean()), "median": float(n.median()),
            "per_channel_mean": {k: float(c[:, i].mean()) for i, k in
                                 enumerate(("v0", "a_long", "yaw_rate", "curvature"))},
            "n": int(n.numel()),
            "_reads": ("‖J‖_F of the 6 s endpoint w.r.t. the measured t0 state; a "
                       "v0-conditioned bank contributes a kinematic term every arm "
                       "shares, so compare ACROSS arms, not to zero")}


def ridge_probe(X, Y, fit_idx, score_idx, lambdas=(1e-3, 1e-2, 1e-1, 1, 10, 100)):
    """Predict Y [n, d] from X [n, k] by ridge; lambda chosen on an INNER val
    carved from FIT (never on the scored split). Returns 1 - R^2 on TEST
    (relative error; 1.0 = the constant predictor), plus n and d."""
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    fit = np.asarray(fit_idx); sc = np.asarray(score_idx)
    n_in = len(fit) // 2
    inner_fit, inner_val = fit[:n_in], fit[n_in:]

    def _fit(idx, lam):
        Xi = np.c_[X[idx], np.ones(len(idx))]
        mu = Y[idx].mean(0)
        A = Xi.T @ Xi + lam * np.eye(Xi.shape[1]); A[-1, -1] -= lam
        return np.linalg.solve(A, Xi.T @ (Y[idx] - mu)), mu

    def _err(idx, Wm, mu):
        Xi = np.c_[X[idx], np.ones(len(idx))]
        pred = Xi @ Wm + mu
        return float(((pred - Y[idx]) ** 2).sum() / ((Y[idx] - Y[idx].mean(0)) ** 2).sum())

    best = min(lambdas, key=lambda l: _err(inner_val, *_fit(inner_fit, l)))
    Wm, mu = _fit(fit, best)
    return {"rel_err_1_minus_r2": _err(sc, Wm, mu), "lambda": best,
            "n_fit": int(len(fit)), "n_score": int(len(sc)), "d_in": int(X.shape[1]),
            "d_out": int(Y.shape[1]),
            "constant_only_control": _err(sc, np.zeros((X.shape[1] + 1, Y.shape[1])),
                                          Y[fit].mean(0))}


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    n_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    n_win = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    DUMPS.mkdir(parents=True, exist_ok=True)
    eps, ds, perm, taus, cfg0, kin = shared_windows(n_eps, n_win)
    G = gt_tensors(ds, perm, cfg0)
    eid = G["eid"].numpy(); n = len(perm)
    fit_idx = list(range(n // 2)); score_idx = list(range(n // 2, n))
    v0 = kin["v0"]; a0 = kin["a0"]; k0 = kin["k0"]
    # controls on the goal rows (taus) and on the 8-slot plan (T8)
    ref = {"ha0": EG.ha0(v0, taus), "ha0_ext": EG.ha0_ext(v0, a0, k0, taus),
           "ha": EG.ha_finite_diff_accel(v0, kin["vprev"], kin["steerprev"], taus)}
    ref8 = {"ha0": EG.ha0(v0, T8)[..., :2], "ha0_ext": EG.ha0_ext(v0, a0, k0, T8)[..., :2],
            "ha": EG.ha_finite_diff_accel(v0, kin["vprev"], kin["steerprev"], T8)[..., :2]}
    const = EG.constant_only_reference(G["goal"])
    straight = kin["man"] == 0
    report = {"_scope": ("RIG RUNG (tiny) on a NON-PARITY corpus — validates the DESIGN "
                         "and the GATE, never a model claim (H-SCALE-2)"),
              "spec": "SPEC.md (H-EGO-LIT-4)", "device": DEV,
              "n_episodes_pool": len(eps), "n_windows": n, "taus_s": taus,
              "plan_slots_s": T8, "estimator": "paired_episode_cluster_bootstrap",
              "n_boot": {"families": N_BOOT_FAM, "probes": N_BOOT_PROBE},
              "splits": {"fit": "windows[:n/2] (ridge lambdas on an inner val carved "
                                "from FIT)", "test": "windows[n/2:], never tuned on"},
              "manoeuvre_census": {"straight": int(straight.sum()),
                                   "non_straight": int((kin["man"] > 0).sum()),
                                   "unlabelled": int((kin["man"] < 0).sum())},
              "arms": {}}

    def _ade_goal(pred, k, m):
        return torch.linalg.vector_norm(pred[m][:, k, :2] - G["goal"][m][:, k, :2],
                                        dim=-1).reshape(-1, 1).numpy()

    # ---- the model-free references, once ----------------------------------
    sv = G["sv"]; wp = G["wp"]
    comps_ref = {nm: ra._components(ref8[nm][:, SLOTS2S].numpy(), wp[:, SLOTS2S].numpy(), 0.5)
                 for nm in ref8}
    report["controls"] = {
        "goal_rows_by_slot": [
            {"slot": k, "tau_s": taus[k], "n": int(G["gval"][:, k].sum()),
             **{nm: float(_ade_goal(ref[nm], k, G["gval"][:, k]).mean()) for nm in ref},
             "constant_only": float(_ade_goal(const, k, G["gval"][:, k]).mean())}
            for k in range(G["goal"].shape[1]) if int(G["gval"][:, k].sum()) >= 32],
        "plan_2s_families": {nm: {mk: float(np.nanmean(c[mk])) for mk in c}
                             for nm, c in comps_ref.items()},
        "ha_equals_ha0_ext_max_abs_m": float((ref["ha"][..., :2] - ref["ha0_ext"][..., :2])
                                             .abs().max()),
        "_reads": "constant_only must read its own ADE; ha == ha0_ext where the "
                  "harness holds (a, kappa) — the finite-difference a differs from "
                  "the corpus ax on some rows, so a small gap is expected, not zero",
    }
    # D4 on the HUMAN plan, once (the reference the copycat signature is read against)
    Xk = torch.stack([v0, a0, v0 * k0], dim=1).numpy()
    report["controls"]["D4_human_plan"] = ridge_probe(Xk, wp.reshape(n, -1).numpy(),
                                                      fit_idx, score_idx)

    arms = sorted(p_ for p_ in ARMS.iterdir() if p_.is_dir())
    store: dict = {}
    for ad in arms:
        if not (ad / "ckpt.pt").exists():
            report["arms"][ad.name] = {"status": "NO_CKPT"}
            continue
        p(f"[{ad.name}] building")
        model, cfg, cfg_j, wb, load = build_model(ad)
        model = model.to(DEV)
        dec = model.core.decoder
        ablate = bool(cfg_j.get("ablate_frames"))
        mode = wb.get("mode", "fixed")
        pool_stamp = None
        if mode == "random":
            pool_stamp = rebuild_random_pool(model, int(cfg_j.get("seed", 0)))["random_pool"]
        regimes = {"k": ("k", mode, False), "w": ("w", mode, False)}
        if mode == "pred":
            regimes["w_fixed"] = ("w", "fixed", False)
            regimes["w_shuf"] = ("w", "pred", True)
        if mode == "random":
            regimes["w_fixed"] = ("w", "fixed", False)
        R = {}
        for tag, (reg, bm, shuf) in regimes.items():
            p(f"[{ad.name}] forward {tag} (regime {reg}, bank {bm}, shuffle {shuf})")
            R[tag] = forward_regime(model, cfg, ds, perm, reg, bm, ablate, shuf,
                                    seed=int(cfg_j.get("seed", 0)))
        store[ad.name] = R
        np.savez_compressed(DUMPS / f"{ad.name}.npz",
                            **{f"{t}_{k}": v.numpy() for t, r in R.items() for k, v in r.items()},
                            eid=eid, v0=v0.numpy(), a0=a0.numpy(), k0=k0.numpy(),
                            wp=wp.numpy(), sv=sv.numpy(), goal=G["goal"].numpy(),
                            gval=G["gval"].numpy())
        arm_rec: dict = {"status": "MEASURED", "config": cfg_j.get("ego"),
                         "withheld_bank": wb, "random_pool_rebuilt": pool_stamp,
                         "size": cfg_j.get("size"), "rig_rung": cfg_j.get("rig_rung"),
                         "ablate_frames": ablate, "load": load, "regimes": {}}

        # ---- gate 1 / goal-row families / pixel floor, per regime ---------
        for tag, r in R.items():
            pred = r["g_tac"]
            rec: dict = {"regime": tag, "gate1": {"slots": []}, "goal_families": []}
            for k in range(G["goal"].shape[1]):
                m = G["gval"][:, k]
                if int(m.sum()) < 32:
                    continue
                refs = {nm: _ade_goal(ref[nm], k, m) for nm in ref}
                refs["constant_only"] = _ade_goal(const, k, m)
                gk = EG.echo_gate(arm_ade=_ade_goal(pred, k, m), references=refs,
                                  eid=eid[m.numpy()], margins={"ha": 0.10, "ha0_ext": 0.10},
                                  n_boot=N_BOOT_FAM)
                row = gk["slots"][0]; row.update(slot=k, tau_s=taus[k], n_windows=int(m.sum()),
                                                 n_episodes=int(len(set(eid[m.numpy()].tolist()))))
                rec["gate1"]["slots"].append(row)
            fam = EG.trajectory_families(pred, G["goal"], taus_s=taus)
            fam_ext = EG.trajectory_families(ref["ha0_ext"], G["goal"], taus_s=taus)
            for k in range(G["goal"].shape[1]):
                m = G["gval"][:, k]
                if int(m.sum()) < 32:
                    continue
                rec["goal_families"].append({
                    "slot": k, "tau_s": taus[k], "n": int(m.sum()),
                    "ade_m": {"arm": float(fam["ade_m"][m][:, k].mean()),
                              "ha0_ext": float(fam_ext["ade_m"][m][:, k].mean())},
                    "longitudinal": {"speed_err_mps": {
                        "arm": float(fam["longitudinal"]["speed_err_mps"][m][:, k].mean()),
                        "ha0_ext": float(fam_ext["longitudinal"]["speed_err_mps"][m][:, k].mean())}},
                    "lateral": {key: {"arm": float(fam["lateral"][key][m][:, k].mean()),
                                      "ha0_ext": float(fam_ext["lateral"][key][m][:, k].mean())}
                                for key in ("heading_err_rad", "curvature_err_invm",
                                            "yaw_rate_err_radps", "cross_track_m")}})
            if tag == "k":
                pix, pixmeta = EG.raw_pixel_floor(G["frlast"], G["goal"], fit_idx=fit_idx,
                                                  score_idx=score_idx, n_pix=8)
                pix = pix.reshape(-1, G["goal"].shape[1], 4)
                msk = G["gval"][score_idx]
                rec["raw_pixel_floor"] = {"meta": pixmeta, "per_slot": [
                    {"slot": k, "tau_s": taus[k], "n": int(msk[:, k].sum()),
                     "floor_ade": float(torch.linalg.vector_norm(
                         pix[msk[:, k]][:, k, :2] - G["goal"][score_idx][msk[:, k]][:, k, :2],
                         dim=-1).mean()),
                     "arm_ade": float(torch.linalg.vector_norm(
                         pred[score_idx][msk[:, k]][:, k, :2]
                         - G["goal"][score_idx][msk[:, k]][:, k, :2], dim=-1).mean())}
                    for k in range(G["goal"].shape[1]) if int(msk[:, k].sum()) > 0]}
            # ---- the selected-trajectory families through the banked instrument
            P = r["traj"]
            comps = {"arm": ra._components(P[:, SLOTS2S].numpy(), wp[:, SLOTS2S].numpy(), 0.5),
                     **comps_ref}
            rec["plan_2s_families"] = {
                "levels": {mk: level(comps["arm"][mk], eid) for mk in comps["arm"]},
                "paired_arm_minus_ha": {mk: paired(comps["arm"][mk], comps["ha"][mk], eid)
                                        for mk in comps["arm"]},
                "paired_arm_minus_ha0_ext": {mk: paired(comps["arm"][mk], comps["ha0_ext"][mk], eid)
                                             for mk in comps["arm"]},
                "_family_of": ra._FAMILY_OF, "grid": {"dt_s": 0.5, "slots_of_8": SLOTS2S},
                "_tier": "T1-style self-action open loop on the selected plan (rig scope)"}
            comps6 = ra._components(P[:, SLOTS6S].numpy(), wp[:, SLOTS6S].numpy(), 1.0)
            rec["plan_6s_families_levels"] = {mk: level(comps6[mk], eid) for mk in comps6}
            rec["_comps2s"] = comps["arm"]; rec["_comps6s"] = comps6
            # ---- the tactical family: the factored heads vs the majority class
            from tanitad.refs import refc_tactical as _tac
            lat_k, lon_k = _tac.window_factored_labels(G["pose_last"], G["fx"])
            tact = {}
            for name, lg, lab in (("lat", r["lat"], lat_k), ("lon", r["lon"], lon_k)):
                m = lab >= 0
                if int(m.sum()) < 32:
                    tact[name] = {"status": "TOO_FEW_LABELLED", "n": int(m.sum())}
                    continue
                pr = lg[m].argmax(-1); y = lab[m]
                cnt = torch.bincount(y, minlength=lg.shape[-1]).float()
                tact[name] = {"n": int(m.sum()), "accuracy": float((pr == y).float().mean()),
                              "majority_class_control": float(cnt.max() / cnt.sum()),
                              "per_class_recall": [
                                  (float((pr[y == c] == c).float().mean()) if int((y == c).sum()) else None)
                                  for c in range(lg.shape[-1])],
                              "predicted_frac": [float((pr == c).float().mean())
                                                 for c in range(lg.shape[-1])]}
            rec["tactical_head"] = tact
            # ---- selection profile + ceilings
            sel = r["sel_idx"].numpy()
            o_dec, a_star = oiv(r["bank"], wp, sv)
            rec["selection"] = {
                "straight_ahead_idx67_frac": float((sel == STRAIGHT_IDX).mean()),
                "n_distinct_selected": int(len(set(sel.tolist()))),
                "astar_straight_frac": float((a_star.numpy() == STRAIGHT_IDX).mean()),
                "agrees_with_own_oracle_frac": float((sel == a_star.numpy()).mean())}
            fam_dir = DUMPS / "families"; fam_dir.mkdir(parents=True, exist_ok=True)
            rec["four_families_artifact"] = str(fam_dir / f"{ad.name}__{tag}.json")
            write_family_artifact(rec["four_families_artifact"], ad.name, tag, comps["arm"],
                                  P[:, SLOTS2S].numpy(), wp[:, SLOTS2S].numpy(), r, G, eid,
                                  sel, a_star.numpy(),
                                  controls={"ha": rec["plan_2s_families"]["paired_arm_minus_ha"],
                                            "ha0_ext": rec["plan_2s_families"]["paired_arm_minus_ha0_ext"],
                                            "_direction": "arm minus control; negative = arm better",
                                            "constant_only_goal_rows": report["controls"]["goal_rows_by_slot"]},
                                  corpus=CORPUS)
            rec["oiv_decoded_bank_m"] = level(o_dec.numpy(), eid)
            rec["oiv_decoded_bank_by_v0_band"] = by_band(
                v0.numpy(), lambda m: {"oiv_m": float(o_dec.numpy()[m].mean())})
            rec["bank_speed_pred_mae_vs_gt2s_mps"] = level(
                np.where(G["gval"][:, 0].numpy(),
                         (r["bsp"] - G["goal"][:, 0, 3]).abs().numpy(), np.nan), eid)
            rec["bank_speed_pred_mae_vs_v0_mps"] = float((r["bsp"] - v0).abs().mean())
            arm_rec["regimes"][tag] = rec

        # ---- the withheld ceilings by v0 band: fixed / own prediction / true v0
        bsp_w = R["w"]["bsp"]
        banks = {"fixed_10ms": roll_at(dec, torch.full_like(v0, dec.anchor_ref_speed)),
                 "own_pred_speed_w": roll_at(dec, bsp_w.clamp(0, dec.anchor_withheld_speed_max)),
                 "true_v0_LEAK_BOUND": roll_at(dec, v0)}
        oivs = {nm: oiv(b_, wp, sv)[0].numpy() for nm, b_ in banks.items()}
        arm_rec["withheld_ceilings"] = {
            "levels": {nm: level(v, eid) for nm, v in oivs.items()},
            "paired_pred_minus_fixed": paired(oivs["own_pred_speed_w"], oivs["fixed_10ms"], eid),
            "paired_pred_minus_true": paired(oivs["own_pred_speed_w"], oivs["true_v0_LEAK_BOUND"], eid),
            "by_v0_band": by_band(v0.numpy(), lambda m: {nm: float(v[m].mean()) for nm, v in oivs.items()}),
            "_tier": "T0 model-free ceiling on the anchor PRIOR (the bank the withheld "
                     "regime WOULD decode); the true-v0 row is the leak bound, reported, "
                     "never a design"}
        # ---- gate 2 / 2b on the kept regime (CPU, as gate_score.py) + scene half withheld
        model = model.cpu()
        b = torch.utils.data.default_collate([ds[i] for i in perm[:BATCH]])
        fr = b["frames"]
        if ablate:
            fr = torch.full_like(fr, float(fr.mean()))
        pose_last = b["pose_last"]
        ego = v3.ego_state_from_batch({"pose_last": pose_last, "actions": b["actions"]})
        gb = {"frames": fr, "nav_cmd": b["nav_cmd"], "v0": pose_last[:, 3],
              "steps": cfg.core.decoder.diffusion_steps, "ego_state": ego}
        dec.anchor_withheld_bank = mode
        g2 = EG.ego_intervention_test(model, gb)

        def predict(frames, ego_state, _gb=gb):
            kw = dict(_gb); kw["frames"] = frames; kw["ego_state"] = ego_state
            with torch.no_grad():
                return model(**kw)["g_tac"].float()

        g2b = EG.source_ablation_test(predict, frames=fr, ego_state=ego,
                                      target=b["goal_tac"].float(),
                                      eid=b["episode_id"].reshape(-1).numpy(), n_boot=N_BOOT_PROBE)
        # the constant predictor MUST read exactly 0.0 / 0.0 on both derangements
        const_pred = b["goal_tac"].float().mean(0, keepdim=True).expand_as(b["goal_tac"].float())
        g2b_const = EG.source_ablation_test(lambda f, e: const_pred, frames=fr, ego_state=ego,
                                            target=b["goal_tac"].float(),
                                            eid=b["episode_id"].reshape(-1).numpy(), n_boot=50)
        ego_w = ego.clone(); ego_w[:, 4] = 0.0
        g2b_w = EG.source_ablation_test(predict, frames=fr, ego_state=ego_w,
                                        target=b["goal_tac"].float(),
                                        eid=b["episode_id"].reshape(-1).numpy(), n_boot=N_BOOT_PROBE)
        g2b_real = None
        if ablate:
            fr_real = torch.utils.data.default_collate([ds[i] for i in perm[:BATCH]])["frames"]
            g2b_real = EG.source_ablation_test(predict, frames=fr_real, ego_state=ego,
                                               target=b["goal_tac"].float(),
                                               eid=b["episode_id"].reshape(-1).numpy(),
                                               n_boot=N_BOOT_PROBE)
        try:
            verdict = EG.assert_not_echoing(arm_rec["regimes"]["k"]["gate1"], g2, gate2b=g2b)
            gate = {"applicable": True, "raised": False, "verdict": verdict}
        except EG.EchoViolation as exc:
            gate = {"applicable": True, "raised": True, "message": str(exc)}
        arm_rec.update(gate2=g2, gate2b=g2b, gate2b_withheld_regime=g2b_w,
                       gate2b_constant_predictor_control=g2b_const,
                       gate2b_realframes=g2b_real, GATE=gate)
        # ---- D3 / D4
        model = model.to(DEV)
        arm_rec["D3_endpoint_gradient"] = d3_endpoint_gradient(model, cfg, ds, perm, ablate)
        arm_rec["D4_plan_predictability"] = {
            tag: ridge_probe(Xk, R[tag]["traj"].reshape(n, -1).numpy(), fit_idx, score_idx)
            for tag in ("k", "w")}
        arm_rec["D4_plan_predictability"]["_reads"] = (
            "1 - R^2 of a ridge from (v0, a0, yaw_rate) alone to the 8-slot plan on TEST; "
            "compare to controls.D4_human_plan — MORE self-predictable than the human "
            "(smaller) is the copycat signature (2010.14876 §5)")
        report["arms"][ad.name] = arm_rec
        p(f"[{ad.name}] gate2={g2.get('verdict')} gate2b={g2b.get('verdict')} "
          f"(scene {g2b['sources']['scene']['degradation_rel']:+.4f}, ego "
          f"{g2b['sources']['ego']['degradation_rel']:+.4f}) gate_raised={gate.get('raised')} "
          f"| w 2s speed_mae {np.nanmean(arm_rec['regimes']['w']['_comps2s']['LON_speed_mae_mps']):.3f}")
        del model
        torch.cuda.empty_cache()

    # ---- paired vs A0, per regime, on the SAME windows ------------------------
    a0n = "A0_fixed"
    if a0n in report["arms"] and report["arms"][a0n].get("status") == "MEASURED":
        for name, rec in report["arms"].items():
            if rec.get("status") != "MEASURED" or name == a0n:
                continue
            rec["paired_vs_A0"] = {}
            for tag in rec["regimes"]:
                if tag not in report["arms"][a0n]["regimes"]:
                    base_tag = "w" if tag.startswith("w") else "k"
                else:
                    base_tag = tag
                c_arm = rec["regimes"][tag]["_comps2s"]
                c_a0 = report["arms"][a0n]["regimes"][base_tag]["_comps2s"]
                rec["paired_vs_A0"][tag] = {
                    "vs_A0_regime": base_tag,
                    "plan_2s": {mk: paired(c_arm[mk], c_a0[mk], eid) for mk in c_arm},
                    "plan_6s": {mk: paired(rec["regimes"][tag]["_comps6s"][mk],
                                           report["arms"][a0n]["regimes"][base_tag]["_comps6s"][mk], eid)
                                for mk in c_arm},
                    "goal_2s_speed_err": paired(
                        np.abs(store[name][tag]["g_tac"][:, 0, 3] - G["goal"][:, 0, 3]).numpy(),
                        np.abs(store[a0n][base_tag]["g_tac"][:, 0, 3] - G["goal"][:, 0, 3]).numpy(), eid),
                    "_direction": f"{name}[{tag}] minus A0[{base_tag}]; negative = {name} better"}
        # A1 vs A2 (the control that must NOT gain the same)
        if all(k in report["arms"] and report["arms"][k].get("status") == "MEASURED"
               for k in ("A1_pred", "A2_random")):
            report["A1_minus_A2"] = {tag: {
                mk: paired(report["arms"]["A1_pred"]["regimes"][tag]["_comps2s"][mk],
                           report["arms"]["A2_random"]["regimes"][tag]["_comps2s"][mk], eid)
                for mk in ra._FAMILY_OF} for tag in ("k", "w")}
    for rec in report["arms"].values():
        for reg in (rec.get("regimes") or {}).values():
            reg.pop("_comps2s", None); reg.pop("_comps6s", None)
    OUT.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    p(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
