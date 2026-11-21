"""CL head read-out check: the measurements of PREREG_CL_READOUT_A1.md (sha256 1f96d833...), zero training, CPU only.

Per CL checkpoint (seed 0 / 1), over the 2,399 eval-141 windows, at T0 = the tick-0 batch the T1-kin rollout feeds (logged history and frames):
  1. v_hat (the output of ``model.lvl_head``, captured by a forward hook) against the logged speed at the tick: R2 (all windows), MAE in the 3..8 m/s band, MAE on the M1 crawl windows of the C0 rollout.
  2. O's sensitivity: a forward pre-hook on ``model.e_level`` adds +0.3 / -0.3 to its input (v_in = v_hat/10 -> (v_hat +- 3)/10), everything else fixed; the change of the plan's mean speed over 0..3 s
     (``traj60`` rows 0..29, step length / 0.1 s, origin as the first knot); and ||e_level(v_in)|| / ||O's ego token||.
Estimator: per-window values, episode-cluster bootstrap B = 2000, seed 0 (taniteval.ci's draw stream: default_rng(0), choice(uniq, n_ep, replace=True)).  Labels per the addendum's committed rules.

Controls that must read known values are run in-process (shift 0.0 -> change EXACTLY 0.0; the e_level input equals v_hat/10 + shift; tick-0 plan equals the banked rollout's tick-0 plan_raw; label function on
constructed inputs).  A failed control is written to the JSON and the run exits non-zero.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

TREE = "F:/TanitAD-scratch/motion_c/cl_read/tree"
TOOL = f"{TREE}/stack/scripts/v8f_t1_eval_ri.py"
ART = "E:/Projects/TanitAD-artifacts"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
TRACKS = f"{ART}/sfresh_inputs/inputs/native_tracks.pt"
EVAL_CACHE = f"{ART}/sfresh_inputs/cache"
CKPT = f"{ART}/v8f-recover/M2b_s{{s}}_recft_CLR/ckpt_last.pt"
C0_ARR = f"{ART}/dagland/recover_next/read/raw/arrays/C0_s{{s}}.npz"
BANKED_TICKS = "F:/TanitAD-scratch/motion_c/clr_read/out/CLR_s{s}/ticks/ticks_CLR_s{s}.npz"
OUT_DIR = f"{ART}/dagland/motion_c/clr_read/readout"

DT = 0.2
BAND = (3.0, 8.0)                 # m/s, inclusive both ends, on the logged speed
SHIFT_MS = 3.0                    # the +-3 m/s shift of v_hat (v_in shifts by +-0.3)
LEVEL_SCALE = 10.0
B_BOOT = 2000
TRACKS_R2, TRACKS_MAE = 0.5, 1.5
NOT_R2, NOT_MAE = 0.3, 2.5
SENS_HI, SENS_LO = 0.5, 0.1


# ------------------------------------------------------------------------------------------------ labels (the addendum's committed rules)
def track_label(r2: float, band_mae: float) -> str:
    if r2 >= TRACKS_R2 and band_mae <= TRACKS_MAE:
        return "TRACKS"
    if r2 < NOT_R2 or band_mae > NOT_MAE:
        return "DOES-NOT-TRACK"
    return "PARTIAL"


def sens_label(abs_change: float) -> str:
    if abs_change >= SENS_HI:
        return "O-SENSITIVE"
    if abs_change < SENS_LO:
        return "O-INSENSITIVE"
    return "WEAK"


def next_arm(tl: str, sl: str) -> str:
    if tl == "TRACKS" and sl in ("O-INSENSITIVE", "WEAK"):
        return "CLR (addendum branch 1): O learned to ignore a level it was handed (the logged history carries the speed exactly in training); CLR = RAPR-P01's argv + --level-head on --w-level 0.2027, its own card (A2) before any run"
    if tl == "TRACKS" and sl == "O-SENSITIVE":
        return "CLR (addendum branch 2): O uses the level and it still does not close the crawl in T1-kin; the level is supplied but O never trained on states where history and level disagree; same CLR, own card (A2)"
    return "lead-aware head (addendum branch 3): the frozen single-frame field does not carry the afforded speed well enough in the crawl band; lead presence and distance from the frame (CRAWL-LEAD), own card"


# ------------------------------------------------------------------------------------------------ estimator
def draws(uniq: np.ndarray, n_boot: int, seed: int = 0) -> np.ndarray:
    """[n_boot, n_ep] episode multiplicities: taniteval.ci's stream (default_rng(seed); choice(uniq, n_ep, replace=True) per draw)."""
    rng = np.random.default_rng(seed)
    n_ep = len(uniq)
    M = np.zeros((n_boot, n_ep))
    for b in range(n_boot):
        pick = rng.choice(uniq, size=n_ep, replace=True)
        M[b] = np.bincount(np.searchsorted(uniq, pick), minlength=n_ep)
    return M


def ep_sums(eid: np.ndarray, uniq: np.ndarray, *cols: np.ndarray) -> np.ndarray:
    """[n_ep, len(cols)] per-episode sums of the columns."""
    idx = np.searchsorted(uniq, eid)
    out = np.zeros((len(uniq), len(cols)))
    for k, c in enumerate(cols):
        out[:, k] = np.bincount(idx, weights=np.asarray(c, np.float64), minlength=len(uniq))
    return out


def ci(vals: np.ndarray) -> list:
    v = vals[np.isfinite(vals)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


def r2_stat(S: np.ndarray) -> np.ndarray:
    """S columns: n, sum y, sum y^2, sum (y - yhat)^2.  R2 = 1 - SSE / SST per row (row = point or a draw)."""
    n, sy, syy, sse = S[..., 0], S[..., 1], S[..., 2], S[..., 3]
    sst = syy - sy * sy / np.maximum(n, 1e-12)
    return 1.0 - sse / np.where(sst > 0, sst, np.nan)


def mean_stat(S: np.ndarray) -> np.ndarray:
    """S columns: n, sum x -> mean x (NaN with n = 0)."""
    return np.where(S[..., 0] > 0, S[..., 1] / np.maximum(S[..., 0], 1e-12), np.nan)


def point_and_ci(S_ep: np.ndarray, M: np.ndarray, fn) -> dict:
    pt = float(fn(S_ep.sum(0, keepdims=True))[0])
    bs = fn(M @ S_ep)
    return {"point": pt, "ci95": ci(bs), "n_boot_finite": int(np.isfinite(bs).sum())}


# ------------------------------------------------------------------------------------------------ the plan statistic
def plan_speed_0_3(traj60: np.ndarray) -> np.ndarray:
    """[B] mean speed (m/s) over 0..3 s of the plan: the 30 steps of 0.1 s from the ego-frame origin through traj60[:, :30]; speed = step length / 0.1."""
    K = np.concatenate([np.zeros((len(traj60), 1, 2)), traj60[:, :30].astype(np.float64)], axis=1)
    return (np.linalg.norm(np.diff(K, axis=1), axis=-1) / 0.1).mean(1)


# ------------------------------------------------------------------------------------------------ the read
def load_tool():
    spec = importlib.util.spec_from_file_location("v8f_t1_eval_ri", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v8f_t1_eval_ri"] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def forward_all(T, model, batch, dev):
    """One window batch -> dict of numpy arrays: baseline plan + v_hat + e_level input/output + ego token norm, then the +3 / -3 / zero-shift / e_level-zeroed forwards.  Hooks only; the model is untouched."""
    import torch
    cap = {"v_hat": None, "e_in": None, "e_out": None}
    state = {"shift": 0.0, "zero_e": False, "add": None}

    def h_lvl(mod, inp, out):
        cap["v_hat"] = out.detach().float().squeeze(-1).clone()

    def pre_e(mod, args):
        x = args[0]
        if state["shift"] != 0.0:
            x = x + state["shift"]
        return (x,)

    def h_e(mod, args, out):
        cap["e_in"] = args[0].detach().float().clone()          # AFTER the pre-hook: the value the layer actually saw
        cap["e_out"] = out.detach().float().clone()
        if state["zero_e"]:
            return torch.zeros_like(out)
        if state["add"] is not None:
            return out + state["add"]

    hs = [model.lvl_head.register_forward_hook(h_lvl), model.e_level.register_forward_pre_hook(pre_e), model.e_level.register_forward_hook(h_e)]
    res = {}
    try:
        def run(shift=0.0, zero_e=False, add=None):
            state["shift"], state["zero_e"], state["add"] = shift, zero_e, add
            with torch.no_grad():
                out = model.forward_infer(batch)
            return out["traj60"].double().cpu().numpy(), out["level_v"].detach().float().cpu().numpy(), cap["v_hat"].cpu().numpy(), cap["e_in"].cpu().numpy(), cap["e_out"].cpu().numpy()
        t0, lv0, vh0, ein0, eout0 = run(0.0)
        res.update(traj60=t0, level_v=lv0, v_hat=vh0, e_in=ein0, e_out=eout0)
        for name, sh in (("p", +SHIFT_MS / LEVEL_SCALE), ("m", -SHIFT_MS / LEVEL_SCALE), ("z", 0.0)):
            t, _, _, ein, _ = run(sh)
            res["traj60_" + name], res["e_in_" + name] = t, ein
        res["traj60_ez"] = run(0.0, zero_e=True)[0]
        g = torch.Generator().manual_seed(7)                                  # REACH CONTROL (instrument only): a fixed random direction of norm 8 (an ego token's size) added to e_level's output must move the plan
        vec = torch.randn(1, model.e_level.out_features, generator=g)
        res["traj60_big"] = run(0.0, add=(vec / vec.norm() * 8.0).to(dev))[0]
        # ego token norm exactly as run_o builds it (i = the current frame)
        Fn = batch["feats"].shape[1]
        last = torch.full((batch["feats"].shape[0],), Fn - 1, dtype=torch.long, device=dev)
        with torch.no_grad():
            ego = model._ego(model._at(batch["p10f"], last)).float().cpu().numpy()
        res["ego_tok"] = ego
        res["frame_ok_now"] = batch["frame_ok"][:, -1].cpu().numpy().astype(bool)
    finally:
        for h in hs:
            h.remove()
    return res


def run_seed(s: int, max_windows: int, threads: int, batch_size: int) -> dict:
    import torch
    torch.set_num_threads(threads)
    T = load_tool()
    ck_path = CKPT.format(s=s)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = T.M.V8FMiniConfig(**ck["cfg"])
    assert cfg.level_head == "on", "checkpoint cfg: level_head is not on"
    model = T.M.V8FMini(cfg, T.M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * T.M.K30)))
    model.load_state_dict(ck["model"])
    dev = torch.device("cpu")
    model.to(dev).eval()
    tracks = T.D.load_native_tracks(TRACKS)
    J = T.D.load_join(f"{DATA}/join_eval141.npz")
    store = T.D.ClipStore(EVAL_CACHE, "eval", lru=32, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    wins_all = T.D.eval_grid_windows(f"{DATA}/eval_grid_windows.json", J)
    sel = list(range(len(wins_all)))
    if max_windows and max_windows < len(wins_all):
        sel = np.linspace(0, len(wins_all) - 1, max_windows).round().astype(int).tolist()
    wins = [wins_all[i] for i in sel]
    proxy = T._OneClipStore()
    ds = T.D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    a_t, a_s = T.ages_for(0, "periodic", 3, 6)
    cols = {k: [] for k in ("traj60", "traj60_p", "traj60_m", "traj60_z", "traj60_ez", "traj60_big", "v_hat", "level_v", "e_in", "e_in_p", "e_in_m", "e_out", "ego_tok", "frame_ok_now")}
    v_cache, v_kin, ci_l, t0_l = [], [], [], []
    t_start = time.time()
    for b0 in range(0, len(wins), batch_size):
        chunk = wins[b0:b0 + batch_size]
        states = T.make_states(store, tracks, J, chunk, "deadreckon")
        items = [T.build_item(ds, proxy, st, 0, a_t, a_s, "logged", "causal", None) for st in states]     # EXACTLY the rollout's tick-0 batch (v8f_t1_eval_ri.rollout)
        full = T.D.collate(items)
        batch = {k: full[k].to(dev) for k in T.MODEL_KEYS}
        r = forward_all(T, model, batch, dev)
        for k in cols:
            cols[k].append(r[k])
        for (ci_, t0_, _, _), st in zip(chunk, states):
            base = store.get(str(J["clip"][ci_]))
            v_cache.append(float(base["v"][t0_]))
            v_kin.append(float(base["kin3"][t0_][0]))
            ci_l.append(int(ci_))
            t0_l.append(int(t0_))
        print(f"[s{s}] {b0 + len(chunk)}/{len(wins)} {time.time() - t_start:.0f}s", flush=True)
    A = {k: np.concatenate(v) for k, v in cols.items()}
    A.update(v_cache=np.array(v_cache), v_kin=np.array(v_kin), ci=np.array(ci_l), t0=np.array(t0_l), sel=np.array(sel))
    A["ckpt_sha256"] = sha256(ck_path)
    A["cfg_level"] = {"level_head": cfg.level_head, "w_level": cfg.w_level}
    return A


def analyze(s: int, A: dict, full_set: bool) -> dict:
    n = len(A["v_hat"])
    ci_ = A["ci"]
    uniq = np.unique(ci_)
    M = draws(uniq, B_BOOT, 0)
    ok_all = np.isfinite(A["v_hat"]) & np.isfinite(A["v_cache"])
    out = {"seed": s, "n_windows": int(n), "n_episodes": int(len(uniq)), "full_eval_set": bool(full_set), "ckpt_sha256": A["ckpt_sha256"], "ckpt_cfg": A["cfg_level"],
           "n_frame_ok_false_now": int((~A["frame_ok_now"]).sum()), "n_nonfinite_vhat_or_target": int((~ok_all).sum())}
    out["target_check"] = {"max_abs_cache_v_minus_kin3_v0": float(np.nanmax(np.abs(A["v_cache"] - A["v_kin"]))), "mean_abs": float(np.nanmean(np.abs(A["v_cache"] - A["v_kin"]))),
                           "n_kin3_nonfinite": int((~np.isfinite(A["v_kin"])).sum()), "logged_speed_range": [float(np.nanmin(A["v_cache"])), float(np.nanmax(A["v_cache"]))], "v_hat_range": [float(A["v_hat"].min()), float(A["v_hat"].max())]}

    def tracking(y, name):
        yh = A["v_hat"]
        e = yh - y
        good = np.isfinite(e)
        r = {}
        S = ep_sums(ci_, uniq, good, np.where(good, y, 0), np.where(good, y * y, 0), np.where(good, e * e, 0))
        r["r2_all"] = point_and_ci(S, M, r2_stat)
        r["pearson_r2_all_beside"] = float(np.corrcoef(yh[good], y[good])[0, 1] ** 2)
        r["bias_mean_vhat_minus_y"] = float(np.mean(e[good]))
        r["mae_all"] = point_and_ci(ep_sums(ci_, uniq, good, np.where(good, np.abs(e), 0)), M, mean_stat)
        band = good & (y >= BAND[0]) & (y <= BAND[1])
        r["n_band"] = int(band.sum())
        r["band_mae"] = point_and_ci(ep_sums(ci_, uniq, band, np.where(band, np.abs(e), 0)), M, mean_stat)
        r["band_bias_mean_vhat_minus_y"] = float(np.mean(e[band])) if band.any() else None
        r["band_mean_y"] = float(np.mean(y[band])) if band.any() else None
        r["band_mean_vhat"] = float(np.mean(yh[band])) if band.any() else None
        r["label"] = track_label(r["r2_all"]["point"], r["band_mae"]["point"])
        r["label_rule"] = f"TRACKS iff R2>={TRACKS_R2} and band MAE<={TRACKS_MAE}; DOES-NOT-TRACK iff R2<{NOT_R2} or band MAE>{NOT_MAE}; else PARTIAL (point estimates)"
        return r

    out["tracking"] = tracking(A["v_cache"], "cache_v")
    out["tracking_beside_target_kin3_v0"] = tracking(A["v_kin"], "kin3")        # the head's own training target (level_stage: kin3[:, 0]); beside only
    out["label_unchanged_under_kin3_target"] = out["tracking"]["label"] == out["tracking_beside_target_kin3_v0"]["label"]

    # ---- M1 crawl windows of the C0 rollout (a_18 = per-window M1 crawl share, rr_lib.m1_window: GT-fast pairs (>= 5.2 m/s) whose own speed is < 5.2 m/s, over the window's GT-fast pairs)
    m1 = {"definition": "per window, C0 rollout's M1 crawl share a_18 (recover_next/read/raw/arrays/C0_s{s}.npz) from rr_lib.m1_window; a crawl window = a_18 > 0 (the rollout's own speed drops below 5.2 m/s while the GT is >= 5.2 m/s); "
                      "beside: the GT-fast set (a_18 finite) and the heavy-crawl set (a_18 >= 0.5)"}
    c0p = C0_ARR.format(s=s)
    if not os.path.exists(c0p):
        m1["status"] = "NOT COMPUTED: C0 array file absent"
    else:
        z = np.load(c0p, allow_pickle=True)
        key = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(z["clip_index"], z["ws"]))}
        idx = np.array([key.get((int(c), int(t)), -1) for c, t in zip(A["ci"], A["t0"])])
        if (idx < 0).any():
            m1["status"] = f"NOT COMPUTED: {int((idx < 0).sum())} windows of the read are absent from the C0 array (clip_index, ws) keys"
        else:
            share = z["a_18"][idx].astype(np.float64)
            m1["v0_alignment_max_abs_dev_vs_kin3"] = float(np.nanmax(np.abs(z["v0"][idx] - A["v_kin"])))
            m1["c0_array_sha256"] = sha256(c0p)
            m1["status"] = "computed"
            y = A["v_cache"]
            e = A["v_hat"] - y
            for nm, mask in (("crawl_share_gt0", np.isfinite(share) & (share > 0)), ("gt_fast_set_finite", np.isfinite(share)), ("heavy_crawl_share_ge_0p5", np.isfinite(share) & (share >= 0.5))):
                good = mask & np.isfinite(e)
                S = ep_sums(ci_, uniq, good, np.where(good, np.abs(e), 0))
                m1[nm] = {"n": int(good.sum()), "n_episodes": int(len(np.unique(ci_[good]))), "mae": point_and_ci(S, M, mean_stat), "bias_mean_vhat_minus_y": float(np.mean(e[good])) if good.any() else None,
                          "mean_logged_speed": float(np.mean(y[good])) if good.any() else None, "mean_vhat": float(np.mean(A["v_hat"][good])) if good.any() else None,
                          "in_band_share": float(np.mean((y[good] >= BAND[0]) & (y[good] <= BAND[1]))) if good.any() else None}
    out["m1_windows"] = m1

    # ---- O's sensitivity
    sp0, spp, spm, spz, spez, spbig = (plan_speed_0_3(A[k]) for k in ("traj60", "traj60_p", "traj60_m", "traj60_z", "traj60_ez", "traj60_big"))
    d_p, d_m, d_z, d_ez, d_big = spp - sp0, spm - sp0, spz - sp0, spez - sp0, spbig - sp0
    sens = {"statistic": "plan mean speed over 0..3 s from traj60 rows 0..29 (origin first), step length / 0.1 s; change = shifted - unshifted, per window", "mean_plan_speed_unshifted": float(sp0.mean())}
    for nm, d in (("plus3", d_p), ("minus3", d_m)):
        S = ep_sums(ci_, uniq, np.ones(n), d, np.abs(d))
        sens[nm] = {"mean_signed_change_ms": point_and_ci(S[:, [0, 1]], M, mean_stat),
                    "mean_abs_change_ms": point_and_ci(S[:, [0, 2]], M, mean_stat),
                    "median_signed_change_ms": float(np.median(d)), "p95_abs_change_ms": float(np.percentile(np.abs(d), 95)), "max_abs_change_ms": float(np.abs(d).max())}
    sens["controls"] = {"zero_shift_change_max_abs": float(np.abs(d_z).max()), "zero_shift_traj_max_abs_diff_m": float(np.abs(A["traj60_z"] - A["traj60"]).max()),
                        "e_in_equals_vhat_over_10_where_frame_ok_max_abs_dev": float(np.abs(A["e_in"][A["frame_ok_now"], 0] - A["v_hat"][A["frame_ok_now"]] / LEVEL_SCALE).max()) if A["frame_ok_now"].any() else None,
                        "e_in_plus_minus_baseline_max_abs_dev_from_0p3": float(max(np.abs(A["e_in_p"] - A["e_in"] - 0.3).max(), np.abs(A["e_in_m"] - A["e_in"] + 0.3).max())),
                        "level_v_equals_hook_vhat_max_abs_dev": float(np.abs(A["level_v"] - A["v_hat"]).max()),
                        "reach_control_big_e_level_output_perturbation": {"mean_abs_plan_speed_change_ms": float(np.abs(d_big).mean()), "max_abs_ms": float(np.abs(d_big).max()), "note": "instrument control, not a prereg measurement: a norm-8 perturbation of e_level's output must move the plan (reads > 0 iff the token position is wired into O's plan in this harness)"}}
    sens["beside_e_level_output_zeroed"] = {"mean_signed_change_ms": point_and_ci(ep_sums(ci_, uniq, np.ones(n), d_ez), M, mean_stat),
                                           "mean_abs_change_ms": point_and_ci(ep_sums(ci_, uniq, np.ones(n), np.abs(d_ez)), M, mean_stat),
                                           "note": "attribution beside (card section 5 names it): e_level's output replaced by zeros at inference; not a committed outcome measurement"}
    # large-shift sanity is not run (it would be a new measurement); the e_level weight sum is read for context
    sd = {k: v.float() for k, v in load_e_level(s).items()}
    sens["e_level_weights"] = {"abs_weight_sum": float(sd["weight"].abs().sum()), "abs_bias_sum": float(sd["bias"].abs().sum())}
    en, gn = np.linalg.norm(A["e_out"], axis=1), np.linalg.norm(A["ego_tok"], axis=1)
    ratio = en / np.where(gn > 0, gn, np.nan)
    gd = np.isfinite(ratio)
    sens["e_level_over_ego_norm"] = {"mean_of_per_window_ratio": point_and_ci(ep_sums(ci_, uniq, gd, np.where(gd, ratio, 0)), M, mean_stat),
                                     "ratio_of_means_beside": float(en.mean() / gn.mean()), "mean_e_level_norm": float(en.mean()), "mean_ego_token_norm": float(gn.mean())}
    sens["label"] = sens_label(abs(sens["plus3"]["mean_signed_change_ms"]["point"]))
    sens["label_alt_mean_abs_reading"] = sens_label(sens["plus3"]["mean_abs_change_ms"]["point"])
    sens["label_rule"] = f"O-SENSITIVE iff |mean plan-speed change| for +3 m/s >= {SENS_HI}; O-INSENSITIVE iff < {SENS_LO}; else WEAK. Decided on |mean signed change| (measurement 2's quantity); the mean-of-absolute reading is reported beside and its label is label_alt_mean_abs_reading"
    out["sensitivity"] = sens
    out["labels"] = {"tracking": out["tracking"]["label"], "sensitivity": sens["label"], "next_arm": next_arm(out["tracking"]["label"], sens["label"]),
                     "sensitivity_label_same_under_both_readings": sens["label"] == sens["label_alt_mean_abs_reading"]}
    return out


def load_e_level(s: int) -> dict:
    import torch
    ck = torch.load(CKPT.format(s=s), map_location="cpu", weights_only=False)
    return {"weight": ck["model"]["e_level.weight"], "bias": ck["model"]["e_level.bias"]}


def banked_tick0_check(s: int, A: dict) -> dict:
    p = BANKED_TICKS.format(s=s)
    if not os.path.exists(p):
        return {"status": "NOT COMPUTED: banked ticks file absent", "path": p}
    z = np.load(p, allow_pickle=True)
    t0rows = np.where(z["tick"] == 0)[0]
    win = z["win"][t0rows]
    pos = {int(w): r for w, r in zip(win, t0rows)}
    rows, mine = [], []
    for k, w in enumerate(A["sel"]):
        if int(w) in pos:
            rows.append(pos[int(w)])
            mine.append(k)
    rows, mine = np.array(rows), np.array(mine)
    mine_traj30 = A["traj60"][mine][:, 1::2]
    ref = z["plan_raw"][rows]
    dev = np.abs(mine_traj30 - ref)
    return {"status": "computed", "path": p, "n_compared": int(len(rows)), "max_abs_dev_m": float(dev.max()), "mean_abs_dev_m": float(dev.mean()), "p99_abs_dev_m": float(np.percentile(dev, 99)),
            "ci_t0_identical": bool(np.array_equal(z["ci"][rows], A["ci"][mine]) and np.array_equal(z["t0"][rows], A["t0"][mine])),
            "v0_max_abs_dev_vs_kin3": float(np.abs(z["v0"][rows] - A["v_kin"][mine]).max()),
            "note": "banked rollout used 16 threads, batch 32 and the rollout's own tick 0; a different thread count / batch changes float rounding only (1e-5 m class); a gap >> 1e-3 m would mean a different tick-0 batch"}


def label_function_controls() -> dict:
    """The label function on constructed inputs: a perfect head must read TRACKS, a constant head DOES-NOT-TRACK, a head off by 2 m/s in the band PARTIAL; the sensitivity labels at the three thresholds."""
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.uniform(0, 30, 2000), rng.uniform(3, 8, 500)])
    eid = np.repeat(np.arange(25), 100)
    uniq = np.unique(eid)
    M = draws(uniq, 50, 0)

    def lab(yh):
        e = yh - y
        S = ep_sums(eid, uniq, np.ones(len(y)), y, y * y, e * e)
        r2 = float(r2_stat(S.sum(0, keepdims=True))[0])
        band = (y >= 3) & (y <= 8)
        return track_label(r2, float(np.abs(e[band]).mean())), r2

    res = {"perfect": lab(y), "constant_mean": lab(np.full_like(y, y.mean())), "plus2_in_band_offset": lab(y + 2.0 * ((y >= 3) & (y <= 8)) + 0.0), "shuffled": lab(rng.permutation(y))}
    res["sens_labels_0p5_0p1_0p3"] = [sens_label(0.5), sens_label(0.099), sens_label(0.3)]
    res["pass"] = bool(res["perfect"][0] == "TRACKS" and res["constant_mean"][0] == "DOES-NOT-TRACK" and res["shuffled"][0] == "DOES-NOT-TRACK" and res["sens_labels_0p5_0p1_0p3"] == ["O-SENSITIVE", "O-INSENSITIVE", "WEAK"])
    res["note"] = "plus2_in_band_offset: R2 stays high while the band MAE is 2.0 m/s: PARTIAL expected (informative, not part of pass)"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    A = run_seed(a.seed, a.max_windows, min(a.threads, 12), a.batch)
    full = a.max_windows == 0
    res = analyze(a.seed, A, full)
    res["banked_tick0_check"] = banked_tick0_check(a.seed, A)
    res["label_function_controls"] = label_function_controls()
    res["wall_s"] = time.time() - t0
    res["prereg"] = {"file": "PREREG_CL_READOUT_A1.md", "sha256": "1f96d833a9c5e32eb7f845d669d2f825485daee5d3b8fcd29bbec45a2523a8bb", "bootstrap": f"episode-cluster, B={B_BOOT}, seed 0"}
    ctrl = res["sensitivity"]["controls"]
    bt = res["banked_tick0_check"]
    fails = []
    if ctrl["zero_shift_change_max_abs"] != 0.0:
        fails.append("zero shift did not read exactly 0.0")
    if ctrl["e_in_plus_minus_baseline_max_abs_dev_from_0p3"] > 1e-5:
        fails.append("the +-0.3 shift did not reach e_level's input")
    if ctrl["level_v_equals_hook_vhat_max_abs_dev"] != 0.0:
        fails.append("forward hook v_hat != out['level_v']")
    if not ctrl["reach_control_big_e_level_output_perturbation"]["mean_abs_plan_speed_change_ms"] > 0.0:
        fails.append("reach control: a norm-8 perturbation of e_level's output did not move the plan")
    if not res["label_function_controls"]["pass"]:
        fails.append("label function controls")
    if bt.get("status") == "computed" and (bt["max_abs_dev_m"] > 1e-2 or not bt["ci_t0_identical"]):
        fails.append("tick-0 plan differs from the banked rollout's tick 0")
    res["control_failures"] = fails
    out = a.out or os.path.join(OUT_DIR, f"CLR_READOUT_s{a.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w", newline="\n"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if full:                                                    # the per-window arrays beside (scratch, for a later re-read; not a deliverable of the card)
        np.savez_compressed(f"F:/TanitAD-scratch/motion_c/clr_read/readout/windows_s{a.seed}.npz", **{k: v for k, v in A.items() if isinstance(v, np.ndarray)})
    print("wrote", out)
    print(json.dumps({"labels": res["labels"], "control_failures": fails}, indent=1))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
