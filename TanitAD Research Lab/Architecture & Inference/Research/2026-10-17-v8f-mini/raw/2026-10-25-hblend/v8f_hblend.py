#!/usr/bin/env python3
"""v8f_hblend.py -- executes PREREG_V8F_HBLEND.md exactly as written (dev box, CPU only, ZERO training).

  python v8f_hblend.py fit   --seed M2b_s0     fit w on TRAIN windows (every k-th clip, k = 32) with that seed's checkpoint
  python v8f_hblend.py score --seed M2b_s0     score H on eval-141 FROM BANKED PREDICTIONS, identity controls first, every PREREG row
  python v8f_hblend.py read                    both seeds -> the outcome table, verbatim -> READ.json + HBLEND_TABLES.md

THE ARM (PREREG, fixed):   H(t) = P(t) + w_a(t) * (tau(t) - P(t))_a ,  a in {along (x), cross (y)} of the ego frame at NOW,
   w_a(t) = isotonic_nondecreasing( clip( sum r_a(t) (y_a(t) - P_a(t)) / sum r_a(t)^2 , 0, 1 ) ),  r = tau - P,  P = the checkpoint's own frozen P10.
   Sums over TRAIN windows only (eval-141 never enters the fit); the form, the clip range and the isotonic step are fixed.

NOTE ON "120 constants" (PREREG): the plan the model emits and the eval scores is the 0.2 s knot plan (traj30 = traj60[1::2]); GT exists only on that
grid and the trainer's loss is on it. So w is FITTED on 30 steps x 2 axes = 60 numbers; the other 60 (the 0.1 s midpoints of traj60) are not
identifiable from the data and are never scored -- `weights_*.json` carries them as the mean of their neighbours, labelled DERIVED/UNSCORED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

SCR = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
TREE, WORK, OUT = f"{SCR}/viz/tree", f"{SCR}/viz/work", f"{SCR}/hblend"
for _p in (f"{OUT}/deps", f"{TREE}/stack"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CKPT = {"M2b_s0": f"{WORK}/ckpt/M2b_s0.pt", "M2b_s1": f"{WORK}/ckpt/M2b_s1.pt"}
SEEDS = ("M2b_s0", "M2b_s1")
B_BOOT = 2000
DT, K = 0.2, 30
F_PREREG = 0.1207      # PREREG's printed seed floor; the bar below uses the EXACT |ADE6(s0) - ADE6(s1)| from the banked npz (F_exact)
SHA = {"M2b_s0": "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8",
       "M2b_s1": "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}


def jdump(o, path):
    def enc(x):
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(o, fh, indent=1, default=enc)


def jround(o):
    """JSON round-trip with the same encoder as jdump (numpy -> python), for exact-equality comparisons of result blocks."""
    def enc(x):
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x))
    return json.loads(json.dumps(o, default=enc))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ the fit (closed form + clip + isotonic)
def pav_nondecreasing(y):
    """Unit-weight pool-adjacent-violators: the least-squares non-decreasing fit to ``y`` (declared: UNWEIGHTED -- the PREREG says 'isotonic'
    and names no weights)."""
    y = [float(v) for v in y]
    vals, cnt = [], []
    for v in y:
        vals.append(v)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            tot = vals[-2] * cnt[-2] + vals[-1] * cnt[-1]
            c = cnt[-2] + cnt[-1]
            vals[-2:] = [tot / c]
            cnt[-2:] = [c]
    out = []
    for v, c in zip(vals, cnt):
        out += [v] * c
    return np.asarray(out)


def fit_weights(tau, P, y):
    """tau, P, y: [N, 30, 2] (ego frame at NOW; axis 0 = along/x, axis 1 = cross/y). -> dict of [30, 2] arrays.
    raw = sum r (y - P) / sum r^2 per step and axis; w = isotonic(clip(raw, 0, 1)) -- the PREREG's order: clip, then monotone."""
    tau, P, y = (np.asarray(v, np.float64) for v in (tau, P, y))
    r, d = tau - P, y - P
    num, den = (r * d).sum(0), (r * r).sum(0)
    if not (den > 0).all():
        raise ValueError("a step/axis with sum r^2 == 0: the closed form is undefined")
    raw = num / den
    clipped = np.clip(raw, 0.0, 1.0)
    iso = np.stack([pav_nondecreasing(clipped[:, a]) for a in range(2)], 1)
    return {"raw": raw, "clipped": clipped, "w": iso, "num": num, "den": den, "n": int(len(r))}


def apply_w(P, tau, w):
    """H = P + w * (tau - P), w broadcast over windows ([30, 2] or scalar)."""
    P, tau = np.asarray(P, np.float64), np.asarray(tau, np.float64)
    return P + np.asarray(w, np.float64) * (tau - P)


def w_midpoints(w30):
    """DERIVED, UNSCORED: the 60 weights of traj60 (0.1 s grid): odd index j = knot j//2 ... even = mean of the neighbouring knots (first: knot 0)."""
    w60 = np.zeros((60, 2))
    w60[1::2] = w30
    mid = np.concatenate([w30[:1], 0.5 * (w30[:-1] + w30[1:])], 0)
    w60[0::2] = mid
    return w60


# ------------------------------------------------------------------------------------------------ model / data loaders
def load_model(name):
    import torch
    from tanitad.models import v8f_mini as M
    ck = torch.load(CKPT[name], map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    return model.eval(), cfg, ck


def cmd_fit(a):
    import torch
    torch.set_num_threads(a.threads)
    from tanitad.data import v8f_mini_data as D
    t0 = time.time()
    model, cfg, ck = load_model(a.seed)
    assert sha256_file(CKPT[a.seed]) == SHA[a.seed], "checkpoint sha256 mismatch"
    fc = json.load(open(f"{OUT}/fit_clips.json"))
    clips = fc["clips"][:a.max_clips] if a.max_clips else fc["clips"]
    J = D.load_join(f"{OUT}/box2/join_train.npz")
    tracks = D.load_native_tracks(f"{WORK}/inputs/native_tracks.pt")
    store = D.ClipStore(f"{OUT}/cache", "train", lru=a.lru, grid=ck.get("grid", "pool4x10"), kin_mode=cfg.kin_mode, tracks=tracks)
    wins = D.train_windows(store, J, clips, hist=cfg.hist, start=cfg.window_start)
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s,
                          window_start=cfg.window_start)
    N = len(wins)
    print(f"[fit {a.seed}] clips {len(clips)} windows {N} (trainer's own builder: train_windows, hist {cfg.hist}, start {ds.window_start})", flush=True)
    outs = {k: [] for k in ("traj30", "p30", "gt", "gt_ok", "ci", "t", "a_t", "a_s")}
    with torch.no_grad():
        for b0 in range(0, N, a.batch):
            items = []
            for i in range(b0, min(b0 + a.batch, N)):
                ci, t, _, _ = wins[i]
                a_t = i % 3                              # the EVAL's own deterministic age schedule (--ages cycle), on the train window index
                a_s = a_t + (i // 3) % 5
                items.append(ds.item(ci, t, a_t, a_s))
            b = D.collate(items)
            o = model.forward_infer(b)
            outs["traj30"].append(o["traj30"].float().numpy())
            outs["p30"].append(o["p10"].float().numpy())
            outs["gt"].append(b["gt"].numpy())
            outs["gt_ok"].append(b["gt_ok"].numpy())
            outs["ci"].append(b["ci"].numpy())
            outs["t"].append(b["t"].numpy())
            outs["a_t"].append(b["a_t"].numpy())
            outs["a_s"].append(b["a_s"].numpy())
            if (b0 // a.batch) % 10 == 0:
                print(f"[fit {a.seed}] {b0 + len(items)}/{N}  {time.time() - t0:.0f}s", flush=True)
    O = {k: np.concatenate(v) for k, v in outs.items()}
    assert O["gt_ok"].all(), "a train window without a 6 s future"
    tag = a.tag or a.seed
    np.savez_compressed(f"{OUT}/fit_windows_{tag}.npz", **O)
    # ---- the fit
    R = fit_weights(O["traj30"], O["p30"], O["gt"])
    d = lambda p: np.linalg.norm(p.astype(np.float64) - O["gt"].astype(np.float64), axis=-1)
    ade = lambda p: d(p).mean(1)
    Hf = apply_w(O["p30"], O["traj30"], R["w"])
    rec = {"seed": a.seed, "ckpt_sha256": SHA[a.seed], "k": fc["k"], "n_clips": len(clips), "fit_clips_sha256": fc["sha256_newline_joined"],
           "n_windows": int(N), "ages": "cycle (a_t = i%3, a_s = a_t + (i//3)%5, i = index in the fit window list; the eval's own schedule)",
           "window_builder": "tanitad.data.v8f_mini_data.train_windows(hist=%d, start=%s) -- the trainer's" % (cfg.hist, cfg.window_start),
           "tier": "T0/T1 open loop, own held outputs (forward_infer: p_gt 0, T's own code, no label read)",
           "w_along_by_step": R["w"][:, 0], "w_cross_by_step": R["w"][:, 1],
           "raw_along": R["raw"][:, 0], "raw_cross": R["raw"][:, 1],
           "clipped_along": R["clipped"][:, 0], "clipped_cross": R["clipped"][:, 1],
           "sum_r2_along": R["den"][:, 0], "sum_r2_cross": R["den"][:, 1],
           "n_clipped_at_0": int((R["raw"] < 0).sum()), "n_clipped_at_1": int((R["raw"] > 1).sum()),
           "n_changed_by_isotonic": int((np.abs(R["w"] - R["clipped"]) > 1e-12).sum()),
           "w60_derived_unscored": w_midpoints(R["w"]),
           "in_sample_train_ade6": {"B_tau": float(ade(O["traj30"]).mean()), "P": float(ade(O["p30"]).mean()), "H": float(ade(Hf).mean())},
           "in_sample_train_ade2": {"B_tau": float(d(O["traj30"])[:, :10].mean()), "P": float(d(O["p30"])[:, :10].mean()),
                                    "H": float(d(Hf)[:, :10].mean())},
           "seconds": time.time() - t0}
    jdump(rec, f"{OUT}/weights_{tag}.json")
    print(f"[fit {a.seed}] DONE {time.time() - t0:.0f}s  train ADE6 tau {rec['in_sample_train_ade6']['B_tau']:.4f} P {rec['in_sample_train_ade6']['P']:.4f} "
          f"H {rec['in_sample_train_ade6']['H']:.4f}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ score
BANK = f"{OUT}/inputs/banked"
P10_FILE = f"{OUT}/inputs/b2_p10_eval_pred6.npy"            # P10-yaw-4512's per-window eval plan (== repo refav1-hj-mini/raw/p10_yaw4512/, same sha256)
LOWER_IS_BETTER = {"tac_lat_acc": False, "tac_lon_acc": False}
FW_KEYS = ("lon_speed_mae", "lon_accel_mae", "lon_along_mae", "lat_cross_mae", "lat_heading_mae_deg", "lat_yaw_rate_mae_degps",
           "lat_curvature_mae_1pm", "tac_lat_acc", "tac_lon_acc")


def eval_world():
    from tanitad.data import v8f_mini_data as D
    grid = json.load(open(f"{WORK}/inputs/eval_grid_windows.json"))
    J = D.load_join(f"{WORK}/inputs/join_eval141.npz")
    X = D.eval_context(grid, f"{WORK}/inputs/eval141_kin.npz", J, kin_mode="yaw")
    return grid, J, X


def load_eval_module():
    import importlib.util
    sp = importlib.util.spec_from_file_location("v8f_mini_eval", f"{TREE}/stack/scripts/v8f_mini_eval.py")
    m = importlib.util.module_from_spec(sp)
    sys.modules["v8f_mini_eval"] = m
    sp.loader.exec_module(m)
    return m


def arm_pw(pred, gt, BE):
    """Per-window scalars of one arm: ADE / FDE at 1..6 s and the family components (taniteval's own `family_windows`)."""
    pred = np.asarray(pred, np.float64)
    d = np.linalg.norm(pred - gt, axis=-1)
    pw = {}
    for h in range(1, 7):
        pw[f"ade{h}"] = d[:, :5 * h].mean(1)
        pw[f"fde{h}"] = d[:, 5 * h - 1]
    for k, v in BE.family_windows(pred, gt, DT).items():
        pw["fw_" + k] = v
    return pw


def single(v, eid, CI, nb):
    v, eid = np.asarray(v, np.float64), np.asarray(eid)
    ok = np.isfinite(v)
    r = CI.episode_cluster_bootstrap(v[ok], eid[ok], n_boot=nb, seed=0)
    return {"mean": float(r["mean"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "n_windows": int(ok.sum())}


def prow(a, b, eid, BE, nb, key=None):
    return BE.paired_row(a, b, eid, nb, lower_is_better=LOWER_IS_BETTER.get((key or "").replace("fw_", ""), True))


def rows_for(pw_a, pw_b, eid, BE, nb):
    return {k: prow(pw_a[k], pw_b[k], eid, BE, nb, k) for k in pw_a}


def deep_diff(x, y, path="", tol=1e-6, out=None):
    out = [] if out is None else out
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                out.append(f"{path}/{k}: only in {'a' if k in x else 'b'}")
            else:
                deep_diff(x[k], y[k], f"{path}/{k}", tol, out)
    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            out.append(f"{path}: len {len(x)} vs {len(y)}")
        else:
            for i, (u, v) in enumerate(zip(x, y)):
                deep_diff(u, v, f"{path}[{i}]", tol, out)
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool):
        if not (x == y or (np.isnan(x) and np.isnan(y)) or abs(x - y) <= tol * max(1.0, abs(x), abs(y))):
            out.append(f"{path}: {x} vs {y}")
    elif x != y:
        out.append(f"{path}: {str(x)[:60]} vs {str(y)[:60]}")
    return out


def extras_of(z, D):
    return {"goal_prob": z["goal_prob"], "lat_logits": z["lat_logits"], "lon_logits": z["lon_logits"],
            "cons": {n: z["cons_pred"][:, D.CONS_IDX[n]] for n in D.CONS_FIELDS}}


def crossfit(tau, P, y, eid, clip_names):
    """SECONDARY (descriptive, NOT deployable): the PREREG's 2-fold episode cross-fit on eval-141 -- fold = parity of the SORTED-episode index."""
    rank = {c: i for i, c in enumerate(sorted(clip_names))}
    par = np.asarray([rank[clip_names[e]] % 2 for e in eid])
    H = np.zeros_like(tau)
    ws = {}
    for f in (0, 1):
        R = fit_weights(tau[par == f], P[par == f], y[par == f])
        ws[f"fit_on_parity_{f}"] = {"w_along": R["w"][:, 0], "w_cross": R["w"][:, 1], "n_windows": int((par == f).sum()),
                                    "n_episodes": int(len({e for e in eid[par == f]}))}
        sel = par != f
        H[sel] = apply_w(P[sel], tau[sel], R["w"])
    return H, ws


def cmd_score(a):
    import torch
    t0 = time.time()
    ev = load_eval_module()
    from taniteval import ci as CI
    from taniteval import v8f_beta_eval as BE
    from tanitad.data import v8f_mini_data as D
    seed, nb = a.seed, a.n_boot
    grid, J, X = eval_world()
    eid, gt = X["eid"], X["gt"].astype(np.float64)
    clip_names = J["clip"].tolist()
    z = np.load(f"{BANK}/eval_{seed}-kind-own.npz")
    bj = json.load(open(f"{BANK}/eval_{seed}-kind-own.json", encoding="utf-8"))
    tau = z["pred"].astype(np.float64)
    other = [s for s in SEEDS if s != seed][0]
    F_exact = abs(float(z["ade6"].mean()) - float(np.load(f"{BANK}/eval_{other}-kind-own.npz")["ade6"].mean()))
    # ---- P = this checkpoint's OWN frozen P10 (needs no image: 8 kinematic features + the kd_x base, both from the eval kin file)
    model, cfg, ck = load_model(seed)
    with torch.no_grad():
        P32 = model.prior(torch.as_tensor(X["kf8"], dtype=torch.float32), torch.as_tensor(X["base"], dtype=torch.float32)).numpy()
    P = P32.astype(np.float64)
    P_file = np.load(P10_FILE).astype(np.float64)
    fl = np.load(f"{OUT}/inputs/full_eval_floors_perwindow.npz")
    W = json.load(open(f"{OUT}/weights_{seed}.json"))
    w_fit = np.stack([np.asarray(W["w_along_by_step"]), np.asarray(W["w_cross_by_step"])], 1)          # [30, 2]
    ctl = {}
    # ---- identity controls (BEFORE anything else)
    dB = np.linalg.norm(tau - gt, axis=-1)
    ctl["banked_pred_eq_traj30"] = bool(np.array_equal(z["pred"], z["traj30"]))
    ctl["banked_ade6_recomputed_maxabs"] = float(np.abs(dB.mean(1) - z["ade6"]).max())
    ctl["banked_ade2_recomputed_maxabs"] = float(np.abs(dB[:, :10].mean(1) - z["ade2"]).max())
    ctl["banked_fde6_recomputed_maxabs"] = float(np.abs(dB[:, -1] - z["fde6"]).max())
    H1 = apply_w(P, tau, 1.0)
    H0 = apply_w(P, tau, 0.0)
    ctl["w1_vs_banked_pred_maxabs_m"] = float(np.abs(H1 - tau).max())
    ctl["w1_ade6_mean"] = float(np.linalg.norm(H1 - gt, axis=-1).mean())
    ctl["w1_ade6_vs_banked_mean_absdiff"] = abs(ctl["w1_ade6_mean"] - float(z["ade6"].mean()))
    ctl["w1_ade6_perwindow_maxabs_vs_banked"] = float(np.abs(np.linalg.norm(H1 - gt, axis=-1).mean(1) - z["ade6"]).max())
    ctl["w0_ade6_mean"] = float(np.linalg.norm(H0 - gt, axis=-1).mean())
    ctl["w0_ade6_vs_prereg_2.4870_absdiff"] = abs(ctl["w0_ade6_mean"] - 2.4870)
    ctl["w0_ade6_perwindow_maxabs_vs_floors_P10_refit"] = float(np.abs(np.linalg.norm(H0 - gt, axis=-1).mean(1) - fl["P10_refit__ade6"]).max())
    ctl["w0_ade2_perwindow_maxabs_vs_floors_P10_refit"] = float(np.abs(np.linalg.norm(H0 - gt, axis=-1)[:, :10].mean(1) - fl["P10_refit__ade2"]).max())
    ctl["model_prior_vs_P10_4512_file_maxabs_m"] = float(np.abs(P - P_file).max())
    ctl["P10_file_ade6_mean"] = float(np.linalg.norm(P_file - gt, axis=-1).mean())
    ctl["P10_file_ade6_perwindow_maxabs_vs_floors"] = float(np.abs(np.linalg.norm(P_file - gt, axis=-1).mean(1) - fl["P10_refit__ade6"]).max())
    ctl["fit_weights_in_range_and_monotone"] = bool(((w_fit >= 0) & (w_fit <= 1)).all() and (np.diff(w_fit, axis=0) >= -1e-12).all())
    ctl["F_exact_m"] = F_exact
    ctl["F_prereg_printed_m"] = F_PREREG
    ctl["PRIMARY_N_bar_m"] = F_exact / 2
    ctl["all_pass"] = bool(ctl["banked_pred_eq_traj30"] and ctl["w1_vs_banked_pred_maxabs_m"] <= 1e-4 and ctl["w1_ade6_vs_banked_mean_absdiff"] <= 1e-4
                           and ctl["w0_ade6_vs_prereg_2.4870_absdiff"] <= 1e-3 and ctl["model_prior_vs_P10_4512_file_maxabs_m"] <= 1e-4
                           and ctl["fit_weights_in_range_and_monotone"])
    print(f"[score {seed}] identity controls all_pass={ctl['all_pass']}  w1 maxabs {ctl['w1_vs_banked_pred_maxabs_m']:.2e}  "
          f"w0 ADE6 {ctl['w0_ade6_mean']:.4f}  prior-vs-file {ctl['model_prior_vs_P10_4512_file_maxabs_m']:.2e}", flush=True)
    if not ctl["all_pass"]:
        jdump({"seed": seed, "identity_controls": ctl}, f"{OUT}/score_{seed}.json")
        raise SystemExit(f"IDENTITY CONTROL FAILED -- refusing to read: {json.dumps(ctl)}")
    # ---- the arm
    H = apply_w(P, tau, w_fit)
    pwB, pwH, pwP = (arm_pw(p, gt, BE) for p in (tau, H, P_file))
    out = {"seed": seed, "ckpt_sha256": SHA[seed], "tier": "T1-class open loop (own held outputs; EVAL_DOCTRINE tier T0/T1 stamp the source eval carries)",
           "estimator": f"episode-cluster bootstrap, B={nb}, 141 episodes, seed 0; deltas PAIRED (same resampled episodes). Answers 'another draw of EPISODES' ONLY; "
                        "training-seed variance only via F = |ADE6(s0)-ADE6(s1)|; inference is deterministic (no inference-seed floor).",
           "n_windows": int(len(eid)), "n_episodes": int(len(set(eid.tolist()))), "identity_controls": ctl,
           "weights_file": f"weights_{seed}.json", "weights_sha256": sha256_file(f"{OUT}/weights_{seed}.json"),
           "route": "BANKED predictions: tau = B's banked `pred` (mirror raw/2026-10-25-kind/run/eval, KIND run own code); P = the checkpoint's own frozen P10 "
                    "evaluated on the eval kin features (no image needed), cross-checked to the P10-yaw-4512 file; GT from the eval context (eval's own).",
           "inputs_sha256": {"banked_npz": sha256_file(f"{BANK}/eval_{seed}-kind-own.npz"), "banked_json": sha256_file(f"{BANK}/eval_{seed}-kind-own.json"),
                             "p10_4512_pred": sha256_file(P10_FILE), "ckpt": SHA[seed]}}
    keys = list(pwB)
    out["single"] = {name: {k: single(pw[k], eid, CI, nb) for k in keys} for name, pw in (("B", pwB), ("H", pwH), ("P10", pwP))}
    out["paired_H_minus_B"] = rows_for(pwH, pwB, eid, BE, nb)
    out["paired_H_minus_P10"] = rows_for(pwH, pwP, eid, BE, nb)
    out["paired_B_minus_P10"] = {k: prow(pwB[k], pwP[k], eid, BE, nb, k) for k in ("ade2", "ade6", "fw_lon_accel_mae")}
    # ---- full eval blocks via the eval's OWN `evaluate` (families, consistency, tactical, self-consistency): H, P10, and the w=1 control
    ex = extras_of(z, D)
    resH, pwH_ev = ev.evaluate(H, X, ex, nb, with_families=not a.no_families)
    resP, _ = ev.evaluate(P_file, X, None, nb, with_families=not a.no_families)
    out["blocks"] = {"H": resH, "P10": resP}
    # CONSISTENCY (the programme's H-MINI-CONS-1 read; NOT in this PREREG's table -- reported BESIDE, per window, paired): H vs B and H vs P10
    ck = ("viol_any", "viol_SPEED", "viol_STOP", "viol_TURN", "viol_LEAD")
    out["consistency_beside"] = {
        "label": "CONSISTENCY(a) violation rates of the PLAN vs the GT-derived constraints (lower is better) and self-consistency (b) violation; not a PREREG bar",
        "rate": {"B": {k: float(z[k].mean()) for k in ck + ("self_viol_any",)}, "H": {k: float(pwH_ev[k].astype(float).mean()) for k in ck + ("self_viol_any",)},
                 "P10": {k: float(fl[f"P10_refit__{k}"].astype(float).mean()) for k in ck}},
        "paired_H_minus_B": {k: BE.paired_row(pwH_ev[k].astype(float), z[k].astype(float), eid, nb) for k in ck + ("self_viol_any",)},
        "paired_H_minus_P10": {k: BE.paired_row(pwH_ev[k].astype(float), fl[f"P10_refit__{k}"].astype(float), eid, nb) for k in ck}}
    if not a.no_families:
        res1, _ = ev.evaluate(H1, X, ex, nb, with_families=True)
        bd = deep_diff(jround({k: v for k, v in res1.items()}), {k: bj["raw"][k] for k in res1 if k in bj["raw"]}, tol=1e-6)
        out["w1_full_block_vs_banked_json"] = {"n_mismatches": len(bd), "mismatches": bd[:20], "compared_keys": sorted(set(res1) & set(bj["raw"]))}
        out["accel_mae_control"] = {"B_mean_fw_vs_banked_family_json": [float(np.nanmean(pwB["fw_lon_accel_mae"])),
                                                                          bj["raw"]["families"]["longitudinal"]["accel_mae_mps2"]],
                                    "P10_mean_fw_vs_PREREG_0.638": [float(np.nanmean(pwP["fw_lon_accel_mae"])), 0.638]}
    # STRUCTURAL IDENTITY: T's decision heads and S's checkpoints are untouched => the tactical block (macro-F1 of T's heads) and the strategic block
    # (S's checkpoint error) are B's, bit for bit. (families.tactical_from_trajectory is read FROM THE PLAN and is NOT covered by it.)
    out["structural_identity"] = {
        "tactical_block_T_heads": {"H_equals_banked_B": jround(resH["tactical"]) == jround(bj["raw"]["tactical"]),
                                   "note": "lat/lon macro-F1 and goal macro-F1 of T's own heads (logits unchanged)"},
        "strategic_block_S_checkpoints": {"H_equals_banked_B": True, "by_construction": "H never touches S; the banked ck_pos_err per window is carried unchanged",
                                          "banked_B_block": bj["raw"]["strategic"],
                                          "status": "PAUSED per PI ruling PI_DECISION_QUEUE ITEM 25 (2026-09-19): strategic layer switched off temporarily; reported as PAUSED, not a pass"},
        "families_tactical_from_trajectory_moves_with_H": ("families" in resH and "tactical_from_trajectory" in resH["families"])}
    # ---- SECONDARY: 2-fold episode cross-fit on eval-141 (descriptive, NOT deployable)
    Hcf, wcf = crossfit(tau, P, gt, eid, clip_names)
    pwCF = arm_pw(Hcf, gt, BE)
    out["crossfit_secondary"] = {"label": "CROSS-FITTED on eval-141 (fold = parity of the sorted-episode index); descriptive, not a deployable weight",
                                 "weights": wcf, "single": {k: single(pwCF[k], eid, CI, nb) for k in ("ade2", "ade6", "fw_lon_accel_mae")},
                                 "paired_vs_B": {k: prow(pwCF[k], pwB[k], eid, BE, nb, k) for k in ("ade1", "ade2", "ade3", "ade4", "ade5", "ade6", "fw_lon_accel_mae")}}
    # ---- diagnostics (declared as such, never a bar): eval-oracle in-sample weights = the ceiling of the lever; a shuffled-residual fit = noise control
    Ro = fit_weights(tau, P, gt)
    pwO = arm_pw(apply_w(P, tau, Ro["w"]), gt, BE)
    out["diag_eval_oracle_in_sample"] = {"label": "T0 DIAGNOSTIC: w fitted ON eval-141 itself (reads GT) -- the ceiling of this lever, never a result",
                                         "w_along": Ro["w"][:, 0], "w_cross": Ro["w"][:, 1],
                                         "single": {k: single(pwO[k], eid, CI, nb) for k in ("ade2", "ade6", "fw_lon_accel_mae")},
                                         "paired_vs_B": {k: prow(pwO[k], pwB[k], eid, BE, nb, k) for k in ("ade2", "ade6", "fw_lon_accel_mae")}}
    fw = np.load(f"{OUT}/fit_windows_{seed}.npz")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(fw["p30"]))
    r_tr = fw["traj30"].astype(np.float64) - fw["p30"].astype(np.float64)
    Rs = fit_weights(fw["p30"].astype(np.float64) + r_tr[perm], fw["p30"], fw["gt"])
    pwS = arm_pw(apply_w(P, tau, Rs["w"]), gt, BE)
    out["diag_shuffled_residual_fit"] = {"label": "NOISE CONTROL: w fitted on train windows with the residual permuted across windows (seed 0): must NOT manufacture a gain",
                                         "w_along": Rs["w"][:, 0], "w_cross": Rs["w"][:, 1],
                                         "single": {k: single(pwS[k], eid, CI, nb) for k in ("ade2", "ade6")},
                                         "paired_vs_B": {k: prow(pwS[k], pwB[k], eid, BE, nb, k) for k in ("ade2", "ade6")}}
    out["seconds"] = time.time() - t0
    jdump(out, f"{OUT}/score_{seed}.json")
    np.savez_compressed(f"{OUT}/score_{seed}_perwindow.npz", H=H, P=P, tau=tau, P_file=P_file, H_crossfit=Hcf, eid=eid,
                        **{f"H__{k}": v for k, v in pwH.items()}, **{f"B__{k}": v for k, v in pwB.items()}, **{f"P10__{k}": v for k, v in pwP.items()})
    pH = out["paired_H_minus_B"]
    print(f"[score {seed}] DONE {time.time() - t0:.0f}s  ADE2 H-B {pH['ade2']['mean']:+.4f} [{pH['ade2']['lo']:+.4f},{pH['ade2']['hi']:+.4f}]  "
          f"ADE6 H-B {pH['ade6']['mean']:+.4f} [{pH['ade6']['lo']:+.4f},{pH['ade6']['hi']:+.4f}]  accel H-B {pH['fw_lon_accel_mae']['mean']:+.4f}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ read: the outcome table, verbatim
def cmd_read(a):
    S = {s: json.load(open(f"{OUT}/score_{s}.json")) for s in SEEDS}
    for s in SEEDS:
        if not S[s]["identity_controls"]["all_pass"]:
            raise SystemExit(f"{s}: identity controls did not pass")
    F_exact = S["M2b_s0"]["identity_controls"]["F_exact_m"]
    bar_n = F_exact / 2
    rows = {}
    for s in SEEDS:
        d = S[s]["paired_H_minus_B"]
        a2, a6, ac = d["ade2"], d["ade6"], d["fw_lon_accel_mae"]
        dom_pt = {"ade2": S[s]["single"]["H"]["ade2"]["mean"] <= S[s]["single"]["P10"]["ade2"]["mean"],
                  "ade6": S[s]["single"]["H"]["ade6"]["mean"] <= S[s]["single"]["P10"]["ade6"]["mean"],
                  "accel": S[s]["single"]["H"]["fw_lon_accel_mae"]["mean"] <= S[s]["single"]["P10"]["fw_lon_accel_mae"]["mean"]}
        dp6 = S[s]["paired_H_minus_P10"]["ade6"]
        rows[s] = {"PRIMARY_S": {"delta": a2["mean"], "lo": a2["lo"], "hi": a2["hi"], "separated": a2["separated"], "pass": bool(a2["mean"] < 0 and a2["separated"])},
                   "PRIMARY_N": {"delta": a6["mean"], "lo": a6["lo"], "hi": a6["hi"], "bar_hi_le": bar_n, "pass": bool(a6["hi"] <= bar_n),
                                 "within_0.0003_of_bar": bool(abs(a6["hi"] - bar_n) < 3e-4),
                                 "hi_le_prereg_printed_0.060": bool(a6["hi"] <= 0.060)},
                   "SECONDARY_A": {"delta": ac["mean"], "lo": ac["lo"], "hi": ac["hi"], "separated": ac["separated"], "pass": bool(ac["mean"] < 0 and ac["separated"])},
                   "DOMINANCE_vs_P10": {"point_estimates_le_P10": dom_pt, "ade6_H_minus_P10": {k: dp6[k] for k in ("mean", "lo", "hi", "separated")},
                                        "H_dominates_P10": bool(all(dom_pt.values()) and dp6["separated"])}}
    s_ok = all(rows[s]["PRIMARY_S"]["pass"] for s in SEEDS)
    n_ok = all(rows[s]["PRIMARY_N"]["pass"] for s in SEEDS)
    if s_ok and n_ok:
        outcome, nxt = "PASS", ("H becomes the deployed v8F output. The shrinkage weights become a model buffer in the next trained arm, and the FLAT and MIXO reads are "
                                "re-scored with H applied to each arm's own fitted weights (descriptive).")
    elif s_ok:
        outcome, nxt = "SHORT ONLY", ("the residual's 6 s signal and its short-horizon error are entangled. Next: a TRAINED horizon-weighted loss arm under its own PREREG (box 2).")
    else:
        outcome, nxt = "NO GAIN", ("the short-horizon loss is not residual noise that shrinkage can remove. Next: a diagnosis of the short-horizon error by component "
                                   "(P10's own error vs the residual vs the speed code), before any training arm.")
    R = {"outcome": outcome, "committed_next_step_verbatim": nxt, "PRIMARY_S_both_seeds": s_ok, "PRIMARY_N_both_seeds": n_ok,
         "SECONDARY_A_both_seeds": all(rows[s]["SECONDARY_A"]["pass"] for s in SEEDS),
         "H_dominates_P10_both_seeds": all(rows[s]["DOMINANCE_vs_P10"]["H_dominates_P10"] for s in SEEDS),
         "F_exact_m": F_exact, "PRIMARY_N_bar_m": bar_n, "per_seed": rows,
         "estimator": S["M2b_s0"]["estimator"], "tier": S["M2b_s0"]["tier"]}
    jdump(R, f"{OUT}/READ.json")
    print(json.dumps({k: R[k] for k in ("outcome", "PRIMARY_S_both_seeds", "PRIMARY_N_both_seeds", "SECONDARY_A_both_seeds", "H_dominates_P10_both_seeds")}, indent=1))
    for s in SEEDS:
        print(s, json.dumps({k: rows[s][k] for k in ("PRIMARY_S", "PRIMARY_N", "SECONDARY_A")}, default=float))
    return 0


def cmd_dk(a):
    """LONGITUDINAL distance keeping (headway / time-gap / min-TTC / LEAD-VIOL) for B (both seeds), H (both seeds) and P10, through the programme's OWN
    reader `v8f_dk_read.read_dk` (-> `taniteval.lead_metrics`, no re-implementation), K = 30 (6 s) block."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("v8f_dk_read", f"{OUT}/deps/v8f_dk_read.py")
    dk = importlib.util.module_from_spec(sp)
    sys.modules["v8f_dk_read"] = dk
    sp.loader.exec_module(dk)
    grid, J, X = eval_world()
    order = list(grid["order"])
    preds, saved = {}, {}
    for s in SEEDS:
        z = np.load(f"{BANK}/eval_{s}-kind-own.npz")
        pw = np.load(f"{OUT}/score_{s}_perwindow.npz")
        k = s.split("_")[1]
        preds[f"B_{k}"], preds[f"H_{k}"] = z["pred"].astype(np.float64), pw["H"]
        saved[f"B_{k}"] = z["ade6"]
    preds["P10"] = np.load(P10_FILE).astype(np.float64)
    fl = np.load(f"{OUT}/inputs/full_eval_floors_perwindow.npz")["P10_refit__ade6"]
    blk = dk.load_block(f"{OUT}/inputs/b1_eval_lead_block_k30.npz")
    pairs = [("H_s0", "B_s0"), ("H_s1", "B_s1"), ("H_s0", "P10"), ("H_s1", "P10"), ("B_s0", "P10"), ("B_s1", "P10")]
    res = dk.read_dk(preds=preds, gt=X["gt"].astype(np.float64), eid=X["eid"], clip=np.array([order[e] for e in X["eid"]]), frame=2 * np.asarray(X["t"], np.int64),
                     v0=np.asarray(X["kin"][:, 0], np.float64), block=blk, k=30, dt=DT, n_boot=a.n_boot, seed=0, saved_ade6=saved, p10_floor_ade6=fl, pairs=pairs)
    jdump(dk._clean(res["read"]), f"{OUT}/dk_read.json")
    np.savez_compressed(f"{OUT}/dk_perwindow.npz", **res["perwindow"])
    r = res["read"]
    print("states", r["join"]["window_states"], "contract", r["human_reference"]["kinematic_contract"]["status"])
    for nm, ab in r["arms"].items():
        m = ab["metrics"]
        print(f"  {nm:5s} headway {m['headway_min_m'].get('mean')} time-gap {m['time_gap_min_s'].get('mean')} TTC {m['min_ttc_s'].get('mean')} LEAD-VIOL at-risk {ab['lead_viol'].get('share_of_at_risk')}")
    return 0


def cmd_xcheck(a):
    """ROUTE CROSS-CHECK on the eval clips whose features are on this box (1 + 4 clips): the CPU forward of THIS box reproduces B's banked GPU `pred`
    (the PREREG's 1e-4 m bar), and the model's embedded `out["p10"]` equals the standalone prior call used by `score` (so `tau - P` is the residual)."""
    import torch
    from pathlib import Path
    torch.set_num_threads(4)
    from tanitad.data import v8f_mini_data as D
    grid, J, X = eval_world()
    tracks = D.load_native_tracks(f"{WORK}/inputs/native_tracks.pt")
    wins = D.eval_grid_windows(f"{WORK}/inputs/eval_grid_windows.json", J)
    rep = {}
    for s in SEEDS:
        model, cfg, ck = load_model(s)
        z = np.load(f"{BANK}/eval_{s}-kind-own.npz")
        with torch.no_grad():
            P = model.prior(torch.as_tensor(X["kf8"], dtype=torch.float32), torch.as_tensor(X["base"], dtype=torch.float32)).numpy()
        dtau, dp, dres, n = 0.0, 0.0, 0.0, 0
        for cache in (f"{WORK}/cache", f"{WORK}/cache_viz"):
            have = {p.stem for p in Path(cache, "eval", "clips").glob("*.pt")}
            store = D.ClipStore(cache, "eval", lru=8, grid="pool4x10", kin_mode="yaw", tracks=tracks)
            ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
            idx = [i for i, w in enumerate(wins) if str(J["clip"][w[0]]) in have]
            for b0 in range(0, len(idx), 24):
                ii = idx[b0:b0 + 24]
                items = []
                for i in ii:
                    ci, t, _, _ = wins[i]
                    a_t = i % 3
                    items.append(ds.item(ci, t, a_t, a_t + (i // 3) % 5))
                with torch.no_grad():
                    o = model.forward_infer(D.collate(items))
                tr, p30 = o["traj30"].numpy().astype(np.float64), o["p10"].numpy().astype(np.float64)
                dtau = max(dtau, float(np.abs(tr - z["pred"][ii]).max()))
                dp = max(dp, float(np.abs(p30 - P[ii]).max()))
                dres = max(dres, float(np.abs((tr - p30) - (z["pred"][ii].astype(np.float64) - P[ii].astype(np.float64))).max()))
                n += len(ii)
        rep[s] = {"n_windows_checked": n, "max_abs_cpu_traj30_vs_banked_pred_m": dtau, "max_abs_embedded_p10_vs_standalone_prior_m": dp,
                  "max_abs_residual_cpu_vs_banked_minus_prior_m": dres, "bar_m": 1e-4, "pass": bool(dtau <= 1e-4 and dp <= 1e-4)}
        print(s, rep[s], flush=True)
    jdump(rep, f"{OUT}/xcheck.json")
    return 0


def register_score_read(sub):
    p = sub.add_parser("score")
    p.add_argument("--seed", required=True, choices=SEEDS)
    p.add_argument("--n-boot", type=int, default=B_BOOT)
    p.add_argument("--no-families", action="store_true", help="smoke only")
    p = sub.add_parser("read")
    p.add_argument("--n-boot", type=int, default=B_BOOT)
    p = sub.add_parser("xcheck")
    p = sub.add_parser("dk")
    p.add_argument("--n-boot", type=int, default=B_BOOT)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--seed", required=True, choices=SEEDS)
    p.add_argument("--tag", default=None)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lru", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-clips", type=int, default=0, help="smoke only")
    register_score_read(sub)
    a = ap.parse_args(argv)
    if a.mode == "fit":
        return cmd_fit(a)
    return {"score": cmd_score, "read": cmd_read, "dk": cmd_dk, "xcheck": cmd_xcheck}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
