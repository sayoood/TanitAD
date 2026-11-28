#!/usr/bin/env python3
"""khyp.py -- executes PREREG_V8F_KHYP.md exactly as written (dev box, CPU only, ZERO training, no GPU).

  hyps  --seed 0 --split eval|train   enumerate the K = 7 longitudinal hypotheses through O for every window -> OUT/hyps/M2b_s0_<split>.npz
                                      (streams clips one at a time, resumable shards; no cache is copied)
  read                                both seeds: controls first (the read REFUSES if a stop-control fails), then HEADROOM / PRIMARY / SECONDARY /
                                      beside rows, the PREREG outcome table applied verbatim -> KHYP_RESULT.md + READ.json

THE ARM (PREREG, fixed).  For each window, T's own code with its longitudinal slice (indices N_GOAL + N_LAT .. + N_LON, 7 classes) replaced by the
one-hot of class k, k = 0..6; goal, lateral and constraint slices stay T's own; O runs through ``forward_diag(code_override=...)`` -> tau_k.
  ORACLE      k* = argmin_k ADE6(tau_k) vs the GT                                       T0 diagnostic: headroom, never a result
  SEL-ARGMAX  k = argmax of T's longitudinal softmax                                    T1-class open loop
  SEL-MIX     sum_k p_k tau_k, p = T's longitudinal softmax                             T1-class open loop
  SEL-LEARN   multinomial logistic regression (L2, C = 1.0, fixed; StandardScaler on the features; nothing tuned on eval) predicting k*, then
              tau_khat. Features (63 = 7 + 7 x 8): T's longitudinal softmax (7); per hypothesis k, in this order: the plan's final speed, its minimum
              and maximum acceleration, its arc length, and its CONSISTENCY(b) violation flag (0 / 1) of T's OWN predicted constraints for each rule
              SPEED, STOP, TURN, LEAD. Fitted per seed on HBLEND's 9,582 training windows (141 clips); never on eval.

DECLARED INTERPRETATIONS (each a choice the PREREG text left open, fixed HERE, before any KHYP number existed):
  1. "ADE6" = the mean over the 30 knots (6 s) of the displacement to the GT, the eval's own ``ade6``; k* ties go to the lowest index (np.argmin).
  2. acceleration = (v[j] - v[j-1]) / 0.2 over consecutive interval speeds of ``taniteval.v8f_consistency.plan_kinematics``; speed = the LAST interval's.
  3. "violation ... per rule" = the 0 / 1 ``violated`` flag of ``taniteval.v8f_consistency.violations(tau_k, constraints_from_pred(T's outputs))``.
  4. LogisticRegression(C = 1.0, max_iter = 1000): max_iter is an optimiser budget, not a modelling choice; lbfgs, multinomial, L2 (sklearn's defaults).
  5. Ages: the eval's own deterministic cycle (a_t = i mod 3, a_s = a_t + (i div 3) mod 5, i = window index) on BOTH splits, as B and HBLEND's fit used.
  6. PRIMARY compares the UNROUNDED point mean(ADE6 LEARN) - mean(ADE6 B) with -F = -0.1207 (the PREREG's printed value); "separated" is the paired
     episode-cluster bootstrap's CI excluding 0 (``taniteval.ci``). SECONDARY: ``worse`` of ``v8f_beta_eval.paired_row`` (separated and in the bad direction).
  7. HEADROOM present  <=>  ORACLE - B <= -2F at BOTH seeds (the programme's both-seeds convention); NO HEADROOM is its complement (a seed split is flagged).
  8. CONTROL 4 is "by construction" only for the selectors that pick one of the 7 plans (ARGMAX, LEARN): a convex combination (MIX) can beat the best
     vertex when the GT lies inside the hull. So control 4 is HARD for ARGMAX / LEARN (any violation stops the read) and REPORTED for MIX (count and size).
  9. CONTROL 3 is one fixed permutation (seed 0) of the training labels across windows; if it separates from SEL-ARGMAX the read says CONTROL 3 FAILED and
     the outcome table is still printed verbatim, flagged, for the Master Mind to adjudicate (it does not silently pass).
 10. CONTROL 1 on real data runs where T's softmax is EXACTLY one-hot (n reported; possibly 0), AND on every window the harness identity
     ``forward_diag(T's own code) == forward_infer`` (exact); the one-hot case is pinned bit-exact by test_khyp.py on a model whose softmax is exactly one-hot.

Every number: MEASURED (ours) at the artifact it is written to; ORACLE = T0 diagnostic (never a result), selectors = T1-class open loop; estimator = paired
episode-cluster bootstrap, B = 2,000, seed 0, 141 episodes (``taniteval/ci.py``) -- it answers "would another draw of EPISODES say this?" and nothing about
training runs, so both seeds ride side by side with F.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
# commit charge: numpy's OpenBLAS reserves ~30 MB x logical CPUs at import (0.74 GB on this 24-thread box, 0.02 GB at 1 thread). BLAS is not the hot path here (torch is).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
SCR = OUT.parent
CODE = Path(os.environ.get("KHYP_CODE", str(OUT / "code")))
for _p in (CODE / "stack" / "scripts", CODE / "taniteval", CODE / "stack"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
from taniteval import v8f_consistency as C  # noqa: E402

K = M.N_LON                                   # 7 longitudinal classes
LON0 = M.N_GOAL + M.N_LAT                     # first longitudinal slot of T's code
DT = 0.2
RULES = C.RULE_NAMES                          # SPEED STOP TURN LEAD
PER_HYP = 4 + len(RULES)                      # final speed, min acc, max acc, arc, 4 flags
N_FEAT = K + K * PER_HYP                      # 63
F_PREREG = 0.1207                             # the PREREG's printed seed floor; 2F = 0.2414
B_BOOT = 2000
SEEDS = (0, 1)
SHA = {0: "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", 1: "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
TIER = {"ORACLE": "T0 diagnostic (reads the GT; headroom, never a result)", "selectors": "T1-class open loop (own held outputs, no GT read at inference)"}
ESTIMATOR = "paired episode-cluster bootstrap, B=2000, seed 0, 141 episodes (taniteval/ci.py)"
FEATURE_NAMES = ([f"p_lon[{k}]" for k in range(K)]
                 + [f"h{k}.{n}" for k in range(K) for n in ("v_end", "a_min", "a_max", "arc") + tuple(f"viol_{r}" for r in RULES)])
ARMS = ("ORACLE", "ARGMAX", "MIX", "LEARN")
NEXT = {  # the PREREG's committed next step, verbatim
    "MULTIMODAL PAYS": "a trained K-hypothesis head (T emits K codes, O plans each, a learned scorer selects) becomes the next box-2 arm, with this result as its prior. "
                       "The stacked HS / H + SEL-LEARN is scored against P10 and FLAT on the dominance row",
    "HEADROOM, NO SELECTOR": "the alternatives exist but the cheap features cannot pick them. Next: a scorer that reads the scene (a trained head; box 2)",
    "NO HEADROOM": "enumerating the longitudinal class does not move the plan enough; O does not use the longitudinal code strongly. Then `H-V8F-LON-1`'s speed-scale "
                   "lever is not reachable through the code, and the next arm targets O's speed profile directly",
}


def jdump(o, path):
    def enc(x):
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (set, tuple)):
            return list(x)
        raise TypeError(type(x))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(o, fh, indent=1, default=enc)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ the hypotheses
def lon_onehot_code(code0: torch.Tensor, k: int) -> torch.Tensor:
    """T's own code with the longitudinal slice replaced by the one-hot of class ``k``; everything else (goal, lateral, constraints) is T's own."""
    c = code0.clone()
    c[:, LON0:LON0 + K] = 0.0
    c[:, LON0 + k] = 1.0
    return c


def argmax_k(p_lon):
    """SEL-ARGMAX's pick: the argmax of T's longitudinal softmax (ties -> the lowest index)."""
    return np.asarray(p_lon).argmax(1)


def oracle_k(ade_k):
    """ORACLE's pick: argmin_k ADE6(tau_k) (ties -> the lowest index)."""
    return np.asarray(ade_k).argmin(1)


@torch.no_grad()
def enumerate_batch(model, b) -> dict:
    """One collated batch -> numpy arrays. ``forward_infer`` gives B's own plan and T's own code; 7 ``forward_diag`` calls give tau_k; one more
    ``forward_diag`` with T's UNCHANGED code is the harness identity (must equal forward_infer exactly)."""
    o0 = model.forward_infer(b)
    code0 = o0["code"]
    taus = []
    for k in range(K):
        ov = lon_onehot_code(code0, k)
        ok = model.forward_diag(b, ov)
        if not torch.equal(ok["code"], ov):
            raise AssertionError("O did not read the override: out['code'] != code_override")
        taus.append(ok["traj30"].float())
    idn = model.forward_diag(b, code0)
    f = lambda t: t.float().cpu().numpy()                                # noqa: E731
    return {"tau": np.stack([f(t) for t in taus], 1), "own": f(o0["traj30"]), "own_diag_maxabs": f((idn["traj30"] - o0["traj30"]).abs().flatten(1).max(1).values),
            "p30": f(o0["p10"]), "p_lon": f(code0[:, LON0:LON0 + K]), "lon_logits": f(o0["lon_logits"]), "goal_prob": f(o0["goal_prob"]),
            "cons_pred": f(o0["cons_pred"]), "gt": b["gt"].numpy(), "gt_ok": b["gt_ok"].numpy(), "limit": b["limit"].numpy(),
            "limit_ok": b["limit_ok"].numpy(), "lead_gap_h": b["lead_gap_h"].numpy(), "s_h": b["s_h"].numpy(),
            "ci": b["ci"].numpy(), "t": b["t"].numpy(), "a_t": b["a_t"].numpy(), "a_s": b["a_s"].numpy()}


# ------------------------------------------------------------------------------------------------ features and selectors
def limit_ms_f64(limit, limit_ok):
    """The eval context's own expression (``eval_context``): float64, NaN where the sign is unknown."""
    return np.where(np.asarray(limit_ok, bool), np.asarray(limit, np.float64) @ (np.array(D.LIMITS_KMH, np.float64) / 3.6), np.nan)


def pred_constraints(d: dict, ctx: dict | None = None) -> dict:
    """T's OWN constraint dict (CONSISTENCY(b)'s), from T's heads only -- identical for all k, so a plan's violation of it is a feature, not a label.
    ``ctx`` (the eval context ``X``) swaps in the evaluator's own float64 limit / lead gap / human arc for REPORTING; the FEATURES always use the stored batch values
    (float32, the same path on the train and the eval windows)."""
    cons = {n: np.asarray(d["cons_pred"])[:, D.CONS_IDX[n]] for n in D.CONS_FIELDS}
    lim, gap, sh = ((ctx["limit_ms"], ctx["lead_gap_h"], ctx["s_h"]) if ctx is not None else (limit_ms_f64(d["limit"], d["limit_ok"]), d["lead_gap_h"], d["s_h"]))
    return C.constraints_from_pred({"goal_prob": d["goal_prob"], "lon_logits": d["lon_logits"], "cons": cons}, lim, gap, sh)


def plan_feats(tau) -> np.ndarray:
    """[N, 30, 2] ego-frame plan -> [N, 4]: final speed (m/s), min and max acceleration (m/s^2), arc length (m)."""
    kin = C.plan_kinematics(np.asarray(tau, np.float64), DT)
    v, s = kin["v"], kin["s"]
    a = np.diff(v, axis=1) / DT
    return np.stack([v[:, -1], a.min(1), a.max(1), s[:, -1]], 1)


def hyp_feature_matrix(d: dict) -> np.ndarray:
    """[N, 63] in :data:`FEATURE_NAMES` order."""
    pc = pred_constraints(d)
    cols = [np.asarray(d["p_lon"], np.float64)]
    for k in range(K):
        tau = np.asarray(d["tau"][:, k], np.float64)
        V = C.violations(tau, pc)
        cols.append(plan_feats(tau))
        cols += [V[r]["violated"].astype(np.float64)[:, None] for r in RULES]
    X = np.concatenate(cols, 1)
    assert X.shape[1] == N_FEAT, X.shape
    return X


def ade6_per_hyp(tau, gt) -> np.ndarray:
    """[N, 7] per-window ADE6 of each hypothesis."""
    return np.linalg.norm(np.asarray(tau, np.float64) - np.asarray(gt, np.float64)[:, None], axis=-1).mean(2)


def pick(tau, k):
    return np.asarray(tau, np.float64)[np.arange(len(k)), np.asarray(k)]


def plan_mix(tau, p_lon):
    return (np.asarray(p_lon, np.float64)[:, :, None, None] * np.asarray(tau, np.float64)).sum(1)


def fit_learn(X, y):
    """SEL-LEARN: StandardScaler + multinomial L2 logistic regression, C = 1.0 (fixed). No tuning."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        raise ValueError(f"SEL-LEARN needs >= 2 observed classes among the training windows' k*; saw {np.unique(y).tolist()}")
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X), y)
    return sc, clf


def predict_learn(model, X):
    sc, clf = model
    return clf.predict(sc.transform(X))


def permute_labels(y, seed=0):
    """CONTROL 3: the labels permuted across windows (same multiset, features and labels decoupled)."""
    return np.asarray(y)[np.random.default_rng(seed).permutation(len(y))]


def hypotheses_distinct(tau, tol=1e-4) -> bool:
    """True iff the 7 hypotheses are pairwise distinct somewhere (a dead override makes all seven the same plan)."""
    tau = np.asarray(tau, np.float64)
    return all(np.abs(tau[:, i] - tau[:, j]).max() > tol for i in range(K) for j in range(i + 1, K))


def control1_identity(tau, own, p_lon):
    """CONTROL 1: on the windows where T's longitudinal softmax is EXACTLY one-hot, tau[argmax] == forward_infer's plan (Delta = 0.0)."""
    p = np.asarray(p_lon)
    exact = (p.max(1) == 1.0) & ((p > 0).sum(1) == 1)
    n = int(exact.sum())
    if n == 0:
        return {"n_exact_onehot_windows": 0, "max_abs_delta_m": None, "pass": None, "note": "vacuous on this data (no exactly one-hot softmax)"}
    sel = pick(tau, argmax_k(p))
    dl = float(np.abs(sel[exact] - np.asarray(own, np.float64)[exact]).max())
    return {"n_exact_onehot_windows": n, "max_abs_delta_m": dl, "pass": bool(dl == 0.0)}


def control4(ade_k, picks: dict) -> dict:
    """CONTROL 4: ORACLE <= every selector that picks one of the 7 plans, on every window (hard); MIX reported separately."""
    ora = np.asarray(ade_k)[np.arange(len(ade_k)), oracle_k(ade_k)]
    out = {}
    for name, k in picks.items():
        a = np.asarray(ade_k)[np.arange(len(ade_k)), np.asarray(k)]
        out[name] = {"n_violations": int((ora > a + 1e-12).sum()), "max_excess_m": float(max((ora - a).max(), 0.0))}
    out["pass"] = bool(all(v["n_violations"] == 0 for v in out.values()))
    return out


# ------------------------------------------------------------------------------------------------ data side
def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    return model.eval(), cfg, ck


def window_list(split, a, J, store, cfg):
    if split == "eval":
        return D.eval_grid_windows(a.eval_grid, J)
    clips = json.load(open(a.fit_clips))["clips"]
    return D.train_windows(store, J, clips, hist=cfg.hist, start=cfg.window_start)


def cmd_hyps(a):
    torch.set_num_threads(max(1, min(a.threads, 6)))
    t0 = time.time()
    model, cfg, ck = load_model(a.ckpt)
    tracks = D.load_native_tracks(a.tracks) if cfg.kin_mode == "yaw" else None
    cache = a.eval_cache if a.split == "eval" else a.train_cache
    J = D.load_join(a.eval_join if a.split == "eval" else a.train_join)
    store = D.ClipStore(cache, a.split, lru=a.lru, grid=ck.get("grid", "pool4x10"), kin_mode=cfg.kin_mode, tracks=tracks)
    wins = window_list(a.split, a, J, store, cfg)
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    N = len(wins)
    tag = f"M2b_s{a.seed}_{a.split}"
    sdir = OUT / "hyps" / f"{tag}_shards"
    sdir.mkdir(parents=True, exist_ok=True)
    shard = max(a.batch, (a.shard // a.batch) * a.batch)
    print(f"[hyps {tag}] windows {N} batch {a.batch} shard {shard} threads {torch.get_num_threads()} lru {a.lru}", flush=True)
    for s0 in range(0, N, shard):
        path = sdir / f"shard_{s0 // shard:05d}.npz"
        if path.exists():
            continue
        parts = []
        for b0 in range(s0, min(s0 + shard, N), a.batch):
            items = []
            for i in range(b0, min(b0 + a.batch, N, s0 + shard)):
                ci, t, _, _ = wins[i]
                a_t = i % 3                                             # the eval's own deterministic cycle, on the global window index
                a_s = a_t + (i // 3) % 5
                items.append(ds.item(ci, t, a_t, a_s))
            parts.append(enumerate_batch(model, D.collate(items)))
        cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        tmp = sdir / f"tmp_{path.name}"
        np.savez(tmp, **cat)
        os.replace(tmp, path)
        print(f"[hyps {tag}] {min(s0 + shard, N)}/{N}  {time.time() - t0:.0f}s", flush=True)
    files = sorted(glob.glob(str(sdir / "shard_*.npz")))
    parts = [dict(np.load(f)) for f in files]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    assert len(cat["tau"]) == N, (len(cat["tau"]), N)
    out = OUT / "hyps" / f"{tag}.npz"
    np.savez_compressed(out, **cat)
    meta = {"tag": tag, "n_windows": N, "ckpt": a.ckpt, "ckpt_sha256": sha256_file(a.ckpt), "split": a.split, "seconds": time.time() - t0,
            "ages": "cycle (a_t = i%3, a_s = a_t + (i//3)%5, i = global window index)", "tier": TIER["selectors"],
            "own_diag_maxabs_over_windows": float(cat["own_diag_maxabs"].max()), "torch_threads": torch.get_num_threads()}
    jdump(meta, OUT / "hyps" / f"{tag}.meta.json")
    print(f"[hyps {tag}] DONE {time.time() - t0:.0f}s own_diag_maxabs {meta['own_diag_maxabs_over_windows']}", flush=True)
    return 0


def load_hyps(path) -> dict:
    return dict(np.load(path))


# ------------------------------------------------------------------------------------------------ the read
class Stop(Exception):
    """A stop-control failed: the read refuses to interpret anything."""


def arm_pw(pred, gt, BE):
    """Per-window scalars of one arm: ADE / FDE at 1..6 s and the family components (taniteval's own ``family_windows``)."""
    pred = np.asarray(pred, np.float64)
    d = np.linalg.norm(pred - gt, axis=-1)
    pw = {}
    for h in range(1, 7):
        pw[f"ade{h}"] = d[:, :5 * h].mean(1)
        pw[f"fde{h}"] = d[:, 5 * h - 1]
    for k, v in BE.family_windows(pred, gt, DT).items():
        pw["fw_" + k] = v
    return pw


LOWER_IS_BETTER = {"tac_lat_acc": False, "tac_lon_acc": False}


def prow(a, b, eid, BE, nb, key):
    r = BE.paired_row(a, b, eid, nb, lower_is_better=LOWER_IS_BETTER.get(key.replace("fw_", ""), True))
    if "mean" in r:
        ok = np.isfinite(a) & np.isfinite(b)
        r["point_unrounded"] = float(np.mean(np.asarray(a)[ok]) - np.mean(np.asarray(b)[ok]))
    return r


def single(v, eid, nb):
    from taniteval.ci import episode_cluster_bootstrap
    v, eid = np.asarray(v, np.float64), np.asarray(eid)
    ok = np.isfinite(v)
    r = episode_cluster_bootstrap(v[ok], eid[ok], n_boot=nb, seed=0)
    return {"mean": float(r["mean"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "n_windows": int(ok.sum())}


def decide(per_seed: dict, F: float = F_PREREG) -> dict:
    """The PREREG outcome table, verbatim. ``per_seed[s]`` = {headroom_point, primary_point, primary_separated, secondary_ok}."""
    prim = {s: bool(r["primary_point"] <= -F and r["primary_separated"]) for s, r in per_seed.items()}
    sec = {s: bool(r["secondary_ok"]) for s, r in per_seed.items()}
    head = {s: bool(r["headroom_point"] <= -2 * F) for s, r in per_seed.items()}
    pays = all(prim.values()) and all(sec.values())
    headroom = all(head.values())
    if pays:
        out = "MULTIMODAL PAYS"
    elif headroom:
        out = "HEADROOM, NO SELECTOR"
    else:
        out = "NO HEADROOM"
    return {"outcome": out, "PRIMARY_per_seed": prim, "SECONDARY_per_seed": sec, "HEADROOM_le_minus_2F_per_seed": head,
            "headroom_seed_split": bool(any(head.values()) and not all(head.values())), "F": F, "bar_primary": -F, "bar_headroom": -2 * F,
            "committed_next_step_quoted": NEXT[out]}


def _eval_modules():
    import v8f_mini_eval as EV
    from taniteval import v8f_beta_eval as BE
    return EV, BE


def _banked_checks(a, s, ev, X, ctl):
    """CONTROL 2 (reproduction) and the input-identity controls; the read REFUSES if any stop-control fails."""
    gt = X["gt"].astype(np.float64)
    ctl["alignment"] = {"gt_maxabs": float(np.abs(ev["gt"].astype(np.float64) - gt).max()),
                        "limit_ms_equal": bool(np.array_equal(limit_ms_f64(ev["limit"], ev["limit_ok"]), X["limit_ms"], equal_nan=True)),
                        "lead_gap_h_equal": bool(np.array_equal(ev["lead_gap_h"].astype(np.float64), X["lead_gap_h"].astype(np.float64), equal_nan=True)),
                        "s_h_maxabs": float(np.abs(ev["s_h"].astype(np.float64) - X["s_h"]).max()), "n": int(len(gt))}
    al = ctl["alignment"]
    if not (al["gt_maxabs"] < 1e-4 and al["limit_ms_equal"] and al["lead_gap_h_equal"] and al["s_h_maxabs"] < 1e-4 and al["n"] == len(ev["tau"])):
        raise Stop(f"alignment of the stored windows with the eval context failed: {al}")
    d6 = np.linalg.norm(ev["own"].astype(np.float64) - gt, axis=-1).mean(1)
    ctl["own_vs_diag_identity_maxabs_m"] = float(ev["own_diag_maxabs"].max())
    if ctl["own_vs_diag_identity_maxabs_m"] != 0.0:
        raise Stop(f"harness identity broken: forward_diag(T's own code) != forward_infer, max abs {ctl['own_vs_diag_identity_maxabs_m']}")
    if a.no_banked:
        ctl["control2_reproduction"] = {"skipped": "smoke (--no-banked)"}
        return d6
    ctl["control2_reproduction"] = {}
    for name, path in (("banked_kind_own", f"{SCR}/hblend/inputs/banked/eval_M2b_s{s}-kind-own.npz"), ("sfresh_cycle_rerun", f"{SCR}/sfresh/eval/eval_M2b_s{s}-sfresh-cycle.npz")):
        z = np.load(path)
        r = {"path": path, "ade6_mean_mine": float(d6.mean()), "ade6_mean_banked": float(z["ade6"].mean()), "ade6_abs_diff_m": float(abs(d6.mean() - z["ade6"].mean())),
             "traj30_maxabs_vs_banked_pred_m": float(np.abs(ev["own"].astype(np.float64) - z["pred"].astype(np.float64)).max()), "stop_bar_m": 1e-3}
        r["pass"] = bool(r["ade6_abs_diff_m"] <= 1e-3 and r["traj30_maxabs_vs_banked_pred_m"] <= 1e-3)
        ctl["control2_reproduction"][name] = r
        if not r["pass"]:
            raise Stop(f"CONTROL 2 (reproduction) failed against {name}: {r}")
    return d6


def _train_list_check(a, s, tr, ctl):
    if a.no_banked:
        ctl["train_window_list"] = {"skipped": "smoke (--no-banked)"}
        return
    z = np.load(f"{SCR}/hblend/fit_windows_M2b_s{s}.npz")
    r = {"n_mine": int(len(tr["ci"])), "n_hblend": int(len(z["ci"]))}
    r.update({f: bool(np.array_equal(tr[f], z[f])) for f in ("ci", "t", "a_t", "a_s")})
    r["gt_maxabs"] = float(np.abs(tr["gt"].astype(np.float64) - z["gt"].astype(np.float64)).max()) if r["n_mine"] == r["n_hblend"] else None
    r["own_traj30_maxabs_vs_hblend_fit_m"] = float(np.abs(tr["own"].astype(np.float64) - z["traj30"].astype(np.float64)).max()) if r["n_mine"] == r["n_hblend"] else None
    r["pass"] = bool(r["n_mine"] == r["n_hblend"] and all(r[f] for f in ("ci", "t", "a_t", "a_s")) and (r["own_traj30_maxabs_vs_hblend_fit_m"] or 1) <= 1e-3)
    ctl["train_window_list"] = r
    if not r["pass"]:
        raise Stop(f"the training window list is not HBLEND's: {r}")


def load_h_weights(path) -> np.ndarray:
    """HBLEND's banked shrinkage weights -> [30, 2] (along, cross) in the ego frame at NOW."""
    w = json.load(open(path))
    return np.stack([np.asarray(w["w_along_by_step"], np.float64), np.asarray(w["w_cross_by_step"], np.float64)], 1)


def apply_h(P, plan, W) -> np.ndarray:
    """H = P + w * (plan - P) (HBLEND's ``apply_w``): P = the checkpoint's own frozen P10, w broadcast over windows."""
    P = np.asarray(P, np.float64)
    return P + np.asarray(W, np.float64) * (np.asarray(plan, np.float64) - P)


def score_seed(a, s, X, EV, BE, nb):
    ev = load_hyps(OUT / "hyps" / f"M2b_s{s}_eval.npz")
    tr = load_hyps(OUT / "hyps" / f"M2b_s{s}_train.npz")
    ctl = {}
    gt = X["gt"].astype(np.float64)
    eid = X["eid"]
    d6_own = _banked_checks(a, s, ev, X, ctl)
    _train_list_check(a, s, tr, ctl)
    # ---- features, labels, SEL-LEARN (fitted on TRAIN only)
    ktr = oracle_k(ade6_per_hyp(tr["tau"], tr["gt"]))
    kev = oracle_k(ade6_per_hyp(ev["tau"], gt))
    Xtr, Xev = hyp_feature_matrix(tr), hyp_feature_matrix(ev)
    lm = fit_learn(Xtr, ktr)
    khat = predict_learn(lm, Xev)
    kperm = predict_learn(fit_learn(Xtr, permute_labels(ktr, 0)), Xev)
    kart = argmax_k(ev["p_lon"])
    ade_k = ade6_per_hyp(ev["tau"], gt)
    ctl["control1"] = control1_identity(ev["tau"], ev["own"], ev["p_lon"])
    ctl["control1"]["harness_identity_maxabs_m"] = ctl["own_vs_diag_identity_maxabs_m"]
    ctl["control4"] = control4(ade_k, {"ARGMAX": kart, "LEARN": khat})
    mixp = plan_mix(ev["tau"], ev["p_lon"])
    d6_mix = np.linalg.norm(mixp - gt, axis=-1).mean(1)
    ora = ade_k[np.arange(len(kev)), kev]
    ctl["control4"]["MIX_reported"] = {"n_windows_mix_beats_oracle": int((d6_mix < ora - 1e-12).sum()), "max_excess_m": float(max((ora - d6_mix).max(), 0.0))}
    ctl["hypotheses_distinct"] = bool(hypotheses_distinct(ev["tau"]))
    if not ctl["control4"]["pass"]:
        raise Stop(f"CONTROL 4 failed: ORACLE is worse than a vertex-picking selector: {ctl['control4']}")
    # ---- plans (all [N, 30, 2] float64)
    B = ev["own"].astype(np.float64)
    plans = {"B": B, "ORACLE": pick(ev["tau"], kev), "ARGMAX": pick(ev["tau"], kart), "MIX": mixp, "LEARN": pick(ev["tau"], khat), "LEARN_PERM": pick(ev["tau"], kperm)}
    # ---- the beside stack: H (HBLEND shrinkage, banked weights) applied to B and to SEL-LEARN's plan (descriptive)
    if not a.no_banked:
        W = load_h_weights(f"{SCR}/hblend/weights_M2b_s{s}.json")
        P = ev["p30"].astype(np.float64)
        plans["H_B"] = apply_h(P, B, W)
        plans["H_LEARN"] = apply_h(P, plans["LEARN"], W)
        zh = np.load(f"{SCR}/hblend/score_M2b_s{s}_perwindow.npz")
        ctl["stack_H_on_B_vs_hblend_banked_H_maxabs_m"] = float(np.abs(plans["H_B"] - zh["H"].astype(np.float64)).max())
    # ---- metrics: the eval's own evaluate() (traj, consistency_a, families) + per-window scalars + paired rows vs B
    R = {"arms": {}, "paired_vs_B": {}, "ctl": ctl}
    PW = {}
    for name, pl in plans.items():
        res, pw = EV.evaluate(pl, X, None, nb, with_families=True)
        pcs = pred_constraints(ev, X)
        sc = C.consistency(pl, pcs, eid=eid, n_boot=nb, seed=0)
        R["arms"][name] = {"evaluate": res, "consistency_b": {k: v for k, v in sc.items() if k != "_per_window"}, "self_viol_any": float(sc["_per_window"]["any_violated"].mean())}
        PW[name] = arm_pw(pl, gt, BE)
        PW[name]["viol_any"] = pw["viol_any"].astype(np.float64)
        PW[name]["self_viol_any"] = sc["_per_window"]["any_violated"].astype(np.float64)
        R["arms"][name]["ade6_single"] = single(PW[name]["ade6"], eid, nb)
    for name in plans:
        if name == "B":
            continue
        R["paired_vs_B"][name] = {k: prow(PW[name][k], PW["B"][k], eid, BE, nb, k) for k in PW[name]}
    R["paired_vs_B"]["LEARN_minus_ARGMAX"] = {k: prow(PW["LEARN"][k], PW["ARGMAX"][k], eid, BE, nb, k) for k in ("ade6", "ade2")}
    R["paired_vs_B"]["LEARN_PERM_minus_ARGMAX"] = {k: prow(PW["LEARN_PERM"][k], PW["ARGMAX"][k], eid, BE, nb, k) for k in ("ade6", "ade2")}
    if "H_LEARN" in PW:
        R["paired_vs_B"]["H_LEARN_minus_H_B"] = {k: prow(PW["H_LEARN"][k], PW["H_B"][k], eid, BE, nb, k) for k in ("ade2", "ade6", "fw_lon_accel_mae", "fw_lon_speed_mae")}
    # ---- selector accuracy vs k*, class distribution
    R["selector_accuracy"] = {"ARGMAX_vs_kstar": float((kart == kev).mean()), "LEARN_vs_kstar": float((khat == kev).mean()), "LEARN_PERM_vs_kstar": float((kperm == kev).mean()),
                              "LEARN_in_sample_train": float((predict_learn(lm, Xtr) == ktr).mean()), "majority_train_class_acc_on_eval": float((kev == np.bincount(ktr, minlength=K).argmax()).mean()),
                              "kstar_eval_counts": np.bincount(kev, minlength=K).tolist(), "kstar_train_counts": np.bincount(ktr, minlength=K).tolist(),
                              "pick_counts_ARGMAX": np.bincount(kart, minlength=K).tolist(), "pick_counts_LEARN": np.bincount(khat, minlength=K).tolist(),
                              "n_eval": int(len(kev)), "n_train": int(len(ktr))}
    # ---- the three reads
    ho, pr = R["paired_vs_B"]["ORACLE"]["ade6"], R["paired_vs_B"]["LEARN"]["ade6"]
    sp, ac = R["paired_vs_B"]["LEARN"]["fw_lon_speed_mae"], R["paired_vs_B"]["LEARN"]["fw_lon_accel_mae"]
    R["HEADROOM"] = {"read": "ADE6 ORACLE - B (T0, descriptive)", **{k: ho.get(k) for k in ("point_unrounded", "mean", "lo", "hi", "separated")}}
    R["PRIMARY"] = {"read": "ADE6 SEL-LEARN - B", "bar": f"<= -{F_PREREG} and separated", **{k: pr.get(k) for k in ("point_unrounded", "mean", "lo", "hi", "separated")}}
    R["SECONDARY"] = {"speed_mae": {k: sp.get(k) for k in ("point_unrounded", "mean", "lo", "hi", "separated", "worse")},
                      "accel_mae": {k: ac.get(k) for k in ("point_unrounded", "mean", "lo", "hi", "separated", "worse")}}
    c3 = R["paired_vs_B"]["LEARN_PERM_minus_ARGMAX"]["ade6"]
    ctl["control3"] = {"perm_minus_argmax_ade6": {k: c3.get(k) for k in ("point_unrounded", "mean", "lo", "hi", "separated")}, "pass": bool(not c3.get("separated", True)),
                       "bar": "SEL-LEARN fitted on permuted labels must NOT separate from SEL-ARGMAX"}
    R["seed_row"] = {"headroom_point": ho["point_unrounded"], "primary_point": pr["point_unrounded"], "primary_separated": bool(pr["separated"]),
                     "secondary_ok": bool((not sp.get("worse", True)) and (not ac.get("worse", True)))}
    R["B_ade6_mean"] = float(d6_own.mean())
    return R, plans


def distance_keeping(a, X, grid, plans, nb):
    """LONGITUDINAL distance keeping (headway / time-gap / min-TTC / LEAD-VIOL) through the programme's OWN reader ``v8f_dk_read.read_dk`` (the tip's file,
    exported to OUT/raw), K = 30 (6 s) lead block. Best effort: an error is RECORDED (a work item), never swallowed into a pass, and it never gates the table."""
    if a.no_banked:
        return {"status": "skipped (smoke --no-banked)"}
    try:
        import importlib.util
        sp = importlib.util.spec_from_file_location("v8f_dk_read", str(OUT / "raw" / "v8f_dk_read.py"))
        dk = importlib.util.module_from_spec(sp)
        sys.modules["v8f_dk_read"] = dk
        sp.loader.exec_module(dk)
        order = list(grid["order"])
        preds, saved = {}, {}
        for s in SEEDS:
            for arm in ("B", "ORACLE", "ARGMAX", "MIX", "LEARN"):
                preds[f"{arm}_s{s}"] = plans[s][arm]
            saved[f"B_s{s}"] = np.load(f"{SCR}/hblend/inputs/banked/eval_M2b_s{s}-kind-own.npz")["ade6"]
        preds["P10"] = np.load(f"{SCR}/sfresh/inputs/p10/p10_eval_pred6.npy").astype(np.float64)
        fl = np.load(f"{SCR}/hblend/inputs/full_eval_floors_perwindow.npz")["P10_refit__ade6"]
        blk = dk.load_block(f"{SCR}/hblend/inputs/b1_eval_lead_block_k30.npz")
        pairs = [(f"{arm}_s{s}", f"B_s{s}") for s in SEEDS for arm in ("ORACLE", "ARGMAX", "MIX", "LEARN")]
        res = dk.read_dk(preds=preds, gt=X["gt"].astype(np.float64), eid=X["eid"], clip=np.array([order[e] for e in X["eid"]]), frame=2 * np.asarray(X["t"], np.int64),
                         v0=np.asarray(X["kin"][:, 0], np.float64), block=blk, k=30, dt=DT, n_boot=nb, seed=0, saved_ade6=saved, p10_floor_ade6=fl, pairs=pairs)
        jdump(dk._clean(res["read"]), OUT / "dk_read.json")
        r = res["read"]
        arms = {}
        for nm, ab in r["arms"].items():
            m = ab["metrics"]
            arms[nm] = {"headway_min_m": m["headway_min_m"].get("mean"), "time_gap_min_s": m["time_gap_min_s"].get("mean"), "min_ttc_s": m["min_ttc_s"].get("mean"),
                        "lead_viol_share_of_at_risk": ab["lead_viol"].get("share_of_at_risk")}
        return {"status": "OK", "artifact": str(OUT / "dk_read.json"), "reader": "TanitAD Research Lab/.../code/v8f_dk_read.py @ 8d6449a7", "arms": arms,
                "pairs_in": "dk_read.json::pairs[<arm>__minus__<B>]"}
    except Exception as e:                                                                # recorded, not dropped
        return {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}


def fmt(r, nd=4):
    if r is None or "mean" not in r:
        return "n/a"
    return f"{r['mean']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]" + (" *" if r.get("separated") else "")


def write_md(path, READ, RES):
    L = []
    o = READ["decision"]
    L.append("# KHYP_RESULT: PREREG_V8F_KHYP executed exactly as written\n")
    L.append(f"**Outcome (the PREREG table applied verbatim): {o['outcome']}.** PRIMARY per seed {o['PRIMARY_per_seed']}; SECONDARY per seed {o['SECONDARY_per_seed']}; "
             f"HEADROOM <= -2F per seed {o['HEADROOM_le_minus_2F_per_seed']}.\n")
    L.append(f"Controls: {'ALL PASS' if READ['controls_all_pass'] else 'CONTROL 3 (deliberate regression) FAILED - read flagged'}.\n")
    L.append(f"**Committed next step, quoted from the PREREG:** *\"{o['committed_next_step_quoted']}\"*\n")
    L.append(f"Tiers: ORACLE = {TIER['ORACLE']}; selectors = {TIER['selectors']}. Estimator: {ESTIMATOR}; answers 'another draw of EPISODES' only, so both seeds are shown. "
             f"F = {F_PREREG} m (PREREG), exact |ADE6(s0)-ADE6(s1)| of B = {READ['F_exact']:.5f} m.\n")
    for s in SEEDS:
        R = RES[s]
        L.append(f"\n## Seed {s}\n")
        L.append("| read | value (m) | CI | bar |\n|---|---|---|---|")
        L.append(f"| HEADROOM ADE6 ORACLE - B | {R['HEADROOM']['point_unrounded']:+.4f} | {fmt(R['HEADROOM'])} | <= {-2 * F_PREREG:.4f} for the headroom outcomes |")
        L.append(f"| PRIMARY ADE6 SEL-LEARN - B | {R['PRIMARY']['point_unrounded']:+.4f} | {fmt(R['PRIMARY'])} | <= -{F_PREREG} and separated |")
        for k, nm in (("speed_mae", "speed MAE"), ("accel_mae", "accel MAE")):
            x = R["SECONDARY"][k]
            L.append(f"| SECONDARY {nm} SEL-LEARN - B | {x['point_unrounded']:+.4f} | {fmt(x)} | not separated-worse (worse = {x.get('worse')}) |")
        L.append("\n| arm (ADE6, m) | level [CI] | arm - B (paired) | ADE2 arm - B | speed MAE | accel MAE | CONS(a) any | CONS(b) any |\n|---|---|---|---|---|---|---|---|")
        for nm in ("B", "ORACLE", "ARGMAX", "MIX", "LEARN", "LEARN_PERM", "H_B", "H_LEARN"):
            if nm not in R["arms"]:
                continue
            ar = R["arms"][nm]
            pv = R["paired_vs_B"].get(nm)
            lv = ar["ade6_single"]
            L.append(f"| {nm} | {lv['mean']:.4f} [{lv['lo']:.4f}, {lv['hi']:.4f}] | {fmt(pv['ade6']) if pv else '-'} | {fmt(pv['ade2']) if pv else '-'} | "
                     f"{fmt(pv['fw_lon_speed_mae']) if pv else '-'} | {fmt(pv['fw_lon_accel_mae']) if pv else '-'} | "
                     f"{ar['evaluate']['consistency_a']['any']['rate_all']:.4f} | {ar['consistency_b']['any']['rate_all']:.4f} |")
        L.append("\nFour families, SEL-LEARN - B (paired; lower is better except the two tactical-agreement rows):\n")
        L.append("| family | metric | SEL-LEARN - B |\n|---|---|---|")
        for key, fam, nm in (("fw_lon_speed_mae", "LONGITUDINAL", "speed MAE"), ("fw_lon_accel_mae", "LONGITUDINAL", "accel MAE"), ("fw_lon_along_mae", "LONGITUDINAL", "along MAE"),
                             ("fw_lat_cross_mae", "LATERAL", "cross-track MAE"), ("fw_lat_heading_mae_deg", "LATERAL", "heading MAE (deg)"),
                             ("fw_lat_yaw_rate_mae_degps", "LATERAL", "yaw-rate MAE (deg/s)"), ("fw_lat_curvature_mae_1pm", "LATERAL", "curvature MAE (1/m)"),
                             ("fw_tac_lat_acc", "TACTICAL (plan-derived)", "lateral decision agreement"), ("fw_tac_lon_acc", "TACTICAL (plan-derived)", "longitudinal decision agreement")):
            L.append(f"| {fam} | {nm} | {fmt(R['paired_vs_B']['LEARN'].get(key), 4)} |")
        L.append("| STRATEGIC | - | PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE ITEM 25): O/T/S unchanged here, reported as paused, not as a pass |")
        sa = R["selector_accuracy"]
        L.append(f"\nSelector accuracy vs k*: ARGMAX {sa['ARGMAX_vs_kstar']:.4f}, LEARN {sa['LEARN_vs_kstar']:.4f} (in-sample train {sa['LEARN_in_sample_train']:.4f}), "
                 f"LEARN_PERM {sa['LEARN_PERM_vs_kstar']:.4f}, majority-class {sa['majority_train_class_acc_on_eval']:.4f}. k* counts (eval) {sa['kstar_eval_counts']}; "
                 f"LEARN picks {sa['pick_counts_LEARN']}; ARGMAX picks {sa['pick_counts_ARGMAX']}.")
        L.append(f"\nControls (seed {s}): {json.dumps(R['ctl'], default=lambda x: x.item() if hasattr(x, 'item') else str(x))[:3500]}")
    L.append("\nDistance keeping proper (headway / time-gap / min-TTC): " + json.dumps(READ["distance_keeping"], default=str))
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def cmd_read(a):
    torch.set_num_threads(max(1, min(a.threads, 6)))
    EV, BE = _eval_modules()
    cfg_kin = a.kin_mode
    grid = json.load(open(a.eval_grid))
    J = D.load_join(a.eval_join)
    X = D.eval_context(grid, a.eval_kin, J, kin_mode=cfg_kin)
    RES, PLANS, stops = {}, {}, {}
    for s in SEEDS:
        try:
            RES[s], PLANS[s] = score_seed(a, s, X, EV, BE, a.n_boot)
        except Stop as e:
            stops[s] = str(e)
    if stops:
        jdump({"status": "STOPPED", "stops": stops}, OUT / "READ.json")
        print("STOPPED:", stops)
        return 2
    decision = decide({s: RES[s]["seed_row"] for s in SEEDS})
    f_exact = abs(RES[0]["B_ade6_mean"] - RES[1]["B_ade6_mean"])
    READ = {"prereg": "PREREG_V8F_KHYP", "decision": decision, "controls_all_pass": bool(all(RES[s]["ctl"]["control3"]["pass"] for s in SEEDS)),
            "F_exact": f_exact, "evidence_class": "MEASURED (ours), artifacts under " + str(OUT), "tier": TIER, "estimator": ESTIMATOR,
            "per_seed": {f"s{s}": {"HEADROOM": RES[s]["HEADROOM"], "PRIMARY": RES[s]["PRIMARY"], "SECONDARY": RES[s]["SECONDARY"], "controls": RES[s]["ctl"],
                                   "selector_accuracy": RES[s]["selector_accuracy"], "B_ade6_mean": RES[s]["B_ade6_mean"]} for s in SEEDS},
            "distance_keeping": distance_keeping(a, X, grid, PLANS, a.n_boot)}
    jdump(READ, OUT / "READ.json")
    jdump({f"s{s}": {k: v for k, v in RES[s].items()} for s in SEEDS}, OUT / "khyp_numbers.json")
    write_md(OUT / "KHYP_RESULT.md", READ, RES)
    print(json.dumps({"outcome": decision["outcome"], "PRIMARY": decision["PRIMARY_per_seed"], "SECONDARY": decision["SECONDARY_per_seed"],
                      "HEADROOM": decision["HEADROOM_le_minus_2F_per_seed"], "controls_all_pass": READ["controls_all_pass"]}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("hyps")
    p.add_argument("--seed", type=int, required=True, choices=SEEDS)
    p.add_argument("--split", required=True, choices=("eval", "train"))
    p.add_argument("--batch", type=int, default=16)         # memprobe.py: batch 16 peaks at 1.11 GB commit, batch 32 at 1.30 GB
    p.add_argument("--shard", type=int, default=640)
    p.add_argument("--lru", type=int, default=2)
    p.add_argument("--ckpt", default=None)
    for q in (p, sub.add_parser("read")):
        q.add_argument("--eval-grid", default=str(OUT / "raw" / "eval_grid_windows.json"))
        q.add_argument("--eval-kin", default=str(OUT / "raw" / "eval141_kin.npz"))
        q.add_argument("--eval-join", default=str(OUT / "raw" / "join_eval141.npz"))
        q.add_argument("--tracks", default=f"{SCR}/sfresh/inputs/native_tracks.pt")
        q.add_argument("--eval-cache", default=f"{SCR}/sfresh/cache")
        q.add_argument("--train-cache", default=f"{SCR}/hblend/cache")
        q.add_argument("--train-join", default=f"{SCR}/hblend/box2/join_train.npz")
        q.add_argument("--fit-clips", default=f"{SCR}/hblend/fit_clips.json")
        q.add_argument("--threads", type=int, default=4)
        q.add_argument("--kin-mode", default="yaw", choices=("yaw", "steer"))
    r = sub.choices["read"]
    r.add_argument("--n-boot", type=int, default=B_BOOT)
    r.add_argument("--no-banked", action="store_true", help="smoke only: skip every control that needs a banked file")
    a = ap.parse_args(argv)
    if a.mode == "hyps":
        if a.ckpt is None:                                             # the banked checkpoint: its identity is asserted, a custom --ckpt (smoke) is not
            a.ckpt = f"{SCR}/sfresh/inputs/M2b_s{a.seed}/ckpt_last.pt"
            if sha256_file(a.ckpt) != SHA[a.seed]:
                raise SystemExit("REFUSED: checkpoint sha256 differs from the banked M2b checkpoint")
        return cmd_hyps(a)
    return cmd_read(a)


if __name__ == "__main__":
    sys.exit(main())
