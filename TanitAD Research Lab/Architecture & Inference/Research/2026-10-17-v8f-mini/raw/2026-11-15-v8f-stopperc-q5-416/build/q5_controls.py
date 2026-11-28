"""q5_controls.py -- AMENDMENT 7 (Q5) fit-side controls: C5 (label-shuffle null) and C3 (a)-(e).  CPU only.

Every function takes the S0-P modules (`F` = s0p_fit, `L` = s0p_lib, imported by `s0p_modules()` from the reused-unchanged s0p dir) so tests can run on synthetic dirs.
`compute_fit_side(cfg)` returns the `controls` keys q5_read.evaluate_controls reads; a key that cannot be computed is None (a MISS: the read is refused).

C5  label shuffle over 20 seeds on V40@416+E linear (the method of s0p/posthoc_shuffle_seed_spread.py: the REAL fit's selected hyperparameters, window-level permutation of the
    TRAIN labels, eval AUROC per draw).  PASS iff the real eval AUROC is above the null MAXIMUM.  The single-draw "CI covers 0.5" is printed beside (the VERDICT_S0P lesson).
C3a GT label appended to E (both sides), linear with E's selected hyperparameters: AUROC >= 0.999.
C3b limit / limit_ok are the causal bucket: max |diff| against the second route (the extractors' meta `c2_second_route_max_abs`, train and eval) == 0.0.
C3c mutation: E's limit block replaced by the logged sidecar `limit`, never fitted: E must change on >= 1 window (a check that cannot see the substitution is BLIND).
C3d clip ids shared between train and eval == 0.
C3e per sample, E / y / clip / ws / eid bitwise equal between the 416 and 256p dirs.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S0P = os.environ.get("Q5_S0P", "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopperc/s0p")
C5_SEEDS = tuple(range(100, 120))                       # 20 draws, the posthoc script's seed range
BIT_KEYS = ("E", "y", "clip", "ws", "eid")
LIMIT_SLICE = slice(13, 17)


def s0p_modules():
    """-> (s0p_fit, s0p_lib), read-only imports from the frozen dagland dir (never edited, never written)."""
    if S0P not in sys.path:
        sys.path.insert(0, S0P)
    import s0p_fit as F
    import s0p_lib as L
    return F, L


# ------------------------------------------------------------------------------------------------------------ helpers
def _best(work, arm, model="linear"):
    return json.load(open(os.path.join(work, f"fit_{arm}_{model}.json"), encoding="utf-8"))


def _features(F, work, arm, cfg, tr, ev):
    g = F.GRID_OF.get(arm)
    Ptr = np.load(os.path.join(work, f"proj_train_{g}.npy"), mmap_mode="r") if g else None
    Pev = np.load(os.path.join(work, f"proj_eval_{g}.npy"), mmap_mode="r") if g else None
    return (F.build_features(arm, Ptr, np.asarray(tr["E"]), cfg["dim"]), F.build_features(arm, Pev, np.asarray(ev["E"]), cfg["dim"]))


def bitwise_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ------------------------------------------------------------------------------------------------------------ C5
def c5_shuffle_null(F, L, work, tr, ev, seeds=C5_SEEDS, arm="V40+E", shuffle=True, ci_mod=None, n_boot=2000):
    """`shuffle=False` is the deliberate-regression arm (labels left as they are): the null then equals the real fit and the control must go RED."""
    rec = _best(work, arm)
    cfg = rec["best"]
    Xtr, Xev = _features(F, work, arm, cfg, tr, ev)
    y = np.asarray(tr["y"], np.int64)
    yev = np.asarray(ev["y"], bool)
    nulls, first = [], None
    for s in seeds:
        lab = np.random.default_rng(int(s)).permutation(y) if shuffle else y
        sc = F._score(F._make_model("linear", cfg).fit(Xtr, lab), Xev)
        nulls.append(L.auroc(sc, yev))
        if first is None:
            first = sc
    real = float(rec["eval_auroc_point"])
    out = {"arm": arm, "model": "linear", "cfg": cfg, "seeds": [int(s) for s in seeds], "null_aurocs": [float(v) for v in nulls], "null_max": float(max(nulls)), "null_min": float(min(nulls)),
           "null_mean": float(np.mean(nulls)), "real_auroc": real, "pass": bool(real > max(nulls)), "shuffled": bool(shuffle)}
    if ci_mod is not None:
        B = ci_mod.stat_cluster_bootstrap(list(range(len(yev))), np.asarray(ev["eid"], np.int64), lambda sel: L.auroc(first[np.asarray(sel)], yev[np.asarray(sel)]), n_boot=n_boot, seed=0)
        out["single_draw"] = {"seed": int(seeds[0]), "auroc": B["point"], "lo": B["lo"], "hi": B["hi"], "ci_covers_0.5": bool(B["lo"] <= 0.5 <= B["hi"]),
                              "note": "printed beside: one draw's episode-cluster CI can cover 0.5 while the 20-draw null maximum still sits below the real fit"}
    return out


# ------------------------------------------------------------------------------------------------------------ C3
def c3a_leak_auroc(F, L, work, tr, ev, append_label=True):
    """GT label appended to E on both sides; `append_label=False` is the regression arm (a column of zeros): the control must go RED."""
    cfg = _best(work, "E")["best"]
    Xtr, Xev = _features(F, work, "E", cfg, tr, ev)
    y, yev = np.asarray(tr["y"], np.int64), np.asarray(ev["y"], bool)
    ctr = y[:, None].astype(np.float32) if append_label else np.zeros((len(y), 1), np.float32)
    cev = yev[:, None].astype(np.float32) if append_label else np.zeros((len(yev), 1), np.float32)
    m = F._make_model("linear", cfg).fit(np.concatenate([Xtr, ctr], 1), y)
    return float(L.auroc(F._score(m, np.concatenate([Xev, cev], 1)), yev))


def c3b_limit_max_abs_diff(tr_meta, ev_meta):
    """Max over train and eval of the extractors' second-route differences for limit and limit_ok.  None (a MISS) if either meta lacks them."""
    vals = []
    for m in (tr_meta, ev_meta):
        c2 = (m or {}).get("c2_second_route_max_abs") or {}
        for k in ("limit", "limit_ok"):
            if c2.get(k) is None:
                return None
            vals.append(float(c2[k]))
    return max(vals)


def second_route_diff(E_a, E_b):
    """Direct form of C3b when a second route's E is at hand: max |diff| over the limit block and limit_ok."""
    a, b = np.asarray(E_a, np.float64), np.asarray(E_b, np.float64)
    return float(np.abs(a[:, 13:18] - b[:, 13:18]).max())


def c3c_windows_changed(E, sidecar_limit, apply=True):
    """The mutation: E's limit block replaced by the logged sidecar `limit` ([N,4]).  The mutated E is NEVER fitted.  Returns the number of windows whose E changes.
    `apply=False` skips the substitution (the regression arm): 0 windows, so the check reads BLIND and the control fails."""
    E = np.asarray(E, np.float32)
    E2 = E.copy()
    if apply:
        sc = np.asarray(sidecar_limit, np.float32)
        assert sc.shape == (len(E), 4), (sc.shape, E.shape)
        E2[:, LIMIT_SLICE] = sc
    return int((E2 != E).any(axis=1).sum())


def c3d_shared_clips(tr_clip, ev_clip):
    return int(len(set(map(str, np.asarray(tr_clip).tolist())) & set(map(str, np.asarray(ev_clip).tolist()))))


def c3e_bitwise(A, B):
    """A, B: two loaded dirs (F.load_any).  True iff E, y, clip, ws, eid are bitwise equal (dtype, shape and bytes)."""
    return all(bitwise_equal(A[k], B[k]) for k in BIT_KEYS)


# ------------------------------------------------------------------------------------------------------------ the driver
def compute_fit_side(cfg, F=None, L=None, ci_mod=None, log=print, c5_seeds=C5_SEEDS):
    """cfg: {"eval416": path, "eval256p": path, "sidecar_limit_eval": optional .npy [N,4], "samples": {S: {"train416","train256p","work416","work256p"}}}.
    -> (controls dict for evaluate_controls, detail dict for the printout).  Per-sample controls are reduced to their WORST case so one value feeds the control."""
    if F is None:
        F, L = s0p_modules()
    ev416, ev256 = F.load_any(cfg["eval416"]), F.load_any(cfg["eval256p"])
    ctl, det = {}, {"per_sample": {}}
    leak, shared, bit, bmax, c5m = [], [], [], [], []
    for s, d in cfg["samples"].items():
        tr416, tr256 = F.load_any(d["train416"]), F.load_any(d["train256p"])
        rec = {}
        rec["c3a"] = c3a_leak_auroc(F, L, d["work416"], tr416, ev416)
        rec["c3b"] = c3b_limit_max_abs_diff(tr416["meta"], ev416["meta"])
        rec["c3d"] = c3d_shared_clips(tr416["clip"], ev416["clip"])
        rec["c3e"] = bool(c3e_bitwise(tr416, tr256) and c3e_bitwise(ev416, ev256))
        c5 = c5_shuffle_null(F, L, d["work416"], tr416, ev416, seeds=c5_seeds, ci_mod=ci_mod)
        rec["c5"] = c5
        det["per_sample"][s] = rec
        leak.append(rec["c3a"]); shared.append(rec["c3d"]); bit.append(rec["c3e"]); bmax.append(rec["c3b"]); c5m.append(c5)
        log(f"  {s}: C3a leak AUROC {rec['c3a']:.4f}  C3b {rec['c3b']}  C3d {rec['c3d']}  C3e {rec['c3e']}  C5 real {c5['real_auroc']:.4f} vs null max {c5['null_max']:.4f} (pass {c5['pass']}); "
            f"single-draw CI covers 0.5: {c5.get('single_draw', {}).get('ci_covers_0.5')}")
    ctl["c3a_leak_auroc"] = min(leak)
    ctl["c3b_limit_max_abs_diff"] = None if any(v is None for v in bmax) else max(bmax)
    ctl["c3d_shared_clips"] = int(sum(shared))
    ctl["c3e_E_y_clip_ws_eid_bitwise"] = bool(all(bit))
    worst = min(c5m, key=lambda c: c["real_auroc"] - c["null_max"])
    ctl["c5_real_auroc"], ctl["c5_null_max"] = worst["real_auroc"], worst["null_max"]
    sp = cfg.get("sidecar_limit_eval")
    if sp and os.path.exists(sp):
        ctl["c3c_E_windows_changed"] = c3c_windows_changed(ev416["E"], np.load(sp))
    else:
        ctl["c3c_E_windows_changed"] = None
        det["c3c"] = (f"BLOCKED: needs the logged sidecar limit [N,4] aligned to the eval windows at {sp!r}; produce it with s0p_extract_eval.py's loop with build_item(..., limit='logged') "
                      "(the extractor asserts limit == 'causal' for E, so this is a second, read-only pass that saves batch['limit'] only)")
    return ctl, det
