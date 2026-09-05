"""CPU analysis of a paired kept/withheld dump (egodrop_dump.py).

Everything paired is `withheld - kept` (or as named), paired episode-cluster
bootstrap (`taniteval.ci.paired_episode_cluster_bootstrap`, n_boot 2000, seed 0);
levels carry `taniteval.ci.episode_cluster_bootstrap`. `overlapping_holdout_se`
is used NOWHERE. The four families come from the SAME instruments the banked
refcv3 record used (t1_eval.analyze + refav1_arm components + the refcv3 arm's
common-grid lead join), on the 2 s grid (dt 0.5, K 4) and the 6 s grid (dt 1.0,
K 6, true-6 s windows only). Smoothness on the model's NATIVE 8-slot grid is this
file's own instrument and is stated as such (slot-resolution, not 10 Hz).
"""
from __future__ import annotations
import argparse
import glob
import importlib.util
import json
import math
import os
import shutil
import sys

import numpy as np

REPO = r"C:\Users\Admin\refcv4b_repo"
for _p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval"),
           os.path.join(REPO, "stack", "scripts"), REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


arm = _load_by_path("refcv3_arm", os.path.join(REPO, "taniteval", "tools", "refcv3_arm.py"))
ra, t1 = arm.ra, arm.t1
import torch                                   # noqa: E402
from taniteval import ci as CI                 # noqa: E402
from taniteval import four_families as ff      # noqa: E402

T8 = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
SLOTS2S = [0, 1, 2, 3]
SLOTS6S = [1, 3, 4, 5, 6, 7]
ARMS = ["os_k", "os_w", "orc_k", "orc_w", "ha", "ha0"]
TIERS = {"os_k": "T1", "os_w": "T1", "orc_k": "T0", "orc_w": "T0", "ha": "T1", "ha0": "T1"}
BANDS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 99))
V_FLOOR, KAPPA_CAP, REF_SPEED, DT, H = 4.0, 0.12, 10.0, 0.1, 60
SLOT_TICKS = [4, 9, 14, 19, 29, 39, 49, 59]
N_BOOT, SEED = 2000, 0


def p(*a):
    print(*a, flush=True)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def PB(a, b, eid):
    """a - b, paired episode-cluster bootstrap. Non-finite pairs dropped (counted)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() == 0:
        return {"status": "UNAVAILABLE", "reason": "no finite pair", "n_windows": 0}
    e = [x for x, k in zip(eid, ok) if k]
    r = CI.paired_episode_cluster_bootstrap(a[ok], b[ok], e, n_boot=N_BOOT, seed=SEED)
    r["n_dropped_nonfinite"] = int((~ok).sum())
    return r


def EB(a, eid):
    a = np.asarray(a, np.float64)
    ok = np.isfinite(a)
    if ok.sum() == 0:
        return {"status": "UNAVAILABLE", "reason": "no finite value", "n_windows": 0}
    e = [x for x, k in zip(eid, ok) if k]
    r = CI.episode_cluster_bootstrap(a[ok], e, n_boot=N_BOOT, seed=SEED)
    r["n_dropped_nonfinite"] = int((~ok).sum())
    return r


def fmt(r, key="delta"):
    if not isinstance(r, dict) or key not in r:
        return str(r)[:60]
    s = f"{r[key]:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]" if key == "delta" else \
        f"{r[key]:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]"
    if "separated" in r:
        s += " SEP" if r["separated"] else " ns"
    return s


# --------------------------------------------------------------------------- #
# loading                                                                      #
# --------------------------------------------------------------------------- #
def load_dump(dump_dir):
    files = sorted(glob.glob(os.path.join(dump_dir, "ep*.npz")))
    dfiles = sorted(glob.glob(os.path.join(dump_dir, "decisions", "ep*.npz")))
    assert files and len(files) == len(dfiles), (len(files), len(dfiles))
    main, dec, eid = {}, {}, []
    for f, df in zip(files, dfiles):
        with np.load(f) as d:
            n = int(d["ws"].shape[0])
            for k in d.files:
                if k in ("eid", "clip_index"):
                    continue
                main.setdefault(k, []).append(np.asarray(d[k]))
            eid += [os.path.splitext(os.path.basename(f))[0]] * n
        with np.load(df) as d:
            assert int(d["ws"].shape[0]) == n
            for k in d.files:
                dec.setdefault(k, []).append(np.asarray(d[k]))
    main = {k: np.concatenate(v) for k, v in main.items()}
    dec = {k: np.concatenate(v) for k, v in dec.items()}
    with open(os.path.join(dump_dir, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    return files, main, dec, eid, man


# --------------------------------------------------------------------------- #
# instruments                                                                  #
# --------------------------------------------------------------------------- #
def masked_ade(P, G, sv):
    """mean over VALID slots of |P-G| per window. P,G [n,K,2]; sv [n,K]."""
    e = np.linalg.norm(P - G, axis=-1)
    return (e * sv).sum(1) / np.maximum(sv.sum(1), 1.0)


def local_frame_offset(P, B):
    """P - B decomposed in the local tangent frame of B (the bank path). [n,K] each."""
    n, K, _ = B.shape
    pts = np.concatenate([np.zeros((n, 1, 2)), B], axis=1)
    d = pts[:, 1:] - pts[:, :-1]
    nrm = np.linalg.norm(d, axis=-1, keepdims=True)
    t = np.where(nrm > 1e-6, d / np.maximum(nrm, 1e-12), np.array([1.0, 0.0])[None, None])
    off = P - B
    along = (off * t).sum(-1)
    lat = t[..., 0] * off[..., 1] - t[..., 1] * off[..., 0]
    return off, along, lat


def smooth_native(P, T, min_ds_mps=ff.MIN_DS_MPS):
    """Slot-resolution smoothness of paths P [n,K,2] given at instants T [K] from the origin.

    Segment k spans (T[k-1], T[k]] with T[-1] = 0. Junction j sits at T[j] between
    segments j and j+1. Rates between junctions are divided by T[j+1]-T[j].
      jerk      = d(along-path accel)/dt      [m/s^3]
      krate     = d(curvature)/dt             [1/(m s)]
      ljerk     = d(v^2 kappa)/dt             [m/s^3]  (lateral jerk)
    Heading/curvature are masked where a segment's mean speed < min_ds_mps
    (four_families' MIN_DS_MPS rule); longitudinal quantities are never masked.
    """
    n, K, _ = P.shape
    T = np.asarray(T, np.float64)
    pts = np.concatenate([np.zeros((n, 1, 2)), P.astype(np.float64)], axis=1)
    d = pts[:, 1:] - pts[:, :-1]
    dtk = np.diff(np.concatenate([[0.0], T]))
    s = np.linalg.norm(d, axis=-1)
    v = s / dtk[None]
    th = np.arctan2(d[..., 1], d[..., 0])
    valid = v > min_ds_mps
    dt_mid = 0.5 * (dtk[:-1] + dtk[1:])
    a = (v[:, 1:] - v[:, :-1]) / dt_mid[None]
    dth = wrap(th[:, 1:] - th[:, :-1])
    yaw = dth / dt_mid[None]
    s_mid = 0.5 * (s[:, 1:] + s[:, :-1])
    kap = dth / np.maximum(s_mid, 1e-8)
    pv = valid[:, 1:] & valid[:, :-1]
    dT = dtk[1:-1]
    jerk = (a[:, 1:] - a[:, :-1]) / dT[None]
    krate = (kap[:, 1:] - kap[:, :-1]) / dT[None]
    v_j = 0.5 * (v[:, 1:] + v[:, :-1])
    alat = v_j ** 2 * kap
    ljerk = (alat[:, 1:] - alat[:, :-1]) / dT[None]
    pvv = pv[:, 1:] & pv[:, :-1]

    def mmean(x, m):
        m = m.astype(np.float64)
        c = m.sum(1)
        return np.where(c > 0, (np.abs(x) * m).sum(1) / np.maximum(c, 1), np.nan)

    def mmax(x, m):
        xx = np.where(m, np.abs(x), -np.inf).max(1)
        return np.where(np.isfinite(xx), xx, np.nan)
    ones_j = np.ones_like(jerk, dtype=bool)
    ones_a = np.ones_like(a, dtype=bool)
    return {
        "jerk_mean_abs_mps3": mmean(jerk, ones_j), "jerk_max_abs_mps3": mmax(jerk, ones_j),
        "jerk_rms_mps3": np.sqrt((jerk ** 2).mean(1)),
        "accel_mean_abs_mps2": mmean(a, ones_a),
        "krate_mean_abs_1pms": mmean(krate, pvv), "krate_max_abs_1pms": mmax(krate, pvv),
        "ljerk_mean_abs_mps3": mmean(ljerk, pvv),
        "kappa_mean_abs_1pm": mmean(kap, pv),
        "yaw_rate_mean_abs_degps": np.degrees(mmean(yaw, pv)),
        "n_junctions_valid": pv.sum(1),
    }


SMOOTH_KEYS = ("jerk_mean_abs_mps3", "jerk_max_abs_mps3", "jerk_rms_mps3", "accel_mean_abs_mps2",
               "krate_mean_abs_1pms", "krate_max_abs_1pms", "ljerk_mean_abs_mps3",
               "kappa_mean_abs_1pm", "yaw_rate_mean_abs_degps")


def roll_bank_np(ctrl, v0v):
    """The decoder's own alat roll (refc.py:1351-1356 derivation, kinematic.rollout_unicycle
    semantics: position advances on the speed at the START of the step, v updates last,
    clamped at 0). ctrl [M,2] = (a_lon, a_lat); v0v [B]. -> [B, M, 8, 2] on the model slots."""
    B, M = len(v0v), len(ctrl)
    kap = np.clip(ctrl[None, :, 1] / np.maximum(v0v, V_FLOOR)[:, None] ** 2, -KAPPA_CAP, KAPPA_CAP)
    acc = np.repeat(ctrl[None, :, 0], B, axis=0)
    x = np.zeros((B, M)); y = np.zeros((B, M)); yaw = np.zeros((B, M))
    v = np.repeat(v0v[:, None], M, axis=1).astype(np.float64)
    wp = np.zeros((B, M, 8, 2))
    for k in range(H):
        x = x + v * np.cos(yaw) * DT
        y = y + v * np.sin(yaw) * DT
        yaw = yaw + v * kap * DT
        v = np.maximum(v + acc * DT, 0.0)
        if k in SLOT_TICKS:
            j = SLOT_TICKS.index(k)
            wp[:, :, j, 0] = x
            wp[:, :, j, 1] = y
    return wp


def oiv_at_speed(ctrl, speeds, G, sv, chunk=256):
    """min over the bank rolled at `speeds` of the masked ADE to G. -> [n]"""
    out = np.zeros(len(speeds))
    for s in range(0, len(speeds), chunk):
        e = min(s + chunk, len(speeds))
        wp = roll_bank_np(ctrl, speeds[s:e])                      # [b,M,8,2]
        dd = np.linalg.norm(wp - G[s:e, None], axis=-1)            # [b,M,8]
        ade = (dd * sv[s:e, None]).sum(-1) / np.maximum(sv[s:e].sum(-1), 1)[:, None]
        out[s:e] = ade.min(1)
    return out


def selection_profile(sel, astar, cls_top1, n_anchors=117):
    counts = np.bincount(sel, minlength=n_anchors)
    modal = int(counts.argmax())
    pr = counts[counts > 0] / counts.sum()
    ent = float(-(pr * np.log(pr)).sum())
    return {"n_windows": int(sel.size), "n_anchors": n_anchors,
            "n_distinct_selected": int(np.count_nonzero(counts)),
            "modal_anchor": modal, "modal_frac": round(float(counts[modal] / sel.size), 4),
            "straight_ahead_idx67_frac": round(float((sel == 67).mean()), 4),
            "entropy_nats": round(ent, 4), "max_entropy_nats": round(float(np.log(n_anchors)), 4),
            "entropy_ratio": round(ent / float(np.log(n_anchors)), 4),
            "agrees_with_own_oracle_frac": round(float((sel == astar).mean()), 4),
            "classifier_top1_acc_vs_own_oracle": round(float((cls_top1 == astar).mean()), 4),
            "degenerate_(modal_frac>=0.9)": bool(counts[modal] / sel.size >= arm.SELECTION_DEGENERATE_FRAC)}


def by_band(v0, fn):
    rows = []
    for lo, hi in BANDS:
        m = (v0 >= lo) & (v0 < hi)
        if not m.any():
            continue
        rows.append({"v0_band_ms": [lo, hi], "n": int(m.sum()), **fn(m)})
    return rows


# --------------------------------------------------------------------------- #
# the four families on a uniform sub-grid, through the banked instruments      #
# --------------------------------------------------------------------------- #
def families_on_grid(files8, man8, out_root, tag, slots, dt, k, keep_mask_fn, lead_block, raw_off):
    """Write an index-selected refcv3_arm-format dump and run t1.analyze + paired families +
    distance keeping + trivial profile on it. keep_mask_fn(sv [n,8]) -> bool [n] window filter."""
    ddir = os.path.join(out_root, f"dump_{tag}")
    if os.path.isdir(ddir):
        shutil.rmtree(ddir)
    os.makedirs(ddir)
    files, episodes, n_kept, n_all = [], [], 0, 0
    for fi, f in enumerate(files8):
        df = os.path.join(os.path.dirname(f), "decisions", os.path.basename(f))
        with np.load(f) as d, np.load(df) as x:
            sv = x["sv"]
            m = keep_mask_fn(sv)
            n_all += int(m.size)
            if not m.any():
                continue
            k_ = len(episodes)
            arrs = {a: d[a][m][:, slots] for a in ARMS}
            np.savez_compressed(os.path.join(ddir, f"ep{k_:03d}.npz"), g=d["g"][m][:, slots],
                                v0=d["v0"][m], ws=d["ws"][m], eid=np.array([k_]),
                                clip_index=d["clip_index"], **arrs)
            files.append(os.path.join(ddir, f"ep{k_:03d}.npz"))
            e = dict(man8["episodes"][fi]); e["file_index"] = k_; e["n_windows"] = int(m.sum())
            episodes.append(e)
            n_kept += int(m.sum())
    man = {"episodes": episodes, "grid": {"dt_s": dt, "k": k, "slots_of_8": slots},
           "corpus": man8["corpus"], "model": man8["model"]}
    with open(os.path.join(ddir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, default=str)
    p(f"[families:{tag}] {n_kept}/{n_all} windows over {len(files)} episodes, dt {dt} K {k}")
    pairs = [("os_k", "os_w", "paired_osw_minus_osk"),
             ("orc_k", "orc_w", "paired_orcw_minus_orck"),
             ("ha0", "os_k", "paired_osk_minus_ha0"), ("ha0", "os_w", "paired_osw_minus_ha0"),
             ("ha", "os_k", "paired_osk_minus_ha"), ("ha", "os_w", "paired_osw_minus_ha"),
             ("os_k", "orc_k", "paired_orck_minus_osk"), ("os_w", "orc_w", "paired_orcw_minus_osw"),
             ("ha0", "ha", "paired_ha_minus_ha0")]
    rec = t1.analyze(files, tiers=TIERS, n_boot=N_BOOT, seed=SEED, dt=dt, paired=pairs)
    G_all, P_all, eid_w = [], {a: [] for a in ARMS}, []
    for f in files:
        with np.load(f) as d:
            G = d["g"][..., :2].astype(np.float64)
            G_all.append(G)
            eid_w += [os.path.splitext(os.path.basename(f))[0]] * G.shape[0]
            for a in ARMS:
                P_all[a].append(d[a][..., :2].astype(np.float64))
    G_all = np.concatenate(G_all)
    P_cat = {a: np.concatenate(P_all[a]) for a in ARMS}
    comps = {a: ra._components(P_cat[a], G_all, dt) for a in ARMS}
    fam_paired = {nm: ra._paired_families(comps, x, y, eid_w, TIERS, N_BOOT, SEED)
                  for x, y, nm in pairs}
    triv = ra.trivial_profile(files, ARMS, dt=dt)
    dk = None
    if lead_block:
        try:
            dk = arm._distance_keeping(rec, files, man, lead_block, ARMS, P_cat, G_all, pairs,
                                       eid_w, N_BOOT, SEED, TIERS, dt, k, raw_off)
        except Exception as ex:                                   # noqa: BLE001
            dk = {"status": "FAILED", "reason": f"{type(ex).__name__}: {ex}"}
    return {"n_windows": int(G_all.shape[0]), "n_windows_all": n_all, "n_episodes": len(files),
            "grid": {"dt_s": dt, "k": k, "slots_of_8": slots},
            "arms": {a: rec["arms"][a] for a in ARMS}, "paired_decision_grade": rec.get("paired_decision_grade"),
            "families_paired": fam_paired, "trivial_profile": triv, "distance_keeping": dk,
            "levels_ade": {a: EB(comps[a]["ade_m"], eid_w) for a in ARMS}}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="<out>/dump8")
    ap.add_argument("--out", required=True, help="output JSON")
    ap.add_argument("--lead-block", default=r"C:\Users\Admin\tanitad-wt\_lead_b1\b1_eval_lead_block.npz")
    ap.add_argument("--anchors-file", default=r"C:\Users\Admin\refcv4b_egodrop\pull\anchors.pt")
    ap.add_argument("--label", default="")
    ap.add_argument("--skip-families", action="store_true")
    a = ap.parse_args()
    files, M, D, eid, man = load_dump(a.dump)
    N = int(M["g"].shape[0])
    v0 = M["v0"].astype(np.float64)
    G = M["g"].astype(np.float64)
    sv = D["sv"].astype(np.float64)
    true6 = sv.min(1) > 0.5
    p(f"[load] {N} windows / {len(files)} episodes; true-6s {int(true6.sum())}; step {man['model']['step']}")
    R = {"label": a.label, "dump": a.dump, "n_windows": N, "n_episodes": len(files),
         "n_true6s": int(true6.sum()), "model_step": man["model"]["step"], "ckpt_md5": man.get("ckpt_md5"),
         "model_checks": man.get("model_checks"), "bit_control": man.get("bit_control"),
         "stamp": man.get("stamp"), "estimator": "paired/episode-cluster bootstrap (taniteval.ci), "
                                                f"n_boot {N_BOOT}, seed {SEED}; overlapping_holdout_se NOWHERE"}

    # ---- A. controls that must read known values --------------------------- #
    ctrl = torch.load(a.anchors_file, map_location="cpu", weights_only=False)["controls"].double().numpy()
    ha0_expect = np.stack([v0[:, None] * T8[None], np.zeros((N, 8))], -1)
    ade2 = {x: np.linalg.norm(M[x][:, SLOTS2S] - G[:, SLOTS2S], axis=-1).mean(1) for x in ARMS}
    zero_path = (np.linalg.norm(G, axis=-1) * sv).sum(1) / np.maximum(sv.sum(1), 1)
    oiv_k = masked_ade(D["bank_astar_k"].astype(np.float64), G, sv)
    oiv_w = masked_ade(D["bank_astar_w"].astype(np.float64), G, sv)
    oiv_np_k = oiv_at_speed(ctrl, v0, G, sv)
    oiv_np_w = oiv_at_speed(ctrl, np.full(N, REF_SPEED), G, sv)
    sm_ha0 = smooth_native(M["ha0"], T8)
    R["controls"] = {
        "ha0_equals_v0_t_straight_max_abs_m": float(np.abs(M["ha0"] - ha0_expect).max()),
        "ade_2s_ha_m": float(ade2["ha"].mean()), "ade_2s_ha0_m": float(ade2["ha0"].mean()),
        "_banked_refcv3_40284_same_windows": {"ha": 0.2996, "ha0": 0.6723,
                                              "src": "GOALS_AND_CLAIMS H-ECHO-1 / RESULT-refcv3-40284-openloop.md"},
        "trainer_astar_masked_ade_kept_m": float(oiv_k.mean()),
        "trainer_astar_masked_ade_withheld_m": float(oiv_w.mean()),
        "_trainer_astar_is": ("masked ADE of bank[a_star] with a_star = argmin of the masked SUM OF SQUARES "
                              "(refc_v3_train.py:546-549, the anchor the trajectory loss supervises) — NOT the "
                              "min-ADE oracle-in-vocabulary; it is >= the OIV by construction"),
        "oiv_8slot_numpy_reroll_at_v0_m": float(oiv_np_k.mean()),
        "oiv_8slot_numpy_reroll_at_10ms_m": float(oiv_np_w.mean()),
        "_oiv_is": "min over the 117 anchors of the masked MEAN L2 (PART6's criterion), bank re-rolled here in numpy",
        "_banked_PART6_same_windows": {"measured_v0": 1.1112, "withheld_10ms": 7.5941,
                                       "delta": "+6.4829 [+4.9994, +8.0756]"},
        "oiv_withheld_minus_kept_paired": PB(oiv_np_w, oiv_np_k, eid),
        "trainer_astar_withheld_minus_kept_paired": PB(oiv_w, oiv_k, eid),
        "constant_only_zero_path_ade_8slot_m": float(zero_path.mean()),
        "reach_survivors_withheld_all_117": bool((D["reach_n_w"] == 117).all()),
        "reach_survivors_kept_mean": float(D["reach_n_k"].mean()),
        "reach_survivors_kept_min": int(D["reach_n_k"].min()),
        "ha0_jerk_max_abs_mps3": float(np.nanmax(sm_ha0["jerk_max_abs_mps3"])),
        "ha0_krate_max_abs_1pms": float(np.nanmax(sm_ha0["krate_max_abs_1pms"])),
        "straight_ahead_control_row": int(np.flatnonzero(np.abs(ctrl).sum(1) == 0)[0]),
        "n": N, "d_anchors": int(ctrl.shape[0]),
    }
    # identity control: where the model's selection IS its own oracle, os == orc bit-for-bit
    for tag in ("k", "w"):
        same = D[f"sel_idx_{tag}"] == D[f"a_star_{tag}"]
        R["controls"][f"os_equals_orc_where_sel_is_oracle_{tag}"] = {
            "n": int(same.sum()),
            "max_abs_m": float(np.abs(M[f"os_{tag}"][same] - M[f"orc_{tag}"][same]).max()) if same.any() else None}
    R["controls"]["_verdicts"] = {
        "ha0_geometry_ok": R["controls"]["ha0_equals_v0_t_straight_max_abs_m"] < 1e-3,
        "same_surface_as_banked_refcv3": (abs(R["controls"]["ade_2s_ha_m"] - 0.2996) < 5e-4
                                          and abs(R["controls"]["ade_2s_ha0_m"] - 0.6723) < 5e-4),
        "numpy_reroll_reproduces_PART6": (abs(R["controls"]["oiv_8slot_numpy_reroll_at_v0_m"] - 1.1112) < 2e-3
                                          and abs(R["controls"]["oiv_8slot_numpy_reroll_at_10ms_m"] - 7.5941) < 2e-3),
        "trainer_astar_ge_oiv": bool((oiv_k >= oiv_np_k - 1e-6).all() and (oiv_w >= oiv_np_w - 1e-6).all()),
        "ha0_smoothness_zero": (R["controls"]["ha0_jerk_max_abs_mps3"] < 1e-3 and R["controls"]["ha0_krate_max_abs_1pms"] < 1e-6),
        "withheld_rows_keep_full_fan": R["controls"]["reach_survivors_withheld_all_117"],
        "os_equals_orc_where_sel_is_oracle": all(
            (R["controls"][f"os_equals_orc_where_sel_is_oracle_{t}"]["max_abs_m"] or 0.0) == 0.0 for t in ("k", "w")),
    }
    p("[controls]", json.dumps(R["controls"]["_verdicts"]))
    p(f"  ha 2s {R['controls']['ade_2s_ha_m']:.4f} (banked 0.2996)  ha0 2s {R['controls']['ade_2s_ha0_m']:.4f} (0.6723)")
    p(f"  OIV numpy re-roll at v0 {oiv_np_k.mean():.4f} (PART6 1.1112)  at 10 m/s {oiv_np_w.mean():.4f} (7.5941); "
      f"trainer-a* masked ADE kept {oiv_k.mean():.4f} / withheld {oiv_w.mean():.4f}")

    # ---- B. offset magnitude ----------------------------------------------- #
    off = {}
    for tag in ("k", "w"):
        P = M[f"os_{tag}"].astype(np.float64)
        Bs = D[f"bank_sel_{tag}"].astype(np.float64)
        Ba = D[f"bank_astar_{tag}"].astype(np.float64)
        O = M[f"orc_{tag}"].astype(np.float64)
        o_sel, al_sel, la_sel = local_frame_offset(P, Bs)
        o_ast, al_ast, la_ast = local_frame_offset(O, Ba)
        need = np.linalg.norm(G - Ba, axis=-1)
        resid = np.linalg.norm(G - O, axis=-1)
        need_m = (need * sv).sum(1) / np.maximum(sv.sum(1), 1)
        resid_m = (resid * sv).sum(1) / np.maximum(sv.sum(1), 1)
        nrm = np.linalg.norm(o_sel, axis=-1)
        nrm_a = np.linalg.norm(o_ast, axis=-1)
        off0 = np.linalg.norm(D[f"off0_sel_{tag}"].astype(np.float64), axis=-1)
        off[tag] = {
            "tot_sel_norm_mean_slots": nrm.mean(1), "tot_sel_norm_2s": nrm[:, 3], "tot_sel_norm_6s": nrm[:, 7],
            "tot_sel_along_6s": al_sel[:, 7], "tot_sel_lat_6s": la_sel[:, 7],
            "tot_sel_along_abs_mean": np.abs(al_sel).mean(1), "tot_sel_lat_abs_mean": np.abs(la_sel).mean(1),
            "off0_sel_norm_mean_slots": off0.mean(1),
            "tot_astar_norm_mean_slots": nrm_a.mean(1), "tot_astar_along_6s": al_ast[:, 7],
            "need_astar_masked_mean": need_m, "resid_astar_masked_mean": resid_m,
            "coverage_ratio_w": 1.0 - resid_m / np.maximum(need_m, 1e-6),
            "fan_minus_bank_mean": D[f"fan_minus_bank_mean_{tag}"].astype(np.float64),
            "off0_fan_mean": D[f"off0_fan_mean_{tag}"].astype(np.float64),
            "traj_norm_6s": np.linalg.norm(P[:, 7], axis=-1),
        }
    OFF_KEYS = ["tot_sel_norm_mean_slots", "tot_sel_norm_2s", "tot_sel_norm_6s", "tot_sel_along_abs_mean",
                "tot_sel_lat_abs_mean", "off0_sel_norm_mean_slots", "tot_astar_norm_mean_slots",
                "need_astar_masked_mean", "resid_astar_masked_mean", "fan_minus_bank_mean", "off0_fan_mean"]
    R["offset"] = {"levels": {t: {k: EB(off[t][k], eid) for k in OFF_KEYS} for t in ("k", "w")},
                   "paired_w_minus_k": {k: PB(off["w"][k], off["k"][k], eid) for k in OFF_KEYS},
                   "burden_coverage_pooled": {t: float(1.0 - (off[t]["resid_astar_masked_mean"] * sv.sum(1)).sum()
                                                       / (off[t]["need_astar_masked_mean"] * sv.sum(1)).sum())
                                              for t in ("k", "w")},
                   "burden_coverage_median_window": {t: float(np.median(off[t]["coverage_ratio_w"])) for t in ("k", "w")},
                   "offset_share_of_6s_displacement": {t: float(np.median(off[t]["tot_sel_norm_6s"] / np.maximum(off[t]["traj_norm_6s"], 1e-6))) for t in ("k", "w")},
                   "_reads": ("tot_sel = out['traj'] - bank[sel] = the TOTAL displacement the decoder adds to the selected "
                              "anchor over its three passes (classifier + 2 refinement steps); off0 = the first pass "
                              "only (out['offset']). need_astar = |GT - bank[a_star]| (what the trajectory loss asks "
                              "the offset head to remove); resid_astar = |GT - anchor_traj[a_star]| (what remains). "
                              "coverage = 1 - resid/need. along/lat are in the LOCAL tangent frame of the bank path.")}
    # the along-track sign test: does the 6 s along-offset track (v0 - 10)*6 on withheld rows?
    x = (v0 - REF_SPEED) * 6.0
    yw = off["w"]["tot_sel_along_6s"]
    yk = off["k"]["tot_sel_along_6s"]
    A = np.stack([x, np.ones_like(x)], 1)
    cw = np.linalg.lstsq(A, yw, rcond=None)[0]
    ck = np.linalg.lstsq(A, yk, rcond=None)[0]
    R["offset"]["along_6s_vs_speed_burden"] = {
        "_x": "(v0 - 10 m/s) * 6 s = along-track displacement of a constant-speed path relative to the fixed 10 m/s roll",
        "withheld": {"slope": float(cw[0]), "intercept": float(cw[1]), "r": float(np.corrcoef(x, yw)[0, 1])},
        "kept": {"slope": float(ck[0]), "intercept": float(ck[1]), "r": float(np.corrcoef(x, yk)[0, 1])},
        "_reads": "slope ~1 with r ~1 on withheld rows = the offset head is re-supplying the speed the bank lacks"}
    R["offset"]["by_v0_band"] = by_band(v0, lambda m: {
        "oiv_kept_bank": float(oiv_k[m].mean()), "oiv_withheld_bank": float(oiv_w[m].mean()),
        "tot_sel_norm_mean_k": float(off["k"]["tot_sel_norm_mean_slots"][m].mean()),
        "tot_sel_norm_mean_w": float(off["w"]["tot_sel_norm_mean_slots"][m].mean()),
        "tot_sel_norm_6s_k": float(off["k"]["tot_sel_norm_6s"][m].mean()),
        "tot_sel_norm_6s_w": float(off["w"]["tot_sel_norm_6s"][m].mean()),
        "need_astar_k": float(off["k"]["need_astar_masked_mean"][m].mean()),
        "need_astar_w": float(off["w"]["need_astar_masked_mean"][m].mean()),
        "resid_astar_k": float(off["k"]["resid_astar_masked_mean"][m].mean()),
        "resid_astar_w": float(off["w"]["resid_astar_masked_mean"][m].mean()),
        "coverage_w_pooled": float(1 - off["w"]["resid_astar_masked_mean"][m].sum() / max(off["w"]["need_astar_masked_mean"][m].sum(), 1e-9)),
        "coverage_k_pooled": float(1 - off["k"]["resid_astar_masked_mean"][m].sum() / max(off["k"]["need_astar_masked_mean"][m].sum(), 1e-9)),
    })
    p("[offset] |traj - bank[sel]| mean-over-slots  kept", fmt(R["offset"]["levels"]["k"]["tot_sel_norm_mean_slots"], "mean"),
      " withheld", fmt(R["offset"]["levels"]["w"]["tot_sel_norm_mean_slots"], "mean"),
      " w-k", fmt(R["offset"]["paired_w_minus_k"]["tot_sel_norm_mean_slots"]))
    p("[offset] need|GT-bank[a*]|  kept", fmt(R["offset"]["levels"]["k"]["need_astar_masked_mean"], "mean"),
      " withheld", fmt(R["offset"]["levels"]["w"]["need_astar_masked_mean"], "mean"),
      "; resid  kept", fmt(R["offset"]["levels"]["k"]["resid_astar_masked_mean"], "mean"),
      " withheld", fmt(R["offset"]["levels"]["w"]["resid_astar_masked_mean"], "mean"),
      "; coverage pooled", R["offset"]["burden_coverage_pooled"])
    p("[offset] along-6s vs (v0-10)*6:", json.dumps(R["offset"]["along_6s_vs_speed_burden"]["withheld"]), "kept:",
      json.dumps(R["offset"]["along_6s_vs_speed_burden"]["kept"]))

    # ---- C. smoothness ------------------------------------------------------ #
    paths = {"os_k": M["os_k"], "os_w": M["os_w"], "orc_k": M["orc_k"], "orc_w": M["orc_w"],
             "bank_sel_k": D["bank_sel_k"], "bank_sel_w": D["bank_sel_w"],
             "bank_astar_k": D["bank_astar_k"], "bank_astar_w": D["bank_astar_w"],
             "ha": M["ha"], "ha0": M["ha0"], "gt": M["g"]}
    sm8 = {k: smooth_native(v.astype(np.float64), T8) for k, v in paths.items()}
    sm2 = {k: smooth_native(v[:, SLOTS2S].astype(np.float64), T8[SLOTS2S]) for k, v in paths.items()}
    sm6 = {k: smooth_native(v[true6][:, SLOTS6S].astype(np.float64), T8[SLOTS6S]) for k, v in paths.items()}
    eid6 = [e for e, t in zip(eid, true6) if t]

    def sm_block(sm, e):
        lv = {arm_: {k: EB(sm[arm_][k], e) for k in SMOOTH_KEYS} for arm_ in sm}
        pairs = [("os_w", "os_k"), ("orc_w", "orc_k"), ("os_k", "bank_sel_k"), ("os_w", "bank_sel_w"),
                 ("os_k", "gt"), ("os_w", "gt"), ("os_k", "ha"), ("os_w", "ha"), ("orc_k", "gt"), ("orc_w", "gt")]
        pr = {f"{x}_minus_{y}": {k: PB(sm[x][k], sm[y][k], e) for k in SMOOTH_KEYS} for x, y in pairs}
        return {"levels": lv, "paired": pr}
    R["smoothness"] = {
        "_instrument": ("slot-resolution finite differences on the emitted waypoints at their NATIVE instants "
                        "(0.5,1,1.5,2,3,4,5,6 s): segment speeds -> junction accel/curvature -> between-junction "
                        "jerk [m/s^3] and curvature-rate [1/(m s)] (+ lateral jerk d(v^2 kappa)/dt). NOT a 10 Hz "
                        "measure; the bank paths (constant a, constant kappa) are the discretisation floor and ha0 "
                        "must read 0. Heading terms masked below MIN_DS_MPS like four_families."),
        "native8": sm_block(sm8, eid), "grid2s_0p5": sm_block(sm2, eid),
        "grid6s_1p0_true6s": {**sm_block(sm6, eid6), "n": int(true6.sum())},
    }
    for k in ("jerk_mean_abs_mps3", "krate_mean_abs_1pms", "ljerk_mean_abs_mps3"):
        p(f"[smooth:native8] {k:22s} os_k {fmt(R['smoothness']['native8']['levels']['os_k'][k], 'mean')}  "
          f"os_w {fmt(R['smoothness']['native8']['levels']['os_w'][k], 'mean')}  gt {fmt(R['smoothness']['native8']['levels']['gt'][k], 'mean')}  "
          f"bank_sel_k {fmt(R['smoothness']['native8']['levels']['bank_sel_k'][k], 'mean')}  "
          f"w-k {fmt(R['smoothness']['native8']['paired']['os_w_minus_os_k'][k])}")

    # ---- D. selection degeneracy ------------------------------------------- #
    sel_k, sel_w = D["sel_idx_k"], D["sel_idx_w"]
    R["selection"] = {
        "kept": selection_profile(sel_k, D["a_star_k"], D["cls_top1_k"]),
        "withheld": selection_profile(sel_w, D["a_star_w"], D["cls_top1_w"]),
        "cross_regime_same_anchor_frac": round(float((sel_k == sel_w).mean()), 4),
        "cross_regime_same_base_anchor_frac": round(float((D["sel_idx_base_k"] == D["sel_idx_base_w"]).mean()), 4),
        "own_oracle_same_across_regimes_frac": round(float((D["a_star_k"] == D["a_star_w"]).mean()), 4),
        "goal_graft_changed_selection_frac": {"kept": round(float((sel_k != D["sel_idx_base_k"]).mean()), 4),
                                              "withheld": round(float((sel_w != D["sel_idx_base_w"]).mean()), 4)},
        "idx67_share_by_v0_band": by_band(v0, lambda m: {"kept": round(float((sel_k[m] == 67).mean()), 4),
                                                         "withheld": round(float((sel_w[m] == 67).mean()), 4),
                                                         "own_oracle_is_67_kept": round(float((D["a_star_k"][m] == 67).mean()), 4),
                                                         "own_oracle_is_67_withheld": round(float((D["a_star_w"][m] == 67).mean()), 4)}),
        "sel_agree_oracle_paired_w_minus_k": PB((sel_w == D["a_star_w"]).astype(float), (sel_k == D["a_star_k"]).astype(float), eid),
        "top_anchors": {"kept": [[int(i), int(c)] for i, c in sorted(enumerate(np.bincount(sel_k, minlength=117)), key=lambda t: -t[1])[:8]],
                        "withheld": [[int(i), int(c)] for i, c in sorted(enumerate(np.bincount(sel_w, minlength=117)), key=lambda t: -t[1])[:8]]},
        "controls_of_top_anchors": {str(i): ctrl[i].tolist() for i in
                                    sorted(set([int(t[0]) for t in sorted(enumerate(np.bincount(sel_k, minlength=117)), key=lambda t: -t[1])[:5]]
                                               + [int(t[0]) for t in sorted(enumerate(np.bincount(sel_w, minlength=117)), key=lambda t: -t[1])[:5]]))},
    }
    p("[selection] kept", json.dumps({k: R["selection"]["kept"][k] for k in ("n_distinct_selected", "modal_anchor", "modal_frac", "straight_ahead_idx67_frac", "entropy_ratio", "agrees_with_own_oracle_frac")}))
    p("[selection] withheld", json.dumps({k: R["selection"]["withheld"][k] for k in ("n_distinct_selected", "modal_anchor", "modal_frac", "straight_ahead_idx67_frac", "entropy_ratio", "agrees_with_own_oracle_frac")}))
    p("[selection] same anchor across regimes", R["selection"]["cross_regime_same_anchor_frac"])

    # ---- E. declared heads (tactical / strategic) -------------------------- #
    heads = {}
    for tag in ("k", "w"):
        lat_ok = D["lat_label"] >= 0
        lon_ok = D["lon_label"] >= 0
        r_ok = D["route_label"] >= 0
        heads[tag] = {"lat_tac_correct": np.where(lat_ok, (D[f"lat_tac_{tag}"] == D["lat_label"]).astype(float), np.nan),
                      "lon_tac_correct": np.where(lon_ok, (D[f"lon_tac_{tag}"] == D["lon_label"]).astype(float), np.nan),
                      "route_correct": np.where(r_ok, (D[f"route_pred_{tag}"] == D["route_label"]).astype(float), np.nan)}
        # strategic goal head bearing vs GT bearing at 6 s (proxy: the g_str label is the LAN route bearing)
        g6 = G[:, 7]
        far = (sv[:, 7] > 0.5) & (np.linalg.norm(g6, axis=-1) > 5.0)
        gs = D[f"g_str_{tag}"].astype(np.float64)
        berr = np.degrees(np.abs(wrap(np.arctan2(gs[:, 1], gs[:, 0]) - np.arctan2(g6[:, 1], g6[:, 0]))))
        heads[tag]["gstr_bearing_err_deg_vs_gt6s"] = np.where(far, berr, np.nan)
        # tactical goal head: predicted speed at 2/4/6 s vs GT speed (the model's OWN speed estimate)
        gt_ = D[f"g_tac_{tag}"].astype(np.float64)
        lab = D["goal_tac_lab"].astype(np.float64)
        gv = D["goal_tac_valid"].astype(bool)
        for j, tau in enumerate((2, 4, 6)):
            heads[tag][f"gtac_speed_abs_err_{tau}s_mps"] = np.where(gv[:, j], np.abs(gt_[:, j, 3] - lab[:, j, 3]), np.nan)
            heads[tag][f"gtac_point_err_{tau}s_m"] = np.where(gv[:, j], np.linalg.norm(gt_[:, j, :2] - lab[:, j, :2], axis=-1), np.nan)
    HK = list(heads["k"].keys())
    R["declared_heads"] = {"levels": {t: {k: EB(heads[t][k], eid) for k in HK} for t in ("k", "w")},
                           "paired_w_minus_k": {k: PB(heads["w"][k], heads["k"][k], eid) for k in HK},
                           "n_labelled": {"lat": int((D["lat_label"] >= 0).sum()), "lon": int((D["lon_label"] >= 0).sum()),
                                          "route": int((D["route_label"] >= 0).sum()),
                                          "gtac_2s": int(D["goal_tac_valid"][:, 0].sum()), "gtac_6s": int(D["goal_tac_valid"][:, 2].sum())},
                           "goal_gate_value": float(D["goal_gate"].mean()),
                           "_caveats": ["the z_tac lat/lon heads are the 8-way v7.2 vocabulary scored on the ~24 % of windows that carry a label",
                                        "route head: 3-way v2.1 route target on route-labelled windows",
                                        "g_str bearing is scored against the GT 6 s bearing as a PROXY (its training label is the LAN route bearing, not dumped)",
                                        "NO nav-shuffle / nav-zero control in this diagnostic (both regimes fed the same v7.2 nav token); strategic numbers are per-regime paired reads, not nav-attribution claims"]}
    for k in ("lat_tac_correct", "lon_tac_correct", "route_correct", "gtac_speed_abs_err_2s_mps", "gtac_speed_abs_err_6s_mps", "gstr_bearing_err_deg_vs_gt6s"):
        p(f"[heads] {k:28s} kept {fmt(R['declared_heads']['levels']['k'][k], 'mean')}  withheld {fmt(R['declared_heads']['levels']['w'][k], 'mean')}  w-k {fmt(R['declared_heads']['paired_w_minus_k'][k])}")

    # ---- F. pricing option 2: roll the withheld bank at the model's OWN predicted speed ---- #
    pred_v2 = D["g_tac_w"].astype(np.float64)[:, 0, 3]          # vision-only (withheld) predicted speed @2 s
    pred_v2k = D["g_tac_k"].astype(np.float64)[:, 0, 3]
    oiv_pred_w = oiv_at_speed(ctrl, np.clip(pred_v2, 0.0, 60.0), G, sv)
    oiv_pred_k = oiv_at_speed(ctrl, np.clip(pred_v2k, 0.0, 60.0), G, sv)
    R["option2_own_predicted_speed"] = {
        "_is": ("MODEL-FREE CEILING on the anchor PRIOR if the withheld-row bank were rolled at the model's own "
                "g_tac speed prediction at 2 s (the only speed the live forward emits — refc1 speed head is OFF), "
                "vs the fixed 10 m/s roll and vs the measured-v0 roll (the LEAK bound, refused as a design)"),
        "pred_speed_2s_withheld_mae_vs_gt_mps": float(np.nanmean(heads["w"]["gtac_speed_abs_err_2s_mps"])),
        "pred_speed_2s_withheld_mae_vs_v0_mps": float(np.mean(np.abs(pred_v2 - v0))),
        "pred_speed_2s_kept_mae_vs_v0_mps": float(np.mean(np.abs(pred_v2k - v0))),
        "oiv_fixed_10ms": float(oiv_w.mean()), "oiv_at_pred_speed_withheld": float(oiv_pred_w.mean()),
        "oiv_at_pred_speed_kept": float(oiv_pred_k.mean()), "oiv_at_true_v0_LEAK_BOUND": float(oiv_k.mean()),
        "paired_predw_minus_fixed": PB(oiv_pred_w, oiv_w, eid),
        "paired_predw_minus_truev0": PB(oiv_pred_w, oiv_k, eid),
        "by_v0_band": by_band(v0, lambda m: {"fixed_10ms": float(oiv_w[m].mean()), "pred_speed_w": float(oiv_pred_w[m].mean()),
                                            "true_v0": float(oiv_k[m].mean()),
                                            "pred_mae_vs_v0": float(np.abs(pred_v2 - v0)[m].mean())}),
    }
    p("[option2] OIV fixed-10 %.4f | at withheld-regime predicted speed %.4f | at true v0 (LEAK bound) %.4f ; pred-speed MAE vs v0 %.3f m/s"
      % (oiv_w.mean(), oiv_pred_w.mean(), oiv_k.mean(), np.mean(np.abs(pred_v2 - v0))))

    # ---- G. four families through the banked instruments ------------------- #
    if not a.skip_families:
        raw_off = int(man["corpus"]["frames"]["provider_to_raw_frame_offset"])
        out_root = os.path.dirname(os.path.abspath(a.out))
        R["four_families_2s"] = families_on_grid(files, man, out_root, f"{a.label}_2s", SLOTS2S, 0.5, 4,
                                                 lambda s: np.ones(s.shape[0], bool), a.lead_block, raw_off)
        R["four_families_6s_true6s"] = families_on_grid(files, man, out_root, f"{a.label}_6s", SLOTS6S, 1.0, 6,
                                                        lambda s: s.min(1) > 0.5, a.lead_block, raw_off)
        for tag in ("four_families_2s", "four_families_6s_true6s"):
            blk = R[tag]
            p(f"[{tag}] n={blk['n_windows']} ep={blk['n_episodes']}")
            for x in ARMS:
                p(f"   ADE {x:6s} {fmt(blk['levels_ade'][x], 'mean')}")
            for nm, fp in blk["families_paired"].items():
                fams = fp["families"]
                s = " | ".join(f"{f}:{mk} {fmt(r)}" for f, d in fams.items() for mk, r in d.items()
                               if mk in ("ade_m", "LON_speed_mae_mps", "LON_along_mae_m", "LAT_cross_mae_m", "LAT_heading_mae_deg", "LAT_curvature_mae_1pm"))
                p(f"   {nm}: {s}")
            if isinstance(blk.get("distance_keeping"), dict):
                dk = blk["distance_keeping"]
                p(f"   distance_keeping status {dk.get('status')} n {dk.get('n')} coverage {json.dumps((dk.get('coverage') or {}).get('counts'))}")

    # ---- H. the criteria view (products/P7-TanitEval/CRITERIA_REGISTRY.json key paths) ------ #
    R["tier"] = "T1"
    R["estimator"] = R["estimator"]
    R["protocol"] = {
        "tier": "T1",
        "loop": "OPEN LOOP (PI ruling 2026-09-02): one forward at the window origin; no action feedback",
        "inference_inputs": ["frames <= t0 (8 observed frames x 9 ch, 256x640)", "nav_cmd = the clip's v7.2 nav token (oracle provenance, as trained)",
                             "v0 = pose_last[:, 3] measured at t0 (PI ruling 2026-09-02) — KEPT regime only",
                             "ego_state @ t0 = (v0, a_long, yaw_rate, curvature, keep) measured at the last OBSERVED frame — KEPT regime only; "
                             "WITHHELD regime: keep=0, every ego value zeroed (the model's own vision-only mode)"],
        "vision_only": ("the WITHHELD regime (os_w) is the vision-only inference arm (ego block zeroed, keep bit 0; nav token still fed); "
                        "the KEPT regime (os_k) consumes the MEASURED t0 ego state, admissible under the 2026-09-02 ruling. "
                        "No future ego quantity reaches any input (refc_v3.ego_state_at_t0 reads [:, -1] of the OBSERVED window)"),
        "goal_source": ("PREDICTED: g_str/g_tac/goal_point_tac are read from pooled frames (+ ego @ t0 in the KEPT regime); "
                        "the situation classifier's output is not in the graph (config.json goal_provenance.contains_situation_classifier_output=false); "
                        "LAN corridor is label-only (E12) and NOT fed here"),
        "corpus": ("B1-v7.2 EVAL v2 cache, 141 clips, provider stride 5 -> 4,823 windows — the banked refcv3 open-loop surface "
                   "(taniteval/results/refcv3-40284-openloop); an EVAL-ONLY diagnostic — the parity key applies to the TRAIN corpus "
                   "(the run's config.json stamps v2_parity.parity=false, B1 train cache)"),
        "parity_key": "n/a (eval-only diagnostic on the B1 EVAL cache; the run trains on the non-parity B1 corpus)",
        "arms_and_tiers": TIERS,
        "scope": "EARLY-TRAINING DIAGNOSTIC of refcv4b-b1-v72-40k at step %s of 40,284 — NOT a capability claim" % man["model"]["step"],
    }
    ff2 = R.get("four_families_2s")
    if ff2:
        fam_k = json.loads(json.dumps(ff2["arms"]["os_k"]["four_families"], default=str))
        fam_w = json.loads(json.dumps(ff2["arms"]["os_w"]["four_families"], default=str))
        strat = {
            "status": "PARTIAL-INSTRUMENTED",
            "decision_accuracy": {"route_head_acc_kept": R["declared_heads"]["levels"]["k"]["route_correct"],
                                  "route_head_acc_withheld": R["declared_heads"]["levels"]["w"]["route_correct"],
                                  "paired_w_minus_k": R["declared_heads"]["paired_w_minus_k"]["route_correct"],
                                  "n": R["declared_heads"]["n_labelled"]["route"],
                                  "label": "v2.1 route target (3-way) on route-labelled windows"},
            "route_quality": {"g_str_bearing_err_deg_vs_gt6s_kept": R["declared_heads"]["levels"]["k"]["gstr_bearing_err_deg_vs_gt6s"],
                              "g_str_bearing_err_deg_vs_gt6s_withheld": R["declared_heads"]["levels"]["w"]["gstr_bearing_err_deg_vs_gt6s"],
                              "paired_w_minus_k": R["declared_heads"]["paired_w_minus_k"]["gstr_bearing_err_deg_vs_gt6s"],
                              "_proxy": "GT 6 s bearing stands in for the LAN route-bearing label, which is not dumped"},
            "_caveat": ("per-regime PAIRED reads of the strategic heads; NO nav-shuffle / nav-zero control in this diagnostic "
                        "(both regimes fed the same v7.2 token) — the nav-attribution controls live in the banked refcv3 record, not here"),
            "n": R["declared_heads"]["n_labelled"]["route"], "tier": "T1"}
        fam_k["strategic"] = strat
        fam_w["strategic"] = dict(strat)
        R["four_families"] = fam_k
        R["four_families_withheld"] = fam_w
        R["strategic"] = strat
        R["tactical"] = {
            "anchor_selection": {"kept": R["selection"]["kept"], "withheld": R["selection"]["withheld"],
                                 "cross_regime_same_anchor_frac": R["selection"]["cross_regime_same_anchor_frac"]},
            "declared_heads_v7_8way": {"lat_acc_kept": R["declared_heads"]["levels"]["k"]["lat_tac_correct"],
                                       "lat_acc_withheld": R["declared_heads"]["levels"]["w"]["lat_tac_correct"],
                                       "lon_acc_kept": R["declared_heads"]["levels"]["k"]["lon_tac_correct"],
                                       "lon_acc_withheld": R["declared_heads"]["levels"]["w"]["lon_tac_correct"],
                                       "n_lat": R["declared_heads"]["n_labelled"]["lat"], "n_lon": R["declared_heads"]["n_labelled"]["lon"]},
            "_trajectory_derived": "four_families.tactical (lateral_decision / longitudinal_decision / maneuver_5way_collapsed) via the canonical labeller",
        }
    R["floors"] = {"ha": "hold-action control (T1)", "ha0": "constant-velocity control (T1)",
                   "constant_only_zero_path_ade_8slot_m": R["controls"]["constant_only_zero_path_ade_8slot_m"],
                   "paired_vs_floor_blocks": "four_families_2s.families_paired.paired_os{k,w}_minus_ha{,0}"}
    R["refused"] = {"strategic.echo_test": "not run in this diagnostic: both regimes were fed the same nav token; the nav-shuffle/nav-zero echo controls are in taniteval/results/refcv3-40284-openloop (refcv3) and are a WORK ITEM for a refcv4b arm record"}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    p(f"[write] {a.out}")
    p("EGODROP_ANALYZE_DONE")


if __name__ == "__main__":
    main()
