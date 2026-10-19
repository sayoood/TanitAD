"""e1_decomp_r8.py -- WHERE on the ceiling-COMPLIANT S2 set does refcv8 EMA-30k lose to (or tie) the hold-action echo `ha0_ext`?
CPU only, banked dumps only, no GPU, no retraining.  NEW file (nothing existing is edited).

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 \
    PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' \
      python -X utf8 e1_decomp_r8.py compute     # -> <PKG>/raw/e1_decomp/E1_DECOMP.json (+ .md)
      python -X utf8 e1_decomp_r8.py render      # re-render E1_DECOMP.md from the JSON (+ the analyst VERDICT_ANALYST.md if present)

THE BAR.  BAR-R8-E1 (SPEC_REFCV8_DRAFT section 5, scored under AMENDMENT A-R8-CLAMP-2, commit b6f7fac): refcv8 `os` minus the echo
`ha0_ext`, S2 ADE 0-2 s (instants 0.5/1.0/1.5/2.0 s), < 0 and SEPARATED at BOTH inference seeds, on the ceiling-compliant windows,
capped refcv8 primary.  On the compliant set capped == uncapped exactly (`raw/compliant_reread/trees/step30000ema_unc.json`:
n_changed_plans_on_compliant_windows 0 at both seeds; re-asserted here as control K), so the panel's own `os` IS the capped primary there.

WHAT.  Per compliant S2 window d = ADE(refcv8 os) - ADE(ha0_ext), the per-window components of `pair_read_r8.components` (the SAME
function the banked read used).  d is decomposed over the PRE-REGISTERED partitions below; per bin: n windows / episodes, the bin's
CONTRIBUTION to the mean delta (sum of the bin's d / N, metres, with its joint-draw CI), its SHARE of the total (contribution / total;
the total's CI spans 0, so a share CI is a ratio over a denominator that changes sign and is printed but NOT read), the bin-mean delta,
and both arms' bin-mean ADE (the echo's own error per bin).  Because the total is near 0, the primary reading is the GROSS split:
the sum of the positive (refcv8 loses) and of the negative (refcv8 wins) bin contributions.

PRE-REGISTERED PARTITIONS (fixed here before any decomposed number was looked at):
  dv2         GT speed change over 2 s (v2 = decisions/gt_future_ext[:, 19, 3]) under the dv-class rule of `s6_lon_decomp.partitions`
              (imported, not copied): start_from_stop | stopping | slow_both | accelerating | decelerating | cruising (+-1.5 m/s).
  dv6         the same rule on v6 = gt_future_ext[:, 59, 3] where the 60-step future is fully valid, else `no full 6-s future`.
  v0bin       GT v0 (panel `v0`): <2 | 2-8 | 8-15 | 15-25 | >=25 m/s (s6_lon_decomp.V0_EDGES).
  yawbin      GT |net yaw rate| over 0-2 s = |wrap(yaw(2.0 s) - yaw(t0))| / 2 s from the WORLD poses (gt_future_ext[:, 19, 2],
              pose_last[:, 2]): straight < 0.02 | mild 0.02-0.08 | curve 0.08-0.20 | sharp >= 0.20 rad/s.  Used only if control X-yaw
              passes (R(-yaw0) (gt_future_ext[:,19,:2] - pose_last[:,:2]) == the ego-frame GT g(2 s) within 0.05 m on every window);
              otherwise the ego-frame chord heading of g's last segment / 1.75 s is used and the fallback is recorded.
  tac_lon     refcv8's tactical lon head argmax (`nav_true.tacv6_lon_logits`, the seed's own extras): the 8 v7 classes.
  tac_brake   head argmax BRAKE_TO vs not.
  tac_correct head argmax == GT `lon_v7` | != | unlabelled (-100).
  echo_regime the echo's OWN error at 2 s (model-free: a function of the logged t0 state and the GT only, exogenous to refcv8):
              echo-good (|along 2 s| < 0.5 m AND |cross 2 s| < 0.3 m) | echo-lon-wrong only | echo-lat-wrong only | echo-both-wrong.
              "steady state where a hold-action prior is right" == echo-good.
  echo_ade    the echo's ADE 0-2 s: <0.1 | 0.1-0.2 | 0.2-0.4 | 0.4-0.8 | >=0.8 m  (descriptive; selecting on the echo's error makes
              the high bins negative by construction -- read the regime partition for attribution, this one for the headroom).
  steady_x_dv echo_regime x dv2 (cruising vs not).
  fl_set      membership in the queued force_lon roll (raw/force_lon/plan_sha12.json, matched on (episode_index, ws), clip sha12 asserted):
              FB selected | FB not selected (the other FB_all windows) | CT selected | head BRAKE_TO, S6 window, not GT-cruising |
              head BRAKE_TO, no full 6-s future (S2-only) | other.  FB_all = full 6-s future AND dv6 cruising AND seed-0 head argmax
              BRAKE_TO (force_lon's definition), re-derived here and asserted to count 460 over all 4,754 windows (control F).
  plan2_vs_gt2 OUTCOME-defined (labelled as such): refcv8's plan longitudinal class over the last S2 segment vs GT's
              (`s6_lon_decomp.plan_lon_class`, +-1.5 m/s vs v0, applied identically to GT).
  ADDED 2026-10-18 AFTER the first decomposed read (which localised the loss to echo-good windows with an ongoing speed change and to
  v0 < 2 / slow_both / stopping) and BEFORE any number of the additions below was computed -- POST-HOC, labelled so, never a bar input:
  a0bin       GT measured longitudinal acceleration at t0 (decisions `ego_t0[:, 1]`, refc_v3.ego_state_at_t0; NOT fed to refcv8, whose
              ego input is v0 only): |a0| < 0.3 | 0.3-1.0 | 1.0-2.0 | >= 2.0 m/s^2.
  per-bin levels: signed along-track at 2 s (P_x - G_x, ego frame) of refcv8 and of the echo, the constant-velocity echo `ha0`'s ADE
              (a = 0, kappa = 0 from v0: exactly the ego information refcv8 is fed), refcv7-50,400's ADE (the panel's embedded arm), and the
              tactical-lon head-argmax histogram.  CV-CAPTURE per bin = (ADE_ha0 - ADE_os) / (ADE_ha0 - ADE_ha0_ext): the fraction of what the
              measured (a0, kappa0) buys over v0 alone that refcv8 recovers from vision.
  LEVER ARMS (zero GPU, banked plans only, EARLY single reads):
    ENS2      the 2-draw ensemble: per-window mean of the seed-0 and seed-1 `os` plans (same weights, same windows; a test-time
              procedure using model outputs only).  By the triangle inequality ADE(mean plan) <= the mean of the two ADEs on EVERY window,
              so the question is HOW MUCH.  ONE read only (both seeds are consumed): the bar's inference replicate needs two more seeds.
    MILESTONE the per-bin refcv8 ADE of the EMA milestones 5k / 10.5k / 15k / 20k / 30k (seed 0) on the SAME compliant windows (panel keys
              asserted equal) -- is the loss in the slow bins a training-trajectory drift?
LON / LAT split of d (exact): per instant the error vector e = P - G is split into its along (x) and cross (y) parts in the ego frame
at t0 (the battery's `four_families` convention: along = x, cross = y).  A(a, c) = mean_t sqrt(a_t^2 + c_t^2).  Two-factor SHAPLEY:
  d_lon = 1/2 [A(a_os, c_ha) - A(a_ha, c_ha) + A(a_os, c_os) - A(a_ha, c_os)],  d_lat = 1/2 [A(a_ha, c_os) - A(a_ha, c_ha) + A(a_os, c_os) - A(a_os, c_ha)]
  d_lon + d_lat == A(os) - A(ha) exactly; both are decomposed by the same partitions.  A GT-path-tangent frame (tangent of the GT's own
  0.5-s segment, held through a stopped segment) is reported beside as a sensitivity.
COUNTERFACTUALS (the lever arithmetic, not an intervention): for a bin B, (i) SHIFT: refcv8's mean excess in B removed, within-bin
spread kept (d' = d - mean_B(d) on B); (ii) TIE: refcv8 == echo on B (d' = 0 on B, optimistic: removes B's variance too).  The E1 cell
of d' on the SAME draws (delta, lo, hi, separated) at both seeds.  These bound what fixing B could do to E1; they are not results.

ESTIMATOR.  Paired episode-cluster bootstrap over the episodes the compliant set contains (138), models FIXED, B 2000, seed 0: the draws
are EXACTLY taniteval.ci._draws (s6_lon_decomp.boot_counts), so every bin statistic is a ratio of resampled sums on the SAME draws and
the bins add up to the total in every draw.  It answers "another draw of EPISODES" only -- NOT another training run (single training
seed), NOT another inference run (seed 1 is the inference replicate beside seed 0).  TIER: T1 self-action OPEN LOOP on logged eval139
frames (not driving performance).  EARLY READ of a banked milestone (EMA step 30,000); the bars are decided at step 48,852.

CONTROLS (the tool writes CONTROL_FAILED.json and exits 3 if any fails; nothing else is written):
  M   the compliant mask, built by compliant_reread_r8.compliant_mask on clamp_posthoc_r8.load_seed_inputs, has sha256 == MASK_SHA at both
      seeds and 4,159 / 4,754 S2 windows, 138 episodes.
  R   the all-bin total reproduces the compliant E1 cell EXACTLY (delta, lo, hi, n_windows, n_episodes, separated) against the banked tree
      json (capped AND uncapped) and against the LITERALS of the brief (seed 0 -0.0085 [-0.0229, +0.0065], seed 1 -0.0076 [-0.0216, +0.0073],
      n 4,159 / 138); the joint-draw total equals ci's at its 4-dp rounding.
  K   capped == uncapped on the compliant S2 rows, read from the compliant tree json (n_changed_plans_on_compliant_windows == 0, both seeds).
  Z   identity contrasts (echo vs itself, refcv8 vs itself): every bin's contribution, mean and CI read EXACTLY 0.0, in d, d_lon and d_lat.
  S   Shapley: max |d_lon + d_lat - d_numpy| < 1e-9; max |numpy ADE - components ade_m| < 1e-4 (both arms).
  P   random partitions (200 permutations of the v0bin labels): the per-bin contribution CI covers frac_b x total in ~95 % of bins.
  F   force_lon's FB_all re-derived == 460 windows, CT pool == 1,279 (select_stats.json), and every plan window found with its clip sha12.
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
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PKG = HERE.parent
OUT = PKG / "raw" / "e1_decomp"
TREE = "step30000ema_unc"
ARCHIVE = "C:/Users/Admin/refcv8_eval_archive/battery"
COMPLIANT_JSON = PKG / "raw" / "compliant_reread" / "trees" / f"{TREE}.json"
FL_PLAN = PKG / "raw" / "force_lon" / "plan_sha12.json"
FL_STATS = PKG / "raw" / "force_lon" / "select_stats.json"
MASK_SHA = "4ebc6b8604f064f69e9e94c15e099062f68339a8ef38d666e8a29d29cb1c4129"
SEEDS = (0, 1)
N_BOOT = 2000
BOOT_SEED = 0
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"
QUESTION = ("paired episode-cluster bootstrap over the episodes the compliant set contains, models FIXED, B 2000, seed 0 (draws identical to "
            "taniteval.ci._draws): answers 'another draw of EPISODES' only; NOT another training run (single training seed), NOT another "
            "inference run (seed 1 shown beside seed 0 as the inference replicate)")
#: the brief's literals (MEASURED in raw/compliant_reread/trees/step30000ema_unc.json); written as literals, never derived from the code under test
BRIEF_E1 = {0: {"delta": -0.0085, "lo": -0.0229, "hi": 0.0065, "n_windows": 4159, "n_episodes": 138},
            1: {"delta": -0.0076, "lo": -0.0216, "hi": 0.0073, "n_windows": 4159, "n_episodes": 138}}
YAW_EDGES = [(0.0, 0.02, "straight <0.02"), (0.02, 0.08, "mild 0.02-0.08"), (0.08, 0.20, "curve 0.08-0.20"), (0.20, 1e9, "sharp >=0.20")]
ECHO_ADE_EDGES = [(0.0, 0.1, "echo ADE <0.1"), (0.1, 0.2, "echo ADE 0.1-0.2"), (0.2, 0.4, "echo ADE 0.2-0.4"), (0.4, 0.8, "echo ADE 0.4-0.8"),
                  (0.8, 1e9, "echo ADE >=0.8")]
ECHO_ALONG_TOL, ECHO_CROSS_TOL = 0.5, 0.3
A0_EDGES = [(0.0, 0.3, "|a0| <0.3"), (0.3, 1.0, "|a0| 0.3-1.0"), (1.0, 2.0, "|a0| 1.0-2.0"), (2.0, 1e9, "|a0| >=2.0")]
MILESTONES = ["step5000ema_unc", "step10500ema_unc", "step15000ema_unc", "step20000ema_unc", "step30000ema_unc"]
LEVEL_PARTS = ("dv2", "v0bin", "echo_regime", "steady_x_dv", "a0bin", "tac_brake", "fl_set", "echo_ade")
REGIMES = ["echo-good", "echo-lon-wrong only", "echo-lat-wrong only", "echo-both-wrong"]
FL_ORDER = ["FB selected (force_lon)", "FB not selected", "CT selected (force_lon)", "head BRAKE_TO, S6, not GT-cruising",
            "head BRAKE_TO, no full 6-s future", "other"]
METRIC_KEYS = [("d", "ADE 0-2 s delta (os - ha0_ext)"), ("d_lon", "LON (along, ego x) Shapley part"), ("d_lat", "LAT (cross, ego y) Shapley part")]


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, p)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def bin_edges(x, edges):
    lab = np.empty(x.size, dtype=object)
    for lo, hi, name in edges:
        lab[(x >= lo) & (x < hi)] = name
    if any(v is None for v in lab):
        raise SystemExit("[e1_decomp] a value fell outside the pre-registered bin edges")
    return lab


# ------------------------------------------------------------------------------------------ exact lon / lat split
def hyb_ade(a, c):
    return np.sqrt(a * a + c * c).mean(1)


def shapley_split(P1, P0, G, frame=None):
    """d = A(P1) - A(P0) split into (d_lon, d_lat); frame None = ego frame at t0 (x along, y cross), else per-instant unit tangents [n,T,2]."""
    e1, e0 = P1 - G, P0 - G
    if frame is None:
        a1, c1, a0, c0 = e1[..., 0], e1[..., 1], e0[..., 0], e0[..., 1]
    else:
        nrm = np.stack([-frame[..., 1], frame[..., 0]], -1)
        a1, a0 = (e1 * frame).sum(-1), (e0 * frame).sum(-1)
        c1, c0 = (e1 * nrm).sum(-1), (e0 * nrm).sum(-1)
    A11, A00, A10, A01 = hyb_ade(a1, c1), hyb_ade(a0, c0), hyb_ade(a1, c0), hyb_ade(a0, c1)
    d_lon = 0.5 * ((A10 - A00) + (A11 - A01))
    d_lat = 0.5 * ((A01 - A00) + (A11 - A10))
    return d_lon, d_lat, A11 - A00, A11, A00


def gt_tangent_frame(G, min_step=0.05):
    """unit tangent of the GT's own segment ending at each instant (origin first); a stopped segment holds the previous tangent (x-axis at start)."""
    Z = np.concatenate([np.zeros_like(G[:, :1]), G], 1)
    d = np.diff(Z, axis=1)
    n = np.linalg.norm(d, axis=-1)
    out = np.zeros_like(d)
    prev = np.tile(np.array([1.0, 0.0]), (G.shape[0], 1))
    for t in range(d.shape[1]):
        ok = n[:, t] > min_step
        cur = np.where(ok[:, None], d[:, t] / np.where(ok, n[:, t], 1.0)[:, None], prev)
        out[:, t] = cur
        prev = cur
    return out


# ------------------------------------------------------------------------------------------ bootstrap helpers (on s6_lon_decomp's joint draws)
def total_cell(d, BC):
    """E1-style cell of the mean of d on the joint draws: unrounded point / lo / hi / separated."""
    import s6_lon_decomp as SD
    C, ep_idx, n_ep = BC
    den = C @ np.bincount(ep_idx, minlength=n_ep).astype(np.float64)
    tb = (C @ np.bincount(ep_idx, weights=d, minlength=n_ep)) / den
    lo, hi = SD.pct(tb)
    return {"point": float(d.mean()), "lo": lo, "hi": hi, "separated_lt0": bool(hi < 0), "separated_gt0": bool(lo > 0),
            "frac_draws_gt0": float((tb > 0).mean())}


def gross(dec):
    pos = sum(v["contrib"] for v in dec["bins"].values() if v.get("n_windows") and v["contrib"] > 0)
    neg = sum(v["contrib"] for v in dec["bins"].values() if v.get("n_windows") and v["contrib"] < 0)
    return {"sum_positive_contrib": pos, "sum_negative_contrib": neg, "gross_abs": pos - neg}


def random_partition_control(d, labels, order, BC, n_perm=200, seed=12345):
    import s6_lon_decomp as SD
    rng = np.random.default_rng(seed)
    total = float(d.mean())
    cov, sep, nb = 0, 0, 0
    for _ in range(n_perm):
        perm = labels[rng.permutation(labels.size)]
        dec = SD.decompose(d, perm, order, BC)
        for b, v in dec["bins"].items():
            if not v.get("n_windows"):
                continue
            nb += 1
            null = v["frac_windows"] * total
            cov += int(v["contrib_lo"] <= null <= v["contrib_hi"])
            sep += int(v["contrib_separated"])
    return {"n_perm": n_perm, "n_bin_reads": nb, "coverage_of_frac_x_total": cov / nb, "rate_contrib_separated_from_0": sep / nb,
            "PASS": bool(0.88 <= cov / nb <= 0.99)}


# ------------------------------------------------------------------------------------------ per-seed load
def BP_label(seed: int) -> str:
    import r8_battery_patch as BP
    return BP.R7_BASE_LABEL.format(s=seed)


def load_seed(seed: int, tools: dict, rc: dict, comp_json: dict) -> dict:
    import clamp_posthoc_r8 as CP
    import compliant_reread_r8 as CR
    NR, PR, sha12, ra = tools["NR"], tools["PR"], tools["sha12"], tools["ra"]
    raw, lim, linfo, dump = CP.load_seed_inputs(TREE, seed, rc)
    comp = CR.compliant_mask(raw["gt_max_speed_6s"], lim)
    ex = NR.load_extras(os.path.dirname(dump))
    clip = raw["clip_index"].astype(np.int64)
    ws_x = ex["ws"].astype(np.int64)
    if not np.array_equal(clip, ex["clip_index"].astype(np.int64)):
        raise SystemExit("[e1_decomp] raw / extras row order disagree")
    sha = CR.mask_sha(clip, ws_x, comp)
    pdir = f"{ARCHIVE}/{TREE}/panel_s{seed}"
    P = CR.load_panel(NR, PR, sha12, ex, pdir)
    rows, eid, A, dt = P["rows"], P["eid"], P["A"], P["dt"]
    z = np.load(dump)
    tac = np.argmax(z["nav_true.tacv6_lon_logits"][rows], axis=1).astype(np.int64)
    lon_v7 = z["lon_v7"][rows].astype(np.int64)
    vmax = z["v_max_ms"][rows].astype(np.float64)
    gtmax = z["gt_max_speed_6s"][rows].astype(np.float64)
    del z
    man, files, ep_of = NR.panel_layout(pdir)
    cid_of = {int(e["file_index"]): str(e["clip_id"]) for e in man["episodes"]}
    v0, v2, v6, full6, yaw0, yaw2, rot_err, ep_index, ws, csha, ha0, a0 = [], [], [], [], [], [], [], [], [], [], [], []
    g_off = 0
    for f in files:
        fi = NR.fi_of(f)
        zz = np.load(f)
        dd = np.load(os.path.join(pdir, "decisions", os.path.basename(f)))
        if not np.array_equal(zz["ws"], dd["ws"]):
            raise SystemExit(f"[e1_decomp] {f}: panel ws != decisions ws")
        ge, va, pl = dd["gt_future_ext"].astype(np.float64), dd["gt_future_valid_ext"], dd["pose_last"].astype(np.float64)
        if float(va[:, :20].min()) < 1.0:
            raise SystemExit(f"[e1_decomp] {f}: a window lacks a valid 2-s future")
        n = len(zz["ws"])
        v0.append(zz["v0"].astype(np.float64))
        ha0.append(zz["ha0"].astype(np.float32))
        a0.append(dd["ego_t0"][:, 1].astype(np.float64))
        v2.append(ge[:, 19, 3])
        v6.append(ge[:, 59, 3])
        full6.append(va.min(1) >= 1.0)
        yaw0.append(pl[:, 2])
        yaw2.append(ge[:, 19, 2])
        # control X-yaw: R(-yaw0) (world displacement at 2 s) == the ego-frame GT g(2 s)
        dw = ge[:, 19, :2] - pl[:, :2]
        cs, sn = np.cos(pl[:, 2]), np.sin(pl[:, 2])
        ego = np.stack([cs * dw[:, 0] + sn * dw[:, 1], -sn * dw[:, 0] + cs * dw[:, 1]], -1)
        rot_err.append(np.linalg.norm(ego - A["g"][g_off:g_off + n, -1].astype(np.float64), axis=-1))
        g_off += n
        ep_index += [int(ep_of[fi])] * n
        ws += [int(w) for w in zz["ws"]]
        csha += [sha12(cid_of[fi])] * n
    ctx = {"seed": seed, "mask_full": comp[rows], "sha": sha, "n_compliant_dump": int(comp.sum()), "n_dump": int(len(comp)),
           "eid": np.asarray(eid), "dt": dt, "A": A, "tac": tac, "lon_v7": lon_v7, "vmax": vmax, "gtmax": gtmax,
           "v0": np.concatenate(v0), "v2": np.concatenate(v2), "v6": np.concatenate(v6), "full6": np.concatenate(full6),
           "yaw0": np.concatenate(yaw0), "yaw2": np.concatenate(yaw2), "rot_err": np.concatenate(rot_err),
           "ep_index": np.asarray(ep_index), "ws": np.asarray(ws), "csha": np.asarray(csha, dtype=object), "linfo": linfo,
           "ha0": np.concatenate(ha0), "a0": np.concatenate(a0), "refarm": BP_label(seed)}
    ctx["comps_os"] = PR.components(ra, A["os"], A["g"], dt)
    ctx["comps_ha"] = PR.components(ra, A["ha0_ext"], A["g"], dt)
    return ctx


# ------------------------------------------------------------------------------------------ partitions
def build_partitions(cx: dict, m: np.ndarray, fl: dict) -> dict:
    import s6_lon_decomp as SD
    from tanitad.models.vocab_v7 import TACTICAL_LON_ACTIONS_V7 as LONV
    v0, v2, v6, full6 = cx["v0"], cx["v2"], cx["v6"], cx["full6"]

    def dvrule(vend):
        T = {"v0": v0, "v6": vend, "v_max_ms": cx["vmax"], "gt_max_speed_6s": cx["gtmax"], "lon_v7": cx["lon_v7"], "r7_v_max_ms": cx["vmax"]}
        with np.errstate(divide="ignore", invalid="ignore"):
            parts, _dv = SD.partitions(T)
        return parts
    p2 = dvrule(v2)
    p6 = dvrule(np.where(full6, v6, v0))
    dv6 = np.where(full6, p6["dvclass"][0], "no full 6-s future").astype(object)
    # yaw: the world-frame net heading change, or the ego-frame fallback if X-yaw failed
    if float(cx["rot_err"].max()) < 0.05:
        yr = np.abs(wrap(cx["yaw2"] - cx["yaw0"])) / 2.0
        yaw_src = "world yaw (X-yaw PASS)"
    else:
        g = cx["A"]["g"].astype(np.float64)
        seg = g[:, -1] - g[:, -2]
        yr = np.abs(np.arctan2(seg[:, 1], seg[:, 0])) / 1.75
        yaw_src = "ego-frame chord heading fallback (X-yaw FAILED)"
    co, ch = cx["comps_os"], cx["comps_ha"]
    e_al, e_cr = np.asarray(ch["LON_along_abs_final_m"], np.float64), np.asarray(ch["LAT_cross_abs_final_m"], np.float64)
    lw, cw = e_al >= ECHO_ALONG_TOL, e_cr >= ECHO_CROSS_TOL
    reg = np.where(~lw & ~cw, REGIMES[0], np.where(lw & ~cw, REGIMES[1], np.where(~lw & cw, REGIMES[2], REGIMES[3]))).astype(object)
    cru = np.where(p2["dvclass"][0] == "cruising", "dv2 cruising", "dv2 not cruising")
    sxd = np.asarray([f"{a} | {b}" for a, b in zip(reg, cru)], dtype=object)
    tacn = np.asarray([LONV[k] for k in cx["tac"]], dtype=object)
    brake_id = list(LONV).index("BRAKE_TO")
    tb = np.where(cx["tac"] == brake_id, "head BRAKE_TO", "head not BRAKE_TO").astype(object)
    lv = cx["lon_v7"]
    tcorr = np.where(lv < 0, "unlabelled", np.where(cx["tac"] == lv, "head == lon_v7", "head != lon_v7")).astype(object)
    eade = bin_edges(np.asarray(ch["ade_m"], np.float64), ECHO_ADE_EDGES)
    pl8 = SD.plan_lon_class(cx["A"]["os"], v0, cx["dt"])
    plg = SD.plan_lon_class(cx["A"]["g"], v0, cx["dt"])
    p2g = np.asarray([f"GT {g} / refcv8 {p}" for g, p in zip(plg, pl8)], dtype=object)
    LC = SD.LONCLS
    parts = {"dv2": (p2["dvclass"][0], SD.DV_ORDER), "dv6": (dv6, SD.DV_ORDER + ["no full 6-s future"]),
             "v0bin": (p2["v0bin"][0], p2["v0bin"][1]), "yawbin": (bin_edges(yr, YAW_EDGES), [n for _a, _b, n in YAW_EDGES]),
             "tac_lon": (tacn, list(LONV)), "tac_brake": (tb, ["head BRAKE_TO", "head not BRAKE_TO"]),
             "tac_correct": (tcorr, ["head == lon_v7", "head != lon_v7", "unlabelled"]),
             "echo_regime": (reg, REGIMES), "echo_ade": (eade, [n for _a, _b, n in ECHO_ADE_EDGES]),
             "steady_x_dv": (sxd, [f"{a} | {b}" for a in REGIMES for b in ("dv2 cruising", "dv2 not cruising")]),
             "fl_set": (fl["labels"], FL_ORDER),
             "plan2_vs_gt2": (p2g, [f"GT {g} / refcv8 {p}" for g in LC for p in LC]),
             "a0bin": (bin_edges(np.abs(cx["a0"]), A0_EDGES), [n for _a, _b, n in A0_EDGES])}
    sub = {k: (np.asarray(lab)[m], order) for k, (lab, order) in parts.items()}
    return sub, {"yaw_source": yaw_src, "x_yaw_max_err_m": float(cx["rot_err"].max()), "x_yaw_p99_err_m": float(np.percentile(cx["rot_err"], 99)),
                 "yaw_rate_all": yr, "dv6_all": dv6, "dv2_all": p2["dvclass"][0]}


def force_lon_sets(cx0: dict, dv6_all: np.ndarray) -> dict:
    """FB_all / CT pool re-derived on ALL S2 windows from the seed-0 head (force_lon's definition) + the plan's selected windows."""
    from tanitad.models.vocab_v7 import TACTICAL_LON_ACTIONS_V7 as LONV
    brake_id = list(LONV).index("BRAKE_TO")
    full6 = cx0["full6"]
    cruis6 = full6 & (dv6_all == "cruising")
    hb = cx0["tac"] == brake_id
    fb_all = cruis6 & hb
    ct_pool = cruis6 & ~hb
    plan = jload(str(FL_PLAN))
    key = {(int(e), int(w)): i for i, (e, w) in enumerate(zip(cx0["ep_index"], cx0["ws"]))}
    sel = {"FB": np.zeros(len(full6), bool), "CT": np.zeros(len(full6), bool)}
    miss, sha_bad, set_bad = 0, 0, 0
    for p in plan:
        i = key.get((int(p["ep_index"]), int(p["ws"])))
        if i is None:
            miss += 1
            continue
        if cx0["csha"][i] != p["clip_sha12"]:
            sha_bad += 1
        sel[p["set"]][i] = True
        if p["set"] == "FB" and not fb_all[i]:
            set_bad += 1
        if p["set"] == "CT" and not ct_pool[i]:
            set_bad += 1
    lab = np.full(len(full6), "other", dtype=object)
    lab[hb & ~full6] = "head BRAKE_TO, no full 6-s future"
    lab[hb & full6 & ~cruis6] = "head BRAKE_TO, S6, not GT-cruising"
    lab[fb_all & ~sel["FB"]] = "FB not selected"
    lab[sel["FB"]] = "FB selected (force_lon)"
    lab[sel["CT"]] = "CT selected (force_lon)"
    st = jload(str(FL_STATS))
    chk = {"FB_all_rederived": int(fb_all.sum()), "FB_all_select_stats": int(st["FB_all"]), "CT_pool_rederived": int(ct_pool.sum()),
           "CT_pool_select_stats": int(st["CT_pool"]), "cruising6_rederived": int(cruis6.sum()), "cruising6_select_stats": int(st["cruising_windows"]),
           "n_plan": len(plan), "n_plan_missing": miss, "n_plan_clip_sha12_mismatch": sha_bad, "n_plan_not_in_its_set": set_bad,
           "n_FB_selected": int(sel["FB"].sum()), "n_CT_selected": int(sel["CT"].sum())}
    chk["PASS"] = bool(chk["FB_all_rederived"] == chk["FB_all_select_stats"] and chk["CT_pool_rederived"] == chk["CT_pool_select_stats"]
                       and chk["cruising6_rederived"] == chk["cruising6_select_stats"] and miss == 0 and sha_bad == 0 and set_bad == 0
                       and chk["n_FB_selected"] == int(st["FB_used"]) and chk["n_CT_selected"] == int(st["CT_used"]))
    return {"labels": lab, "fb_all": fb_all, "sel_fb": sel["FB"], "sel_ct": sel["CT"], "hb": hb, "chk": chk}


# ------------------------------------------------------------------------------------------ one seed's analysis
def analyse_seed(cx: dict, fl: dict, comp_json: dict) -> dict:  # noqa: C901
    import pair_read_r8 as PR
    import s6_lon_decomp as SD
    seed = cx["seed"]
    m = cx["mask_full"]
    e = cx["eid"][m]
    co, ch = cx["comps_os"], cx["comps_ha"]
    os_ade, ha_ade = np.asarray(co["ade_m"], np.float64)[m], np.asarray(ch["ade_m"], np.float64)[m]
    d = os_ade - ha_ade
    out = {"seed": seed, "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(e))), "controls": {}}
    # ---- R: exact reproduction of the compliant E1 cell
    cell = PR.paired_cell(os_ade, ha_ade, e, N_BOOT)
    flds = ("delta", "lo", "hi", "n_windows", "n_episodes", "separated", "n_dropped_nonfinite")
    S2c = comp_json["seeds"][str(seed)]["surfaces"]["S2"]["compliant"]
    rep = {"reread": {f: cell[f] for f in flds}}
    for v in ("capped", "uncapped"):
        b = S2c[v]["rows"]["ade_m"]["os_minus_ha0_ext"]
        rep[f"banked_{v}"] = {f: b.get(f) for f in flds}
        rep[f"exact_vs_banked_{v}"] = bool(all(cell[f] == b.get(f) for f in flds))
    lit = BRIEF_E1[seed]
    rep["exact_vs_brief_literal"] = bool(all(cell[f] == lit[f] for f in lit) and cell["separated"] is False)
    BC = SD.boot_counts(e)
    jt = total_cell(d, BC)
    rep["joint_draws_total"] = jt
    rep["joint_equals_ci_at_4dp"] = bool(round(jt["point"], 4) == cell["delta"] and round(jt["lo"], 4) == cell["lo"] and round(jt["hi"], 4) == cell["hi"])
    out["controls"]["R"] = rep
    out["controls"]["R_PASS"] = bool(rep["exact_vs_banked_capped"] and rep["exact_vs_banked_uncapped"] and rep["exact_vs_brief_literal"]
                                     and rep["joint_equals_ci_at_4dp"])
    out["E1_cell"] = cell
    # ---- K: capped == uncapped on the compliant set (from the compliant tree json)
    sd = comp_json["seeds"][str(seed)]
    out["controls"]["K"] = {"n_changed_plans_on_compliant_windows": sd["n_changed_plans_on_compliant_windows"],
                            "n_binding_on_compliant_windows": sd["n_binding_on_compliant_windows"],
                            "PASS": bool(sd["n_changed_plans_on_compliant_windows"] == 0 and sd["n_binding_on_compliant_windows"] == 0)}
    # ---- the lon / lat split (exact Shapley) in the ego frame + the GT-tangent frame
    Pos, Pha, G = (cx["A"][k][m].astype(np.float64) for k in ("os", "ha0_ext", "g"))
    d_lon, d_lat, d_np, A_os, A_ha = shapley_split(Pos, Pha, G)
    t_lon, t_lat, _t_np, _a, _b = shapley_split(Pos, Pha, G, frame=gt_tangent_frame(G))
    out["controls"]["S"] = {"max_abs_lon_plus_lat_minus_d": float(np.abs(d_lon + d_lat - d_np).max()),
                            "max_abs_tangent_lon_plus_lat_minus_d": float(np.abs(t_lon + t_lat - d_np).max()),
                            "max_abs_numpy_vs_components_ade_os": float(np.abs(A_os - os_ade).max()),
                            "max_abs_numpy_vs_components_ade_ha": float(np.abs(A_ha - ha_ade).max())}
    s = out["controls"]["S"]
    s["PASS"] = bool(s["max_abs_lon_plus_lat_minus_d"] < 1e-9 and s["max_abs_tangent_lon_plus_lat_minus_d"] < 1e-9
                     and s["max_abs_numpy_vs_components_ade_os"] < 1e-4 and s["max_abs_numpy_vs_components_ade_ha"] < 1e-4)
    D = {"d": d, "d_lon": d_lon, "d_lat": d_lat}
    out["lonlat_totals"] = {k: total_cell(v, BC) for k, v in D.items()}
    out["lonlat_totals"]["tangent_frame_d_lon"] = total_cell(t_lon, BC)
    out["lonlat_totals"]["tangent_frame_d_lat"] = total_cell(t_lat, BC)
    # four-family context cells (os - echo) on the compliant set, same estimator
    ctxm = {}
    for k in ("LON_along_mae_m", "LAT_cross_mae_m", "LON_speed_mae_mps", "LON_along_abs_final_m", "LAT_cross_abs_final_m", "LAT_heading_mae_deg",
              "fde_m"):
        if k in co:
            ctxm[k] = PR.paired_cell(np.asarray(co[k], np.float64)[m], np.asarray(ch[k], np.float64)[m], e, N_BOOT)
    out["component_cells_os_minus_echo"] = ctxm
    # ---- context arms for the per-bin levels (numpy ADE; refcv7-50,400 = the panel's embedded arm, ha0 = constant velocity from v0)
    from tanitad.models.vocab_v7 import TACTICAL_LON_ACTIONS_V7 as LONV
    cv_ade = np.linalg.norm(cx["ha0"][m].astype(np.float64) - G, axis=-1).mean(1)
    r7_ade = np.linalg.norm(cx["A"][cx["refarm"]][m].astype(np.float64) - G, axis=-1).mean(1)
    sa_os, sa_ha = Pos[:, -1, 0] - G[:, -1, 0], Pha[:, -1, 0] - G[:, -1, 0]
    tac_m, lv7_m = cx["tac"][m], cx["lon_v7"][m]
    out["cv_echo_cells"] = {"os_minus_ha0_CV": PR.paired_cell(os_ade, cv_ade, e, N_BOOT), "ha0_ext_minus_ha0_CV": PR.paired_cell(ha_ade, cv_ade, e, N_BOOT),
                            "refcv7_minus_ha0_ext": PR.paired_cell(r7_ade, ha_ade, e, N_BOOT)}
    # ---- partitions
    parts, pinfo = build_partitions(cx, m, fl)
    out["_labels_for_levers"] = {k: parts[k][0].tolist() for k in ("echo_regime", "dv2", "v0bin", "steady_x_dv", "a0bin")}
    if seed == 0:
        dv2c, sxd = parts["dv2"][0], parts["steady_x_dv"][0]
        ep_m, ws_m, cs_m = cx["ep_index"][m], cx["ws"][m], cx["csha"][m]
        sets = {"S2A_slow_or_stopping (dv2 slow_both | stopping)": np.isin(dv2c, ["slow_both", "stopping"]),
                "S2B_decelerating (dv2 decelerating)": dv2c == "decelerating",
                "S2C_echo_good_speed_changing (echo-good & dv2 not cruising)": sxd == "echo-good | dv2 not cruising"}
        out["s2_widen_sets"] = {}
        for nm, sm in sets.items():
            out["s2_widen_sets"][nm] = {"n_windows": int(sm.sum()), "n_episodes": int(len(np.unique(e[sm]))),
                                        "contrib_seed0": float(d[sm].sum() / d.size),
                                        "head_argmax_hist": {LONV[k]: int(((tac_m == k) & sm).sum()) for k in range(len(LONV)) if ((tac_m == k) & sm).any()},
                                        "lon_v7_hist": {("unlabelled" if k < 0 else LONV[k]): int(((lv7_m == k) & sm).sum()) for k in sorted(set(lv7_m[sm].tolist()))},
                                        "windows": [{"clip_sha12": str(c), "ep_index": int(a), "ws": int(w), "dv2": str(v), "head_argmax_v7": int(h), "lon_v7": int(l)}
                                                    for c, a, w, v, h, l, k in zip(cs_m, ep_m, ws_m, dv2c, tac_m, lv7_m, sm) if k]}
    out["yaw_source"] = {k: pinfo[k] for k in ("yaw_source", "x_yaw_max_err_m", "x_yaw_p99_err_m")}
    out["partitions"] = {}
    for pname, (lab, order) in parts.items():
        ent = {}
        for k, _l in METRIC_KEYS:
            dec = SD.decompose(D[k], lab, order, BC)
            dec["gross"] = gross(dec)
            ent[k] = dec
        lv = {}
        for b in order:
            mb = lab == b
            if not mb.any():
                continue
            lv[b] = {"n": int(mb.sum()), "echo_ade_mean": float(ha_ade[mb].mean()), "refcv8_ade_mean": float(os_ade[mb].mean()),
                     "frac_refcv8_worse": float((d[mb] > 0).mean())}
            if pname in LEVEL_PARTS:
                dn = cv_ade[mb] - ha_ade[mb]
                lv[b].update({"cv_ha0_ade_mean": float(cv_ade[mb].mean()), "refcv7_ade_mean": float(r7_ade[mb].mean()),
                              "refcv8_along_signed_2s_mean": float(sa_os[mb].mean()), "echo_along_signed_2s_mean": float(sa_ha[mb].mean()),
                              "frac_refcv8_too_far_2s": float((sa_os[mb] > 0).mean()),
                              "cv_capture": (float((cv_ade[mb] - os_ade[mb]).mean() / dn.mean()) if abs(dn.mean()) > 1e-9 else None),
                              "head_argmax_hist": {LONV[k]: int(((tac_m == k) & mb).sum()) for k in range(len(LONV)) if ((tac_m == k) & mb).any()},
                              "head_eq_lon_v7_frac_of_labelled": (float((tac_m[mb] == lv7_m[mb])[lv7_m[mb] >= 0].mean()) if (lv7_m[mb] >= 0).any() else None),
                              "lon_v7_hist": {("unlabelled" if k < 0 else LONV[k]): int(((lv7_m == k) & mb).sum()) for k in sorted(set(lv7_m[mb].tolist()))}})
        ent["_levels"] = lv
        out["partitions"][pname] = ent
    # ---- Z: identity contrasts in every partition and every metric
    zok = True
    zero = np.zeros_like(d)
    for pname, (lab, order) in parts.items():
        for zname, zd in (("echo_vs_itself", ha_ade - ha_ade), ("refcv8_vs_itself", os_ade - os_ade), ("lon_split_identity", shapley_split(Pha, Pha, G)[0]),
                          ("lat_split_identity", shapley_split(Pos, Pos, G)[1])):
            dec = SD.decompose(zd, lab, order, BC)
            for b, v in dec["bins"].items():
                if v.get("n_windows") and any(x != 0.0 for x in (v["contrib"], v["contrib_lo"], v["contrib_hi"], v["mean_delta"], v["mean_lo"], v["mean_hi"])):
                    zok = False
            if any(x != 0.0 for x in (dec["total"]["point"], dec["total"]["lo"], dec["total"]["hi"])):
                zok = False
    del zero
    out["controls"]["Z_PASS"] = bool(zok)
    # ---- P: random partitions
    out["controls"]["P"] = random_partition_control(d, parts["v0bin"][0], parts["v0bin"][1], BC)
    # ---- counterfactual lever arithmetic: every bin with a positive contribution, plus the force_lon sets
    cf = {}
    for pname, (lab, order) in parts.items():
        if pname == "plan2_vs_gt2":
            continue
        for b in order:
            mb = lab == b
            if not mb.any():
                continue
            c_b = float(d[mb].sum() / d.size)
            if c_b <= 0 and pname != "fl_set":
                continue
            sh = np.where(mb, d - d[mb].mean(), d)
            ti = np.where(mb, 0.0, d)
            cf[f"{pname} :: {b}"] = {"n": int(mb.sum()), "contrib": c_b, "shift": total_cell(sh, BC), "tie": total_cell(ti, BC)}
    # union levers: everything that loses in the head-BRAKE_TO set, the whole force_lon FB, steady regime
    unions = {"head BRAKE_TO (all compliant)": parts["tac_brake"][0] == "head BRAKE_TO",
              "force_lon FB_all (selected + not selected)": np.isin(parts["fl_set"][0], ["FB selected (force_lon)", "FB not selected"]),
              "echo-good (steady) all": parts["echo_regime"][0] == "echo-good",
              "echo-good AND dv2 cruising": parts["steady_x_dv"][0] == "echo-good | dv2 cruising"}
    for nm, mb in unions.items():
        c_b = float(d[mb].sum() / d.size)
        cf[f"UNION :: {nm}"] = {"n": int(mb.sum()), "contrib": c_b, "shift": total_cell(np.where(mb, d - d[mb].mean(), d), BC),
                                "tie": total_cell(np.where(mb, 0.0, d), BC)}
    # the E1 gap: the uniform per-window improvement needed to make hi < 0 (CI width held) -- arithmetic, not a result
    out["gap_to_separation_m"] = float(-jt["hi"]) if jt["hi"] is not None else None
    out["counterfactuals"] = cf
    out["n_s2_compliant_without_full6"] = int((~cx["full6"][m]).sum())
    return out


def lever_ens2(keep: dict, labels: dict, PR) -> dict:
    """the 2-draw ensemble (mean of the seed-0 and seed-1 os plans) vs the echo on the compliant S2 set -- ONE read."""
    import s6_lon_decomp as SD
    k0, k1 = keep[0], keep[1]
    for a in ("g", "ha", "mask", "eid", "ws"):
        if not np.array_equal(k0[a], k1[a]):
            raise SystemExit(f"[e1_decomp] ENS2: seed panels differ in {a}")
    m, e, G = k0["mask"], k0["eid"][k0["mask"]], k0["g"][k0["mask"]]

    def ade(P):
        return np.linalg.norm(P[m] - G, axis=-1).mean(1)
    a0_, a1_, ah = ade(k0["os"]), ade(k1["os"]), ade(k0["ha"])
    ae = ade(0.5 * (k0["os"] + k1["os"]))
    BC = SD.boot_counts(e)
    out = {"definition": "per-window mean of the seed-0 and seed-1 refcv8 os plans (EMA-30k), scored on the compliant S2 set vs ha0_ext; numpy ADE",
           "status": "EARLY single read, zero GPU; consumes BOTH inference seeds, so the bar's inference replicate needs two more seeds (GPU)",
           "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(e))),
           "triangle_inequality_holds_every_window": bool((ae <= 0.5 * (a0_ + a1_) + 1e-12).all()),
           "per_window_seed_to_seed_abs_ade_diff_mean": float(np.abs(a0_ - a1_).mean()),
           "per_window_seed_to_seed_abs_ade_diff_p90": float(np.percentile(np.abs(a0_ - a1_), 90)),
           "frac_windows_seed_ade_differs_gt_0p05": float((np.abs(a0_ - a1_) > 0.05).mean()),
           "plan_l2_seed0_vs_seed1_mean_m": float(np.linalg.norm(k0["os"][m] - k1["os"][m], axis=-1).mean()),
           "E1_ens2_minus_echo": PR.paired_cell(ae, ah, e, N_BOOT),
           "E1_seed0_numpy": PR.paired_cell(a0_, ah, e, N_BOOT), "E1_seed1_numpy": PR.paired_cell(a1_, ah, e, N_BOOT),
           "ens2_minus_seed0": PR.paired_cell(ae, a0_, e, N_BOOT), "ens2_minus_seed1": PR.paired_cell(ae, a1_, e, N_BOOT),
           "decomposition": {}}
    for p in ("echo_regime", "dv2", "v0bin"):
        order = sorted(set(labels[p].tolist()), key=lambda x: str(x))
        dec = SD.decompose(ae - ah, labels[p], order, BC)
        dec["gross"] = gross(dec)
        out["decomposition"][p] = dec
    return out


def lever_retime(keep: dict, labels: dict, PR) -> dict:
    """2-fold EPISODE cross-fitted speed offset (s6_lon_decomp.retime, reused) conditioned on INFERENCE-ADMISSIBLE information only:
    the measured v0 bin x refcv8's own tactical-lon head argmax.  Fit on one fold's mean ADE (refcv8 alone; the echo never enters the fit),
    applied to the other fold.  DIAGNOSTIC (fit on eval episodes), NOT deployable: a deployable version is fit on TRAIN."""
    import s6_lon_decomp as SD
    from tanitad.models.vocab_v7 import TACTICAL_LON_ACTIONS_V7 as LONV
    out = {"definition": "plan moved along its own path by beta m/s, beta per (v0bin x head argmax) group, 2 episode folds (eid % 2), grid -2..2 step 0.05",
           "status": "DIAGNOSTIC EARLY read (fit on eval episodes, scored only on the other fold); not deployable", "seeds": {}}
    for sd, k in keep.items():
        m = k["mask"]
        G, e, dt = k["g"][m], k["eid"][m], k["dt"]
        P = k["os"][m]
        ah = np.linalg.norm(k["ha"][m] - G, axis=-1).mean(1)
        grp = np.asarray([f"{a} | {LONV[t]}" for a, t in zip(labels["v0bin"], k["tac"][m])], dtype=object)
        fold = (e % 2).astype(int)
        Rg = np.stack([np.linalg.norm(SD.retime(P, float(b), dt).astype(np.float64) - G, axis=-1).mean(1) for b in SD.BETA_GRID])
        beta_w = np.zeros(len(e))
        chosen = {}
        for f in (0, 1):
            fit, app = fold != f, fold == f
            for g in np.unique(grp):
                gm = grp == g
                fm = fit & gm
                if fm.sum() < 10:
                    continue
                b = float(SD.BETA_GRID[int(np.argmin(Rg[:, fm].mean(1)))])
                chosen[f"fold{f}:{g}"] = {"beta": b, "n_fit": int(fm.sum())}
                beta_w[app & gm] = b
        Pc = np.empty_like(P)
        for b in np.unique(beta_w):
            mm = beta_w == b
            Pc[mm] = SD.retime(P[mm], float(b), dt)
        ac = np.linalg.norm(Pc.astype(np.float64) - G, axis=-1).mean(1)
        a0_ = np.linalg.norm(P - G, axis=-1).mean(1)
        out["seeds"][str(sd)] = {"E1_calibrated_minus_echo": PR.paired_cell(ac, ah, e, N_BOOT), "calibrated_minus_raw": PR.paired_cell(ac, a0_, e, N_BOOT),
                                 "E1_raw_numpy": PR.paired_cell(a0_, ah, e, N_BOOT), "betas": chosen,
                                 "known_value_beta0_identity_max_abs": float(np.abs(SD.retime(P, 0.0, dt).astype(np.float64) - P).max())}
    return out


def lever_milestones(k0: dict, labels: dict, PR, sha12) -> dict:
    """per-bin refcv8 ADE of the EMA milestones (seed 0) on the SAME compliant S2 windows (keys asserted), vs the echo."""
    import s6_lon_decomp as SD
    m = k0["mask"]
    G, e = k0["g"][m], k0["eid"][m]
    ah = np.linalg.norm(k0["ha"][m] - G, axis=-1).mean(1)
    key30 = [(str(a), int(b)) for a, b in zip(k0["csha"].tolist(), k0["ws"].tolist())]
    BC = SD.boot_counts(e)
    out = {"definition": "EMA milestones, inference seed 0, panel `os`, numpy ADE on the compliant S2 windows of step30000ema (same keys)", "trees": {}}
    for t in MILESTONES:
        pdir = f"{ARCHIVE}/{t}/panel_s0"
        if not os.path.exists(os.path.join(pdir, "manifest.json")):
            out["trees"][t] = {"status": "ABSENT"}
            continue
        _man, A, _eid, keys = PR.load_panel_keyed(pdir, sha12)
        if [(str(a), int(b)) for a, b in keys] != key30:
            out["trees"][t] = {"status": "WINDOW KEYS DIFFER from step30000ema -- not read"}
            continue
        if not np.array_equal(A["g"].astype(np.float64), k0["g"]) or not np.array_equal(A["ha0_ext"].astype(np.float64), k0["ha"]):
            out["trees"][t] = {"status": "GT or echo differ from step30000ema -- not read"}
            continue
        ao = np.linalg.norm(A["os"].astype(np.float64)[m] - G, axis=-1).mean(1)
        rec = {"status": "OK", "E1_os_minus_echo": PR.paired_cell(ao, ah, e, N_BOOT), "bins": {}}
        for p in ("echo_regime", "dv2", "v0bin", "steady_x_dv"):
            order = sorted(set(labels[p].tolist()), key=lambda x: str(x))
            dec = SD.decompose(ao - ah, labels[p], order, BC)
            rec["bins"][p] = {b: {"n": v["n_windows"], "contrib": v.get("contrib"), "contrib_lo": v.get("contrib_lo"), "contrib_hi": v.get("contrib_hi"),
                                  "refcv8_ade_mean": float(ao[labels[p] == b].mean()), "echo_ade_mean": float(ah[labels[p] == b].mean())}
                              for b, v in dec["bins"].items() if v.get("n_windows")}
        out["trees"][t] = rec
        del A
    return out


def compute() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[e1_decomp] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    t0 = time.time()
    import clamp_posthoc_r8 as CP
    NR, BP, PR, ra, sha12 = CP.get_readers()
    tools = {"NR": NR, "PR": PR, "ra": ra, "sha12": sha12}
    rc = CP.run_cfg()
    comp_json = jload(str(COMPLIANT_JSON))
    if not comp_json["controls"]["ALL_PASS"]:
        raise SystemExit("[e1_decomp] the compliant reread's own controls did not pass: refusing")
    res = {"tool": "e1_decomp_r8.py", "generated": time.strftime("%FT%T"), "tier": TIER, "estimator": QUESTION, "n_boot": N_BOOT,
           "boot_seed": BOOT_SEED, "tree": f"{ARCHIVE}/{TREE}", "surface": "S2 0-2 s (instants 0.5/1.0/1.5/2.0 s, dt 0.5)",
           "bar": "BAR-R8-E1: refcv8 os - ha0_ext, S2 ADE 0-2 s, < 0 and separated at both inference seeds, ceiling-compliant windows (A-R8-CLAMP-2, b6f7fac)",
           "status": "EARLY READ of a banked milestone (EMA step 30,000); the bars are decided at step 48,852",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "inputs": {"compliant_tree_json": str(COMPLIANT_JSON), "force_lon_plan": str(FL_PLAN), "force_lon_select_stats": str(FL_STATS)},
           "partition_rules": __doc__.split("PRE-REGISTERED PARTITIONS")[1].split("ESTIMATOR.")[0].strip(), "seeds": {}, "controls": {}}
    fl, ctl_ok, seed_res, keep = None, True, {}, {}
    for seed in SEEDS:
        cx = load_seed(seed, tools, rc, comp_json)
        mk = {"sha": cx["sha"], "sha_equal_expected": cx["sha"] == MASK_SHA, "n_compliant_dump": cx["n_compliant_dump"], "n_dump": cx["n_dump"],
              "n_compliant_panel": int(cx["mask_full"].sum()), "n_panel": int(len(cx["mask_full"])),
              "n_episodes_compliant": int(len(np.unique(cx["eid"][cx["mask_full"]])))}
        mk["PASS"] = bool(mk["sha_equal_expected"] and mk["n_compliant_panel"] == 4159 and mk["n_panel"] == 4754 and mk["n_episodes_compliant"] == 138)
        if seed == 0:
            _p, pinfo = build_partitions(cx, cx["mask_full"], {"labels": np.full(len(cx["mask_full"]), "other", dtype=object)})
            fl = force_lon_sets(cx, pinfo["dv6_all"])
            fl0_tac = cx["tac"].copy()
            fl0_key = np.stack([cx["ep_index"], cx["ws"]])
        else:
            # the force_lon labels were built on the seed-0 panel order: the seed-1 panel must hold the SAME windows in the SAME order
            if not np.array_equal(np.stack([cx["ep_index"], cx["ws"]]), fl0_key):
                raise SystemExit("[e1_decomp] seed-1 panel window order != seed-0: refusing to carry the seed-0 labels across")
            fl["head_argmax_differs_seed1_vs_seed0"] = int((cx["tac"] != fl0_tac).sum())
        r = analyse_seed(cx, fl, comp_json)
        r["controls"]["M"] = mk
        seed_res[seed] = r
        keep[seed] = {"os": cx["A"]["os"].astype(np.float64), "g": cx["A"]["g"].astype(np.float64), "ha": cx["A"]["ha0_ext"].astype(np.float64),
                      "mask": cx["mask_full"].copy(), "eid": cx["eid"].copy(), "csha": cx["csha"].copy(), "ws": cx["ws"].copy(),
                      "tac": cx["tac"].copy(), "dt": cx["dt"]}
        print(f"[{time.time() - t0:6.1f}s] seed {seed} done  R={r['controls']['R_PASS']} S={r['controls']['S']['PASS']} Z={r['controls']['Z_PASS']} "
              f"P={r['controls']['P']['PASS']} K={r['controls']['K']['PASS']} M={mk['PASS']}", flush=True)
        del cx
    labels = {k: np.asarray(v, dtype=object) for k, v in seed_res[0].pop("_labels_for_levers").items()}
    seed_res[1].pop("_labels_for_levers", None)
    res["levers"] = {"ENS2": lever_ens2(keep, labels, PR), "MILESTONE": lever_milestones(keep[0], labels, PR, sha12),
                     "RT_XFIT": lever_retime(keep, labels, PR)}
    print(f"[{time.time() - t0:6.1f}s] levers done", flush=True)
    res["seeds"] = {str(k): v for k, v in seed_res.items()}
    res["controls"]["F_force_lon_sets"] = fl["chk"]
    res["controls"]["F_head_argmax_differs_seed1_vs_seed0"] = fl.get("head_argmax_differs_seed1_vs_seed0")
    for k in ("R_PASS", "Z_PASS"):
        res["controls"][k] = bool(all(v["controls"][k] for v in seed_res.values()))
    for k in ("S", "P", "K", "M"):
        res["controls"][f"{k}_PASS"] = bool(all(v["controls"][k]["PASS"] for v in seed_res.values()))
    res["controls"]["F_PASS"] = bool(fl["chk"]["PASS"])
    res["controls"]["ALL_PASS"] = bool(all(res["controls"][f"{k}_PASS"] for k in ("R", "Z", "S", "P", "K", "M", "F")))
    res["wall_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    if not res["controls"]["ALL_PASS"]:
        jdump(res, OUT / "CONTROL_FAILED.json")
        print("CONTROL FAILED", json.dumps({k: v for k, v in res["controls"].items() if k.endswith("_PASS")}))
        return 3
    jdump(res, OUT / "E1_DECOMP.json")
    render()
    print("WROTE", OUT / "E1_DECOMP.json", f"{res['wall_s']}s")
    return 0


# ------------------------------------------------------------------------------------------ markdown
def f4(x, dp=4):
    return "n/a" if x is None else f"{x:+.{dp}f}"


def ccell(v, dp=4):
    if not v or not v.get("n_windows"):
        return "-"
    sep = " SEP" if v["contrib_separated"] else ""
    return f"{f4(v['contrib'], dp)} [{f4(v['contrib_lo'], dp)}, {f4(v['contrib_hi'], dp)}]{sep}"


def tcell(t, dp=4):
    s = ""
    if t["separated_lt0"]:
        s = " SEP<0"
    elif t["separated_gt0"]:
        s = " SEP>0"
    return f"{f4(t['point'], dp)} [{f4(t['lo'], dp)}, {f4(t['hi'], dp)}]{s}"


def render() -> int:
    R = jload(str(OUT / "E1_DECOMP.json"))
    L = [f"# E1 decomposition -- refcv8 EMA-30k vs the hold-action echo on the ceiling-compliant S2 set", "",
         f"Tool `code/e1_decomp_r8.py` (sha256 `{R['script_sha256'][:16]}`), generated {R['generated']}.  **{R['tier']}.**  {R['status']}.",
         f"Estimator: {R['estimator']}.", f"Bar: {R['bar']}.", "",
         "Controls: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in R["controls"].items() if k.endswith("_PASS")), ""]
    for sd, S in R["seeds"].items():
        c = S["E1_cell"]
        L += [f"## Inference seed {sd}", "",
              f"E1 cell (reproduced exactly): **{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]**, separated {c['separated']}, "
              f"n {c['n_windows']:,} windows / {c['n_episodes']} episodes.  Gap to separation (hi -> < 0): {S['gap_to_separation_m']:+.4f} m.", "",
              "| split | total [95 % CI] |", "|---|---|"]
        for k, t in S["lonlat_totals"].items():
            L.append(f"| {k} | {tcell(t)} |")
        L += ["", "| component (os - echo) | delta [CI] | n |", "|---|---|---|"]
        for k, c2 in S["component_cells_os_minus_echo"].items():
            L.append(f"| {k} | {c2['delta']:+.4f} [{c2['lo']:+.4f}, {c2['hi']:+.4f}]{' SEP' if c2['separated'] else ''} | {c2['n_windows']} |")
        L.append("")
        for pname, ent in S["partitions"].items():
            g = ent["d"]["gross"]
            L += [f"### {pname} (seed {sd})  -- gross: losses {g['sum_positive_contrib']:+.4f}, wins {g['sum_negative_contrib']:+.4f} m", "",
                  "| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for b, v in ent["d"]["bins"].items():
                if not v.get("n_windows"):
                    continue
                lv = ent["_levels"][b]
                sh = "n/a" if v["share"] is None else f"{100 * v['share']:.0f} %"
                L.append(f"| {b} | {v['n_windows']} / {v['n_episodes']} | {100 * v['frac_windows']:.1f} % | {ccell(v)} | {sh} | "
                         f"{ccell(ent['d_lon']['bins'][b])} | {ccell(ent['d_lat']['bins'][b])} | {f4(v['mean_delta'])} [{f4(v['mean_lo'])}, {f4(v['mean_hi'])}] | "
                         f"{lv['echo_ade_mean']:.4f} | {lv['refcv8_ade_mean']:.4f} | {lv['frac_refcv8_worse']:.2f} |")
            L.append("")
        L += [f"### Counterfactual lever arithmetic (seed {sd}) -- NOT results: what E1 would read if refcv8's excess in the bin were removed", "",
              "| bin | n | contrib | SHIFT (excess removed, spread kept) | TIE (== echo on the bin) |", "|---|---|---|---|---|"]
        for k, v in sorted(S["counterfactuals"].items(), key=lambda kv: -kv[1]["contrib"]):
            L.append(f"| {k} | {v['n']} | {v['contrib']:+.4f} | {tcell(v['shift'])} | {tcell(v['tie'])} |")
        L.append("")
    S0 = R["seeds"]["0"]
    L += ["## Per-bin levels (seed 0): echo / refcv8 / constant-velocity `ha0` / refcv7-50,400 ADE, signed along-track at 2 s, head argmax", "",
          "| partition :: bin | n | echo | refcv8 | CV ha0 | refcv7 | refcv8 signed along 2 s | echo signed along 2 s | CV-capture | head argmax | GT lon_v7 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for pname in LEVEL_PARTS:
        for b, lv in S0["partitions"][pname]["_levels"].items():
            if "cv_ha0_ade_mean" not in lv:
                continue
            cap = "n/a" if lv["cv_capture"] is None else f"{lv['cv_capture']:.2f}"
            L.append(f"| {pname} :: {b} | {lv['n']} | {lv['echo_ade_mean']:.3f} | {lv['refcv8_ade_mean']:.3f} | {lv['cv_ha0_ade_mean']:.3f} | "
                     f"{lv['refcv7_ade_mean']:.3f} | {lv['refcv8_along_signed_2s_mean']:+.3f} | {lv['echo_along_signed_2s_mean']:+.3f} | {cap} | "
                     f"{lv['head_argmax_hist']} | {lv['lon_v7_hist']} |")
    L.append("")
    LV = R.get("levers", {})
    if LV:
        E = LV["ENS2"]
        L += ["## Lever arms run in this turn (zero GPU, banked plans)", "", f"**ENS2** -- {E['definition']}.  {E['status']}.", "",
              f"seed-to-seed per-window |ADE diff| mean {E['per_window_seed_to_seed_abs_ade_diff_mean']:.4f} m (p90 {E['per_window_seed_to_seed_abs_ade_diff_p90']:.4f}); "
              f"{100 * E['frac_windows_seed_ade_differs_gt_0p05']:.1f} % of windows differ by > 0.05 m; plan L2 seed0-vs-seed1 {E['plan_l2_seed0_vs_seed1_mean_m']:.4f} m; "
              f"triangle inequality holds on every window: {E['triangle_inequality_holds_every_window']}.", ""]
        for k in ("E1_ens2_minus_echo", "E1_seed0_numpy", "E1_seed1_numpy", "ens2_minus_seed0", "ens2_minus_seed1"):
            c = E[k]
            L.append(f"* {k}: {c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] separated {c['separated']} (n {c['n_windows']} / {c['n_episodes']})")
        RT = LV["RT_XFIT"]
        L += ["", f"**RT_XFIT** -- {RT['definition']}.  {RT['status']}.", ""]
        for sd, r in RT["seeds"].items():
            for k in ("E1_calibrated_minus_echo", "calibrated_minus_raw"):
                c = r[k]
                L.append(f"* seed {sd} {k}: {c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] separated {c['separated']}")
        MI = LV["MILESTONE"]
        L += ["", f"**MILESTONE** -- {MI['definition']}.", "", "| tree | E1 (seed 0) | echo-good | echo-good & dv2 cruising | echo-good & dv2 not cruising | slow_both | stopping | decelerating | v0<2 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for t, rec in MI["trees"].items():
            if rec.get("status") != "OK":
                L.append(f"| {t} | {rec.get('status')} |")
                continue
            c = rec["E1_os_minus_echo"]
            g = lambda p, b: f"{rec['bins'][p][b]['contrib']:+.4f} ({rec['bins'][p][b]['refcv8_ade_mean']:.3f})"  # noqa: E731
            L.append(f"| {t} | {c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | {g('echo_regime', 'echo-good')} | {g('steady_x_dv', 'echo-good | dv2 cruising')} | "
                     f"{g('steady_x_dv', 'echo-good | dv2 not cruising')} | {g('dv2', 'slow_both')} | {g('dv2', 'stopping')} | {g('dv2', 'decelerating')} | {g('v0bin', 'v0<2')} |")
        L.append("")
    if "s2_widen_sets" in S0:
        L += ["## S2 widening sets for the next force roll (seed-0 head; the window lists are in the JSON, sha12-keyed)", "",
              "| set | n win / ep | contrib (seed 0) | head argmax | GT lon_v7 |", "|---|---|---|---|---|"]
        for nm, w in S0["s2_widen_sets"].items():
            L.append(f"| {nm} | {w['n_windows']} / {w['n_episodes']} | {w['contrib_seed0']:+.4f} | {w['head_argmax_hist']} | {w['lon_v7_hist']} |")
        L.append("")
    L += ["## Controls (detail)", "", "```", json.dumps(R["controls"], indent=1, default=str)[:6000], "```", ""]
    for sd, S in R["seeds"].items():
        L += [f"seed {sd}: " + json.dumps({k: (v if not isinstance(v, dict) or len(json.dumps(v)) < 600 else {kk: vv for kk, vv in v.items() if kk != 'joint_draws_total'})
                                           for k, v in S["controls"].items()}, default=str)[:3000], "", f"yaw source: {S['yaw_source']}", ""]
    md = "\n".join(L)
    va = OUT / "VERDICT_ANALYST.md"
    if va.exists():
        md = va.read_text(encoding="utf-8") + "\n\n---\n\n" + md
    (OUT / "E1_DECOMP.md").write_text(md, encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["compute", "render"])
    a = ap.parse_args()
    return compute() if a.stage == "compute" else render()


if __name__ == "__main__":
    sys.exit(main())
