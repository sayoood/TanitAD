"""refcv8 v9-CONSTRAINT-head accuracy vs controls (STEP 2b), CPU only, read-only on every input.

What exists: refcv8's trainer EVALUATES the 26-dim v9 constraint heads (lat_c 12 / lon_c 10 / speed 4) every 500 steps
on 8 FIXED eval batches (16 windows each = 128 windows, 80 of the 139 eval episodes; g_ev seed 12345) and LOGS the
mean |pred - GT| per group in the heads' NORMALISED units (metrics.jsonl `ema_eval_r8_v9_mae_*`; EMA weights).  Per-window
head outputs were never banked, so the HEAD side is the logged aggregate; the CONTROL side is computed here on the very
same 128 windows (the trainer's own cached batches), under the trainer's own mask and reduction.

Definitions are SOURCE-LITERAL (stack/tanitad/refs/refcv8_conditioning.py @ launch commit fb47a58, lines 746-822):
  normalised target = clamp(raw / scale, +-6); supervised iff finite, the row's v9 class is EXACT (v7 id >= 0), and the
  turn columns 1..9 only on a TURN class (v7 6/7); the head is read on the GT-ACTIVE class's query (teacher-forced).
Controls (all on the same masked entries, same unit):
  GT-as-prediction (must be exactly 0) | constant = TRAIN mean per field (the asked control) | TRAIN median |
  class-conditional TRAIN mean | ego-scalar ridge floor (v0, a0 fit on TRAIN only; lambda by clip-grouped 5-fold CV on
  TRAIN) | the same with a degree-2 polynomial (nonlinear floor).  Yaw rate is NOT in the v9 release, so it is not a
  floor feature on the TRAIN side (stated in RESULT.md).
Estimator: episode-cluster bootstrap over the 80 episodes present (taniteval.ci's own draw generator), n_boot 2000,
seed 0.  It answers another draw of EPISODES at this checkpoint only; it says nothing about another training run.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, "E:/Projects/TanitAD/taniteval")
from taniteval import ci as TCI  # noqa: E402

OUT = "C:/Users/Admin/refcv8_pf/tacconstr"
TRAIN_REL = "C:/Users/Admin/qland/hold_v9train4572/work/release_train4572/v9_labels_train.npz"
TRAIN_MD5 = "83e30bd3791f0767707eade5f4671759"            # == --r8-v9-md5 in refcv8-r101-s0.argv.json @ fb47a58
G0 = "C:/Users/Admin/refcv8_eval_archive/g0cache/step40000"
METRICS = "C:/Users/Admin/qland/work/watch8/metrics.jsonl"
N_BOOT, SEED = 2000, 0

LAT_F = ("lat_theta_deg", "turn_t_start_s", "turn_t_end_s", "turn_d_start_m", "turn_len_m", "turn_dyaw_deg",
         "turn_r_arc_m", "turn_exit_x_m", "turn_exit_y_m", "turn_exit_psi_deg", "lc_t_cross_s", "lc_d_cross_m")
LON_F = ("lon_v_target_ms", "lon_t_reach_s", "lon_d_reach_m", "lon_t_start_s", "stop_x_m", "stop_y_m", "lead_gap_m",
         "lead_tg_s", "lead_gap_min_m", "lead_tg_min_s")
SPD_F = ("v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
SCALE = np.array((90.0, 8.0, 8.0, 50.0, 50.0, 90.0, 50.0, 50.0, 50.0, 90.0, 8.0, 50.0,
                  15.0, 8.0, 50.0, 8.0, 50.0, 50.0, 50.0, 8.0, 50.0, 8.0,
                  15.0, 15.0, 15.0, 15.0), dtype=np.float32)
CLAMP = 6.0
FIELDS = LAT_F + LON_F + SPD_F
G_SL = {"lat_c": slice(0, 12), "lon_c": slice(12, 22), "speed": slice(22, 26)}
# the v8F-mini 26-dim code groups, expressed on refcv8's three heads (8 + 6 + 12 = 26)
V8F_GROUP_COLS = {"latcons": list(range(0, 12)), "lonspeed": list(range(12, 16)) + list(range(22, 26)),
                  "stoplead": list(range(16, 22))}
TURN_COLS = list(range(1, 10))


def norm_targets(lat_c, lon_c, spd, lat_cls, lon_cls):
    """Source-literal v9_constraint_targets (numpy float32, same op order as the torch code)."""
    raw = np.concatenate([lat_c, lon_c, spd], 1).astype(np.float32)
    fin = np.isfinite(raw)
    y = np.nan_to_num(np.clip(raw / SCALE, -CLAMP, CLAMP), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    m = fin.copy()
    lat_cls = np.asarray(lat_cls).reshape(-1)
    lon_cls = np.asarray(lon_cls).reshape(-1)
    m[:, 0:12] &= (lat_cls >= 0)[:, None]
    is_turn = np.isin(lat_cls, (6, 7))
    tcols = np.zeros(12, bool)
    tcols[TURN_COLS] = True
    m[:, 0:12] &= (~tcols[None, :] | is_turn[:, None])
    m[:, 12:] &= (lon_cls >= 0)[:, None]
    return y, m


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ TRAIN (release)
assert md5(TRAIN_REL) == TRAIN_MD5, "train release md5 != the one refcv8 launched with"
Z = np.load(TRAIN_REL, allow_pickle=True)
lat_c = np.stack([Z["row__" + f] for f in LAT_F], 1)
lon_c = np.stack([Z["row__" + f] for f in LON_F], 1)
spd_c = np.stack([Z["row__" + f] for f in SPD_F], 1)
lat_cls_tr, lon_cls_tr = Z["row__lat_v7id_a"], Z["row__lon_v7id"]          # variant 'a' = --r8-v9-lat-variant a
Ytr, Mtr = norm_targets(lat_c, lon_c, spd_c, lat_cls_tr, lon_cls_tr)
row0, nrow = Z["clip__row0"], Z["clip__n_rows"]
N_TR = Ytr.shape[0]
clip_of_row = np.repeat(np.arange(len(row0)), nrow)
assert clip_of_row.shape[0] == N_TR
v_now = Z["row__v_now_ms"].astype(np.float64)
DT = float(np.median(Z["clip__dt_s"]))
a0_tr = np.full(N_TR, np.nan)
same = np.zeros(N_TR, bool)
same[2:] = clip_of_row[2:] == clip_of_row[:-2]
a0_tr[2:] = (v_now[2:] - v_now[:-2]) / (2 * DT)
a0_tr[~same] = np.nan
ego_ok_tr = np.isfinite(v_now) & np.isfinite(a0_tr)

# ------------------------------------------------------------------------------------------------ EVAL (G0 cache)
import torch  # noqa: E402

B = []
for i in range(8):
    b = torch.load(f"{G0}/batch_{i}.pt", map_location="cpu", mmap=True, weights_only=False)
    keep = {k: b[k].numpy() for k in ("r8_v9_lat_c", "r8_v9_lon_c", "r8_v9_speed", "lat_v7", "lon_v7", "episode_id",
                                       "pose_last", "pose_hist")}
    B.append(keep)
cat = {k: np.concatenate([x[k] for x in B], 0) for k in B[0]}
batch_id = np.repeat(np.arange(8), [x["lat_v7"].shape[0] for x in B])
Yev, Mev = norm_targets(cat["r8_v9_lat_c"], cat["r8_v9_lon_c"], cat["r8_v9_speed"], cat["lat_v7"], cat["lon_v7"])
eid = np.asarray([str(x) for x in cat["episode_id"]])
N_EV = Yev.shape[0]
v0_ev = cat["pose_last"][:, 3].astype(np.float64)
a0_ev = ((cat["pose_hist"][:, -1, 3] - cat["pose_hist"][:, -3, 3]) / (2 * DT)).astype(np.float64)

# ---- count cross-check against the TRAINER's own logged counts (a different channel: metrics.jsonl)
rows = [json.loads(l) for l in open(METRICS, encoding="utf-8") if l.strip()]
last_ema = [r for r in rows if r.get("step") == 48852 and "ema_eval_r8_v9_mae_lat_c" in r][-1]
cnt = {g: [int(Mev[batch_id == b, s].sum()) for b in range(8)] for g, s in G_SL.items()}
logged_n = {"lat_c": last_ema["ema_eval_n_v9_lat_c"], "lon_c": last_ema["ema_eval_n_v9_lon_c"],
            "speed": last_ema["ema_eval_n_v9_speed"]}
count_check = {g: {"mine_mean_per_batch": float(np.mean(cnt[g])), "logged": logged_n[g],
                   "match": bool(abs(np.mean(cnt[g]) - logged_n[g]) < 1e-6)} for g in G_SL}
assert all(c["match"] for c in count_check.values()), f"mask replication != trainer counts: {count_check}"


# ------------------------------------------------------------------------------------------------ reductions
def group_err(pred, g):
    """per-entry |pred - y| restricted to the group's supervised entries -> (abs_err [N, d], mask [N, d])."""
    s = G_SL[g]
    return np.abs(pred[:, s] - Yev[:, s]) * Mev[:, s], Mev[:, s]


def trainer_reduction(pred, g):
    """mean over the 8 batches of (sum |e| m / max(sum m, 1)) -- the trainer's own eval reduction."""
    e, m = group_err(pred, g)
    return float(np.mean([e[batch_id == b].sum() / max(m[batch_id == b].sum(), 1.0) for b in range(8)]))


def pooled(pred, g):
    e, m = group_err(pred, g)
    return float(e.sum() / max(m.sum(), 1.0))


UNIQ, IDX = TCI.episode_index(eid)
DRAWS = list(TCI._draws(UNIQ, IDX, N_BOOT, SEED))      # the SAME episode resamples for every arm -> paired by construction


def boot_pooled(pred, g, pred_b=None):
    """episode-cluster bootstrap of the pooled entry-mean MAE (or, with pred_b, of the paired difference pred - pred_b)."""
    e, m = group_err(pred, g)
    es, ms = e.sum(1), m.sum(1)
    if pred_b is not None:
        eb, mb = group_err(pred_b, g)
        ebs = eb.sum(1)
    vals = []
    for d in DRAWS:
        den = max(ms[d].sum(), 1.0)
        v = es[d].sum() / den
        if pred_b is not None:
            v = (es[d].sum() - ebs[d].sum()) / den
        vals.append(v)
    vals = np.asarray(vals)
    pt = float(es.sum() / max(ms.sum(), 1.0)) if pred_b is None else float((es.sum() - ebs.sum()) / max(ms.sum(), 1.0))
    return {"point": pt, "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "n_boot": N_BOOT, "n_episodes": int(len(UNIQ)), "estimator": "episode_cluster_bootstrap (taniteval.ci draws)"}


# ------------------------------------------------------------------------------------------------ the controls
def tile(vec):
    return np.tile(vec[None, :].astype(np.float32), (N_EV, 1))


controls = {}
controls["GT_as_prediction"] = Yev.copy()
# TRAIN mean per field over supervised TRAIN entries; recomputed a second, independent way and asserted equal
mean_tr = np.array([Ytr[Mtr[:, f], f].astype(np.float64).mean() for f in range(26)])
mean_tr2 = (Ytr.astype(np.float64) * Mtr).sum(0) / Mtr.sum(0)
assert np.allclose(mean_tr, mean_tr2, atol=1e-9), "train-mean two derivations disagree"
med_tr = np.array([np.median(Ytr[Mtr[:, f], f]) for f in range(26)])
controls["const_TRAIN_mean"] = tile(mean_tr)
controls["const_TRAIN_median"] = tile(med_tr)
# class-conditional TRAIN mean (lat_c by the lateral class, lon_c/speed by the longitudinal class)
cc = np.tile(mean_tr[None, :].astype(np.float32), (N_EV, 1))
for cls in np.unique(lat_cls_tr[lat_cls_tr >= 0]):
    sel = cat["lat_v7"] == cls
    for f in range(0, 12):
        mk = Mtr[:, f] & (lat_cls_tr == cls)
        if mk.sum() > 0:
            cc[sel, f] = Ytr[mk, f].astype(np.float64).mean()
for cls in np.unique(lon_cls_tr[lon_cls_tr >= 0]):
    sel = cat["lon_v7"] == cls
    for f in range(12, 26):
        mk = Mtr[:, f] & (lon_cls_tr == cls)
        if mk.sum() > 0:
            cc[sel, f] = Ytr[mk, f].astype(np.float64).mean()
controls["const_classcond_TRAIN_mean"] = cc


# ridge floors fit on TRAIN only
def feats(v, a, poly):
    x = np.stack([v, a], 1)
    if poly:
        x = np.concatenate([x, x ** 2, (v * a)[:, None]], 1)
    return x


def fit_ridge_cv(f, poly, lams=(1e-2, 1e0, 1e2, 1e4), k=5):
    """per-field ridge on TRAIN entries (supervised & ego defined). lambda by clip-grouped k-fold CV MAE on TRAIN."""
    sel = Mtr[:, f] & ego_ok_tr
    X = feats(v_now[sel], a0_tr[sel], poly)
    y = Ytr[sel, f].astype(np.float64)
    g = clip_of_row[sel]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    Xs = np.concatenate([Xs, np.ones((len(Xs), 1))], 1)
    fold = (g % k)
    best, best_l = None, None
    for lam in lams:
        err = 0.0
        for q in range(k):
            tr, te = fold != q, fold == q
            A = Xs[tr].T @ Xs[tr] + lam * np.diag(np.r_[np.ones(Xs.shape[1] - 1), 0.0])
            w = np.linalg.solve(A, Xs[tr].T @ y[tr])
            err += np.abs(Xs[te] @ w - y[te]).sum()
        if best is None or err < best:
            best, best_l = err, lam
    A = Xs.T @ Xs + best_l * np.diag(np.r_[np.ones(Xs.shape[1] - 1), 0.0])
    w = np.linalg.solve(A, Xs.T @ y)
    return w, mu, sd, best_l, int(sel.sum()), X.shape[1]


for tag, poly in (("ridge_v0a0_TRAINfit", False), ("ridge_v0a0_poly2_TRAINfit", True)):
    P = np.zeros((N_EV, 26), np.float32)
    meta = {}
    X_ev = feats(v0_ev, a0_ev, poly)
    for f in range(26):
        w, mu, sd, lam, n, d = fit_ridge_cv(f, poly)
        P[:, f] = (np.concatenate([(X_ev - mu) / sd, np.ones((N_EV, 1))], 1) @ w).astype(np.float32)
        meta[FIELDS[f]] = {"lambda": lam, "n_train_entries": n, "d": d}
    controls[tag] = P
    controls[tag + "__meta"] = meta
CTRL_NAMES = [k for k in controls if not k.endswith("__meta")]

# ------------------------------------------------------------------------------------------------ the table
head_log = {"lat_c": last_ema["ema_eval_r8_v9_mae_lat_c"], "lon_c": last_ema["ema_eval_r8_v9_mae_lon_c"],
            "speed": last_ema["ema_eval_r8_v9_mae_speed"]}
raw_row = [r for r in rows if r.get("step") == 48852 and "eval_r8_v9_mae_lat_c" in r and "ema_eval_r8_v9_mae_lat_c" in r][-1]
head_raw = {"lat_c": raw_row["eval_r8_v9_mae_lat_c"], "lon_c": raw_row["eval_r8_v9_mae_lon_c"], "speed": raw_row["eval_r8_v9_mae_speed"]}

table = {}
for g in G_SL:
    t = {"n_windows": N_EV, "n_episodes": int(len(UNIQ)),
         "n_supervised_entries": int(Mev[:, G_SL[g]].sum()), "d_outputs": int(G_SL[g].stop - G_SL[g].start),
         "head_EMA48852_trainer_logged_normalised_MAE": head_log[g],
         "head_RAW48852_trainer_logged_normalised_MAE": head_raw[g], "controls": {}}
    for c in CTRL_NAMES:
        t["controls"][c] = {"trainer_reduction": trainer_reduction(controls[c], g), "pooled": boot_pooled(controls[c], g)}
        if c != "GT_as_prediction":
            t["controls"][c]["head_over_control_ratio"] = head_log[g] / t["controls"][c]["trainer_reduction"]
            t["controls"][c]["head_gap_closed_vs_control"] = 1.0 - head_log[g] / t["controls"][c]["trainer_reduction"]
    # paired: const-mean minus ridge, same windows (does ego dynamics alone beat the constant?)
    t["paired_const_minus_ridge_v0a0"] = boot_pooled(controls["const_TRAIN_mean"], g, controls["ridge_v0a0_TRAINfit"])
    t["paired_const_minus_classcond"] = boot_pooled(controls["const_TRAIN_mean"], g, controls["const_classcond_TRAIN_mean"])
    table[g] = t

# per-field controls (the head has no per-field log): constant & ridge MAE per field, and which v8F code group it belongs to
field_tab = []
for f in range(26):
    e_const = (np.abs(controls["const_TRAIN_mean"][:, f] - Yev[:, f]) * Mev[:, f]).sum() / max(Mev[:, f].sum(), 1)
    e_ridge = (np.abs(controls["ridge_v0a0_TRAINfit"][:, f] - Yev[:, f]) * Mev[:, f]).sum() / max(Mev[:, f].sum(), 1)
    e_cc = (np.abs(controls["const_classcond_TRAIN_mean"][:, f] - Yev[:, f]) * Mev[:, f]).sum() / max(Mev[:, f].sum(), 1)
    grp = [k for k, v in V8F_GROUP_COLS.items() if f in v][0]
    field_tab.append({"field": FIELDS[f], "head": ("lat_c" if f < 12 else "lon_c" if f < 22 else "speed"), "v8f_group": grp,
                      "n_entries": int(Mev[:, f].sum()), "scale": float(SCALE[f]),
                      "const_MAE_norm": float(e_const), "const_MAE_physical": float(e_const * SCALE[f]),
                      "classcond_MAE_norm": float(e_cc), "ridge_v0a0_MAE_norm": float(e_ridge)})

# ------------------------------------------------------------------------------------------------ controls that must read known values
ctl = {}
ctl["GT_as_prediction_MAE_all_groups"] = {g: float(np.abs(controls["GT_as_prediction"][:, G_SL[g]] - Yev[:, G_SL[g]]).__mul__(Mev[:, G_SL[g]]).sum()) for g in G_SL}
ctl["GT_exactly_zero"] = all(v == 0.0 for v in ctl["GT_as_prediction_MAE_all_groups"].values())
ctl["const_equals_TRAIN_mean"] = bool(np.allclose(controls["const_TRAIN_mean"][0], mean_tr.astype(np.float32), atol=1e-7)
                                      and np.all(controls["const_TRAIN_mean"] == controls["const_TRAIN_mean"][0]))
ctl["mask_count_vs_trainer_log"] = count_check
ctl["train_release_md5"] = TRAIN_MD5
ctl["n_train_rows"], ctl["n_train_clips"] = int(N_TR), int(len(row0))
ctl["ego_ok_train_rows"] = int(ego_ok_tr.sum())
# discriminating control: a MUTANT head (the constant shifted by 1 sd) must read WORSE than the constant, and a head that
# is GT + noise(0.05) must read ~0.04; both through the same reduction -> the reduction is live, not a constant
rng = np.random.default_rng(0)
mut = controls["const_TRAIN_mean"] + 1.0
noi = Yev + rng.normal(0, 0.05, Yev.shape).astype(np.float32)
ctl["mutant_const_plus_1_worse_than_const"] = {g: trainer_reduction(mut, g) > trainer_reduction(controls["const_TRAIN_mean"], g) for g in G_SL}
ctl["gt_plus_noise0p05_reads"] = {g: trainer_reduction(noi, g) for g in G_SL}

# ------------------------------------------------------------------------------------------------ the logged trajectory
traj = []
for r in rows:
    if "ema_eval_r8_v9_mae_lat_c" in r:
        traj.append({"step": r["step"], "lat_c": r["ema_eval_r8_v9_mae_lat_c"], "lon_c": r["ema_eval_r8_v9_mae_lon_c"],
                     "speed": r["ema_eval_r8_v9_mae_speed"], "huber": r["ema_eval_r8_v9_cons"]})
tr_rows = [r for r in rows if "r8_v9_mae_lat_c" in r and "ema_eval_r8_v9_mae_lat_c" not in r and r.get("step", 0) > 47000]
train_tail = {k: float(np.mean([r[f"r8_v9_mae_{k}"] for r in tr_rows])) for k in ("lat_c", "lon_c", "speed")} if tr_rows else None

result = {"schema": "tacconstr-r8-v1",
          "what": "refcv8 v9-constraint heads: trainer-logged EMA-48852 normalised MAE (GT-active class query, T0) vs controls on the SAME 128 windows",
          "tier": "T0 (teacher-forced: the head is read on the GT-active class's query). NOT driving performance.",
          "evidence": {"head": "INHERITED trainer log (metrics.jsonl step 48852, ema_eval_*) -- aggregate only, no per-window values",
                       "controls": "MEASURED here (G0 cache of the same 8 fixed eval batches; v9 train release md5 83e30bd3...)"},
          "estimator": f"episode_cluster_bootstrap B={N_BOOT} seed={SEED} over {len(UNIQ)} episodes (controls only); answers another draw of EPISODES; "
                       "the head has NO interval (one aggregate number); one training seed",
          "n_windows": N_EV, "n_episodes": int(len(UNIQ)), "table": table, "per_field_controls": field_tab, "controls": ctl,
          "ridge_meta": {k: controls[k + "__meta"] for k in ("ridge_v0a0_TRAINfit", "ridge_v0a0_poly2_TRAINfit")},
          "head_logged_trajectory_ema": traj, "head_train_tail_mean_last_1k_steps": train_tail,
          "head_log_source": METRICS}
json.dump(result, open(f"{OUT}/TACCONSTR_R8_v9heads.json", "w"), indent=1)

# ------------------------------------------------------------------------------------------------ print
print("controls:", json.dumps({k: ctl[k] for k in ("GT_exactly_zero", "const_equals_TRAIN_mean")}))
print("counts:", json.dumps(count_check))
print("mutant worse:", ctl["mutant_const_plus_1_worse_than_const"], " gt+noise:", ctl["gt_plus_noise0p05_reads"])
for g in G_SL:
    t = table[g]
    print(f"\n{g}: n_win {t['n_windows']} ep {t['n_episodes']} entries {t['n_supervised_entries']} d {t['d_outputs']}")
    print(f"  HEAD EMA-48852 (trainer log): {t['head_EMA48852_trainer_logged_normalised_MAE']:.4f}   raw weights: {t['head_RAW48852_trainer_logged_normalised_MAE']:.4f}")
    for c in CTRL_NAMES:
        x = t["controls"][c]
        extra = "" if c == "GT_as_prediction" else f"  head/ctrl {x['head_over_control_ratio']:.3f}  gap-closed {x['head_gap_closed_vs_control']:+.3f}"
        print(f"  {c:30s} trainer-red {x['trainer_reduction']:.4f}  pooled {x['pooled']['point']:.4f} [{x['pooled']['lo']:.4f},{x['pooled']['hi']:.4f}]{extra}")
    p = t["paired_const_minus_ridge_v0a0"]
    print(f"  paired const-ridge(v0,a0): {p['point']:+.4f} [{p['lo']:+.4f},{p['hi']:+.4f}]")
    p = t["paired_const_minus_classcond"]
    print(f"  paired const-classcond   : {p['point']:+.4f} [{p['lo']:+.4f},{p['hi']:+.4f}]")
print("\ntrain tail (last 1k steps, trainer train rows):", train_tail)
print("per-field:")
for r in field_tab:
    print(f"  {r['field']:16s} {r['head']:6s} {r['v8f_group']:8s} n {r['n_entries']:3d} const {r['const_MAE_norm']:.3f} ({r['const_MAE_physical']:.2f}) cc {r['classcond_MAE_norm']:.3f} ridge {r['ridge_v0a0_MAE_norm']:.3f}")
