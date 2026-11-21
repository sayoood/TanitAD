"""refcv8 EMA-48,852 tactical CONSTRAINT heads, PER WINDOW, on the 4,754 per-frame eval windows (139 episodes) vs GT and
controls.  CPU only; read-only on every input.

HEAD OUTPUTS: E:/refcv8_eval_kit/videos/heads_48852_pf/heads_ep000..138.npz (a post-battery `dump_heads` pass, EMA-48,852,
REAL_CHECKPOINT_ON_REAL_CORPUS, perframe_full nav feed; the BATTERY dumps do NOT carry them -- STEP1_DUMP_AUDIT.json).
  v9 heads  out_r8_v9_lat_c [35,8,12] / out_r8_v9_lon_c [35,8,10] / out_r8_v9_speed [35,8,4]  (normalised v9 units, one row per v7 class)
  probe     out_r8_cons_lat [35,8,2] / out_r8_cons_lon [35,8,2]                               (psi_term/pi, t_onset/6 | log1p(prog6)/log1p(120), v_end/30)
GT: v9 EVAL release (md5 6b5c7f20..., the one refcv8 trained/evaluated with); window -> row k = ws + 2 (v_now_ms vs dump v0:
mean |d| printed; class labels agree except where the battery masked a window).  The head is read on the GT-ACTIVE class's query
(the trainer's rule, `v9_constraint_loss`), so this is T0 / teacher-forced.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "C:/Users/Admin/refcv8_pf/navsim_ckpt/thor_bundle/stage/tree/stack/scripts")
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/navsim_ckpt/thor_bundle/stage/tree/stack")
import torch  # noqa: E402

from tanitad.refs import refcv8_conditioning as r8c  # noqa: E402
import refb_labels  # noqa: E402

OUT = "C:/Users/Admin/refcv8_pf/tacconstr"
HEADS = "E:/refcv8_eval_kit/videos/heads_48852_pf"
BATT = "C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/dump_s0/decisions"
EVAL_REL = "C:/Users/Admin/qland/work/refcv8_hf_v9/release/v9_labels_eval139.npz"
EVAL_MD5 = "6b5c7f207cffc3b7eb3cd527fd433599"
TRAIN_REL = "C:/Users/Admin/qland/hold_v9train4572/work/release_train4572/v9_labels_train.npz"
TRAIN_MD5 = "83e30bd3791f0767707eade5f4671759"
TRAINFIT = "C:/Users/Admin/refcv8_pf/trainfit/step30000ema_train139_pf/dump_s0/decisions"
N_BOOT, SEED = 2000, 0
OFFSET = 2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hashlib  # noqa: E402


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


assert md5(EVAL_REL) == EVAL_MD5 and md5(TRAIN_REL) == TRAIN_MD5

LAT_F = ("lat_theta_deg", "turn_t_start_s", "turn_t_end_s", "turn_d_start_m", "turn_len_m", "turn_dyaw_deg",
         "turn_r_arc_m", "turn_exit_x_m", "turn_exit_y_m", "turn_exit_psi_deg", "lc_t_cross_s", "lc_d_cross_m")
LON_F = ("lon_v_target_ms", "lon_t_reach_s", "lon_d_reach_m", "lon_t_start_s", "stop_x_m", "stop_y_m", "lead_gap_m",
         "lead_tg_s", "lead_gap_min_m", "lead_tg_min_s")
SPD_F = ("v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
FIELDS = LAT_F + LON_F + SPD_F
SCALE = np.array((90.0, 8.0, 8.0, 50.0, 50.0, 90.0, 50.0, 50.0, 50.0, 90.0, 8.0, 50.0, 15.0, 8.0, 50.0, 8.0, 50.0, 50.0, 50.0, 8.0,
                  50.0, 8.0, 15.0, 15.0, 15.0, 15.0), np.float32)
G_SL = {"lat_c": slice(0, 12), "lon_c": slice(12, 22), "speed": slice(22, 26)}
V8F = {"latcons": list(range(0, 12)), "lonspeed": list(range(12, 16)) + list(range(22, 26)), "stoplead": list(range(16, 22))}
ALLG = {**{k: list(range(v.start, v.stop)) for k, v in G_SL.items()}, **V8F, "ALL26": list(range(26))}


def norm_targets(lat_c, lon_c, spd, lat_cls, lon_cls):
    raw = np.concatenate([lat_c, lon_c, spd], 1).astype(np.float32)
    fin = np.isfinite(raw)
    y = np.nan_to_num(np.clip(raw / SCALE, -6.0, 6.0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    m = fin.copy()
    m[:, 0:12] &= (lat_cls >= 0)[:, None]
    tcols = np.zeros(12, bool)
    tcols[1:10] = True
    m[:, 0:12] &= (~tcols[None, :] | np.isin(lat_cls, (6, 7))[:, None])
    m[:, 12:] &= (lon_cls >= 0)[:, None]
    return y, m


def rows_of(R):
    lc = np.stack([R["row__" + f] for f in LAT_F], 1)
    oc = np.stack([R["row__" + f] for f in LON_F], 1)
    sc = np.stack([R["row__" + f] for f in SPD_F], 1)
    return lc, oc, sc


# ------------------------------------------------------------------------------------------- EVAL windows
RE = np.load(EVAL_REL, allow_pickle=True)
row0, k0, nrow = RE["clip__row0"], RE["clip__k0"], RE["clip__n_rows"]
H = {k: [] for k in ("v9lat", "v9lon", "v9spd", "pcl", "pco")}
W = {k: [] for k in ("ep", "ws", "ridx", "v0h", "pose_last", "gt_fut", "gt_fv", "ego_t0", "lat_pred", "lon_pred", "lat_label", "lon_label", "v0b")}
for ep in range(139):
    h = np.load(f"{HEADS}/heads_ep{ep:03d}.npz", allow_pickle=True)
    b = np.load(f"{BATT}/ep{ep:03d}.npz", allow_pickle=True)
    assert np.array_equal(h["ws"], b["ws"]), f"ws differ in ep {ep}"
    H["v9lat"].append(h["out_r8_v9_lat_c"]); H["v9lon"].append(h["out_r8_v9_lon_c"]); H["v9spd"].append(h["out_r8_v9_speed"])
    H["pcl"].append(h["out_r8_cons_lat"]); H["pco"].append(h["out_r8_cons_lon"])
    n = len(h["ws"])
    W["ep"].append(np.full(n, ep)); W["ws"].append(h["ws"]); W["ridx"].append(row0[ep] + (h["ws"] + OFFSET - k0[ep]))
    W["v0h"].append(h["v0"]); W["pose_last"].append(b["pose_last"]); W["gt_fut"].append(b["gt_future_ext"]); W["gt_fv"].append(b["gt_future_valid_ext"])
    W["ego_t0"].append(b["ego_t0"]); W["lat_pred"].append(b["lat_pred_nav_true"]); W["lon_pred"].append(b["lon_pred_nav_true"])
    W["lat_label"].append(b["lat_label"]); W["lon_label"].append(b["lon_label"])
    W["v0b"].append(b["pose_last"][:, 3])
H = {k: np.concatenate(v, 0) for k, v in H.items()}
W = {k: np.concatenate(v, 0) for k, v in W.items()}
N = len(W["ws"])
assert N == 4754
ridx = W["ridx"]
assert np.all((ridx >= 0) & (ridx < len(RE["row__k"])))
v_rel = RE["row__v_now_ms"][ridx]
join_check = {"n_windows": int(N), "mean_abs_v0_dump_vs_release_v_now": float(np.abs(v_rel - W["v0h"]).mean()),
              "max_abs": float(np.abs(v_rel - W["v0h"]).max()), "battery_pose_v_vs_heads_v0_max_abs": float(np.abs(W["v0b"] - W["v0h"]).max())}
lat_cls = RE["row__lat_v7id_a"][ridx].astype(np.int64)
lon_cls = RE["row__lon_v7id"][ridx].astype(np.int64)
lc, oc, sc = rows_of(RE)
Yev, Mev = norm_targets(lc[ridx], oc[ridx], sc[ridx], lat_cls, lon_cls)
# independent derivation control: the SOURCE torch function on the same raw rows must give identical targets/masks
tg = r8c.v9_constraint_targets(torch.as_tensor(lc[ridx], dtype=torch.float32), torch.as_tensor(oc[ridx], dtype=torch.float32),
                               torch.as_tensor(sc[ridx], dtype=torch.float32), torch.as_tensor(lat_cls), torch.as_tensor(lon_cls))
Ysrc = torch.cat([tg["lat"], tg["lon"], tg["speed"]], 1).numpy()
Msrc = torch.cat([tg["lat_m"], tg["lon_m"], tg["speed_m"]], 1).numpy()
join_check["numpy_targets_equal_source_torch_targets"] = bool(np.array_equal(Msrc, Mev) and np.allclose(Ysrc[Mev], Yev[Mev], atol=0.0))

# head prediction on the GT-active class's query
ar = np.arange(N)
P_head = np.concatenate([H["v9lat"][ar, np.clip(lat_cls, 0, None)], H["v9lon"][ar, np.clip(lon_cls, 0, None)],
                         H["v9spd"][ar, np.clip(lon_cls, 0, None)]], 1).astype(np.float32)

# ------------------------------------------------------------------------------------------- TRAIN controls
RT = np.load(TRAIN_REL, allow_pickle=True)
lt, ot, st = rows_of(RT)
Ytr, Mtr = norm_targets(lt, ot, st, RT["row__lat_v7id_a"], RT["row__lon_v7id"])
lat_cls_tr, lon_cls_tr = RT["row__lat_v7id_a"], RT["row__lon_v7id"]
mean_tr = np.array([Ytr[Mtr[:, f], f].astype(np.float64).mean() for f in range(26)])
mean_tr2 = (Ytr.astype(np.float64) * Mtr).sum(0) / Mtr.sum(0)
assert np.allclose(mean_tr, mean_tr2, atol=1e-9)
med_tr = np.array([np.median(Ytr[Mtr[:, f], f]) for f in range(26)])
C = {}
C["const_TRAIN_mean"] = np.tile(mean_tr[None].astype(np.float32), (N, 1))
C["const_TRAIN_median"] = np.tile(med_tr[None].astype(np.float32), (N, 1))
cc = C["const_TRAIN_mean"].copy()
for cls in np.unique(lat_cls_tr[lat_cls_tr >= 0]):
    for f in range(0, 12):
        mk = Mtr[:, f] & (lat_cls_tr == cls)
        if mk.any():
            cc[lat_cls == cls, f] = Ytr[mk, f].astype(np.float64).mean()
for cls in np.unique(lon_cls_tr[lon_cls_tr >= 0]):
    for f in range(12, 26):
        mk = Mtr[:, f] & (lon_cls_tr == cls)
        if mk.any():
            cc[lon_cls == cls, f] = Ytr[mk, f].astype(np.float64).mean()
C["const_classcond_TRAIN_mean"] = cc
# ego-scalar floors fit on TRAIN rows (v0, a0 from v_now_ms); eval features from the EVAL release by the SAME function
DT = float(np.median(RT["clip__dt_s"]))


def ego_from_release(R, clip_of_row, rows=None):
    v = R["row__v_now_ms"].astype(np.float64)
    a = np.full(len(v), np.nan)
    same = np.zeros(len(v), bool)
    same[2:] = clip_of_row[2:] == clip_of_row[:-2]
    a[2:] = (v[2:] - v[:-2]) / (2 * DT)
    a[~same] = np.nan
    return v, a


cl_tr = np.repeat(np.arange(len(RT["clip__row0"])), RT["clip__n_rows"])
cl_ev = np.repeat(np.arange(len(RE["clip__row0"])), RE["clip__n_rows"])
v_tr, a_tr = ego_from_release(RT, cl_tr)
v_ev_all, a_ev_all = ego_from_release(RE, cl_ev)
v_e, a_e = v_ev_all[ridx], a_ev_all[ridx]
ego_ok_tr = np.isfinite(v_tr) & np.isfinite(a_tr)
ego_ok_ev = np.isfinite(a_e)


def feats(v, a, poly):
    x = np.stack([v, a], 1)
    return np.concatenate([x, x ** 2, (v * a)[:, None]], 1) if poly else x


def ridge_fit_predict(poly, lams=(1e-2, 1e0, 1e2, 1e4), k=5):
    P = np.zeros((N, 26), np.float32)
    meta = {}
    Xe_all = feats(np.where(ego_ok_ev, v_e, 0.0), np.where(ego_ok_ev, a_e, 0.0), poly)
    for f in range(26):
        sel = Mtr[:, f] & ego_ok_tr
        X = feats(v_tr[sel], a_tr[sel], poly)
        y = Ytr[sel, f].astype(np.float64)
        g = cl_tr[sel]
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
        reg = np.diag(np.r_[np.ones(Xs.shape[1] - 1), 0.0])
        best, bl = None, None
        for lam in lams:
            err = 0.0
            for q in range(k):
                a_, b_ = (g % k) != q, (g % k) == q
                w = np.linalg.solve(Xs[a_].T @ Xs[a_] + lam * reg, Xs[a_].T @ y[a_])
                err += np.abs(Xs[b_] @ w - y[b_]).sum()
            if best is None or err < best:
                best, bl = err, lam
        w = np.linalg.solve(Xs.T @ Xs + bl * reg, Xs.T @ y)
        P[:, f] = (np.concatenate([(Xe_all - mu) / sd, np.ones((N, 1))], 1) @ w).astype(np.float32)
        meta[FIELDS[f]] = {"lambda": bl, "n_train_entries": int(sel.sum()), "d": int(X.shape[1])}
    P[~ego_ok_ev] = C["const_TRAIN_mean"][~ego_ok_ev]       # a0 undefined for the first rows: fall back to the constant
    return P, meta


C["ridge_v0a0_TRAINfit"], meta_r = ridge_fit_predict(False)
C["ridge_v0a0_poly2_TRAINfit"], meta_p = ridge_fit_predict(True)
C["GT_as_prediction"] = Yev.copy()

# ------------------------------------------------------------------------------------------- the statistics
ep_ids = np.array([f"{e:03d}" for e in W["ep"]])
uniq = np.unique(ep_ids)
epi = np.searchsorted(uniq, ep_ids)
E = len(uniq)
rng = np.random.default_rng(SEED)
PICKS = np.stack([np.searchsorted(uniq, rng.choice(uniq, size=E, replace=True)) for _ in range(N_BOOT)])     # == taniteval.ci._draws' RNG stream


def ep_sums(x):
    s = np.zeros(E)
    np.add.at(s, epi, x)
    return s


def stat(P_a, cols, P_b=None):
    """pooled entry-mean |e| of arm a (or the paired difference a - b) over the columns, episode-cluster bootstrap."""
    m = Mev[:, cols]
    ea = (np.abs(P_a[:, cols] - Yev[:, cols]) * m).sum(1)
    if P_b is not None:
        ea = ea - (np.abs(P_b[:, cols] - Yev[:, cols]) * m).sum(1)
    num, den = ep_sums(ea), ep_sums(m.sum(1).astype(float))
    vals = num[PICKS].sum(1) / np.maximum(den[PICKS].sum(1), 1.0)
    pt = float(num.sum() / max(den.sum(), 1.0))
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    return {"point": pt, "lo": lo, "hi": hi, "n_entries": int(m.sum()), "n_windows_any": int((m.sum(1) > 0).sum()), "d": len(cols),
            "n_episodes": int(E), "n_boot": N_BOOT, "separated": bool(P_b is not None and (lo > 0 or hi < 0))}


ARMS = ["const_TRAIN_mean", "const_TRAIN_median", "const_classcond_TRAIN_mean", "ridge_v0a0_TRAINfit", "ridge_v0a0_poly2_TRAINfit"]
table = {}
for g, cols in ALLG.items():
    t = {"head": stat(P_head, cols), "GT_as_prediction": stat(C["GT_as_prediction"], cols), "controls": {}, "paired_head_minus": {}}
    for a in ARMS:
        t["controls"][a] = stat(C[a], cols)
        t["paired_head_minus"][a] = stat(P_head, cols, C[a])
        t["controls"][a]["head_over_control"] = t["head"]["point"] / t["controls"][a]["point"]
        t["controls"][a]["gap_closed"] = 1.0 - t["head"]["point"] / t["controls"][a]["point"]
    table[g] = t
# per-field (head vs controls), physical-unit MAE of the head
per_field = []
for f in range(26):
    m = Mev[:, f]
    if m.sum() == 0:
        continue
    row = {"field": FIELDS[f], "head_group": ("lat_c" if f < 12 else "lon_c" if f < 22 else "speed"),
           "v8f_group": [k for k, v in V8F.items() if f in v][0], "n": int(m.sum()), "scale": float(SCALE[f])}
    for nm, P in (("head", P_head), ("const", C["const_TRAIN_mean"]), ("classcond", C["const_classcond_TRAIN_mean"]),
                  ("ridge", C["ridge_v0a0_TRAINfit"]), ("gt", C["GT_as_prediction"])):
        row[nm + "_mae_norm"] = float((np.abs(P[:, f] - Yev[:, f])[m]).mean())
    row["head_mae_physical"] = row["head_mae_norm"] * float(SCALE[f])
    row["head_beats_const"] = row["head_mae_norm"] < row["const_mae_norm"]
    row["head_beats_ridge"] = row["head_mae_norm"] < row["ridge_mae_norm"]
    per_field.append(row)

# ------------------------------------------------------------------------------------------- controls that must read known values
ctl = {"GT_as_prediction_MAE_exactly_zero": all(table[g]["GT_as_prediction"]["point"] == 0.0 for g in ALLG),
       "const_equals_TRAIN_mean": bool(np.all(C["const_TRAIN_mean"] == C["const_TRAIN_mean"][0]) and
                                       np.allclose(C["const_TRAIN_mean"][0], mean_tr.astype(np.float32), atol=1e-7)),
       "join": join_check, "n_train_rows": int(len(Ytr)), "n_eval_windows": int(N), "train_release_md5": TRAIN_MD5, "eval_release_md5": EVAL_MD5}
# mutant: const + 1 must read worse than const; GT + noise(0.05) must read ~0.04
rngm = np.random.default_rng(1)
ctl["mutant_const_plus1_worse"] = bool(stat(C["const_TRAIN_mean"] + 1.0, ALLG["ALL26"])["point"] > table["ALL26"]["controls"]["const_TRAIN_mean"]["point"])
ctl["gt_plus_noise0p05_reads"] = stat(Yev + rngm.normal(0, 0.05, Yev.shape).astype(np.float32), ALLG["ALL26"])["point"]

# the lateral/longitudinal DECISION head agreement re-read from THIS dump (same model as the battery? equality + 0.9415)
dec_checks = {}
Hh = {k: [] for k in ("lat8", "lon8", "tlat8", "tlon8")}
for ep in range(139):
    h = np.load(f"{HEADS}/heads_ep{ep:03d}.npz", allow_pickle=True)
    Hh["lat8"].append(h["out_lat_logits_tac"]); Hh["lon8"].append(h["out_lon_logits_tac"])
    Hh["tlat8"].append(h["out_tacv6_lat_logits"]); Hh["tlon8"].append(h["out_tacv6_lon_logits"])
Hh = {k: np.concatenate(v, 0) for k, v in Hh.items()}
for nm, lg, pred in (("lat", Hh["tlat8"], W["lat_pred"]), ("lon", Hh["tlon8"], W["lon_pred"]),
                     ("lat_alt", Hh["lat8"], W["lat_pred"]), ("lon_alt", Hh["lon8"], W["lon_pred"])):
    dec_checks[nm + "_argmax_equals_battery_pred_frac"] = float((lg.argmax(1) == pred).mean())
# the fields whose argmax equals the battery's *_pred_nav_true on 100 % of the 4,754 windows are out_lat_logits_tac / out_lon_logits_tac
# (the tacv6_* 8-way logits are a different head: 94.6 % / 72.5 % -- NOT the battery's decision), so the control is read from those
ok_l = W["lat_label"] >= 0
dec_checks["lat_acc_from_heads_dump_vs_battery_label"] = float((Hh["lat8"].argmax(1)[ok_l] == W["lat_label"][ok_l]).mean())
ok_o = W["lon_label"] >= 0
dec_checks["lon_acc_from_heads_dump_vs_battery_label"] = float((Hh["lon8"].argmax(1)[ok_o] == W["lon_label"][ok_o]).mean())
cl0 = np.bincount(W["lat_label"][ok_l]).argmax()
dec_checks["lat_constant_class"] = int(cl0)
dec_checks["lat_acc_constant_from_label_counts"] = float((W["lat_label"][ok_l] == cl0).mean())
dec_checks["lat_n"] = int(ok_l.sum()); dec_checks["lon_n"] = int(ok_o.sum())
dec_checks["lat_acc_banked_TAC_CONTROLS_seed0"] = 0.9415
dec_checks["lon_acc_banked_TAC_CONTROLS_seed0"] = 0.6103
dec_checks["lat_reads_banked_value"] = bool(abs(dec_checks["lat_acc_from_heads_dump_vs_battery_label"] - 0.9415) < 5e-5)
dec_checks["lon_reads_banked_value"] = bool(abs(dec_checks["lon_acc_from_heads_dump_vs_battery_label"] - 0.6103) < 5e-5)
ctl["decision_head_cross_check"] = dec_checks

# ------------------------------------------------------------------------------------------- PROBE-LEVEL (cons_lat / cons_lon) at the PATH-TAGGED GT hypothesis
HOR = (5, 10, 15, 20, 30, 40, 50, 60)
SLOT_T = [h * 0.1 for h in HOR]
pl = torch.as_tensor(W["pose_last"], dtype=torch.float32)
fu = torch.as_tensor(W["gt_fut"], dtype=torch.float32)
fv = torch.as_tensor(W["gt_fv"] > 0.5)
tt = refb_labels.waypoint_targets(pl, fu, HOR)
svv = torch.stack([fv[:, h - 1] for h in HOR], 1)
gl, go = r8c.tag_paths(tt[:, None], pl[:, 3], SLOT_T)
gl, go = gl[:, 0], go[:, 0]
okk = svv[:, -1]
gl = torch.where(okk, gl, torch.full_like(gl, -1)); go = torch.where(okk, go, torch.full_like(go, -1))
gl7 = torch.where(gl >= 0, torch.as_tensor(r8c.LAT3_V7_IDS)[gl.clamp_min(0)], torch.full_like(gl, -100)).numpy()
go7 = torch.where(go >= 0, torch.as_tensor(r8c.LON6_V7_IDS)[go.clamp_min(0)], torch.full_like(go, -100)).numpy()
ct = r8c.constraint_targets(tt, svv, pl, fu, fv)
lt_, lv_, ot_, ov_ = r8c.normalise_constraints(ct)
Yp = torch.cat([torch.nan_to_num(lt_), torch.nan_to_num(ot_)], 1).numpy()
Mp = torch.cat([lv_ & torch.as_tensor(gl7 >= 0)[:, None], ov_ & torch.as_tensor(go7 >= 0)[:, None]], 1).numpy()
Pp_head = np.concatenate([H["pcl"][ar, np.clip(gl7, 0, None)], H["pco"][ar, np.clip(go7, 0, None)]], 1).astype(np.float32)
# TRAIN-139 dump windows (the windows the model trained on): constant / class-cond / ego-3 ridge (v0, a_long, yaw_rate)
tr = {k: [] for k in ("pose_last", "gt_future_ext", "gt_future_valid_ext", "ego_t0")}
for f in sorted(os.listdir(TRAINFIT)):
    if f.startswith("ep") and f.endswith(".npz"):
        z = np.load(f"{TRAINFIT}/{f}", allow_pickle=True)
        for k in tr:
            tr[k].append(z[k])
tr = {k: np.concatenate(v, 0) for k, v in tr.items()}
pl_t = torch.as_tensor(tr["pose_last"], dtype=torch.float32)
fu_t = torch.as_tensor(tr["gt_future_ext"], dtype=torch.float32)
fv_t = torch.as_tensor(tr["gt_future_valid_ext"] > 0.5)
tt_t = refb_labels.waypoint_targets(pl_t, fu_t, HOR)
sv_t = torch.stack([fv_t[:, h - 1] for h in HOR], 1)
gl_t, go_t = r8c.tag_paths(tt_t[:, None], pl_t[:, 3], SLOT_T)
gl_t, go_t = gl_t[:, 0], go_t[:, 0]
ok_t = sv_t[:, -1]
gl_t = torch.where(ok_t, gl_t, torch.full_like(gl_t, -1)); go_t = torch.where(ok_t, go_t, torch.full_like(go_t, -1))
gl7_t = torch.where(gl_t >= 0, torch.as_tensor(r8c.LAT3_V7_IDS)[gl_t.clamp_min(0)], torch.full_like(gl_t, -100)).numpy()
go7_t = torch.where(go_t >= 0, torch.as_tensor(r8c.LON6_V7_IDS)[go_t.clamp_min(0)], torch.full_like(go_t, -100)).numpy()
ct_t = r8c.constraint_targets(tt_t, sv_t, pl_t, fu_t, fv_t)
a_, b_, c_, d_ = r8c.normalise_constraints(ct_t)
Yp_t = torch.cat([torch.nan_to_num(a_), torch.nan_to_num(c_)], 1).numpy()
Mp_t = torch.cat([b_ & torch.as_tensor(gl7_t >= 0)[:, None], d_ & torch.as_tensor(go7_t >= 0)[:, None]], 1).numpy()
mean_p = np.array([Yp_t[Mp_t[:, f], f].astype(np.float64).mean() for f in range(4)])
CP = {"GT_as_prediction": Yp.copy(), "const_TRAIN_mean": np.tile(mean_p[None].astype(np.float32), (N, 1))}
ccp = CP["const_TRAIN_mean"].copy()
for cls in np.unique(gl7_t[gl7_t >= 0]):
    for f in (0, 1):
        mk = Mp_t[:, f] & (gl7_t == cls)
        if mk.any():
            ccp[gl7 == cls, f] = Yp_t[mk, f].mean()
for cls in np.unique(go7_t[go7_t >= 0]):
    for f in (2, 3):
        mk = Mp_t[:, f] & (go7_t == cls)
        if mk.any():
            ccp[go7 == cls, f] = Yp_t[mk, f].mean()
CP["const_classcond_TRAIN_mean"] = ccp


def ridge_p(poly, lams=(1e-2, 1e0, 1e2, 1e4), k=5):
    P = np.zeros((N, 4), np.float32)
    g_ep = np.arange(len(tr["pose_last"])) // 35
    ft = lambda E_: (np.concatenate([E_[:, :3], E_[:, :3] ** 2, (E_[:, 0] * E_[:, 1])[:, None], (E_[:, 0] * E_[:, 2])[:, None]], 1) if poly else E_[:, :3]).astype(np.float64)
    for f in range(4):
        sel = Mp_t[:, f]
        X = ft(tr["ego_t0"][sel]); y = Yp_t[sel, f].astype(np.float64); g = g_ep[sel]
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
        reg = np.diag(np.r_[np.ones(Xs.shape[1] - 1), 0.0])
        best, bl = None, None
        for lam in lams:
            err = 0.0
            for q in range(k):
                aa, bb = (g % k) != q, (g % k) == q
                w = np.linalg.solve(Xs[aa].T @ Xs[aa] + lam * reg, Xs[aa].T @ y[aa])
                err += np.abs(Xs[bb] @ w - y[bb]).sum()
            if best is None or err < best:
                best, bl = err, lam
        w = np.linalg.solve(Xs.T @ Xs + bl * reg, Xs.T @ y)
        P[:, f] = (np.concatenate([(ft(W["ego_t0"]) - mu) / sd, np.ones((N, 1))], 1) @ w).astype(np.float32)
    return P


CP["ridge_ego3_TRAIN139fit"] = ridge_p(False)
CP["ridge_ego3_poly2_TRAIN139fit"] = ridge_p(True)


def stat_p(P_a, cols, P_b=None):
    m = Mp[:, cols]
    ea = (np.abs(P_a[:, cols] - Yp[:, cols]) * m).sum(1)
    if P_b is not None:
        ea = ea - (np.abs(P_b[:, cols] - Yp[:, cols]) * m).sum(1)
    num, den = ep_sums(ea), ep_sums(m.sum(1).astype(float))
    vals = num[PICKS].sum(1) / np.maximum(den[PICKS].sum(1), 1.0)
    return {"point": float(num.sum() / max(den.sum(), 1.0)), "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "n_entries": int(m.sum()), "d": len(cols), "n_episodes": int(E), "n_boot": N_BOOT}


PNAMES = ("psi_term/pi", "t_onset/6", "log1p(prog6)/log1p(120)", "v_end/30")
PG = {"lat(psi_term,t_onset)": [0, 1], "lon(prog6,v_end)": [2, 3], "ALL4": [0, 1, 2, 3], "psi_term": [0], "t_onset": [1], "v_end": [3], "prog6": [2]}
PARMS = ["const_TRAIN_mean", "const_classcond_TRAIN_mean", "ridge_ego3_TRAIN139fit", "ridge_ego3_poly2_TRAIN139fit"]
ptab = {}
for g, cols in PG.items():
    t = {"head": stat_p(Pp_head, cols), "GT_as_prediction": stat_p(CP["GT_as_prediction"], cols), "controls": {}, "paired_head_minus": {}}
    for a in PARMS:
        t["controls"][a] = stat_p(CP[a], cols)
        t["paired_head_minus"][a] = stat_p(Pp_head, cols, CP[a])
        t["controls"][a]["gap_closed"] = 1.0 - t["head"]["point"] / t["controls"][a]["point"]
    ptab[g] = t
ctl["probe_GT_exactly_zero"] = all(ptab[g]["GT_as_prediction"]["point"] == 0.0 for g in PG)
ctl["probe_const_equals_TRAIN_mean"] = bool(np.all(CP["const_TRAIN_mean"] == CP["const_TRAIN_mean"][0]))
ctl["probe_n_train139_windows"] = int(len(tr["pose_last"]))

# ------------------------------------------------------------------------------------------- mispaired control (the head vs ANOTHER window's GT)
# a head paired with the GT of a window ~1000 windows away must read near / above the constant level, far above its true-pair error;
# if it did not, the instrument would be insensitive to the pairing (a silent join defect would pass)
_sh = 1000
_Yr, _Mr = np.roll(Yev, _sh, 0), np.roll(Mev, _sh, 0)
_mis = float((np.abs(P_head - _Yr) * _Mr).sum() / max(_Mr.sum(), 1))
_cmis = float((np.abs(C["const_TRAIN_mean"] - _Yr) * _Mr).sum() / max(_Mr.sum(), 1))
ctl["mispaired_roll1000_head_MAE_ALL26"] = _mis
ctl["mispaired_roll1000_const_MAE_ALL26"] = _cmis
ctl["true_pair_head_MAE_ALL26"] = table["ALL26"]["head"]["point"]
ctl["mispaired_head_is_worse_than_true_pair_by_x"] = _mis / table["ALL26"]["head"]["point"]
ctl["mispaired_head_not_better_than_const"] = bool(_mis >= 0.9 * _cmis)

# ------------------------------------------------------------------------------------------- EXPLORATORY strata (descriptive; NOT a committed bar)
# does the lonspeed head carry the scene-driven speed INTENTION an ego-scalar floor cannot? stratify by the GT speed change ahead:
# |lon_v_target - v_now| >= 2 m/s ("speed-change windows") vs < 2 ("steady"); labels may use the future (stratification only).
vt_raw = oc[ridx, 0].astype(np.float64)
chg = np.abs(vt_raw - v_rel)
strata = {"speed_change_ge2ms": np.isfinite(chg) & (chg >= 2.0), "steady_lt2ms": np.isfinite(chg) & (chg < 2.0)}


def stat_w(P_a, cols, wmask, P_b=None):
    m = Mev[:, cols] & wmask[:, None]
    ea = (np.abs(P_a[:, cols] - Yev[:, cols]) * m).sum(1)
    if P_b is not None:
        ea = ea - (np.abs(P_b[:, cols] - Yev[:, cols]) * m).sum(1)
    num, den = ep_sums(ea), ep_sums(m.sum(1).astype(float))
    vals = num[PICKS].sum(1) / np.maximum(den[PICKS].sum(1), 1.0)
    return {"point": float(num.sum() / max(den.sum(), 1.0)), "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "n_entries": int(m.sum()), "n_windows": int((m.sum(1) > 0).sum()), "d": len(cols), "n_episodes": int(len(np.unique(W["ep"][wmask & (m.sum(1) > 0)])))}


strata_res = {}
for sn, wm in strata.items():
    r = {"n_windows_in_stratum": int(wm.sum())}
    for g in ("lonspeed", "speed"):
        cols = ALLG[g]
        r[g] = {"head": stat_w(P_head, cols, wm), "const": stat_w(C["const_TRAIN_mean"], cols, wm),
                "ridge_v0a0": stat_w(C["ridge_v0a0_TRAINfit"], cols, wm), "ridge_poly2": stat_w(C["ridge_v0a0_poly2_TRAINfit"], cols, wm),
                "paired_head_minus_ridge": stat_w(P_head, cols, wm, C["ridge_v0a0_TRAINfit"]),
                "paired_head_minus_poly2": stat_w(P_head, cols, wm, C["ridge_v0a0_poly2_TRAINfit"])}
    strata_res[sn] = r

res = {"schema": "tacconstr-r8-heads-v1", "tier": "T0 (teacher-forced: the head is read on the GT-active class's query). NOT driving performance.",
       "checkpoint": "EMA-48,852 (E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt, heads dump E:/refcv8_eval_kit/videos/heads_48852_pf, perframe_full feed)",
       "estimator": f"episode_cluster_bootstrap B={N_BOOT} seed={SEED} over {E} episodes (same resample stream as taniteval.ci._draws), pooled entry-mean |e| "
                    "(ratio of sums); paired = same resamples. Answers another draw of EPISODES at this checkpoint; ONE training seed; the heads are "
                    "deterministic given the inputs (seed-0 and seed-1 decision heads read identical).",
       "n_windows": int(N), "n_episodes": int(E), "v9_groups": table, "per_field": per_field, "probe_level": ptab, "controls": ctl,
       "exploratory_strata_lonspeed": strata_res, "ridge_meta_v0a0": meta_r, "head_units": "normalised (raw/scale, clamp +-6), scales per field in per_field[].scale"}
json.dump(res, open(f"{OUT}/TACCONSTR_R8_heads_perwindow.json", "w"), indent=1)

print("controls:", json.dumps({k: v for k, v in ctl.items()}, indent=0)[:1800])
for g in ALLG:
    t = table[g]
    print(f"\n[{g}] n_entries {t['head']['n_entries']} windows {t['head']['n_windows_any']} d {t['head']['d']}  HEAD {t['head']['point']:.4f} [{t['head']['lo']:.4f},{t['head']['hi']:.4f}]")
    for a in ARMS:
        c, p = t["controls"][a], t["paired_head_minus"][a]
        print(f"   {a:28s} {c['point']:.4f} [{c['lo']:.4f},{c['hi']:.4f}]  gap-closed {c['gap_closed']:+.3f}  paired head-ctrl {p['point']:+.4f} [{p['lo']:+.4f},{p['hi']:+.4f}] {'SEP' if p['separated'] else 'ns'}")
print("\nPROBE-LEVEL")
for g in PG:
    t = ptab[g]
    print(f"[{g}] n {t['head']['n_entries']} HEAD {t['head']['point']:.4f} [{t['head']['lo']:.4f},{t['head']['hi']:.4f}]")
    for a in PARMS:
        c, p = t["controls"][a], t["paired_head_minus"][a]
        print(f"   {a:30s} {c['point']:.4f} gap-closed {c['gap_closed']:+.3f} paired {p['point']:+.4f} [{p['lo']:+.4f},{p['hi']:+.4f}]")
print("\nper-field:")
for r in per_field:
    print(f"  {r['field']:16s} {r['v8f_group']:8s} n {r['n']:4d} head {r['head_mae_norm']:.3f} ({r['head_mae_physical']:.2f}) const {r['const_mae_norm']:.3f} cc {r['classcond_mae_norm']:.3f} ridge {r['ridge_mae_norm']:.3f}  beats const {r['head_beats_const']} ridge {r['head_beats_ridge']}")

print("EXPLORATORY strata:")
for sn, r in strata_res.items():
    for g in ("lonspeed", "speed"):
        x = r[g]
        print(f"  {sn:20s} {g:9s} windows {x['head']['n_windows']:5d} head {x['head']['point']:.4f} const {x['const']['point']:.4f} ridge {x['ridge_v0a0']['point']:.4f} poly2 {x['ridge_poly2']['point']:.4f}  "
              f"head-ridge {x['paired_head_minus_ridge']['point']:+.4f} [{x['paired_head_minus_ridge']['lo']:+.4f},{x['paired_head_minus_ridge']['hi']:+.4f}]  "
              f"head-poly2 {x['paired_head_minus_poly2']['point']:+.4f} [{x['paired_head_minus_poly2']['lo']:+.4f},{x['paired_head_minus_poly2']['hi']:+.4f}]")
