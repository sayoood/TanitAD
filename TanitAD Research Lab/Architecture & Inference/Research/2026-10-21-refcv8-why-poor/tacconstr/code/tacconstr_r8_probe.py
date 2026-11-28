"""refcv8's PROBE-LEVEL constraint heads (cons_lat [psi_term, t_onset], cons_lon [log-progress, v_end]) -- the 4 numbers the
planner's FactorizedSelection / allocation actually READ (refc.py:2500-2551, 3834) -- vs controls.  CPU only.

HEAD side: trainer-logged `ema_eval_r8_cons` (Huber beta 0.05, GT-active PATH-TAGGED hypothesis query, normalised) at step
48852 on the 8 fixed eval batches (128 windows).  CONTROL side: the REAL source functions (imported from the launch-tree
bundle whose blobs equal fb47a58) -- constraint_targets / normalise_constraints / tag_paths / constraint_head_loss -- so
the control's loss is the trainer's own function evaluated on a constant / ridge prediction.
TRAIN set for the constant & ridge: the 139 TRAIN-episode dump windows (`trainfit/step30000ema_train139_pf`, windows the
model was trained on).  Features for the ridge floor: ego_t0 = (v0, a_long, yaw_rate) from the SAME function on both sides.
"""
import json
import os
import sys

import numpy as np

STACK = "C:/Users/Admin/refcv8_pf/navsim_ckpt/thor_bundle/stage/tree/stack"
sys.path.insert(0, STACK + "/scripts")
sys.path.insert(0, STACK)
sys.path.insert(0, "E:/Projects/TanitAD/taniteval")
import torch  # noqa: E402

from tanitad.refs import refcv8_conditioning as r8c  # noqa: E402
from tanitad.refs import refc_v3 as v3mod  # noqa: E402
import refb_labels  # noqa: E402
from taniteval import ci as TCI  # noqa: E402

assert os.path.normpath(r8c.__file__).startswith(os.path.normpath(STACK)), f"imported {r8c.__file__}"  # the editable-install trap
OUT = "C:/Users/Admin/refcv8_pf/tacconstr"
G0 = "C:/Users/Admin/refcv8_eval_archive/g0cache/step40000"
TRAINFIT = "C:/Users/Admin/refcv8_pf/trainfit/step30000ema_train139_pf/dump_s0/decisions"
METRICS = "C:/Users/Admin/qland/work/watch8/metrics.jsonl"
HOR = (5, 10, 15, 20, 30, 40, 50, 60)
SLOT_T = [h * 0.1 for h in HOR]
N_BOOT, SEED = 2000, 0
NAMES = ("psi_term/pi", "t_onset/6", "log1p(prog6)/log1p(120)", "v_end/30")


def build(pose_last, fut_ext, fut_valid):
    """source-literal r8_before_forward target block (refcv8_train.py:540-551)."""
    pl = torch.as_tensor(pose_last, dtype=torch.float32)
    fu = torch.as_tensor(fut_ext, dtype=torch.float32)
    fv = torch.as_tensor(fut_valid).to(torch.bool)
    tt = refb_labels.waypoint_targets(pl, fu, HOR)
    svv = torch.stack([fv[:, h - 1] for h in HOR], dim=1)
    v0 = pl[:, 3]
    gl, go = r8c.tag_paths(tt[:, None], v0, SLOT_T)
    gl, go = gl[:, 0], go[:, 0]
    ok = svv[:, -1].to(torch.bool)
    gl = torch.where(ok, gl, torch.full_like(gl, -1))
    go = torch.where(ok, go, torch.full_like(go, -1))
    gl7 = torch.where(gl >= 0, torch.as_tensor(r8c.LAT3_V7_IDS)[gl.clamp_min(0)], torch.full_like(gl, -100))
    go7 = torch.where(go >= 0, torch.as_tensor(r8c.LON6_V7_IDS)[go.clamp_min(0)], torch.full_like(go, -100))
    ct = r8c.constraint_targets(tt, svv, pl, fu, fv)
    lt, lv, ot, ov = r8c.normalise_constraints(ct)
    return {"gl7": gl7, "go7": go7, "lat_t": torch.nan_to_num(lt), "lat_v": lv, "lon_t": torch.nan_to_num(ot), "lon_v": ov}


# ---------------------------------------------------------------- eval cache (128 windows, 8 batches of 16)
bs = []
for i in range(8):
    b = torch.load(f"{G0}/batch_{i}.pt", map_location="cpu", mmap=True, weights_only=False)
    bs.append({k: b[k].clone() for k in ("pose_last", "future_poses_ext", "future_valid_ext", "actions", "episode_id")})
ev = {k: torch.cat([x[k] for x in bs], 0) for k in bs[0]}
batch_id = np.repeat(np.arange(8), [x["pose_last"].shape[0] for x in bs])
T_ev = build(ev["pose_last"], ev["future_poses_ext"], ev["future_valid_ext"])
N_EV = int(ev["pose_last"].shape[0])
eid = np.asarray([str(x) for x in ev["episode_id"].numpy()])


def ego(pose_last, actions):
    pl = torch.as_tensor(pose_last, dtype=torch.float32)
    ac = torch.as_tensor(actions, dtype=torch.float32)
    return v3mod.ego_state_at_t0(pl[:, None].expand(pl.shape[0], ac.shape[1], 4), ac)[:, :4].numpy()


E_ev = ego(ev["pose_last"], ev["actions"])

# ---------------------------------------------------------------- TRAIN-139 dump windows
tr = {k: [] for k in ("pose_last", "gt_future_ext", "gt_future_valid_ext", "ego_t0")}
for f in sorted(os.listdir(TRAINFIT)):
    if f.startswith("ep") and f.endswith(".npz"):
        z = np.load(f"{TRAINFIT}/{f}", allow_pickle=True)
        for k in tr:
            tr[k].append(z[k])
tr = {k: np.concatenate(v, 0) for k, v in tr.items()}
T_tr = build(tr["pose_last"], tr["gt_future_ext"], tr["gt_future_valid_ext"] > 0.5)
N_TR = tr["pose_last"].shape[0]
# the ego_t0 the TRAIN dumps wrote must equal the function applied to what I feed the eval side (same definition both sides):
# check on the eval cache via the same function (self-consistency of the call) and range-compare the two populations
ego_ranges = {"train_dump_ego_t0_mean": tr["ego_t0"][:, :3].mean(0).tolist(), "eval_cache_ego_t0_mean": E_ev[:, :3].mean(0).tolist(),
              "train_dump_ego_t0_std": tr["ego_t0"][:, :3].std(0).tolist(), "eval_cache_ego_t0_std": E_ev[:, :3].std(0).tolist()}

# ---------------------------------------------------------------- counts vs the trainer's own log (separate channel)
rows = [json.loads(l) for l in open(METRICS, encoding="utf-8") if l.strip()]
last = [r for r in rows if r.get("step") == 48852 and "ema_eval_r8_cons" in r][-1]
ml = (T_ev["lat_v"].float() * (T_ev["gl7"] >= 0)[:, None].float())
mo = (T_ev["lon_v"].float() * (T_ev["go7"] >= 0)[:, None].float())
cnt_lat = [float(ml[batch_id == b].sum()) for b in range(8)]
cnt_lon = [float(mo[batch_id == b].sum()) for b in range(8)]
counts = {"lat_mine_mean_per_batch": float(np.mean(cnt_lat)), "lat_logged": last["ema_eval_n_cons_lat"],
          "lon_mine_mean_per_batch": float(np.mean(cnt_lon)), "lon_logged": last["ema_eval_n_cons_lon"]}
counts["match"] = bool(abs(counts["lat_mine_mean_per_batch"] - counts["lat_logged"]) < 1e-6 and
                       abs(counts["lon_mine_mean_per_batch"] - counts["lon_logged"]) < 1e-6)

# ---------------------------------------------------------------- the controls, through the REAL loss function
# ent: per-entry values for 4 fields; masks m [N, 4]
Y_tr = torch.cat([T_tr["lat_t"], T_tr["lon_t"]], 1).numpy()
M_tr = torch.cat([(T_tr["lat_v"] & (T_tr["gl7"] >= 0)[:, None]), (T_tr["lon_v"] & (T_tr["go7"] >= 0)[:, None])], 1).numpy()
Y_ev = torch.cat([T_ev["lat_t"], T_ev["lon_t"]], 1).numpy()
M_ev = torch.cat([(T_ev["lat_v"] & (T_ev["gl7"] >= 0)[:, None]), (T_ev["lon_v"] & (T_ev["go7"] >= 0)[:, None])], 1).numpy()
mean_tr = np.array([Y_tr[M_tr[:, f], f].astype(np.float64).mean() for f in range(4)])
mean_tr_b = (Y_tr.astype(np.float64) * M_tr).sum(0) / M_tr.sum(0)
assert np.allclose(mean_tr, mean_tr_b, atol=1e-9)
med_tr = np.array([np.median(Y_tr[M_tr[:, f], f]) for f in range(4)])
gl7_tr, go7_tr = T_tr["gl7"].numpy(), T_tr["go7"].numpy()
gl7_ev, go7_ev = T_ev["gl7"].numpy(), T_ev["go7"].numpy()
cc = np.tile(mean_tr[None].astype(np.float32), (N_EV, 1))
for cls in np.unique(gl7_tr[gl7_tr >= 0]):
    for f in (0, 1):
        mk = M_tr[:, f] & (gl7_tr == cls)
        if mk.sum() > 0:
            cc[gl7_ev == cls, f] = Y_tr[mk, f].mean()
for cls in np.unique(go7_tr[go7_tr >= 0]):
    for f in (2, 3):
        mk = M_tr[:, f] & (go7_tr == cls)
        if mk.sum() > 0:
            cc[go7_ev == cls, f] = Y_tr[mk, f].mean()


def feats(E, poly):
    x = E[:, :3].astype(np.float64)
    return np.concatenate([x, x ** 2, (x[:, 0] * x[:, 1])[:, None], (x[:, 0] * x[:, 2])[:, None]], 1) if poly else x


def ridge(poly, lams=(1e-2, 1e0, 1e2, 1e4), k=5):
    P = np.zeros((N_EV, 4), np.float32)
    meta = {}
    ep_tr = np.repeat(np.arange(139), 35)[:N_TR] if N_TR == 139 * 35 else (np.arange(N_TR) // 35)
    for f in range(4):
        sel = M_tr[:, f]
        X = feats(tr["ego_t0"][sel], poly)
        y = Y_tr[sel, f].astype(np.float64)
        g = ep_tr[sel]
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
        best, bl = None, None
        for lam in lams:
            err = 0.0
            for q in range(k):
                a, b = (g % k) != q, (g % k) == q
                A = Xs[a].T @ Xs[a] + lam * np.diag(np.r_[np.ones(Xs.shape[1] - 1), 0.0])
                w = np.linalg.solve(A, Xs[a].T @ y[a])
                err += np.abs(Xs[b] @ w - y[b]).sum()
            if best is None or err < best:
                best, bl = err, lam
        A = Xs.T @ Xs + bl * np.diag(np.r_[np.ones(Xs.shape[1] - 1), 0.0])
        w = np.linalg.solve(A, Xs.T @ y)
        Xe = np.concatenate([(feats(E_ev, poly) - mu) / sd, np.ones((N_EV, 1))], 1)
        P[:, f] = (Xe @ w).astype(np.float32)
        meta[NAMES[f]] = {"lambda": bl, "n_train_entries": int(sel.sum()), "d": int(X.shape[1])}
    return P, meta


P_r, meta_r = ridge(False)
P_p, meta_p = ridge(True)
controls = {"GT_as_prediction": Y_ev.copy(), "const_TRAIN_mean": np.tile(mean_tr[None].astype(np.float32), (N_EV, 1)),
            "const_TRAIN_median": np.tile(med_tr[None].astype(np.float32), (N_EV, 1)),
            "const_classcond_TRAIN_mean": cc, "ridge_v0_a_yawrate_TRAINfit": P_r, "ridge_poly2_TRAINfit": P_p}


def real_loss(pred4):
    """trainer-style: the REAL constraint_head_loss per batch (cons_lat8 / cons_lon8 filled so the GT-active query reads pred4), mean over 8."""
    vals = []
    for b in range(8):
        sl = batch_id == b
        pl = torch.as_tensor(pred4[sl, 0:2])[:, None, :].expand(-1, 8, 2).contiguous()
        po = torch.as_tensor(pred4[sl, 2:4])[:, None, :].expand(-1, 8, 2).contiguous()
        l, _ = r8c.constraint_head_loss(pl, po, T_ev["gl7"][sl], T_ev["go7"][sl], T_ev["lat_t"][sl], T_ev["lat_v"][sl],
                                        T_ev["lon_t"][sl], T_ev["lon_v"][sl])
        vals.append(float(l))
    return float(np.mean(vals))


def huber(e, beta=0.05):
    a = np.abs(e)
    return np.where(a < beta, 0.5 * a * a / beta, a - 0.5 * beta)


UNIQ, IDX = TCI.episode_index(eid)
DRAWS = list(TCI._draws(UNIQ, IDX, N_BOOT, SEED))


def pooled(pred4, cols, kind="mae", ref4=None):
    e = (np.abs(pred4[:, cols] - Y_ev[:, cols]) if kind == "mae" else huber(pred4[:, cols] - Y_ev[:, cols])) * M_ev[:, cols]
    es, ms = e.sum(1), M_ev[:, cols].sum(1).astype(float)
    if ref4 is not None:
        r = (np.abs(ref4[:, cols] - Y_ev[:, cols]) if kind == "mae" else huber(ref4[:, cols] - Y_ev[:, cols])) * M_ev[:, cols]
        es = es - r.sum(1)
    vals = np.asarray([es[d].sum() / max(ms[d].sum(), 1.0) for d in DRAWS])
    return {"point": float(es.sum() / max(ms.sum(), 1.0)), "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "n_entries": int(ms.sum()), "n_boot": N_BOOT, "n_episodes": int(len(UNIQ))}


COLS = {"lat(psi_term,t_onset)": [0, 1], "lon(prog6,v_end)": [2, 3], "ALL4": [0, 1, 2, 3]}
tab = {}
for c, P in controls.items():
    tab[c] = {"real_constraint_head_loss_trainer_reduction": real_loss(P),
              "mae": {g: pooled(P, cols) for g, cols in COLS.items()},
              "huber_pooled_ALL4": pooled(P, COLS["ALL4"], "huber")}
    if c != "GT_as_prediction":
        tab[c]["per_field_mae_norm"] = {NAMES[f]: float(np.abs(P[:, f] - Y_ev[:, f])[M_ev[:, f]].mean()) for f in range(4)}
head_logged = last["ema_eval_r8_cons"]
head_raw = [r for r in rows if r.get("step") == 48852 and "eval_r8_cons" in r][-1]["eval_r8_cons"]
for c in controls:
    if c != "GT_as_prediction":
        tab[c]["head_over_control_ratio_huber"] = head_logged / tab[c]["real_constraint_head_loss_trainer_reduction"]
        tab[c]["head_gap_closed_vs_control"] = 1.0 - head_logged / tab[c]["real_constraint_head_loss_trainer_reduction"]

# controls that must read known values
ctl = {"counts_vs_trainer_log": counts,
       "GT_as_prediction_loss_exact_zero": tab["GT_as_prediction"]["real_constraint_head_loss_trainer_reduction"] == 0.0,
       "const_equals_TRAIN_mean": bool(np.all(controls["const_TRAIN_mean"] == controls["const_TRAIN_mean"][0]) and
                                       np.allclose(controls["const_TRAIN_mean"][0], mean_tr.astype(np.float32), atol=1e-7)),
       "n_train_windows": int(N_TR), "n_eval_windows": N_EV, "ego_t0_populations": ego_ranges,
       "mutant_const_plus_0p5_worse_than_const": bool(real_loss(controls["const_TRAIN_mean"] + 0.5) > tab["const_TRAIN_mean"]["real_constraint_head_loss_trainer_reduction"]),
       "imported_module_file": r8c.__file__}
res = {"schema": "tacconstr-r8-probe-v1",
       "what": "refcv8 probe-level constraint heads (the 4 dims the planner reads) vs controls, 128 windows",
       "tier": "T0 teacher-forced on the PATH-TAGGED GT hypothesis's query; NOT driving performance",
       "head_EMA48852_logged_huber": head_logged, "head_RAW48852_logged_huber": head_raw,
       "estimator": f"episode_cluster_bootstrap B={N_BOOT} seed={SEED}, {len(UNIQ)} episodes, controls only; head has NO interval",
       "controls_table": tab, "controls": ctl, "ridge_meta": {"linear": meta_r, "poly2": meta_p},
       "field_names": NAMES}
json.dump(res, open(f"{OUT}/TACCONSTR_R8_probe.json", "w"), indent=1)
print("counts:", counts)
print("controls:", {k: v for k, v in ctl.items() if k not in ("ego_t0_populations",)})
print("ego pops:", ego_ranges)
print(f"HEAD EMA-48852 logged Huber {head_logged:.5f} (raw {head_raw:.5f})")
for c in controls:
    t = tab[c]
    s = f"  {c:30s} loss {t['real_constraint_head_loss_trainer_reduction']:.5f}"
    if c != "GT_as_prediction":
        s += f"  head/ctrl {t['head_over_control_ratio_huber']:.3f} gap-closed {t['head_gap_closed_vs_control']:+.3f}"
    s += "  mae " + " | ".join(f"{g}: {t['mae'][g]['point']:.4f}[{t['mae'][g]['lo']:.4f},{t['mae'][g]['hi']:.4f}] n{t['mae'][g]['n_entries']}" for g in COLS)
    print(s)
for c in controls:
    if c != "GT_as_prediction":
        print(c, {k: round(v, 4) for k, v in tab[c]["per_field_mae_norm"].items()})
