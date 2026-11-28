"""rbjoint.py -- RB-JOINT: zero-training decodability of the roundabout lateral decision (TAC-LAT), with power and a positive control.
Card: PREREG_RBJOINT.md (written and sha256-hashed before any number).  Tier T0 decodability (NOT driving performance).  CPU only.
Stages:  python rbjoint.py --stage extract|cv|all [--smoke] [--out DIR]    sentinel: <out>/RBJOINT.done (first line DONE or FAIL)
Inputs at inference of every arm: frames <= t of the cached DINOv3 pooled grid (+ route-text features for ED / VED8).  No ego / kinematic input
(yaw_rate is only the TARGET of the positive control).  Labels may use future ego.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = "F:/TanitAD-scratch/strat1"
sys.path.insert(0, S1 + "/rbdist")
sys.path.insert(0, S1 + "/rbdist/src")
sys.path.insert(0, S1 + "/rbvis")
import rbdist_zt as RD  # noqa: E402
import rbvis as RV  # noqa: E402

TRAIN_CACHE, EVAL_CACHE = RV.TRAIN_CACHE, RV.EVAL_CACHE
CARD = HERE + "/PREREG_RBJOINT.md"
MARGIN = 0.0383
NB, SEED = 2000, 0
NFOLD = 5
STRIDE = 2
FREE_GB_FLOOR = 22.0
YAW_THR = 0.05
POS_YAW_MIN = 0.10
EPOCHS = 6
CKPTS = (3, 6)
BS, LR, WD, WARM, DIM = 64, 1e-3, 0.01, 50, 256
EXIT_BINS = 9
ED_DIM = 1 + EXIT_BINS + EXIT_BINS + 2
ARMS_VIS = {  # name: (feature file, target, uses_ed, shuffled)
    "RAW4": ("X4", "tac", False, False), "RAW8": ("X8", "tac", False, False), "VED8": ("X8", "tac", True, False),
    "YAW4": ("X4", "yaw", False, False), "YAW8": ("X8", "yaw", False, False),
    "SHUF_RAW4": ("X4", "tac", False, True), "SHUF_VED8": ("X8", "tac", True, True)}
SHAPES = {"X4": (4, 40), "X8": (2, 160)}


def sha(p):
    return RD.sha(p)


# ------------------------------------------------------------------ pure rules (each is a mutation target of the tests)
def fold_of(clip_id, salt="rbjoint-fold", k=NFOLD):
    return int(hashlib.sha256((salt + "|" + str(clip_id)).encode()).hexdigest(), 16) % k


def window_frames(t, grid, _shift=0):
    """cache frame indices read for window t.  X4: t-3..t of pool4x10;  X8: {t-3, t} of pool8x20.  _shift is for the mutant test only."""
    if grid == "X4":
        return [int(t) - 3 + _shift + j for j in range(4)]
    return [int(t) - 3 + _shift, int(t) + _shift]


def read_window(x, t, grid, _shift=0):
    fr = window_frames(t, grid, _shift)
    assert min(fr) >= 0 and max(fr) < x.shape[0], (t, fr, x.shape)
    return x[fr]


def yaw_class(yaw, thr=YAW_THR):
    """0 straight, 1 L (yaw_rate > +thr), 2 R (< -thr); the sign convention does not matter for accuracy"""
    yaw = np.asarray(yaw, np.float64)
    return np.where(yaw > thr, 1, np.where(yaw < -thr, 2, 0)).astype(np.int64)


def shuffle_labels(y, seed):
    """window-level permutation of the fold-train label vector (labels move between train clips)"""
    return np.asarray(y)[np.random.RandomState(seed).permutation(len(y))]


def ed_matrix(X):
    """X = rbdist features [state, exit_cur, exit_last, dist/100, dist_valid] -> [state, onehot(exit_cur,9), onehot(exit_last,9), dist, dist_valid]"""
    X = np.asarray(X, np.float64)
    out = np.zeros((len(X), ED_DIM), np.float32)
    out[:, 0] = X[:, 0]
    ec = np.clip(X[:, 1].astype(int), 0, EXIT_BINS - 1)
    el = np.clip(X[:, 2].astype(int), 0, EXIT_BINS - 1)
    out[np.arange(len(X)), 1 + ec] = 1.0
    out[np.arange(len(X)), 1 + EXIT_BINS + el] = 1.0
    out[:, 1 + 2 * EXIT_BINS] = X[:, 3]
    out[:, 2 + 2 * EXIT_BINS] = X[:, 4]
    return out


def carries(d, margin=MARGIN):
    return bool(d["delta"] >= margin and d["separated"])


def labels_rule(P, validity_ok):
    """P: paired dicts {delta, separated} for VED8-S, VED8-ED, VED8-RAW8, RAW8-S, RAW8-RAW4, RAW4-S, ED-S.  -> (labels holding, next lever = first in order)"""
    if not validity_ok:
        return ["INVALID"], "INVALID"
    held = []
    if carries(P["VED8-S"]) and carries(P["VED8-ED"]) and carries(P["VED8-RAW8"]):
        held.append("JOINT")
    if carries(P["RAW8-S"]) and carries(P["RAW8-RAW4"]):
        held.append("GRID")
    if carries(P["RAW4-S"]) or carries(P["RAW8-S"]):
        held.append("VIS")
    if carries(P["ED-S"]):
        held.append("TEXT")
    if not held:
        held.append("SILENT")
    return held, held[0]


def pos_yaw_pass(gain_over_majority):
    return bool(gain_over_majority["delta"] >= POS_YAW_MIN and gain_over_majority["separated"])


def free_gb(path):
    import shutil
    return shutil.disk_usage(path).free / 2 ** 30


def need_free(path, need_gb, what):
    f = free_gb(path)
    assert f - need_gb >= FREE_GB_FLOOR, f"{what}: F: free {f:.1f} GB minus {need_gb:.1f} GB < floor {FREE_GB_FLOOR} GB"


# ------------------------------------------------------------------ windows (rbdist selection code, t >= 3)
def build_windows():
    tpath, jt, je = S1 + "/read/inputs/navtok_table.npz", S1 + "/ckpt120build/in/join_train.npz", S1 + "/read/inputs/join_eval141.npz"
    assert sha(tpath) == RD.PIN_TABLE
    tab = np.load(tpath, allow_pickle=False)
    JT, JE = dict(np.load(jt, allow_pickle=False)), dict(np.load(je, allow_pickle=False))
    for k in ("clip", "off", "n"):
        assert np.array_equal(tab["train_" + k], JT[k]) and np.array_equal(tab["eval_" + k], JE[k]), k
    tr_clips, ev_clips = [str(c) for c in JT["clip"]], [str(c) for c in JE["clip"]]
    disj = {"table_train_vs_eval": len(set(tr_clips) & set(ev_clips)),
            "train_cache_files_vs_eval": len({f[:-3] for f in os.listdir(TRAIN_CACHE + "/clips")} & set(ev_clips)),
            "eval_cache_files_vs_train": len({f[:-3] for f in os.listdir(EVAL_CACHE + "/clips")} & set(tr_clips))}
    assert all(v == 0 for v in disj.values()), disj
    ex_t = RD.text_arrays(S1 + "/rbdist/src/texts.json")
    rb_t, fl_t, ex_tr, di_tr = RD.split_arrays(tab, "train", ex_t)
    off_t, n_t = np.asarray(JT["off"], np.int64), np.asarray(JT["n"], np.int64)
    clip_row = np.repeat(np.arange(len(n_t)), n_t)
    ytr_all = RD.label_rule(JT["has"], JT["lat_cls"])
    rows = np.where((fl_t == 0) & rb_t & (ytr_all != RD.IGNORE))[0]
    assert len(rows) == 24722 and len(set(clip_row[rows])) == 426, (len(rows), len(set(clip_row[rows])))
    X = RD.feature_matrix(rb_t, fl_t, ex_tr, di_tr, rows, off_t[clip_row[rows]])
    c, t = clip_row[rows], rows - off_t[clip_row[rows]]
    ok = t >= 3
    dropped_lt3 = int((~ok).sum())
    rows, c, t, X = rows[ok], c[ok], t[ok], X[ok]
    st = RV.subsample(rows, c, STRIDE)
    # eval (rbvis selection)
    d = S1 + "/read/t1/dump_STRAT1-C0_s0_m10_causal_heads"
    zs = [np.load(os.path.join(d, f"ep{i:04d}.npz")) for i in range(141)]
    eid = np.concatenate([z["eid"] for z in zs]); cidx = np.concatenate([z["clip_index"] for z in zs]); ws = np.concatenate([z["ws"] for z in zs])
    rows_e = np.asarray(JE["off"], np.int64)[cidx] + ws.astype(np.int64)
    rb_e, fl_e, ex_e, di_e = RD.split_arrays(tab, "eval", ex_t)
    state_e = np.array([RD.rb_state(rb_e, fl_e, int(r)) for r in rows_e])
    yall = RD.label_rule(JE["has"][rows_e], JE["lat_cls"][rows_e])
    assert int(state_e.sum()) == 235 and len(set(eid[state_e])) == 16
    sel = state_e & (yall != RD.IGNORE)
    assert int(sel.sum()) == 202 and len(set(eid[sel])) == 16
    off_e = np.asarray(JE["off"], np.int64)[cidx]
    Xe = RD.feature_matrix(rb_e, fl_e, ex_e, di_e, rows_e[sel], off_e[sel])
    ev = {"ci": cidx[sel].astype(np.int64), "t": ws[sel].astype(np.int64), "y": yall[sel], "eid": eid[sel], "X": Xe}
    tr = {"clip": c, "t": t, "y": ytr_all[rows], "X": X, "stride": st}
    info = {"disjointness": disj, "train_rb_windows_labelled": 24722, "train_episodes": 426, "dropped_t_lt3": dropped_lt3, "windows_t_ge3": int(len(rows)),
            "episodes_t_ge3": int(len(set(c))), "stride": STRIDE, "windows_strided": int(st.sum()),
            "inputs_sha256": {"navtok_table.npz": RD.PIN_TABLE, "join_train.npz": sha(jt), "join_eval141.npz": sha(je)}}
    return tr, ev, tr_clips, ev_clips, info


# ------------------------------------------------------------------ stage 1: features
def extract(out, smoke=False, log=print):
    import torch
    os.makedirs(out + "/feat", exist_ok=True)
    tr, ev, tr_clips, ev_clips, info = build_windows()
    # bad frame-count clips (cache vs table) are dropped whole and counted
    bad = set()
    n_tab = np.asarray(np.load(S1 + "/ckpt120build/in/join_train.npz")["n"])
    for c in np.unique(tr["clip"]):
        if int(RV.load_clip(TRAIN_CACHE, tr_clips[c]).shape[0]) != int(n_tab[c]):
            bad.add(int(c))
    keep = ~np.isin(tr["clip"], list(bad))
    tr = {k: v[keep] for k, v in tr.items()}
    info["dropped_bad_frame_count_clips"] = len(bad)
    info["windows_after_bad_drop"] = int(len(tr["y"]))
    sidx = np.where(tr["stride"])[0]
    if smoke:
        sidx = sidx[np.random.RandomState(0).permutation(len(sidx))[:300]]
        sidx.sort()
    N, NE = len(sidx), (len(ev["y"]) if not smoke else 20)
    info.update({"N_vision_windows": int(N), "N_eval_windows": int(NE), "smoke": smoke})
    need_free(out, (N * (4 * 40 + 2 * 160) * 1024) / 2 ** 30 + 0.2, "feature memmap")
    mm = {n: np.lib.format.open_memmap(f"{out}/feat/{n}.npy", mode="w+", dtype=np.uint8, shape=(N, SHAPES[g][0], SHAPES[g][1], 1024))
          for n, g in (("X4", "X4"), ("X8", "X8"))}
    ev_mm = {n: np.lib.format.open_memmap(f"{out}/feat/EV{n[1]}.npy", mode="w+", dtype=np.uint8, shape=(NE, SHAPES[n][0], SHAPES[n][1], 1024)) for n in ("X4", "X8")}
    yaw = np.zeros(N, np.float32)
    by = {}
    for k, i in enumerate(sidx):
        by.setdefault(int(tr["clip"][i]), []).append(k)
    t0 = time.time()
    for ci, (c, ks) in enumerate(by.items()):
        z = torch.load(RV.clip_path(TRAIN_CACHE, tr_clips[c]), map_location="cpu", mmap=True, weights_only=False)
        for k in ks:
            t = int(tr["t"][sidx[k]])
            mm["X4"][k] = read_window(z["pool4x10"], t, "X4").view(torch.uint8).numpy()
            mm["X8"][k] = read_window(z["pool8x20"], t, "X8").view(torch.uint8).numpy()
            yaw[k] = float(z["yaw_rate"][t])
        if ci % 50 == 0:
            log(f"extract train clip {ci}/{len(by)} {time.time() - t0:.0f}s")
    byE = {}
    for k in range(NE):
        byE.setdefault(int(ev["ci"][k]), []).append(k)
    for c, ks in byE.items():
        z = torch.load(RV.clip_path(EVAL_CACHE, ev_clips[c]), map_location="cpu", mmap=True, weights_only=False)
        for k in ks:
            t = int(ev["t"][k])
            ev_mm["X4"][k] = read_window(z["pool4x10"], t, "X4").view(torch.uint8).numpy()
            ev_mm["X8"][k] = read_window(z["pool8x20"], t, "X8").view(torch.uint8).numpy()
    for m in list(mm.values()) + list(ev_mm.values()):
        m.flush()
    ne = slice(0, NE)
    np.savez(out + "/feat/meta.npz", tr_clip=tr["clip"], tr_t=tr["t"], tr_y=tr["y"], tr_X=tr["X"], sidx=sidx, yaw=yaw,
             ev_ci=ev["ci"][ne], ev_t=ev["t"][ne], ev_y=ev["y"][ne], ev_eid=ev["eid"][ne], ev_X=ev["X"][ne], tr_clip_ids=np.array(tr_clips), ev_clip_ids=np.array(ev_clips))
    json.dump(info, open(out + "/windows_info.json", "w"), indent=1)
    log("EXTRACT", json.dumps(info), f"{time.time() - t0:.0f}s")


# ------------------------------------------------------------------ probe
def make_net():
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self, grid, n_out, ed_dim, seed):
            super().__init__()
            torch.manual_seed(seed)
            F, P = SHAPES[grid]
            self.ln, self.lin = nn.LayerNorm(1024), nn.Linear(1024, DIM)
            self.pos, self.time = nn.Parameter(torch.zeros(P, DIM)), nn.Parameter(torch.zeros(F, DIM))
            self.n1 = nn.LayerNorm(DIM)
            self.q = nn.Parameter(torch.randn(1, 1, DIM) * 0.02)
            self.att = nn.MultiheadAttention(DIM, 4, batch_first=True)
            self.n2 = nn.LayerNorm(DIM)
            self.head = nn.Sequential(nn.Linear(DIM + ed_dim, 128), nn.GELU(), nn.Linear(128, n_out))

        def forward(self, f8, ed=None):
            tok = (self.lin(self.ln(f8.float())) + self.pos[None, None] + self.time[None, :, None]).flatten(1, 2)
            x = self.n1(tok)
            o, _ = self.att(self.q.expand(x.shape[0], -1, -1), x, x, need_weights=False)
            h = self.n2(o[:, 0])
            if ed is not None:
                h = torch.cat([h, ed], 1)
            return self.head(h)
    return Net


def to_f8(u8):
    import torch
    return u8.view(torch.float8_e4m3fn)


def fit_arm(grid, F, y, fit_idx, hold_idx, pred_sets, n_out, ED, seed, log):
    """F: uint8 tensor [N,f,p,1024]; y: np labels (already permuted for shuffled arms); ED: float tensor [N, ED_DIM] or None.
    Train on fit_idx for EPOCHS epochs (cosine); at epochs CKPTS score hold_idx and every pred set.  The epoch is chosen by hold accuracy (ties -> fewer), no refit.
    pred_sets: {name: (F_tensor, ED_tensor_or_None, idx array)}.  -> dict(E, hold{E}, logits{name})"""
    import torch
    import torch.nn as nn
    Net = make_net()
    net = Net(grid, n_out, ED_DIM if ED is not None else 0, seed)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * ((len(fit_idx) + BS - 1) // BS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, steps - 1)))))
    rng = np.random.RandomState(seed)
    yt = torch.from_numpy(np.asarray(y, np.int64))

    def predict(Fx, EDx, idx):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                ix = torch.from_numpy(idx[i:i + 256])
                out.append(net(to_f8(Fx[ix]), None if EDx is None else EDx[ix]).float().numpy())
        net.train()
        return np.concatenate(out) if out else np.zeros((0, n_out), np.float32)
    res = {"hold": {}, "logits": {}}
    for ep in range(1, EPOCHS + 1):
        perm = rng.permutation(fit_idx)
        tot, t0 = 0.0, time.time()
        for i in range(0, len(perm), BS):
            ix = torch.from_numpy(perm[i:i + BS])
            loss = nn.functional.cross_entropy(net(to_f8(F[ix]), None if ED is None else ED[ix]), yt[ix])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(ix)
        log(f"      ep {ep}/{EPOCHS} loss {tot / len(perm):.4f} {time.time() - t0:.0f}s")
        if ep in CKPTS:
            hl = predict(F, ED, hold_idx)
            res["hold"][ep] = float((hl.argmax(1) == np.asarray(y)[hold_idx]).mean()) if len(hold_idx) else 0.0
            res["logits"][ep] = {n: predict(Fx, EDx, idx) for n, (Fx, EDx, idx) in pred_sets.items()}
    E = max(CKPTS, key=lambda e: (res["hold"][e], -e))
    return {"E": E, "hold": res["hold"], "logits": res["logits"][E]}


def timing_cls(y):
    return RV.timing_cls(y)


def cells(pred, y):
    turn = np.isin(y, (1, 2))
    return {"timing3_accuracy": float((timing_cls(pred) == timing_cls(y)).mean()),
            "side_given_turn": {"n": int(turn.sum()), "accuracy": float((pred[turn] == y[turn]).mean()) if turn.any() else None}}


# ------------------------------------------------------------------ stage 2: cross-validation
def cv(out, smoke=False, log=print):
    import torch
    torch.set_num_threads(int(os.environ.get("RBJOINT_THREADS", "8")))
    sys.path.insert(0, S1 + "/navrb2/read/tree/taniteval")
    from taniteval import ci as CI
    t0 = time.time()
    M = np.load(out + "/feat/meta.npz", allow_pickle=False)
    clip, sidx = M["tr_clip"], M["sidx"]
    y_all, X_all = M["tr_y"], M["tr_X"]
    yv, clip_v, Xv = y_all[sidx], clip[sidx], X_all[sidx]
    yawv = yaw_class(M["yaw"])
    ids = M["tr_clip_ids"]
    fold_all = np.array([fold_of(ids[c]) for c in clip])
    fold_v = fold_all[sidx]
    N = len(sidx)
    EDv = torch.from_numpy(ed_matrix(Xv))
    res_dir = out + "/cv"
    os.makedirs(res_dir, exist_ok=True)
    disjoint = {f"fold{k}": len(set(clip[fold_all == k]) & set(clip[fold_all != k])) for k in range(NFOLD)}
    fold_clips = {f"fold{k}": int(len(set(clip[fold_all == k]))) for k in range(NFOLD)}
    # ---- HGB arms (S, ED, shuffled ED): fit on ALL fold-train windows, score the strided held-out windows
    oof = {n: np.full(N, -1, np.int64) for n in ("S", "ED", "SHUF_ED")}
    S_const = []
    for k in range(NFOLD):
        trm, tem = fold_all != k, fold_v == k
        ytr = y_all[trm]
        for n, cols, yy in (("S", RD.FEATS["S"], ytr), ("ED", RD.FEATS["ED"], ytr), ("SHUF_ED", RD.FEATS["ED"], shuffle_labels(ytr, k))):
            oof[n][tem] = RD.fit_predict(X_all[trm], yy, Xv[tem], cols)
        S_const.append({"fold": k, "train_majority": int(np.bincount(ytr, minlength=5).argmax()), "S_pred_unique": sorted(set(oof["S"][tem].tolist()))})
    log("HGB done", f"{time.time() - t0:.0f}s")
    oofv = {n: None for n in ARMS_VIS}
    yaw_maj_pred = np.zeros(N, np.int64)

    def arm_target(a):
        return yawv if ARMS_VIS[a][1] == "yaw" else yv

    # ---- vision arms, in feature-file order so only one big tensor is resident
    ev_logits_store = {}
    for fname in ("X4", "X8"):
        FT = torch.from_numpy(np.load(f"{out}/feat/{fname}.npy"))
        EVF = torch.from_numpy(np.load(f"{out}/feat/EV{fname[1]}.npy"))
        EVED = torch.from_numpy(ed_matrix(M["ev_X"]))
        for a, (fn, tgt, use_ed, shuf) in ARMS_VIS.items():
            if fn != fname:
                continue
            n_out = 3 if tgt == "yaw" else 5
            y = arm_target(a)
            oofl = np.full(N, -1, np.int64)
            for k in range(NFOLD):
                p = f"{res_dir}/{a}_f{k}.npz"
                if os.path.exists(p):
                    z = np.load(p)
                    oofl[np.where(fold_v == k)[0]] = z["pred"]
                    continue
                tr_i, te_i = np.where(fold_v != k)[0], np.where(fold_v == k)[0]
                yf = y.copy()
                if shuf:
                    yf[tr_i] = shuffle_labels(y[tr_i], k)
                inner = np.array([fold_of(ids[clip_v[i]], salt="rbjoint-inner", k=5) == 0 for i in tr_i])
                fit_i, hold_i = tr_i[~inner], tr_i[inner]
                log(f"  {a} fold {k}: fit {len(fit_i)} hold {len(hold_i)} test {len(te_i)}")
                r = fit_arm(fname, FT, yf, fit_i, hold_i, {"test": (FT, EDv if use_ed else None, te_i)}, n_out, EDv if use_ed else None, 0, log)
                pred = r["logits"]["test"].argmax(1)
                oofl[te_i] = pred
                np.savez(p, pred=pred, E=r["E"], hold=json.dumps({str(e): v for e, v in r["hold"].items()}))
                log(f"  {a} fold {k}: E {r['E']} hold {r['hold']} test acc {(pred == y[te_i]).mean():.4f} {time.time() - t0:.0f}s")
            oofv[a] = oofl
            # eval-202: fit on all strided train windows, inner holdout for E, score eval windows (tac arms only, not shuffled)
            if tgt == "tac" and not shuf:
                p = f"{res_dir}/{a}_eval.npz"
                if os.path.exists(p):
                    ev_logits_store[a] = np.load(p)["logits"]
                else:
                    inner = np.array([fold_of(ids[clip_v[i]], salt="rbjoint-inner", k=5) == 0 for i in range(N)])
                    fit_i, hold_i = np.where(~inner)[0], np.where(inner)[0]
                    log(f"  {a} eval fit: fit {len(fit_i)} hold {len(hold_i)}")
                    r = fit_arm(fname, FT, y, fit_i, hold_i, {"ev": (EVF, EVED if use_ed else None, np.arange(len(EVF)))}, 5, EDv if use_ed else None, 0, log)
                    ev_logits_store[a] = r["logits"]["ev"]
                    np.savez(p, logits=r["logits"]["ev"], E=r["E"])
        del FT, EVF
    # ---- scoring
    assert all((v >= 0).all() for v in oof.values()) and all((v >= 0).all() for v in oofv.values() if v is not None)
    pred = {"S": oof["S"], "ED": oof["ED"], "RAW4": oofv["RAW4"], "RAW8": oofv["RAW8"], "VED8": oofv["VED8"], "SHUF_ED": oof["SHUF_ED"],
            "SHUF_RAW4": oofv["SHUF_RAW4"], "SHUF_VED8": oofv["SHUF_VED8"]}
    corr = {n: (p == yv).astype(np.float64) for n, p in pred.items()}
    arms = {}
    for n in pred:
        arms[n] = {"accuracy": RD.level(CI, corr[n], clip_v), **cells(pred[n], yv), "pred_counts": np.bincount(pred[n], minlength=5).tolist()}
    P = {}
    for a, b in (("VED8", "S"), ("VED8", "ED"), ("VED8", "RAW8"), ("RAW8", "S"), ("RAW8", "RAW4"), ("RAW4", "S"), ("ED", "S"), ("RAW4", "ED"), ("RAW8", "ED"), ("VED8", "RAW4")):
        P[f"{a}-{b}"] = RD.paired(CI, corr[a], corr[b], clip_v)
    # secondary (no bar): timing and side-given-turn paired vs S
    sec = {}
    for n in ("ED", "RAW4", "RAW8", "VED8"):
        ct = (timing_cls(pred[n]) == timing_cls(yv)).astype(float) - (timing_cls(pred["S"]) == timing_cls(yv)).astype(float)
        sec[n] = {"timing_gain_over_S": float(ct.mean())}
        turn = np.isin(yv, (1, 2))
        if turn.any():
            sec[n]["side_given_turn_acc"] = float((pred[n][turn] == yv[turn]).mean())
    # full-window S/ED (all 24k windows) beside, same folds
    # controls
    maj_all = int(np.bincount(yv, minlength=5).argmax())
    kconst = {"majority_class": maj_all, "S_accuracy": float(corr["S"].mean()), "oof_share_of_majority": float((yv == maj_all).mean()), "folds": S_const}
    kconst["pass"] = bool(all(f["S_pred_unique"] == [f["train_majority"]] for f in S_const) and all(f["train_majority"] == maj_all for f in S_const)
                          and kconst["S_accuracy"] == kconst["oof_share_of_majority"])
    shuf = {n: RD.paired(CI, corr[n], corr["S"], clip_v) for n in ("SHUF_ED", "SHUF_RAW4", "SHUF_VED8")}
    shuf_pass = all(d["delta"] < MARGIN for d in shuf.values())
    yaw = {}
    for a, base in (("YAW4", "RAW4"), ("YAW8", "RAW8")):
        pr = oofv[a]
        cm = np.zeros(N)
        for k in range(NFOLD):
            tem = fold_v == k
            maj = int(np.bincount(yawv[fold_v != k], minlength=3).argmax())
            cm[tem] = (yawv[tem] == maj).astype(float)
        cr = (pr == yawv).astype(float)
        g = RD.paired(CI, cr, cm, clip_v)
        yaw[a] = {"accuracy": RD.level(CI, cr, clip_v), "majority_rate": float(cm.mean()), "gain_over_majority": g, "pass": pos_yaw_pass(g),
                  "class_counts_LstraightR": np.bincount(yawv, minlength=3).tolist()}
    pos_ok = all(v["pass"] for v in yaw.values())
    validity = {"POS-YAW": {"pass": pos_ok, **yaw}, "R-SHUF-TRAIN": {"pass": bool(shuf_pass), **shuf}, "K-CONST": kconst,
                "fold_clip_disjointness": {"pass": all(v == 0 for v in disjoint.values()), "overlaps": disjoint, "clips_per_fold": fold_clips}}
    validity_ok = pos_ok and shuf_pass and kconst["pass"] and validity["fold_clip_disjointness"]["pass"]
    held, nxt = labels_rule(P, validity_ok)
    # eval-202 beside (no label)
    ev = {}
    ye, ee = M["ev_y"], M["ev_eid"]
    pe = {"S": RD.fit_predict(X_all, y_all, M["ev_X"], RD.FEATS["S"]), "ED": RD.fit_predict(X_all, y_all, M["ev_X"], RD.FEATS["ED"])}
    for a in ("RAW4", "RAW8", "VED8"):
        if a in ev_logits_store:
            pe[a] = ev_logits_store[a].argmax(1)
    ce = {n: (p == ye).astype(float) for n, p in pe.items()}
    for n in pe:
        ev[n] = {"accuracy": RD.level(CI, ce[n], ee), **cells(pe[n], ye)}
        if n != "S":
            ev[n]["gain_over_S_paired"] = RD.paired(CI, ce[n], ce["S"], ee)
    res = {"tool": "rbjoint.py", "card": "PREREG_RBJOINT.md", "card_sha256": (sha(CARD) if os.path.exists(CARD) else "absent"), "smoke": smoke,
           "evidence_class": "MEASURED (ours; frozen cached DINOv3 pooled features, small probe, 5-fold clip-grouped CV, CPU, zero model training)",
           "tier": "T0 decodability (NOT driving performance)", "estimator": "taniteval.ci.paired_episode_cluster_bootstrap on per-window correctness, B=2000, seed 0, episodes = train clips; answers 'another draw of EPISODES' only",
           "margin": MARGIN, "n_windows_oof": int(N), "n_episodes": int(len(set(clip_v))), "arms": arms, "paired": P, "secondary": sec, "validity": validity,
           "validity_ok": bool(validity_ok), "labels_held": held, "next_lever_from": nxt, "eval202_beside_no_label": ev, "elapsed_s": round(time.time() - t0, 1)}
    with open(out + "/RESULT_RBJOINT.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=HERE + "/out")
    ap.add_argument("--skip-card-check", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    LOG = open(a.out + "/rbjoint_log.txt", "a", encoding="utf-8")

    def log(*x):
        s = time.strftime("%H:%M:%S ") + " ".join(str(v) for v in x)
        print(s, flush=True)
        LOG.write(s + "\n")
        LOG.flush()
    done = a.out + "/RBJOINT.done"
    try:
        if not a.smoke:
            assert sha(CARD) == open(HERE + "/PREREG_RBJOINT.sha256").read().split()[0], "card changed after hashing"
        if a.stage in ("extract", "all"):
            extract(a.out, a.smoke, log)
        if a.stage in ("cv", "all"):
            r = cv(a.out, a.smoke, log)
            log("VERDICT", json.dumps({"labels": r["labels_held"], "validity_ok": r["validity_ok"], "elapsed_s": r["elapsed_s"]}))
        open(done, "w").write("DONE\n" + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    except BaseException:
        tb = traceback.format_exc()
        log(tb)
        open(done, "w").write("FAIL\n" + tb)
        raise


if __name__ == "__main__":
    main()
